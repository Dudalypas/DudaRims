from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from retrieval_experiment_core import (
    TOP_KS,
    aggregate_classification_fold_metrics,
    aggregate_classification_per_class_metrics,
    aggregate_fold_metrics,
    build_classification_metrics,
    build_reference_db,
    evaluate_retrieval,
    extract_embedding_model,
    infer_embeddings,
    read_records_csv,
    save_json,
    train_one_run,
    write_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "trained_cropped_classifier" / "retrieval_experiment_v5"
DEFAULT_FOLDS_CSV = DEFAULT_EXPERIMENT_ROOT / "folds" / "dev_with_folds.csv"
BASELINE_VAL = {"recall@1": 0.5007, "recall@3": 0.8027, "recall@5": 0.8912}


@dataclass
class CVExperimentResult:
    name: str
    description: str
    config: dict[str, Any]
    output_dir: Path
    cv_summary_path: Path | None
    recall_at_1_mean: float | None
    recall_at_1_std: float | None
    recall_at_3_mean: float | None
    recall_at_3_std: float | None
    recall_at_5_mean: float | None
    recall_at_5_std: float | None
    success: bool
    macro_precision_mean: float | None = None
    macro_precision_std: float | None = None
    macro_recall_mean: float | None = None
    macro_recall_std: float | None = None
    macro_f1_mean: float | None = None
    macro_f1_std: float | None = None
    weighted_precision_mean: float | None = None
    weighted_precision_std: float | None = None
    weighted_recall_mean: float | None = None
    weighted_recall_std: float | None = None
    weighted_f1_mean: float | None = None
    weighted_f1_std: float | None = None
    error_msg: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5-fold cross-validation on TRAIN+VAL data"
    )
    parser.add_argument("--folds-csv", type=str, default=str(DEFAULT_FOLDS_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_EXPERIMENT_ROOT / "cv"))

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=5)

    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--triplet-weight", type=float, default=0.5)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--reference-mode", choices=["all", "limited"], default="limited")
    parser.add_argument("--max-refs-per-class", type=int, default=20)
    parser.add_argument("--retrieval-modes", nargs="+", choices=["centroid", "multi_max", "multi_topn_avg"], default=["centroid", "multi_max", "multi_topn_avg"])
    parser.add_argument("--topn", type=int, default=1)

    return parser.parse_args()


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def compute_top1_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, float]:
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = [str(label) for label in labels]

    if not y_true:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
        }

    return {
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def merge_aggregate_rows(
    recall_rows: list[dict[str, Any]],
    classification_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    classification_by_key = {
        (str(row["retrieval_mode"]), int(row["topn"])): row for row in classification_rows
    }

    merged_rows: list[dict[str, Any]] = []
    for row in recall_rows:
        key = (str(row["retrieval_mode"]), int(row["topn"]))
        merged = dict(row)
        classification_row = classification_by_key.get(key)
        if classification_row is not None:
            merged.update(classification_row)
        merged_rows.append(merged)
    return merged_rows


def format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "n/a"
    return f"{mean:.4f}±{std:.4f}"


def serialize_result(result: CVExperimentResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["output_dir"] = str(result.output_dir)
    payload["cv_summary_path"] = None if result.cv_summary_path is None else str(result.cv_summary_path)
    return payload


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_records_csv(Path(args.folds_csv))
    fold_ids = sorted({r.fold for r in records if r.fold is not None})
    if not fold_ids:
        raise RuntimeError(f"No folds found in {args.folds_csv}")

    fold_metric_rows: list[dict[str, object]] = []
    fold_classification_rows: list[dict[str, object]] = []
    fold_per_class_classification_rows: list[dict[str, object]] = []
    fold_confusion_rows: list[dict[str, object]] = []
    top1_prediction_rows: list[dict[str, object]] = []
    top1_compare_rows: list[dict[str, object]] = []
    per_class_recall_rows: list[dict[str, object]] = []

    for fold in fold_ids:
        train_records = [r for r in records if r.fold != fold]
        val_records = [r for r in records if r.fold == fold]

        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        model, val_metrics, _ = train_one_run(
            train_records=train_records,
            val_records=val_records,
            output_dir=fold_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed + int(fold),
            embedding_dim=args.embedding_dim,
            learning_rate=args.learning_rate,
            triplet_margin=args.triplet_margin,
            triplet_weight=args.triplet_weight,
            ce_weight=args.ce_weight,
            dropout=args.dropout,
            early_stopping_patience=args.early_stopping_patience,
            restore_best_weights=True,
        )

        embedding_model = extract_embedding_model(model)

        train_emb, train_labels, _ = infer_embeddings(
            embedding_model=embedding_model,
            records=train_records,
            img_size=args.img_size,
            batch_size=args.batch_size,
        )
        val_emb, val_labels, _ = infer_embeddings(
            embedding_model=embedding_model,
            records=val_records,
            img_size=args.img_size,
            batch_size=args.batch_size,
        )

        ref_db = build_reference_db(
            embeddings=train_emb,
            labels=train_labels,
            mode=args.reference_mode,
            max_refs_per_class=args.max_refs_per_class,
            seed=args.seed + int(fold),
        )

        fold_eval_summary = {
            "fold": fold,
            "train_size": len(train_records),
            "val_size": len(val_records),
            "keras_val": val_metrics,
            "retrieval": [],
        }

        for mode in args.retrieval_modes:
            topn = args.topn if mode == "multi_topn_avg" else 1
            summary, per_class, predictions = evaluate_retrieval(
                query_embeddings=val_emb,
                query_labels=val_labels,
                reference_db=ref_db,
                retrieval_mode=mode,
                topn=topn,
            )
            y_true = [str(item["true_label"]) for item in predictions]
            y_pred = [str(item["top1_predicted_label"]) for item in predictions]
            classification_metrics = compute_top1_classification_metrics(
                y_true,
                y_pred,
                labels=sorted(ref_db.keys()),
            )
            classification = build_classification_metrics(predictions, labels=sorted(ref_db.keys()))

            row = {
                "fold": int(fold),
                "retrieval_mode": mode,
                "topn": int(topn),
                "recall@1": float(summary["recall@1"]),
                "recall@3": float(summary["recall@3"]),
                "recall@5": float(summary["recall@5"]),
                "macro_precision": float(classification_metrics["macro_precision"]),
                "macro_recall": float(classification_metrics["macro_recall"]),
                "macro_f1": float(classification_metrics["macro_f1"]),
                "weighted_precision": float(classification_metrics["weighted_precision"]),
                "weighted_recall": float(classification_metrics["weighted_recall"]),
                "weighted_f1": float(classification_metrics["weighted_f1"]),
            }
            fold_metric_rows.append(row)
            fold_eval_summary["retrieval"].append(row)

            for item in per_class:
                per_class_recall_rows.append(
                    {
                        "fold": int(fold),
                        "retrieval_mode": mode,
                        "topn": int(topn),
                        **item,
                    }
                )

            for item in predictions:
                top1_prediction_rows.append(
                    {
                        "fold": int(fold),
                        "retrieval_mode": mode,
                        "topn": int(topn),
                        **item,
                    }
                )
                top1_compare_rows.append(
                    {
                        "fold": int(fold),
                        "retrieval_mode": mode,
                        "topn": int(topn),
                        "true_label": item["true_label"],
                        "pred_label": item["top1_predicted_label"],
                        "count": 1,
                    }
                )

            report_rows = classification["classification_report_rows"]
            for item in report_rows:
                fold_classification_rows.append(
                    {
                        "fold": int(fold),
                        "retrieval_mode": mode,
                        "topn": int(topn),
                        **item,
                    }
                )

            per_class_classification_rows = classification["per_class_metrics_rows"]
            for item in per_class_classification_rows:
                fold_per_class_classification_rows.append(
                    {
                        "fold": int(fold),
                        "retrieval_mode": mode,
                        "topn": int(topn),
                        **item,
                    }
                )

            cm = classification["confusion_matrix"]
            labels = classification["labels"]
            for i, true_label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    fold_confusion_rows.append(
                        {
                            "fold": int(fold),
                            "retrieval_mode": mode,
                            "topn": int(topn),
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "count": int(cm[i, j]),
                        }
                    )

        save_json(fold_eval_summary, fold_dir / "fold_eval_summary.json")
        write_csv(fold_dir / "classification_report.csv", [r for r in fold_classification_rows if r["fold"] == int(fold)], fieldnames=["fold", "retrieval_mode", "topn", "label", "precision", "recall", "f1_score", "support"])
        write_csv(fold_dir / "confusion_matrix.csv", [r for r in fold_confusion_rows if r["fold"] == int(fold)], fieldnames=["fold", "retrieval_mode", "topn", "true_label", "pred_label", "count"])
        write_csv(fold_dir / "top1_predictions.csv", [r for r in top1_prediction_rows if r["fold"] == int(fold)], fieldnames=["fold", "retrieval_mode", "topn", "query_index", "true_label", "top1_predicted_label", "top_k_labels", "top_k_scores"])

    aggregate = aggregate_fold_metrics(fold_metric_rows)
    classification_aggregate = aggregate_classification_fold_metrics(fold_metric_rows)
    aggregate_per_class = aggregate_classification_per_class_metrics(fold_per_class_classification_rows)
    merged_aggregate = merge_aggregate_rows(aggregate, classification_aggregate)

    centroid_row = next(
        (row for row in merged_aggregate if str(row["retrieval_mode"]).lower() == "centroid"),
        merged_aggregate[0] if merged_aggregate else None,
    )
    if centroid_row is None:
        raise RuntimeError("No CV aggregate rows were produced.")

    experiment_result = CVExperimentResult(
        name="retrieval_experiment_cv",
        description="5-fold CV retrieval experiment",
        config={
            "folds_csv": str(args.folds_csv),
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "early_stopping_patience": args.early_stopping_patience,
            "restore_best_weights": True,
            "embedding_dim": args.embedding_dim,
            "learning_rate": args.learning_rate,
            "triplet_margin": args.triplet_margin,
            "triplet_weight": args.triplet_weight,
            "ce_weight": args.ce_weight,
            "dropout": args.dropout,
            "reference_mode": args.reference_mode,
            "max_refs_per_class": args.max_refs_per_class,
            "retrieval_modes": list(args.retrieval_modes),
            "topn": args.topn,
        },
        output_dir=output_dir,
        cv_summary_path=output_dir / "cv_summary.json",
        recall_at_1_mean=float(centroid_row["mean_recall@1"]),
        recall_at_1_std=float(centroid_row["std_recall@1"]),
        recall_at_3_mean=float(centroid_row["mean_recall@3"]),
        recall_at_3_std=float(centroid_row["std_recall@3"]),
        recall_at_5_mean=float(centroid_row["mean_recall@5"]),
        recall_at_5_std=float(centroid_row["std_recall@5"]),
        success=True,
        macro_precision_mean=float(centroid_row["macro_precision_mean"]),
        macro_precision_std=float(centroid_row["macro_precision_std"]),
        macro_recall_mean=float(centroid_row["macro_recall_mean"]),
        macro_recall_std=float(centroid_row["macro_recall_std"]),
        macro_f1_mean=float(centroid_row["macro_f1_mean"]),
        macro_f1_std=float(centroid_row["macro_f1_std"]),
        weighted_precision_mean=float(centroid_row["weighted_precision_mean"]),
        weighted_precision_std=float(centroid_row["weighted_precision_std"]),
        weighted_recall_mean=float(centroid_row["weighted_recall_mean"]),
        weighted_recall_std=float(centroid_row["weighted_recall_std"]),
        weighted_f1_mean=float(centroid_row["weighted_f1_mean"]),
        weighted_f1_std=float(centroid_row["weighted_f1_std"]),
        error_msg=None,
    )
    result_payload = serialize_result(experiment_result)

    baseline_compare_rows: list[dict[str, object]] = []
    for row in merged_aggregate:
        for metric in ("recall@1", "recall@3", "recall@5"):
            mean_key = f"mean_{metric}"
            baseline_compare_rows.append(
                {
                    "retrieval_mode": row["retrieval_mode"],
                    "topn": row["topn"],
                    "metric": metric,
                    "cv_mean": row[mean_key],
                    "baseline_val": BASELINE_VAL[metric],
                    "delta": float(row[mean_key]) - float(BASELINE_VAL[metric]),
                }
            )

    top1_compare_aggregated: dict[tuple[int, str, int, str, str], int] = defaultdict(int)
    for row in top1_compare_rows:
        key = (
            int(row["fold"]),
            str(row["retrieval_mode"]),
            int(row["topn"]),
            str(row["true_label"]),
            str(row["pred_label"]),
        )
        top1_compare_aggregated[key] += int(row["count"])
    top1_compare_rows_out = [
        {
            "fold": fold,
            "retrieval_mode": mode,
            "topn": topn,
            "true_label": true_label,
            "pred_label": pred_label,
            "count": count,
        }
        for (fold, mode, topn, true_label, pred_label), count in sorted(top1_compare_aggregated.items())
    ]

    write_csv(output_dir / "cv_fold_metrics.csv", fold_metric_rows)
    write_csv(output_dir / "cv_per_class_metrics.csv", per_class_recall_rows)
    write_csv(output_dir / "cv_top1_compare.csv", top1_compare_rows_out)
    write_csv(output_dir / "cv_aggregate_metrics.csv", merged_aggregate)
    write_csv(output_dir / "cv_baseline_comparison.csv", baseline_compare_rows)
    write_csv(output_dir / "per_fold_classification_report.csv", fold_classification_rows)
    write_csv(output_dir / "per_fold_confusion_matrix.csv", fold_confusion_rows)
    write_csv(output_dir / "aggregate_classification_metrics.csv", classification_aggregate)
    write_csv(output_dir / "aggregate_per_class_metrics.csv", aggregate_per_class)
    write_csv(output_dir / "top1_predictions.csv", top1_prediction_rows)

    summary = {
        **result_payload,
        "top_ks": list(TOP_KS),
        "folds_csv": args.folds_csv,
        "settings": {
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "early_stopping_patience": args.early_stopping_patience,
            "restore_best_weights": True,
            "embedding_dim": args.embedding_dim,
            "learning_rate": args.learning_rate,
            "triplet_margin": args.triplet_margin,
            "triplet_weight": args.triplet_weight,
            "ce_weight": args.ce_weight,
            "dropout": args.dropout,
            "reference_mode": args.reference_mode,
            "max_refs_per_class": args.max_refs_per_class,
            "retrieval_modes": args.retrieval_modes,
            "topn": args.topn,
        },
        "aggregate": merged_aggregate,
        "classification_aggregate": classification_aggregate,
        "baseline_val": BASELINE_VAL,
        "baseline_comparison_csv": str(output_dir / "cv_baseline_comparison.csv"),
        "artifacts": {
            "cv_fold_metrics_csv": str(output_dir / "cv_fold_metrics.csv"),
            "cv_per_class_metrics_csv": str(output_dir / "cv_per_class_metrics.csv"),
            "cv_top1_compare_csv": str(output_dir / "cv_top1_compare.csv"),
            "cv_aggregate_metrics_csv": str(output_dir / "cv_aggregate_metrics.csv"),
            "cv_baseline_comparison_csv": str(output_dir / "cv_baseline_comparison.csv"),
            "per_fold_classification_report_csv": str(output_dir / "per_fold_classification_report.csv"),
            "per_fold_confusion_matrix_csv": str(output_dir / "per_fold_confusion_matrix.csv"),
            "aggregate_classification_metrics_csv": str(output_dir / "aggregate_classification_metrics.csv"),
            "aggregate_per_class_metrics_csv": str(output_dir / "aggregate_per_class_metrics.csv"),
            "top1_predictions_csv": str(output_dir / "top1_predictions.csv"),
        },
    }
    save_json(summary, output_dir / "cv_summary.json")

    print("[cv] fold metrics:", output_dir / "cv_fold_metrics.csv")
    print("[cv] aggregate:", output_dir / "cv_aggregate_metrics.csv")
    print("[cv] class aggregate:", output_dir / "aggregate_classification_metrics.csv")
    print("[cv] summary:", output_dir / "cv_summary.json")
    print()
    print("[cv] summary (mean±std):")
    for row in merged_aggregate:
        print(
            f"  {row['retrieval_mode']} topn={row['topn']}: "
            f"recall@1={format_mean_std(row['mean_recall@1'], row['std_recall@1'])}, "
            f"recall@3={format_mean_std(row['mean_recall@3'], row['std_recall@3'])}, "
            f"recall@5={format_mean_std(row['mean_recall@5'], row['std_recall@5'])}, "
            f"macro_precision={format_mean_std(row['macro_precision_mean'], row['macro_precision_std'])}, "
            f"macro_recall={format_mean_std(row['macro_recall_mean'], row['macro_recall_std'])}, "
            f"macro_f1={format_mean_std(row['macro_f1_mean'], row['macro_f1_std'])}, "
            f"weighted_precision={format_mean_std(row['weighted_precision_mean'], row['weighted_precision_std'])}, "
            f"weighted_recall={format_mean_std(row['weighted_recall_mean'], row['weighted_recall_std'])}, "
            f"weighted_f1={format_mean_std(row['weighted_f1_mean'], row['weighted_f1_std'])}"
        )


if __name__ == "__main__":
    main()
