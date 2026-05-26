from __future__ import annotations

import argparse
import random
from pathlib import Path

from retrieval_experiment_core import (
    TOP_KS,
    ImageRecord,
    build_classification_metrics,
    build_reference_db,
    evaluate_retrieval,
    export_embedding_tflite,
    extract_embedding_model,
    infer_embeddings,
    read_records_csv,
    save_json,
    save_reference_json,
    set_global_seed,
    train_one_run,
    write_csv,
    write_records_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "trained_cropped_classifier" / "retrieval_experiment_v3"
DEFAULT_DEV_CSV = DEFAULT_EXPERIMENT_ROOT / "folds" / "dev_with_folds.csv"
DEFAULT_TEST_CSV = DEFAULT_EXPERIMENT_ROOT / "folds" / "test_holdout.csv"

BASELINE_VAL = {"recall@1": 0.5007, "recall@3": 0.8027, "recall@5": 0.8912}
BASELINE_TEST = {"recall@1": 0.5388, "recall@3": 0.7891, "recall@5": 0.8830}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train final retrieval embedding model on full TRAIN+VAL development set, "
            "evaluate once on untouched TEST, and export deployment artifacts."
        )
    )
    parser.add_argument("--dev-csv", type=str, default=str(DEFAULT_DEV_CSV))
    parser.add_argument("--test-csv", type=str, default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_EXPERIMENT_ROOT / "final"))

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--internal-val-ratio", type=float, default=0.1)

    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--triplet-weight", type=float, default=0.5)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--reference-mode", choices=["all", "limited"], default="limited")
    parser.add_argument("--max-refs-per-class", type=int, default=20)

    parser.add_argument("--retrieval-mode", choices=["centroid", "multi_max", "multi_topn_avg"], default="centroid")
    parser.add_argument("--topn", type=int, default=3)

    parser.add_argument(
        "--tflite-output",
        type=str,
        default=str(DEFAULT_EXPERIMENT_ROOT / "artifacts" / "wheel_embedding_experiment_v2_float32.tflite"),
    )
    parser.add_argument(
        "--tflite-fp16-output",
        type=str,
        default=str(DEFAULT_EXPERIMENT_ROOT / "artifacts" / "wheel_embedding_experiment_v2_fp16.tflite"),
    )
    parser.add_argument(
        "--reference-json-output",
        type=str,
        default=str(DEFAULT_EXPERIMENT_ROOT / "artifacts" / "wheel_reference_embeddings_experiment_v2.json"),
    )

    return parser.parse_args()


def split_dev_internal(
    dev_records: list[ImageRecord],
    val_ratio: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if not (0.0 < val_ratio < 0.5):
        raise ValueError("internal val ratio should be in (0, 0.5)")

    set_global_seed(seed)

    by_label: dict[str, list[ImageRecord]] = {}
    for r in dev_records:
        by_label.setdefault(r.label, []).append(r)

    train_rows: list[ImageRecord] = []
    val_rows: list[ImageRecord] = []

    for label in sorted(by_label):
        rows = by_label[label][:]
        rows.sort(key=lambda x: str(x.path))
        label_seed = seed + sum(ord(ch) for ch in label)
        rng = random.Random(label_seed)
        rng.shuffle(rows)

        n = len(rows)
        n_val = max(1, int(round(n * val_ratio))) if n > 1 else 1
        n_val = min(n_val, max(1, n - 1)) if n > 1 else 1

        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])

    return train_rows, val_rows


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_csv_path = Path(args.dev_csv)
    test_csv_path = Path(args.test_csv)
    if not dev_csv_path.exists() or not test_csv_path.exists():
        missing = []
        if not dev_csv_path.exists():
            missing.append(str(dev_csv_path))
        if not test_csv_path.exists():
            missing.append(str(test_csv_path))
        raise FileNotFoundError(
            "Missing retrieval manifest CSV file(s): "
            + ", ".join(missing)
            + "\nRegenerate folds first"
        )

    dev_records = read_records_csv(dev_csv_path)
    test_records = read_records_csv(test_csv_path)

    dev_records = [ImageRecord(path=r.path, label=r.label, source_split=r.source_split) for r in dev_records]
    test_records = [ImageRecord(path=r.path, label=r.label, source_split=r.source_split) for r in test_records]

    train_records, val_records = split_dev_internal(dev_records, val_ratio=args.internal_val_ratio, seed=args.seed)

    write_records_csv(train_records, output_dir / "internal_train.csv")
    write_records_csv(val_records, output_dir / "internal_val.csv")

    model, val_metrics, _ = train_one_run(
        train_records=train_records,
        val_records=val_records,
        output_dir=output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        triplet_margin=args.triplet_margin,
        triplet_weight=args.triplet_weight,
        ce_weight=args.ce_weight,
        dropout=args.dropout,
    )

    embedding_model = extract_embedding_model(model)
    
    dev_emb, dev_labels, _ = infer_embeddings(
        embedding_model=embedding_model,
        records=dev_records,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )
    test_emb, test_labels, _ = infer_embeddings(
        embedding_model=embedding_model,
        records=test_records,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    ref_db = build_reference_db(
        embeddings=dev_emb,
        labels=dev_labels,
        mode=args.reference_mode,
        max_refs_per_class=args.max_refs_per_class,
        seed=args.seed,
    )

    topn = args.topn if args.retrieval_mode == "multi_topn_avg" else 1
    test_summary, per_class_rows, top1_predictions = evaluate_retrieval(
        query_embeddings=test_emb,
        query_labels=test_labels,
        reference_db=ref_db,
        retrieval_mode=args.retrieval_mode,
        topn=topn,
    )
    test_classification = build_classification_metrics(top1_predictions, labels=sorted(ref_db.keys()))

    write_csv(output_dir / "test_per_class_metrics.csv", per_class_rows)

    top1_compare_agg = {}
    for row in top1_predictions:
        key = (row["true_label"], row["top1_predicted_label"])
        top1_compare_agg[key] = top1_compare_agg.get(key, 0) + 1
    top1_compare_rows = [
        {"true_label": true_label, "pred_label": pred_label, "count": count}
        for (true_label, pred_label), count in sorted(top1_compare_agg.items())
    ]
    write_csv(output_dir / "test_top1_compare.csv", top1_compare_rows)
    write_csv(output_dir / "test_classification_report.csv", test_classification["classification_report_rows"])
    write_csv(output_dir / "test_confusion_matrix.csv", [
        {
            "true_label": true_label,
            "pred_label": pred_label,
            "count": int(test_classification["confusion_matrix"][i, j]),
        }
        for i, true_label in enumerate(test_classification["labels"])
        for j, pred_label in enumerate(test_classification["labels"])
    ])
    write_csv(output_dir / "test_top1_predictions.csv", top1_predictions)
    write_csv(output_dir / "test_classification_metrics.csv", [
        {
            "metric": "macro_precision",
            "value": test_classification["aggregate_metrics"]["macro_precision"],
        },
        {
            "metric": "macro_recall",
            "value": test_classification["aggregate_metrics"]["macro_recall"],
        },
        {
            "metric": "macro_f1",
            "value": test_classification["aggregate_metrics"]["macro_f1"],
        },
        {
            "metric": "weighted_precision",
            "value": test_classification["aggregate_metrics"]["weighted_precision"],
        },
        {
            "metric": "weighted_recall",
            "value": test_classification["aggregate_metrics"]["weighted_recall"],
        },
        {
            "metric": "weighted_f1",
            "value": test_classification["aggregate_metrics"]["weighted_f1"],
        },
    ])

    tflite_out = Path(args.tflite_output)
    tflite_fp16_out = Path(args.tflite_fp16_output)
    ref_json_out = Path(args.reference_json_output)

    export_embedding_tflite(embedding_model, tflite_out, fp16=False)
    export_embedding_tflite(embedding_model, tflite_fp16_out, fp16=True)

    save_reference_json(
        output_path=ref_json_out,
        reference_db=ref_db,
        embedding_dim=args.embedding_dim,
        source_split="train+val",
        model_path=str(tflite_out),
        img_size=args.img_size,
        mode=args.reference_mode,
        max_refs_per_class=args.max_refs_per_class,
    )

    baseline_compare_rows = []
    for k in TOP_KS:
        key = f"recall@{k}"
        baseline = BASELINE_TEST[key]
        new_val = float(test_summary[key])
        baseline_compare_rows.append(
            {
                "split": "test",
                "metric": key,
                "baseline": baseline,
                "new_model": new_val,
                "delta": new_val - baseline,
            }
        )
    write_csv(output_dir / "baseline_comparison_test.csv", baseline_compare_rows)

    run_summary = {
        "phase": "final_train_val_plus_test_eval",
        "rules_respected": {
            "test_not_used_for_cv_or_training": True,
            "references_built_from_dev_only": True,
            "app_runtime_assets_not_replaced": True,
        },
        "settings": {
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "internal_val_ratio": args.internal_val_ratio,
            "embedding_dim": args.embedding_dim,
            "learning_rate": args.learning_rate,
            "triplet_margin": args.triplet_margin,
            "triplet_weight": args.triplet_weight,
            "ce_weight": args.ce_weight,
            "dropout": args.dropout,
            "reference_mode": args.reference_mode,
            "max_refs_per_class": args.max_refs_per_class,
            "retrieval_mode": args.retrieval_mode,
            "topn": topn,
        },
        "sizes": {
            "dev_total": len(dev_records),
            "internal_train": len(train_records),
            "internal_val": len(val_records),
            "test_total": len(test_records),
        },
        "keras_val_metrics": val_metrics,
        "test_retrieval": test_summary,
        "test_classification": test_classification["aggregate_metrics"],
        "baseline": {
            "val": BASELINE_VAL,
            "test": BASELINE_TEST,
        },
        "artifacts": {
            "best_keras": str(output_dir / "best.keras"),
            "tflite_float32": str(tflite_out),
            "tflite_fp16": str(tflite_fp16_out),
            "reference_json": str(ref_json_out),
            "test_per_class_metrics_csv": str(output_dir / "test_per_class_metrics.csv"),
            "test_top1_compare_csv": str(output_dir / "test_top1_compare.csv"),
            "test_classification_report_csv": str(output_dir / "test_classification_report.csv"),
            "test_confusion_matrix_csv": str(output_dir / "test_confusion_matrix.csv"),
            "test_top1_predictions_csv": str(output_dir / "test_top1_predictions.csv"),
            "test_classification_metrics_csv": str(output_dir / "test_classification_metrics.csv"),
            "baseline_comparison_csv": str(output_dir / "baseline_comparison_test.csv"),
        },
    }
    save_json(run_summary, output_dir / "final_summary.json")

    print("[final] test:", test_summary)
    print("[final] summary:", output_dir / "final_summary.json")
    print("[final] tflite:", tflite_out)
    print("[final] refs:", ref_json_out)


if __name__ == "__main__":
    main()
