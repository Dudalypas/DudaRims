#!/usr/bin/env python3
"""
Sukuria ir ivertina retrieval modelio abliacijos eksperimentus

Ivestis: bazine konfiguraicija ir vertinimo duomenu failai
Isvestis: rezultatu CSV/JSON failai ir trumpa suvestine
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "trained_cropped_classifier" / "retrieval_ablation_v1"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "PROD_cropped"
DEFAULT_FOLDS_CSV = PROJECT_ROOT / "trained_cropped_classifier" / "retrieval_experiment_v2" / "folds" / "dev_with_folds.csv"

# Bazine konfiguracija
BASELINE_CONFIG = {
    "name": "baseline",
    "description": "Baseline: embedding_dim=256, lr=1e-3, triplet_margin=0.2, dropout=0.2",
    "img_size": 224,
    "batch_size": 32,
    "epochs": 32,
    "seed": 42,
    "early_stopping_patience": 5,
    "embedding_dim": 256,
    "learning_rate": 1e-3,
    "triplet_margin": 0.2,
    "triplet_weight": 0.5,
    "ce_weight": 1.0,
    "dropout": 0.2,
    "reference_mode": "limited",
    "max_refs_per_class": 20,
    "retrieval_mode": "centroid",
    "topn": 1,
}

# Vieno faktoriaus abliacijos variantai (kiekvienas keicia VIENA parametra nuo pradinio lygio)
ABLATION_CONFIGS = [
    {
        "name": "embedding_dim_128",
        "description": "Ablation: embedding_dim=128 (smaller)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "embedding_dim": 128,
    },
    {
        "name": "embedding_dim_512",
        "description": "Ablation: embedding_dim=512 (larger)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "embedding_dim": 512,
    },
    {
        "name": "learning_rate_3e4",
        "description": "Ablation: learning_rate=3e-4 (slower)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "learning_rate": 3e-4,
    },
    {
        "name": "triplet_margin_03",
        "description": "Ablation: triplet_margin=0.3 (larger margin)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "triplet_margin": 0.3,
    },
    {
        "name": "dropout_03",
        "description": "Ablation: dropout=0.3 (more regularization)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "dropout": 0.3,
    },
    {
        "name": "batch_size_16",
        "description": "Ablation: batch_size=16 (smaller batches, sensitivity check)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "batch_size": 16,
    },
    {
        "name": "img_size_192",
        "description": "Ablation: img_size=192 (smaller resolution, sensitivity check)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "img_size": 192,
    },
]

# Smoke testui
QUICK_MODE_CONFIGS = [c for c in [BASELINE_CONFIG] + ABLATION_CONFIGS if c["name"] in ["baseline", "embedding_dim_128", "learning_rate_3e4"]]


@dataclass
class ExperimentResult:
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
        description=(
            "Run ablation variants for the retrieval model"
        )
    )
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT), help="Root folder with split data")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_EXPERIMENT_ROOT), help="Output directory for runs")
    parser.add_argument("--folds-csv", type=str, default=str(DEFAULT_FOLDS_CSV), help="Development folds CSV")
    parser.add_argument("--folds", type=int, default=5, help="Fold count for normal mode")
    parser.add_argument("--epochs", type=int, default=32, help="Max epochs for normal mode")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small smoke test set",
    )
    parser.add_argument("--quick-epochs", type=int, default=2, help="Epoch count for quick test")
    parser.add_argument("--quick-folds", type=int, default=2, help="Fold count for quick test")
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=0.0,
        help="Run timeout in hours, 0 disables it",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing run folders")
    return parser.parse_args()


def extract_cv_metrics(cv_summary_path: Path) -> dict[str, float | None]:
    """Nuskaito centroid metrikas is cv_summary.json"""

    def none_metrics() -> dict[str, float | None]:
        return {
            "recall_at_1_mean": None,
            "recall_at_1_std": None,
            "recall_at_3_mean": None,
            "recall_at_3_std": None,
            "recall_at_5_mean": None,
            "recall_at_5_std": None,
            "macro_precision_mean": None,
            "macro_precision_std": None,
            "macro_recall_mean": None,
            "macro_recall_std": None,
            "macro_f1_mean": None,
            "macro_f1_std": None,
            "weighted_precision_mean": None,
            "weighted_precision_std": None,
            "weighted_recall_mean": None,
            "weighted_recall_std": None,
            "weighted_f1_mean": None,
            "weighted_f1_std": None,
        }

    if not cv_summary_path.exists():
        return none_metrics()

    def pick_value(d: dict[str, Any], candidates: list[str]) -> float | None:
        for c in candidates:
            if c in d and d[c] is not None:
                try:
                    return float(d[c])
                except Exception:
                    continue
        return None

    try:
        payload = json.loads(cv_summary_path.read_text(encoding="utf-8"))
    except Exception:
        payload = None

    def extract_from_rows(rows: list[Any], metric_prefix: str) -> dict[str, float | None] | None:
        if not rows:
            return None
        centroid_row = None
        for r in rows:
            if not isinstance(r, dict):
                continue
            if str(r.get("retrieval_mode", "")).lower() == "centroid":
                centroid_row = r
                break
        if centroid_row is None:
            return None
        if metric_prefix == "recall":
            return {
                "recall_at_1_mean": pick_value(centroid_row, ["recall_at_1_mean", "mean_recall@1", "recall@1_mean", "mean_recall_1"]),
                "recall_at_1_std": pick_value(centroid_row, ["recall_at_1_std", "std_recall@1", "recall@1_std", "std_recall_1"]),
                "recall_at_3_mean": pick_value(centroid_row, ["recall_at_3_mean", "mean_recall@3", "recall@3_mean", "mean_recall_3"]),
                "recall_at_3_std": pick_value(centroid_row, ["recall_at_3_std", "std_recall@3", "recall@3_std", "std_recall_3"]),
                "recall_at_5_mean": pick_value(centroid_row, ["recall_at_5_mean", "mean_recall@5", "recall@5_mean", "mean_recall_5"]),
                "recall_at_5_std": pick_value(centroid_row, ["recall_at_5_std", "std_recall@5", "recall@5_std", "std_recall_5"]),
            }
        return {
            "macro_precision_mean": pick_value(centroid_row, ["macro_precision_mean", "precision_macro_mean", "macro_precision"]),
            "macro_precision_std": pick_value(centroid_row, ["macro_precision_std", "precision_macro_std"]),
            "macro_recall_mean": pick_value(centroid_row, ["macro_recall_mean", "macro_recall"]),
            "macro_recall_std": pick_value(centroid_row, ["macro_recall_std"]),
            "macro_f1_mean": pick_value(centroid_row, ["macro_f1_mean", "macro_f1"]),
            "macro_f1_std": pick_value(centroid_row, ["macro_f1_std"]),
            "weighted_precision_mean": pick_value(centroid_row, ["weighted_precision_mean", "weighted_precision"]),
            "weighted_precision_std": pick_value(centroid_row, ["weighted_precision_std"]),
            "weighted_recall_mean": pick_value(centroid_row, ["weighted_recall_mean", "weighted_recall"]),
            "weighted_recall_std": pick_value(centroid_row, ["weighted_recall_std"]),
            "weighted_f1_mean": pick_value(centroid_row, ["weighted_f1_mean", "weighted_f1"]),
            "weighted_f1_std": pick_value(centroid_row, ["weighted_f1_std"]),
        }

    if isinstance(payload, dict):
        recall_rows = None
        for key in ("aggregate", "aggregate_metrics", "cv_aggregate_metrics", "metrics"):
            val = payload.get(key)
            if isinstance(val, list):
                recall_rows = val
                break
        recall_metrics = extract_from_rows(recall_rows or [], "recall")
        if recall_metrics is not None:
            classification_rows = None
            for key in ("classification_aggregate", "aggregate_classification_metrics", "classification_metrics"):
                val = payload.get(key)
                if isinstance(val, list):
                    classification_rows = val
                    break
            classification_metrics = extract_from_rows(classification_rows or [], "classification")
            if classification_metrics is None:
                classification_metrics = none_metrics()
            recall_metrics.update(classification_metrics)
            return recall_metrics

    csv_path = cv_summary_path.parent / "cv_aggregate_metrics.csv"
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str((row.get("retrieval_mode") or "")).lower() == "centroid":
                        metrics = none_metrics()
                        metrics.update(
                            {
                                "recall_at_1_mean": pick_value(row, ["recall_at_1_mean", "mean_recall@1", "recall@1_mean", "mean_recall_1"]),
                                "recall_at_1_std": pick_value(row, ["recall_at_1_std", "std_recall@1", "recall@1_std", "std_recall_1"]),
                                "recall_at_3_mean": pick_value(row, ["recall_at_3_mean", "mean_recall@3", "recall@3_mean", "mean_recall_3"]),
                                "recall_at_3_std": pick_value(row, ["recall_at_3_std", "std_recall@3", "recall@3_std", "std_recall_3"]),
                                "recall_at_5_mean": pick_value(row, ["recall_at_5_mean", "mean_recall@5", "recall@5_mean", "mean_recall_5"]),
                                "recall_at_5_std": pick_value(row, ["recall_at_5_std", "std_recall@5", "recall@5_std", "std_recall_5"]),
                            }
                        )
                        classification_path = cv_summary_path.parent / "aggregate_classification_metrics.csv"
                        if classification_path.exists():
                            with classification_path.open("r", encoding="utf-8") as cf:
                                class_reader = csv.DictReader(cf)
                                for class_row in class_reader:
                                    if str((class_row.get("retrieval_mode") or "")).lower() == "centroid":
                                        metrics.update(
                                            {
                                                "macro_precision_mean": pick_value(class_row, ["macro_precision_mean"]),
                                                "macro_precision_std": pick_value(class_row, ["macro_precision_std"]),
                                                "macro_recall_mean": pick_value(class_row, ["macro_recall_mean"]),
                                                "macro_recall_std": pick_value(class_row, ["macro_recall_std"]),
                                                "macro_f1_mean": pick_value(class_row, ["macro_f1_mean"]),
                                                "macro_f1_std": pick_value(class_row, ["macro_f1_std"]),
                                                "weighted_precision_mean": pick_value(class_row, ["weighted_precision_mean"]),
                                                "weighted_precision_std": pick_value(class_row, ["weighted_precision_std"]),
                                                "weighted_recall_mean": pick_value(class_row, ["weighted_recall_mean"]),
                                                "weighted_recall_std": pick_value(class_row, ["weighted_recall_std"]),
                                                "weighted_f1_mean": pick_value(class_row, ["weighted_f1_mean"]),
                                                "weighted_f1_std": pick_value(class_row, ["weighted_f1_std"]),
                                            }
                                        )
                                        break
                        return metrics
        except Exception:
            pass

    return none_metrics()


def config_copy(base: dict[str, Any], **updates: Any) -> dict[str, Any]:
    copied = dict(base)
    copied.update(updates)
    return copied


def build_configs(quick: bool, epochs: int, quick_epochs: int) -> list[dict[str, Any]]:
    baseline = config_copy(BASELINE_CONFIG, epochs=quick_epochs if quick else epochs)
    ablations = [config_copy(cfg, epochs=quick_epochs if quick else epochs) for cfg in ABLATION_CONFIGS]
    if quick:
        wanted = {"baseline", "embedding_dim_128", "learning_rate_3e4"}
        return [baseline] + [cfg for cfg in ablations if cfg["name"] in wanted]
    return [baseline] + ablations


def read_fold_ids(folds_csv: Path) -> list[int]:
    fold_ids: set[int] = set()
    with folds_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold_text = (row.get("fold") or "").strip()
            if not fold_text:
                continue
            fold_ids.add(int(fold_text))
    return sorted(fold_ids)


def limit_folds_csv(source_csv: Path, target_csv: Path, max_folds: int, dry_run: bool) -> Path:
    """Atrenka pirmus N fold id i nauja CSV"""
    if max_folds <= 0:
        raise ValueError("max_folds must be >= 1")

    fold_ids = read_fold_ids(source_csv)
    selected = fold_ids[:max_folds]
    if not selected:
        raise RuntimeError(f"No fold ids found in {source_csv}")

    if dry_run:
        print(f"[ablation] dry-run folds={selected}: {target_csv}")
        return target_csv

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    keep = set(selected)
    with source_csv.open("r", encoding="utf-8") as src, target_csv.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise RuntimeError(f"Missing CSV header in {source_csv}")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            fold_text = (row.get("fold") or "").strip()
            if fold_text and int(fold_text) in keep:
                writer.writerow(row)

    return target_csv


def existing_run_metrics(output_dir: Path) -> dict[str, float | None] | None:
    cv_summary_path = output_dir / "cv_summary.json"
    if not cv_summary_path.exists():
        return None

    metrics = extract_cv_metrics(cv_summary_path)
    # Require the three mean metrics to be present to consider this a valid existing run
    if metrics["recall_at_1_mean"] is None or metrics["recall_at_3_mean"] is None or metrics["recall_at_5_mean"] is None:
        return None
    return metrics


def run_cv_for_config(
    config: dict[str, Any],
    output_dir: Path,
    folds_csv: Path,
    cv_script: Path,
    timeout_hours: float = 0.0,
    dry_run: bool = False,
) -> tuple[bool, str | None]:
    """Paleidzia retrieval_experiment_cv.py su duota konfiguraicija"""
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(cv_script),
        "--folds-csv", str(folds_csv),
        "--output-dir", str(output_dir),
        "--img-size", str(config["img_size"]),
        "--batch-size", str(config["batch_size"]),
        "--epochs", str(config["epochs"]),
        "--seed", str(config["seed"]),
        "--early-stopping-patience", str(config.get("early_stopping_patience", 5)),
        "--embedding-dim", str(config["embedding_dim"]),
        "--learning-rate", str(config["learning_rate"]),
        "--triplet-margin", str(config["triplet_margin"]),
        "--triplet-weight", str(config["triplet_weight"]),
        "--ce-weight", str(config["ce_weight"]),
        "--dropout", str(config["dropout"]),
        "--reference-mode", str(config["reference_mode"]),
        "--max-refs-per-class", str(config["max_refs_per_class"]),
        "--retrieval-modes", "centroid",  # Pagal cv, pickintas centroidas
        "--topn", str(config["topn"]),
    ]

    if dry_run:
        print(f"[ablation] dry-run {' '.join(cmd)}")
        return True, None

    print(f"\n[ablation] run: {config['name']}")
    print(f"[ablation] out: {output_dir}")

    try:
        run_kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
        }
        if timeout_hours > 0:
            run_kwargs["timeout"] = timeout_hours * 3600

        result = subprocess.run(cmd, **run_kwargs)
        if result.returncode == 0:
            print(f"[ablation] ok: {config['name']}")
            return True, None
        else:
            err = result.stderr if result.stderr else result.stdout
            msg = f"CV run failed with return code {result.returncode}: {err[:200]}"
            print(f"[ablation] fail: {config['name']} {msg}")
            return False, msg
    except subprocess.TimeoutExpired:
        msg = f"CV run timed out after 1 hour."
        print(f"[ablation] timeout: {config['name']}")
        return False, msg
    except Exception as e:
        msg = f"Exception during CV run: {e}"
        print(f"[ablation] error: {config['name']} {msg}")
        return False, msg


def select_best_config(results: list[ExperimentResult]) -> tuple[ExperimentResult | None, str]:
    """Parenka geriausia konfiguraicija pagal recall@1 ir recall@3"""
    successful = [r for r in results if r.success and r.recall_at_1_mean is not None]
    if not successful:
        return None, "No successful runs."

    # Rikiuojam pagal recall@1, tada recall@3, tada mazesni embedding_dim, o galiausiai baseline
    def sort_key(r: ExperimentResult) -> tuple:
        recall1 = r.recall_at_1_mean or -1.0
        recall3 = r.recall_at_3_mean or -1.0
        embedding_dim = r.config["embedding_dim"]
        is_baseline = 1 if r.name == "baseline" else 0
        return (-recall1, -recall3, embedding_dim, -is_baseline)

    successful.sort(key=sort_key)
    best = successful[0]

    justification = (
        f"Selected '{best.name}' based on: "
        f"recall@1={best.recall_at_1_mean:.4f}±{best.recall_at_1_std:.4f}, "
        f"recall@3={best.recall_at_3_mean:.4f}±{best.recall_at_3_std:.4f}. "
    )

    if best.recall_at_1_mean == (successful[1].recall_at_1_mean if len(successful) > 1 else None):
        justification += "(Tied on recall@1; used recall@3 and model complexity as tie-breaker.)"

    return best, justification


def write_results(
    results: list[ExperimentResult],
    output_root: Path,
) -> None:
    """Isveda rezultatus i CSV, JSON ir best_config faila"""
    output_root.mkdir(parents=True, exist_ok=True)

    # CSV results
    csv_path = output_root / "ablation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "config_name",
            "description",
            "embedding_dim",
            "learning_rate",
            "triplet_margin",
            "dropout",
            "batch_size",
            "img_size",
            "success",
            "recall@1_mean",
            "recall@1_std",
            "recall@3_mean",
            "recall@3_std",
            "recall@5_mean",
            "recall@5_std",
            "macro_precision_mean",
            "macro_precision_std",
            "macro_recall_mean",
            "macro_recall_std",
            "macro_f1_mean",
            "macro_f1_std",
            "weighted_precision_mean",
            "weighted_precision_std",
            "weighted_recall_mean",
            "weighted_recall_std",
            "weighted_f1_mean",
            "weighted_f1_std",
            "error_msg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "config_name": r.name,
                "description": r.description,
                "embedding_dim": r.config["embedding_dim"],
                "learning_rate": r.config["learning_rate"],
                "triplet_margin": r.config["triplet_margin"],
                "dropout": r.config["dropout"],
                "batch_size": r.config["batch_size"],
                "img_size": r.config["img_size"],
                "success": int(r.success),
                "recall@1_mean": r.recall_at_1_mean,
                "recall@1_std": r.recall_at_1_std,
                "recall@3_mean": r.recall_at_3_mean,
                "recall@3_std": r.recall_at_3_std,
                "recall@5_mean": r.recall_at_5_mean,
                "recall@5_std": r.recall_at_5_std,
                "macro_precision_mean": r.macro_precision_mean,
                "macro_precision_std": r.macro_precision_std,
                "macro_recall_mean": r.macro_recall_mean,
                "macro_recall_std": r.macro_recall_std,
                "macro_f1_mean": r.macro_f1_mean,
                "macro_f1_std": r.macro_f1_std,
                "weighted_precision_mean": r.weighted_precision_mean,
                "weighted_precision_std": r.weighted_precision_std,
                "weighted_recall_mean": r.weighted_recall_mean,
                "weighted_recall_std": r.weighted_recall_std,
                "weighted_f1_mean": r.weighted_f1_mean,
                "weighted_f1_std": r.weighted_f1_std,
                "error_msg": r.error_msg or "",
            })
    print(f"[ablation] csv: {csv_path}")

    # JSON results
    json_path = output_root / "ablation_results.json"
    json_data = {
        "experiment": "retrieval_ablation_v1",
        "date": str(Path(__file__).resolve().stat().st_mtime),
        "results": [
            {
                "name": r.name,
                "description": r.description,
                "config": r.config,
                "success": r.success,
                "metrics": {
                    "recall@1": {"mean": r.recall_at_1_mean, "std": r.recall_at_1_std},
                    "recall@3": {"mean": r.recall_at_3_mean, "std": r.recall_at_3_std},
                    "recall@5": {"mean": r.recall_at_5_mean, "std": r.recall_at_5_std},
                    "macro_precision": {"mean": r.macro_precision_mean, "std": r.macro_precision_std},
                    "macro_recall": {"mean": r.macro_recall_mean, "std": r.macro_recall_std},
                    "macro_f1": {"mean": r.macro_f1_mean, "std": r.macro_f1_std},
                    "weighted_precision": {"mean": r.weighted_precision_mean, "std": r.weighted_precision_std},
                    "weighted_recall": {"mean": r.weighted_recall_mean, "std": r.weighted_recall_std},
                    "weighted_f1": {"mean": r.weighted_f1_mean, "std": r.weighted_f1_std},
                },
                "error_msg": r.error_msg,
                "cv_summary_path": str(r.cv_summary_path) if r.cv_summary_path else None,
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"[ablation] json: {json_path}")

    # Best config
    best_result, best_justification = select_best_config(results)
    if best_result:
        best_config_path = output_root / "best_config.json"
        best_config_data = {
            "selected_config": best_result.name,
            "justification": best_justification,
            "metrics": {
                "recall@1": {"mean": best_result.recall_at_1_mean, "std": best_result.recall_at_1_std},
                "recall@3": {"mean": best_result.recall_at_3_mean, "std": best_result.recall_at_3_std},
                "recall@5": {"mean": best_result.recall_at_5_mean, "std": best_result.recall_at_5_std},
                "macro_precision": {"mean": best_result.macro_precision_mean, "std": best_result.macro_precision_std},
                "macro_recall": {"mean": best_result.macro_recall_mean, "std": best_result.macro_recall_std},
                "macro_f1": {"mean": best_result.macro_f1_mean, "std": best_result.macro_f1_std},
                "weighted_precision": {"mean": best_result.weighted_precision_mean, "std": best_result.weighted_precision_std},
                "weighted_recall": {"mean": best_result.weighted_recall_mean, "std": best_result.weighted_recall_std},
                "weighted_f1": {"mean": best_result.weighted_f1_mean, "std": best_result.weighted_f1_std},
            },
            "config": best_result.config,
        }
        best_config_path.write_text(json.dumps(best_config_data, indent=2), encoding="utf-8")
        print(f"[ablation] best: {best_config_path}")


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_dir)
    cv_script = PROJECT_ROOT / "xtools" / "retrieval_experiment_cv.py"

    if not cv_script.exists():
        print(f"Error: {cv_script} not found.")
        sys.exit(1)

    if not Path(args.folds_csv).exists():
        print(f"Error: Folds CSV not found: {args.folds_csv}")
        sys.exit(1)

    configs_to_run = build_configs(quick=args.quick, epochs=args.epochs, quick_epochs=args.quick_epochs)

    fold_limit = args.quick_folds if args.quick else args.folds
    if fold_limit <= 0:
        raise ValueError("Fold limit must be >= 1")

    available_folds = read_fold_ids(Path(args.folds_csv))
    if not available_folds:
        raise RuntimeError(f"No fold ids found in {args.folds_csv}")
    fold_limit = min(fold_limit, len(available_folds))

    active_folds_csv = limit_folds_csv(
        source_csv=Path(args.folds_csv),
        target_csv=output_root / "_tmp" / f"folds_first{fold_limit}.csv",
        max_folds=fold_limit,
        dry_run=args.dry_run,
    )

    print(f"\n[ablation] start")
    print(f"[ablation] out: {output_root}")
    print(f"[ablation] configs: {len(configs_to_run)} folds={fold_limit} timeout_h={args.timeout_hours} quick={args.quick} dry={args.dry_run}\n")

    results: list[ExperimentResult] = []

    for config in configs_to_run:
        config_output_dir = output_root / config["name"]

        existing_metrics = existing_run_metrics(config_output_dir) if args.skip_existing and config_output_dir.exists() else None
        if existing_metrics is not None:
            print(f"[ablation] skip: {config['name']}")
            result = ExperimentResult(
                name=config["name"],
                description=config["description"],
                config=config,
                output_dir=config_output_dir,
                cv_summary_path=config_output_dir / "cv_summary.json",
                recall_at_1_mean=existing_metrics.get("recall_at_1_mean"),
                recall_at_1_std=existing_metrics.get("recall_at_1_std"),
                recall_at_3_mean=existing_metrics.get("recall_at_3_mean"),
                recall_at_3_std=existing_metrics.get("recall_at_3_std"),
                recall_at_5_mean=existing_metrics.get("recall_at_5_mean"),
                recall_at_5_std=existing_metrics.get("recall_at_5_std"),
                macro_precision_mean=existing_metrics.get("macro_precision_mean"),
                macro_precision_std=existing_metrics.get("macro_precision_std"),
                macro_recall_mean=existing_metrics.get("macro_recall_mean"),
                macro_recall_std=existing_metrics.get("macro_recall_std"),
                macro_f1_mean=existing_metrics.get("macro_f1_mean"),
                macro_f1_std=existing_metrics.get("macro_f1_std"),
                weighted_precision_mean=existing_metrics.get("weighted_precision_mean"),
                weighted_precision_std=existing_metrics.get("weighted_precision_std"),
                weighted_recall_mean=existing_metrics.get("weighted_recall_mean"),
                weighted_recall_std=existing_metrics.get("weighted_recall_std"),
                weighted_f1_mean=existing_metrics.get("weighted_f1_mean"),
                weighted_f1_std=existing_metrics.get("weighted_f1_std"),
                success=True,
            )
            results.append(result)
            continue

        # Paleidziam CV sitai konfiguracijai
        success, error_msg = run_cv_for_config(
            config=config,
            output_dir=config_output_dir,
            folds_csv=active_folds_csv,
            cv_script=cv_script,
            timeout_hours=args.timeout_hours,
            dry_run=args.dry_run,
        )

        # Isgaunam metrikas
        cv_summary_path = config_output_dir / "cv_summary.json"
        if success and not args.dry_run:
            metrics = extract_cv_metrics(cv_summary_path)
        else:
            metrics = {
                "recall_at_1_mean": None,
                "recall_at_1_std": None,
                "recall_at_3_mean": None,
                "recall_at_3_std": None,
                "recall_at_5_mean": None,
                "recall_at_5_std": None,
                "macro_precision_mean": None,
                "macro_precision_std": None,
                "macro_recall_mean": None,
                "macro_recall_std": None,
                "macro_f1_mean": None,
                "macro_f1_std": None,
                "weighted_precision_mean": None,
                "weighted_precision_std": None,
                "weighted_recall_mean": None,
                "weighted_recall_std": None,
                "weighted_f1_mean": None,
                "weighted_f1_std": None,
            }

        print(f"[ablation] metrics: {metrics}")

        # Jei training reportina sekme bet nepavyksta isgauti metriku, laikom kaip nesekme, bet tesiam toliau su kitais testais
        if success and not args.dry_run:
            if (
                metrics["recall_at_1_mean"] is None
                or metrics["recall_at_3_mean"] is None
                or metrics["recall_at_5_mean"] is None
            ):
                msg = "metrics extraction failed"
                print(f"[ablation] warn: {config['name']} {msg}")
                success = False
                if error_msg:
                    error_msg = f"{error_msg}; {msg}"
                else:
                    error_msg = msg

        result = ExperimentResult(
            name=config["name"],
            description=config["description"],
            config=config,
            output_dir=config_output_dir,
            cv_summary_path=cv_summary_path if (success or cv_summary_path.exists()) else None,
            recall_at_1_mean=metrics.get("recall_at_1_mean"),
            recall_at_1_std=metrics.get("recall_at_1_std"),
            recall_at_3_mean=metrics.get("recall_at_3_mean"),
            recall_at_3_std=metrics.get("recall_at_3_std"),
            recall_at_5_mean=metrics.get("recall_at_5_mean"),
            recall_at_5_std=metrics.get("recall_at_5_std"),
            macro_precision_mean=metrics.get("macro_precision_mean"),
            macro_precision_std=metrics.get("macro_precision_std"),
            macro_recall_mean=metrics.get("macro_recall_mean"),
            macro_recall_std=metrics.get("macro_recall_std"),
            macro_f1_mean=metrics.get("macro_f1_mean"),
            macro_f1_std=metrics.get("macro_f1_std"),
            weighted_precision_mean=metrics.get("weighted_precision_mean"),
            weighted_precision_std=metrics.get("weighted_precision_std"),
            weighted_recall_mean=metrics.get("weighted_recall_mean"),
            weighted_recall_std=metrics.get("weighted_recall_std"),
            weighted_f1_mean=metrics.get("weighted_f1_mean"),
            weighted_f1_std=metrics.get("weighted_f1_std"),
            success=success,
            error_msg=error_msg,
        )
        results.append(result)

    # Raso rezultatus ir suvestine
    if not args.dry_run:
        write_results(results, output_root)

        best_result, _ = select_best_config(results)
        if best_result:
            print(f"\n[ablation] best: {best_result.name}")
            print(f"[ablation] r1={best_result.recall_at_1_mean:.4f} r3={best_result.recall_at_3_mean:.4f} r5={best_result.recall_at_5_mean:.4f}\n")

    print("[ablation] done")


if __name__ == "__main__":
    main()
