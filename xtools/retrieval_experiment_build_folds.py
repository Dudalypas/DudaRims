from __future__ import annotations

import argparse
from pathlib import Path

from retrieval_experiment_core import (
    class_distribution,
    combine_dev_records,
    fold_distribution,
    save_json,
    stratified_kfold_assign,
    write_csv,
    write_records_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "PROD_cropped"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "trained_cropped_classifier" / "retrieval_experiment_v2" / "folds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build stratified 5 fold development splits from TRAIN+VAL while preserving TEST"
        )
    )
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT), help="Root folder with split data")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_records, test_records = combine_dev_records(dataset_root)
    dev_with_folds = stratified_kfold_assign(dev_records, n_folds=args.n_folds, seed=args.seed)

    dev_csv = output_dir / "dev_with_folds.csv"
    test_csv = output_dir / "test_holdout.csv"

    write_records_csv(dev_with_folds, dev_csv)
    write_records_csv(test_records, test_csv)

    dev_counts = class_distribution(dev_with_folds)
    test_counts = class_distribution(test_records)
    fold_counts = fold_distribution(dev_with_folds, n_folds=args.n_folds)

    per_class_rows = []
    labels = sorted(set(dev_counts) | set(test_counts))
    for label in labels:
        row = {
            "label": label,
            "dev_total": dev_counts.get(label, 0),
            "test_total": test_counts.get(label, 0),
        }
        for fold_idx in range(args.n_folds):
            row[f"fold_{fold_idx}"] = fold_counts[fold_idx].get(label, 0)
        per_class_rows.append(row)

    write_csv(output_dir / "class_distribution_report.csv", per_class_rows)

    summary = {
        "dataset_root": str(dataset_root),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "dev_total_images": len(dev_with_folds),
        "test_total_images": len(test_records),
        "dev_num_classes": len(dev_counts),
        "test_num_classes": len(test_counts),
        "dev_with_folds_csv": str(dev_csv),
        "test_holdout_csv": str(test_csv),
        "class_distribution_csv": str(output_dir / "class_distribution_report.csv"),
        "fold_distribution": {str(k): v for k, v in fold_counts.items()},
    }
    save_json(summary, output_dir / "folds_summary.json")

    print("[folds] dev:", dev_csv)
    print("[folds] test:", test_csv)
    print("[folds] report:", output_dir / "class_distribution_report.csv")


if __name__ == "__main__":
    main()
