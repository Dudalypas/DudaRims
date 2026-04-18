from __future__ import annotations

import argparse
from pathlib import Path

from retrieval_experiment_core import ImageRecord, read_records_csv, train_one_run

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "trained_cropped_classifier" / "retrieval_experiment_v2"
DEFAULT_DEV_CSV = DEFAULT_EXPERIMENT_ROOT / "folds" / "dev_with_folds.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one retrieval embedding experiment run from CSV records. "
            "Useful for single-fold debugging or controlled ablation runs."
        )
    )
    parser.add_argument("--records-csv", type=str, default=str(DEFAULT_DEV_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_EXPERIMENT_ROOT / "single_run"))
    parser.add_argument("--train-folds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--val-folds", nargs="+", type=int, default=[4])

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--triplet-weight", type=float, default=0.5)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = read_records_csv(Path(args.records_csv))
    train_rows = [r for r in records if r.fold in set(args.train_folds)]
    val_rows = [r for r in records if r.fold in set(args.val_folds)]

    train_records = [ImageRecord(path=r.path, label=r.label, source_split=r.source_split) for r in train_rows]
    val_records = [ImageRecord(path=r.path, label=r.label, source_split=r.source_split) for r in val_rows]

    _, val_metrics, _ = train_one_run(
        train_records=train_records,
        val_records=val_records,
        output_dir=Path(args.output_dir),
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

    print("Single run complete.")
    print("Train size:", len(train_records))
    print("Val size:", len(val_records))
    print("Validation metrics:", val_metrics)


if __name__ == "__main__":
    main()
