#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split YOLO detection dataset into train/val/test with optional group-aware splitting."
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Source images directory")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Source labels directory (.txt files)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset directory")

    parser.add_argument("--train", type=float, default=0.80, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.10, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.10, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", choices=["copy", "move"], default="copy", help="Copy or move files")

    parser.add_argument(
        "--group-csv",
        type=Path,
        default=None,
        help="CSV file mapping filename/stem to group_id. Best option.",
    )
    parser.add_argument(
        "--group-regex",
        type=str,
        default=None,
        help=r"Regex applied to filename stem. First capture group becomes group_id. Example: ^(.+?)_\d+$",
    )
    parser.add_argument(
        "--group-by-parent",
        action="store_true",
        help="Use immediate parent folder name as group_id",
    )
    parser.add_argument(
        "--no-group-split",
        action="store_true",
        help="Treat every image as its own group (not recommended if you have same listing/session images)",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search images recursively",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Skip images without matching label instead of failing",
    )

    return parser.parse_args()


def validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}")


def load_group_csv(csv_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]

        # Accepted columns:
        # filename, stem, file, image, name
        # group_id, group, listing, session
        filename_keys = {"filename", "stem", "file", "image", "name"}
        group_keys = {"group_id", "group", "listing", "session"}

        file_col = next((h for h in headers if h.lower() in filename_keys), None)
        group_col = next((h for h in headers if h.lower() in group_keys), None)

        if not file_col or not group_col:
            raise ValueError(
                "CSV must contain a filename/stem column and a group column.\n"
                "Example headers: filename,group_id"
            )

        for row in reader:
            file_key = row[file_col].strip()
            group_id = row[group_col].strip()
            if not file_key or not group_id:
                continue

            p = Path(file_key)
            mapping[p.name] = group_id
            mapping[p.stem] = group_id

    return mapping


def find_images(images_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    images = [
        p for p in images_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images)


def corresponding_label(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def infer_group_id(
    image_path: Path,
    group_map: Dict[str, str] | None,
    group_regex: re.Pattern[str] | None,
    group_by_parent: bool,
    no_group_split: bool,
) -> str:
    if no_group_split:
        return image_path.stem

    if group_map:
        if image_path.name in group_map:
            return group_map[image_path.name]
        if image_path.stem in group_map:
            return group_map[image_path.stem]
        raise KeyError(f"No group mapping found for: {image_path.name}")

    if group_by_parent:
        return image_path.parent.name

    if group_regex:
        m = group_regex.search(image_path.stem)
        if not m:
            raise ValueError(
                f"Regex did not match filename stem: {image_path.stem}"
            )
        return m.group(1) if m.groups() else m.group(0)

    raise ValueError(
        "You must choose one grouping method:\n"
        "--group-csv OR --group-regex OR --group-by-parent OR --no-group-split"
    )


def compute_targets(total_items: int, ratios: Dict[str, float]) -> Dict[str, int]:
    raw = {k: total_items * v for k, v in ratios.items()}
    base = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = total_items - sum(base.values())

    fractions = sorted(
        ((k, raw[k] - base[k]) for k in raw),
        key=lambda x: x[1],
        reverse=True,
    )

    for i in range(remainder):
        base[fractions[i][0]] += 1

    return base


def assign_groups_to_splits(
    groups: Dict[str, List[Tuple[Path, Path]]],
    ratios: Dict[str, float],
    seed: int,
) -> Dict[str, List[Tuple[Path, Path]]]:
    random.seed(seed)

    total_images = sum(len(v) for v in groups.values())
    targets = compute_targets(total_images, ratios)

    group_items = list(groups.items())
    random.shuffle(group_items)
    group_items.sort(key=lambda x: len(x[1]), reverse=True)

    split_data: Dict[str, List[Tuple[Path, Path]]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for group_id, items in group_items:
        size = len(items)

        # Prefer the split with the biggest remaining deficit.
        deficits = {s: targets[s] - counts[s] for s in counts}
        best_split = max(deficits, key=lambda s: deficits[s])

        # If all are over target already, choose minimal overflow.
        if all(deficits[s] < size * -1 for s in deficits):
            best_split = min(
                counts,
                key=lambda s: abs((counts[s] + size) - targets[s])
            )

        split_data[best_split].extend(items)
        counts[best_split] += size

    return split_data


def ensure_dirs(output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def transfer_file(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def write_report(
    output_dir: Path,
    groups: Dict[str, List[Tuple[Path, Path]]],
    splits: Dict[str, List[Tuple[Path, Path]]],
) -> None:
    group_sizes = {gid: len(items) for gid, items in groups.items()}
    report = {
        "total_groups": len(groups),
        "total_images": sum(group_sizes.values()),
        "group_sizes": group_sizes,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_files": {
            k: [img.name for img, _ in v] for k, v in splits.items()
        },
    }

    with (output_dir / "split_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with (output_dir / "split_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image_count"])
        for split, items in splits.items():
            writer.writerow([split, len(items)])


def main() -> None:
    args = parse_args()
    validate_ratios(args.train, args.val, args.test)

    group_methods = sum([
        bool(args.group_csv),
        bool(args.group_regex),
        bool(args.group_by_parent),
        bool(args.no_group_split),
    ])
    if group_methods != 1:
        raise ValueError(
            "Choose exactly one grouping method:\n"
            "--group-csv OR --group-regex OR --group-by-parent OR --no-group-split"
        )

    group_map = load_group_csv(args.group_csv) if args.group_csv else None
    group_regex = re.compile(args.group_regex) if args.group_regex else None

    images = find_images(args.images_dir, args.recursive)
    if not images:
        raise FileNotFoundError("No images found.")

    groups: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    skipped = []

    for img_path in images:
        label_path = corresponding_label(img_path, args.labels_dir)

        if not label_path.exists():
            if args.allow_missing_labels:
                skipped.append(img_path.name)
                continue
            raise FileNotFoundError(
                f"Missing label for image: {img_path.name}\nExpected: {label_path}"
            )

        group_id = infer_group_id(
            image_path=img_path,
            group_map=group_map,
            group_regex=group_regex,
            group_by_parent=args.group_by_parent,
            no_group_split=args.no_group_split,
        )
        groups[group_id].append((img_path, label_path))

    if not groups:
        raise RuntimeError("No valid image-label pairs found.")

    splits = assign_groups_to_splits(
        groups=groups,
        ratios={"train": args.train, "val": args.val, "test": args.test},
        seed=args.seed,
    )

    ensure_dirs(args.output_dir)

    for split, items in splits.items():
        for img_path, label_path in items:
            out_img = args.output_dir / "images" / split / img_path.name
            out_lbl = args.output_dir / "labels" / split / label_path.name
            transfer_file(img_path, out_img, args.mode)
            transfer_file(label_path, out_lbl, args.mode)

    write_report(args.output_dir, groups, splits)

    print("\nDone.")
    print(f"Total valid pairs: {sum(len(v) for v in groups.values())}")
    print(f"Total groups: {len(groups)}")
    for split, items in splits.items():
        print(f"{split}: {len(items)}")
    if skipped:
        print(f"Skipped images without labels: {len(skipped)}")
        print("See your source folders if that was not intended.")


if __name__ == "__main__":
    main()