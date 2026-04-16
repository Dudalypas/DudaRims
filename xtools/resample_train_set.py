from __future__ import annotations

import argparse
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff", ".avif"}


def list_class_images(train_root: Path) -> Dict[str, List[Path]]:
    if not train_root.exists():
        raise FileNotFoundError(f"Train root not found: {train_root}")

    class_to_files: Dict[str, List[Path]] = {}
    for class_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if files:
            class_to_files[class_dir.name] = files

    if not class_to_files:
        raise RuntimeError(f"No class folders with images found in: {train_root}")

    return class_to_files


def compute_target(counts: List[int], mode: str, fixed_target: int | None) -> int:
    if fixed_target is not None:
        if fixed_target <= 0:
            raise ValueError("fixed target must be > 0")
        return fixed_target

    sorted_counts = sorted(counts)

    if mode == "max":
        return sorted_counts[-1]
    if mode == "median":
        mid = len(sorted_counts) // 2
        if len(sorted_counts) % 2 == 1:
            return sorted_counts[mid]
        return int(round((sorted_counts[mid - 1] + sorted_counts[mid]) / 2))
    if mode == "p75":
        idx = int(math.ceil(0.75 * len(sorted_counts))) - 1
        idx = max(0, min(idx, len(sorted_counts) - 1))
        return sorted_counts[idx]

    raise ValueError(f"Unknown mode: {mode}")


def safe_remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, prefer_hardlink: bool = True) -> None:
    if dst.exists():
        dst.unlink()

    if prefer_hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass

    shutil.copy2(src, dst)


def make_balanced_train_dir(
    train_root: Path,
    output_root: Path,
    target: int,
    seed: int,
    prefer_hardlink: bool,
) -> None:
    rng = random.Random(seed)
    class_to_files = list_class_images(train_root)

    safe_remove_dir(output_root)
    ensure_dir(output_root)

    print("Original class counts:")
    for class_name, files in class_to_files.items():
        print(f"  {class_name}: {len(files)}")

    print(f"\nTarget per class: {target}\n")

    for class_name, files in class_to_files.items():
        class_out = output_root / class_name
        ensure_dir(class_out)

        original_count = len(files)

        if original_count >= target:
            selected = rng.sample(files, target)
        else:
            selected = list(files)
            needed = target - original_count
            selected.extend(rng.choices(files, k=needed))

        for idx, src in enumerate(selected, start=1):
            suffix = src.suffix.lower()

            if idx <= original_count and src in files:
                # Preserve obvious originals when possible.
                dst_name = f"{class_name}_{idx:04d}{suffix}"
            else:
                dst_name = f"{class_name}_{idx:04d}__dup{suffix}"

            dst = class_out / dst_name
            link_or_copy(src, dst, prefer_hardlink=prefer_hardlink)

        print(f"  Wrote {class_name}: {original_count} -> {len(selected)}")

    total_out = sum(
        1
        for p in output_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    print(f"\nDone. Output train set: {output_root}")
    print(f"Total images in resampled train set: {total_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a balanced/resampled image train folder."
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        required=True,
        help="Path to original train folder, with one subfolder per class.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Path where resampled train folder will be created.",
    )
    parser.add_argument(
        "--mode",
        choices=["median", "p75", "max"],
        default="p75",
        help="How to choose target count if --target is not given.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Explicit target images per class. Overrides --mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of trying hardlinks first.",
    )

    args = parser.parse_args()

    class_to_files = list_class_images(args.train_root)
    counts = [len(v) for v in class_to_files.values()]
    target = compute_target(counts, args.mode, args.target)

    make_balanced_train_dir(
        train_root=args.train_root,
        output_root=args.output_root,
        target=target,
        seed=args.seed,
        prefer_hardlink=not args.copy,
    )


if __name__ == "__main__":
    main()