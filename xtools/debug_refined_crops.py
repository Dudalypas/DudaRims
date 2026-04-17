from __future__ import annotations

import argparse
from pathlib import Path

from generate_reference_embeddings import DetectorCropRunner, iter_images, refined_crop
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = Path(r"C:\Users\vilja\Desktop\Training_Mixed_V1\val")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "trained_cropped_classifier" / "crop_debug"
DEFAULT_DETECTOR_MODEL = PROJECT_ROOT / "assets" / "models" / "best_float16.tflite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save refined post-detection crop examples for inspection.")
    parser.add_argument("--input-root", type=str, default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--detector-model", type=str, default=str(DEFAULT_DETECTOR_MODEL))
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.04)
    parser.add_argument("--crop-tighten-ratio", type=float, default=0.94)
    parser.add_argument("--crop-enforce-square", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    detector_model = Path(args.detector_model)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not detector_model.exists():
        raise FileNotFoundError(f"Detector model not found: {detector_model}")

    detector = DetectorCropRunner(detector_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class dirs found under {input_root}")

    saved_total = 0
    for class_dir in class_dirs:
        images = iter_images(class_dir)
        if not images:
            continue

        class_out = output_dir / class_dir.name
        class_out.mkdir(parents=True, exist_ok=True)

        saved = 0
        for image_path in images:
            if saved >= args.per_class:
                break

            with Image.open(image_path) as raw:
                oriented = ImageOps.exif_transpose(raw).convert("RGB")

            box = detector.detect_best(oriented)
            if box is None:
                crop = oriented
            else:
                crop = refined_crop(
                    oriented=oriented,
                    box=box,
                    padding_ratio=args.crop_padding_ratio,
                    tighten_ratio=args.crop_tighten_ratio,
                    enforce_square=args.crop_enforce_square,
                )

            crop.save(class_out / f"{image_path.stem}.jpg", format="JPEG", quality=94)
            saved += 1
            saved_total += 1

    print("Saved refined crops:", saved_total)
    print("Output dir:", output_dir)


if __name__ == "__main__":
    main()
