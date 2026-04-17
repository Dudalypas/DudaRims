from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path(r"C:\Users\vilja\Desktop\Training_Mixed_V1")
DEFAULT_EMBEDDING_MODEL = PROJECT_ROOT / "assets" / "models" / "wheel_embedding_cropped_float32.tflite"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "assets" / "data" / "wheel_reference_embeddings.json"
DEFAULT_DETECTOR_MODEL = PROJECT_ROOT / "assets" / "models" / "best_float16.tflite"

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class DetectorCropRunner:
    def __init__(
        self,
        model_path: Path,
        input_size: int = 640,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> None:
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _letterbox(self, image: Image.Image) -> tuple[np.ndarray, float, float, float]:
        w, h = image.size
        scale = min(self.input_size / w, self.input_size / h)
        rw = round(w * scale)
        rh = round(h * scale)
        resized = image.resize((rw, rh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.input_size, self.input_size), (114, 114, 114))
        pad_x = (self.input_size - rw) / 2.0
        pad_y = (self.input_size - rh) / 2.0
        canvas.paste(resized, (round(pad_x), round(pad_y)))
        inp = np.asarray(canvas, dtype=np.float32) / 255.0
        return inp, scale, pad_x, pad_y

    def _parse_candidates(
        self,
        out: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        original_w: int,
        original_h: int,
    ) -> list[tuple[float, float, float, float, float]]:
        if out.ndim != 3:
            return []

        out0 = out[0]
        boxes: list[tuple[float, float, float, float, float]] = []

        def try_add(cx: float, cy: float, bw: float, bh: float, score: float) -> None:
            if not np.isfinite(score) or score < self.confidence_threshold:
                return

            normalized = abs(cx) <= 1.5 and abs(cy) <= 1.5 and abs(bw) <= 1.5 and abs(bh) <= 1.5
            if normalized:
                cx *= self.input_size
                cy *= self.input_size
                bw *= self.input_size
                bh *= self.input_size

            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0

            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            x1 = float(np.clip(x1, 0.0, float(original_w)))
            y1 = float(np.clip(y1, 0.0, float(original_h)))
            x2 = float(np.clip(x2, 0.0, float(original_w)))
            y2 = float(np.clip(y2, 0.0, float(original_h)))

            if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                return

            boxes.append((x1, y1, x2, y2, float(score)))

        if out.shape[1] == 5:
            for i in range(out.shape[2]):
                try_add(float(out0[0, i]), float(out0[1, i]), float(out0[2, i]), float(out0[3, i]), float(out0[4, i]))
        elif out.shape[2] == 5:
            for i in range(out.shape[1]):
                try_add(float(out0[i, 0]), float(out0[i, 1]), float(out0[i, 2]), float(out0[i, 3]), float(out0[i, 4]))

        boxes.sort(key=lambda x: x[4], reverse=True)
        kept: list[tuple[float, float, float, float, float]] = []
        for box in boxes:
            overlaps = False
            for k in kept:
                if self._iou(box[:4], k[:4]) > self.nms_threshold:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(box)
        return kept

    def detect_best(self, image: Image.Image) -> tuple[float, float, float, float] | None:
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        w, h = oriented.size
        inp, scale, pad_x, pad_y = self._letterbox(oriented)
        inp = np.expand_dims(inp, axis=0).astype(self.input_details["dtype"])
        self.interpreter.set_tensor(self.input_details["index"], inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details["index"])
        candidates = self._parse_candidates(np.asarray(out, dtype=np.float32), scale, pad_x, pad_y, w, h)
        if not candidates:
            return None
        x1, y1, x2, y2, _ = candidates[0]
        return x1, y1, x2, y2


def refined_crop(
    oriented: Image.Image,
    box: tuple[float, float, float, float],
    padding_ratio: float,
    tighten_ratio: float,
    enforce_square: bool,
) -> Image.Image:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    x1 -= w * padding_ratio
    x2 += w * padding_ratio
    y1 -= h * padding_ratio
    y2 += h * padding_ratio

    w = x2 - x1
    h = y2 - y1

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    if enforce_square:
        side = max(w, h) * max(tighten_ratio, 1e-6)
        side = min(side, float(min(oriented.size)))
        crop_w = side
        crop_h = side
    else:
        crop_w = min(float(oriented.width), w * max(tighten_ratio, 1e-6))
        crop_h = min(float(oriented.height), h * max(tighten_ratio, 1e-6))

    left = cx - crop_w / 2.0
    top = cy - crop_h / 2.0
    left = max(0.0, min(left, oriented.width - crop_w))
    top = max(0.0, min(top, oriented.height - crop_h))

    x = int(np.clip(round(left), 0, oriented.width - 1))
    y = int(np.clip(round(top), 0, oriented.height - 1))
    cw = int(np.clip(round(crop_w), 1, oriented.width - x))
    ch = int(np.clip(round(crop_h), 1, oriented.height - y))
    return oriented.crop((x, y, x + cw, y + ch))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate retrieval reference embeddings from train split."
    )
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model", type=str, default=str(DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--reference-mode", choices=["all", "limited", "centroid"], default="limited")
    parser.add_argument("--max-refs-per-class", type=int, default=20)
    parser.add_argument("--sampling", choices=["first", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-centroid", action="store_true")
    parser.add_argument("--detector-model", type=str, default=str(DEFAULT_DETECTOR_MODEL))
    parser.add_argument("--use-detector-crop", action="store_true")
    parser.add_argument("--crop-padding-ratio", type=float, default=0.04)
    parser.add_argument("--crop-tighten-ratio", type=float, default=0.94)
    parser.add_argument("--crop-enforce-square", action="store_true")
    parser.add_argument("--save-crops-dir", type=str, default="")
    parser.add_argument("--save-crops-per-class", type=int, default=6)
    return parser.parse_args()


def iter_images(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(p)
    return files


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def preprocess_image(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        oriented = ImageOps.exif_transpose(img).convert("RGB")
        resized = oriented.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    return arr


def create_interpreter(model_path: Path) -> tuple[tf.lite.Interpreter, dict, dict]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return interpreter, input_details, output_details


def embed_image(
    interpreter: tf.lite.Interpreter,
    input_details: dict,
    output_details: dict,
    image_path: Path,
    size: int,
) -> np.ndarray:
    arr = preprocess_image(image_path, size)
    inp = np.expand_dims(arr, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details["index"], inp)
    interpreter.invoke()

    out = interpreter.get_tensor(output_details["index"])
    vec = np.asarray(out, dtype=np.float32).reshape(-1)
    return l2_normalize(vec)


def maybe_refine_image(
    image_path: Path,
    detector: DetectorCropRunner | None,
    padding_ratio: float,
    tighten_ratio: float,
    enforce_square: bool,
) -> Image.Image:
    with Image.open(image_path) as raw:
        oriented = ImageOps.exif_transpose(raw).convert("RGB")

    if detector is None:
        return oriented

    box = detector.detect_best(oriented)
    if box is None:
        return oriented

    return refined_crop(
        oriented=oriented,
        box=box,
        padding_ratio=padding_ratio,
        tighten_ratio=tighten_ratio,
        enforce_square=enforce_square,
    )


def preprocess_image_object(image: Image.Image, size: int) -> np.ndarray:
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def sample_paths(paths: list[Path], mode: str, max_refs: int, seed: int, sampling: str) -> list[Path]:
    if mode == "all":
        return paths
    if mode == "centroid":
        return paths

    if max_refs <= 0 or len(paths) <= max_refs:
        return paths

    if mode == "limited" and max_refs > 0:
        if max_refs >= len(paths):
            return paths
        if max_refs < 1:
            return paths[:1]

        if seed >= 0:
            rnd = random.Random(seed)
        else:
            rnd = random.Random()

        if sampling == "first":
            return paths[:max_refs]
        return sorted(rnd.sample(paths, k=max_refs))

    return paths


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    split_dir = dataset_root / args.split
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Embedding model not found: {model_path}")
    if args.split.lower() != "train":
        raise ValueError("Reference generation must use TRAIN split only.")

    np.random.seed(args.seed)
    random.seed(args.seed)

    interpreter, input_details, output_details = create_interpreter(model_path)
    detector: DetectorCropRunner | None = None
    if args.use_detector_crop:
        detector_path = Path(args.detector_model)
        if not detector_path.exists():
            raise FileNotFoundError(f"Detector model not found: {detector_path}")
        detector = DetectorCropRunner(detector_path)

    class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {split_dir}")

    class_records: list[dict[str, object]] = []
    embedding_size: int | None = None
    crop_debug_dir = Path(args.save_crops_dir) if args.save_crops_dir.strip() else None
    if crop_debug_dir is not None:
        crop_debug_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        images = iter_images(class_dir)
        if not images:
            print(f"Skipping empty class: {class_dir.name}")
            continue

        sampled_paths = sample_paths(
            images,
            mode=args.reference_mode,
            max_refs=args.max_refs_per_class,
            seed=args.seed,
            sampling=args.sampling,
        )

        vectors: list[np.ndarray] = []
        saved_crops = 0
        for image_path in sampled_paths:
            try:
                refined = maybe_refine_image(
                    image_path=image_path,
                    detector=detector,
                    padding_ratio=args.crop_padding_ratio,
                    tighten_ratio=args.crop_tighten_ratio,
                    enforce_square=args.crop_enforce_square,
                )

                if crop_debug_dir is not None and saved_crops < args.save_crops_per_class:
                    class_crop_dir = crop_debug_dir / class_dir.name
                    class_crop_dir.mkdir(parents=True, exist_ok=True)
                    refined.save(class_crop_dir / f"{image_path.stem}.jpg", format="JPEG", quality=94)
                    saved_crops += 1

                arr = preprocess_image_object(refined, args.img_size)
                inp = np.expand_dims(arr, axis=0).astype(np.float32)
                interpreter.set_tensor(input_details["index"], inp)
                interpreter.invoke()
                out = interpreter.get_tensor(output_details["index"])
                vec = l2_normalize(np.asarray(out, dtype=np.float32).reshape(-1))
                vectors.append(vec)
            except Exception as exc:
                print(f"WARN: Failed embedding for {image_path}: {exc}")

        if not vectors:
            print(f"Skipping class without valid embeddings: {class_dir.name}")
            continue

        stack = np.stack(vectors, axis=0)
        centroid = l2_normalize(np.mean(stack, axis=0))
        embedding_size = int(centroid.shape[0])

        references_payload: list[dict[str, object]]
        if args.reference_mode == "centroid":
            references_payload = [{"id": f"{class_dir.name}__centroid", "embedding": centroid.tolist()}]
        else:
            references_payload = [
                {"id": f"{class_dir.name}__{i:04d}", "embedding": v.astype(np.float32).tolist()}
                for i, v in enumerate(vectors, start=1)
            ]

        class_records.append(
            {
                "label": class_dir.name,
                "sample_count": len(vectors),
                "centroid_embedding": (
                    centroid.astype(np.float32).tolist()
                    if (args.include_centroid or args.reference_mode == "centroid")
                    else None
                ),
                "references": references_payload,
                # Backward compatibility for older centroid-only readers.
                "embedding": centroid.astype(np.float32).tolist(),
            }
        )

    if not class_records:
        raise RuntimeError("No reference embeddings were generated.")

    payload = {
        "version": 2,
        "method": "multi_reference_per_class" if args.reference_mode != "centroid" else "centroid_per_class",
        "reference_mode": args.reference_mode,
        "source_split": args.split,
        "dataset_root": str(dataset_root),
        "model": str(model_path),
        "img_size": int(args.img_size),
        "embedding_size": embedding_size,
        "sampling": args.sampling,
        "max_refs_per_class": args.max_refs_per_class,
        "include_centroid": bool(args.include_centroid or args.reference_mode == "centroid"),
        "preprocessing": {
            "exif_transpose": True,
            "resize": [args.img_size, args.img_size],
            "channel_order": "RGB",
            "dtype": "float32",
            "input_scale": "0..255",
            "l2_normalize_embedding": True,
        },
        "crop": {
            "use_detector_crop": args.use_detector_crop,
            "detector_model": str(args.detector_model) if args.use_detector_crop else None,
            "padding_ratio": args.crop_padding_ratio,
            "tighten_ratio": args.crop_tighten_ratio,
            "enforce_square": args.crop_enforce_square,
        },
        "classes": class_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Saved reference embeddings:", output_path)
    print("Classes:", len(class_records))
    print("Embedding size:", embedding_size)


if __name__ == "__main__":
    main()
