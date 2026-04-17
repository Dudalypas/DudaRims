from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

SOURCE_SPLIT_ROOT = Path(r"C:\Users\vilja\Desktop\dataset_split")
CROPPED_SPLIT_ROOT = Path(r"C:\Users\vilja\Desktop\Training_Mixed")

PROJECT_ROOT = Path(__file__).resolve().parent
DETECTOR_MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "best_float16.tflite"

ARTIFACTS_DIR = PROJECT_ROOT / "trained_cropped_classifier"
CROP_LOG_CSV = ARTIFACTS_DIR / "crop_generation_log.csv"
CROP_ERROR_LOG = ARTIFACTS_DIR / "crop_generation_errors.log"
TRAIN_HISTORY_CSV = ARTIFACTS_DIR / "training_history.csv"
TRAIN_SUMMARY_JSON = ARTIFACTS_DIR / "training_summary.json"
VAL_METRICS_TXT = ARTIFACTS_DIR / "final_val_metrics.txt"
LABELS_PATH = ARTIFACTS_DIR / "labels.txt"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best.keras"
FINAL_MODEL_PATH = ARTIFACTS_DIR / "final.keras"
BEST_TFLITE_FLOAT32_PATH = ARTIFACTS_DIR / "wheel_classifier_cropped_best_float32.tflite"
BEST_TFLITE_FP16_PATH = ARTIFACTS_DIR / "wheel_classifier_cropped_best_fp16.tflite"
TRAIN_CURVES_PNG = ARTIFACTS_DIR / "training_curves.png"
PREVIEW_DIR = ARTIFACTS_DIR / "crop_preview"

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".gif", ".tiff"}


@dataclass
class DetectionResult:
    left: float
    top: float
    right: float
    bottom: float
    score: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


class DetectorRunner:
    """Detector behavior ported minimally from WheelDetectorService (Dart)."""

    def __init__(
        self,
        model_path: Path,
        input_size: int = 640,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Detector model not found: {model_path}")

        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    def _letterbox(self, source: Image.Image) -> tuple[Image.Image, float, float, float]:
        src_w, src_h = source.size
        scale = min(self.input_size / src_w, self.input_size / src_h)

        resized_w = round(src_w * scale)
        resized_h = round(src_h * scale)
        resized = source.resize((resized_w, resized_h), Image.Resampling.BILINEAR)

        canvas = Image.new("RGB", (self.input_size, self.input_size), (114, 114, 114))
        pad_x = (self.input_size - resized_w) / 2.0
        pad_y = (self.input_size - resized_h) / 2.0

        canvas.paste(resized, (round(pad_x), round(pad_y)))
        return canvas, scale, pad_x, pad_y

    def _parse_candidates(
        self,
        out: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        original_w: int,
        original_h: int,
    ) -> list[DetectionResult]:
        shape = out.shape
        if len(shape) != 3:
            raise ValueError(f"Unexpected detector output shape: {shape}")

        out0 = out[0]
        candidates: list[DetectionResult] = []

        if shape[1] == 5:
            count = shape[2]
            for i in range(count):
                self._try_add_candidate(
                    candidates,
                    cx=float(out0[0, i]),
                    cy=float(out0[1, i]),
                    w=float(out0[2, i]),
                    h=float(out0[3, i]),
                    score=float(out0[4, i]),
                    scale=scale,
                    pad_x=pad_x,
                    pad_y=pad_y,
                    original_w=original_w,
                    original_h=original_h,
                )
        elif shape[2] == 5:
            count = shape[1]
            for i in range(count):
                self._try_add_candidate(
                    candidates,
                    cx=float(out0[i, 0]),
                    cy=float(out0[i, 1]),
                    w=float(out0[i, 2]),
                    h=float(out0[i, 3]),
                    score=float(out0[i, 4]),
                    scale=scale,
                    pad_x=pad_x,
                    pad_y=pad_y,
                    original_w=original_w,
                    original_h=original_h,
                )
        else:
            raise ValueError(f"Unsupported detector output shape: {shape}")

        candidates.sort(key=lambda d: d.score, reverse=True)
        return candidates

    def _try_add_candidate(
        self,
        out: list[DetectionResult],
        cx: float,
        cy: float,
        w: float,
        h: float,
        score: float,
        scale: float,
        pad_x: float,
        pad_y: float,
        original_w: int,
        original_h: int,
    ) -> None:
        if not math.isfinite(score) or score < self.confidence_threshold:
            return

        normalized = (
            abs(cx) <= 1.5
            and abs(cy) <= 1.5
            and abs(w) <= 1.5
            and abs(h) <= 1.5
        )

        box_cx, box_cy, box_w, box_h = cx, cy, w, h
        if normalized:
            box_cx *= self.input_size
            box_cy *= self.input_size
            box_w *= self.input_size
            box_h *= self.input_size

        x1_model = box_cx - box_w / 2.0
        y1_model = box_cy - box_h / 2.0
        x2_model = box_cx + box_w / 2.0
        y2_model = box_cy + box_h / 2.0

        left = (x1_model - pad_x) / scale
        top = (y1_model - pad_y) / scale
        right = (x2_model - pad_x) / scale
        bottom = (y2_model - pad_y) / scale

        left = float(np.clip(left, 0.0, float(original_w)))
        top = float(np.clip(top, 0.0, float(original_h)))
        right = float(np.clip(right, 0.0, float(original_w)))
        bottom = float(np.clip(bottom, 0.0, float(original_h)))

        if right - left < 2.0 or bottom - top < 2.0:
            return

        out.append(
            DetectionResult(
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                score=score,
            )
        )

    @staticmethod
    def _iou(a: DetectionResult, b: DetectionResult) -> float:
        inter_left = max(a.left, b.left)
        inter_top = max(a.top, b.top)
        inter_right = min(a.right, b.right)
        inter_bottom = min(a.bottom, b.bottom)

        inter_w = max(0.0, inter_right - inter_left)
        inter_h = max(0.0, inter_bottom - inter_top)
        inter_area = inter_w * inter_h

        area_a = max(0.0, a.width) * max(0.0, a.height)
        area_b = max(0.0, b.width) * max(0.0, b.height)
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    def _nms(self, boxes: list[DetectionResult]) -> list[DetectionResult]:
        if not boxes:
            return []
        sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
        selected: list[DetectionResult] = []

        while sorted_boxes:
            current = sorted_boxes.pop(0)
            selected.append(current)
            sorted_boxes = [b for b in sorted_boxes if self._iou(current, b) <= self.nms_threshold]

        return selected

    def detect_best(self, image_path: Path) -> DetectionResult | None:
        with Image.open(image_path) as img:
            oriented = ImageOps.exif_transpose(img).convert("RGB")
            original_w, original_h = oriented.size

            prepared, scale, pad_x, pad_y = self._letterbox(oriented)
            inp = np.asarray(prepared, dtype=np.float32) / 255.0
            inp = np.expand_dims(inp, axis=0).astype(self.input_details["dtype"])

            self.interpreter.set_tensor(self.input_details["index"], inp)
            self.interpreter.invoke()

            out = self.interpreter.get_tensor(self.output_details["index"])
            out = np.asarray(out, dtype=np.float32)

            candidates = self._parse_candidates(out, scale, pad_x, pad_y, original_w, original_h)
            kept = self._nms(candidates)
            return kept[0] if kept else None


def crop_detected_wheel(image_path: Path, detection: DetectionResult, padding_ratio: float = 0.08) -> Image.Image:
    """Crop logic ported from WheelCropService (Dart)."""
    with Image.open(image_path) as img:
        oriented = ImageOps.exif_transpose(img).convert("RGB")

    left = detection.left
    top = detection.top
    right = detection.right
    bottom = detection.bottom

    width = right - left
    height = bottom - top

    pad_x = width * padding_ratio
    pad_y = height * padding_ratio

    left -= pad_x
    right += pad_x
    top -= pad_y
    bottom += pad_y

    width = right - left
    height = bottom - top

    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    side = max(width, height)

    img_w, img_h = oriented.size
    side = min(side, min(float(img_w), float(img_h)))

    crop_left = cx - side / 2.0
    crop_top = cy - side / 2.0

    if crop_left < 0:
        crop_left = 0.0
    if crop_top < 0:
        crop_top = 0.0
    if crop_left + side > img_w:
        crop_left = img_w - side
    if crop_top + side > img_h:
        crop_top = img_h - side

    x = int(np.clip(round(crop_left), 0, img_w - 1))
    y = int(np.clip(round(crop_top), 0, img_h - 1))
    s = int(np.clip(round(side), 1, min(img_w - x, img_h - y)))

    return oriented.crop((x, y, x + s, y + s))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cropped dataset and train classifier.")
    parser.add_argument("--source-root", type=str, default=str(SOURCE_SPLIT_ROOT))
    parser.add_argument("--cropped-root", type=str, default=str(CROPPED_SPLIT_ROOT))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--preview-per-class", type=int, default=8)
    parser.add_argument("--skip-crop-generation", action="store_true")
    return parser.parse_args()


def iter_split_images(split_dir: Path) -> Iterable[Path]:
    for p in split_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            yield p


def infer_label_from_split_path(image_path: Path, split_dir: Path) -> str:
    rel = image_path.relative_to(split_dir)
    if len(rel.parts) < 2:
        raise ValueError(f"Image is not inside class folder: {image_path}")
    return rel.parts[0]


def unique_target_path(base_target: Path, original_path: Path) -> Path:
    if not base_target.exists():
        return base_target
    digest = hashlib.md5(str(original_path).encode("utf-8")).hexdigest()[:8]
    return base_target.with_name(f"{base_target.stem}_{digest}{base_target.suffix}")


def generate_cropped_dataset(source_root: Path, cropped_root: Path, detector: DetectorRunner) -> tuple[int, int, int]:
    train_dir = source_root / "train"
    val_dir = source_root / "val"
    test_dir = source_root / "test"

    for required in (train_dir, val_dir, test_dir):
        if not required.exists():
            raise FileNotFoundError(f"Required split folder missing: {required}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if cropped_root.exists():
        shutil.rmtree(cropped_root)
    cropped_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    errors: list[str] = []

    processed = 0
    cropped_count = 0
    fallback_count = 0

    for split_name, split_src in (("train", train_dir), ("val", val_dir), ("test", test_dir)):
        split_images = sorted(iter_split_images(split_src))
        if not split_images:
            raise RuntimeError(f"No images found in split: {split_src}")

        for idx, image_path in enumerate(split_images, start=1):
            try:
                true_label = infer_label_from_split_path(image_path, split_src)
                rel = image_path.relative_to(split_src)
                rel_subpath = rel.parts[1:]

                out_dir = cropped_root / split_name / true_label
                if len(rel_subpath) > 1:
                    out_dir = out_dir.joinpath(*rel_subpath[:-1])
                out_dir.mkdir(parents=True, exist_ok=True)

                target_name = f"{Path(rel_subpath[-1]).stem}.jpg"
                out_path = unique_target_path(out_dir / target_name, image_path)

                print(
                    f"[{split_name}] {idx}/{len(split_images)} class={true_label} image={image_path}"
                )

                detector_found = False
                detector_conf = ""
                fallback_used = False

                try:
                    det = detector.detect_best(image_path)
                except Exception as det_err:
                    det = None
                    fallback_used = True
                    errors.append(
                        f"DETECTOR_FAIL | {image_path} | {det_err}\n{traceback.format_exc()}"
                    )

                if det is not None:
                    detector_found = True
                    detector_conf = f"{det.score:.6f}"
                    try:
                        out_img = crop_detected_wheel(image_path, det)
                        cropped_count += 1
                    except Exception as crop_err:
                        fallback_used = True
                        errors.append(
                            f"CROP_FAIL | {image_path} | {crop_err}\n{traceback.format_exc()}"
                        )
                        with Image.open(image_path) as src_img:
                            out_img = ImageOps.exif_transpose(src_img).convert("RGB")
                else:
                    fallback_used = True
                    with Image.open(image_path) as src_img:
                        out_img = ImageOps.exif_transpose(src_img).convert("RGB")

                if fallback_used:
                    fallback_count += 1

                out_img.save(out_path, format="JPEG", quality=94)
                crop_w, crop_h = out_img.size
                out_img.close()

                rows.append(
                    {
                        "original_path": str(image_path),
                        "output_path": str(out_path),
                        "split": split_name,
                        "true_label": true_label,
                        "detector_found": detector_found,
                        "detector_conf": detector_conf,
                        "fallback_used": fallback_used,
                        "crop_width": crop_w,
                        "crop_height": crop_h,
                    }
                )
                processed += 1
            except Exception as img_err:
                errors.append(f"IMAGE_FAIL | {image_path} | {img_err}\n{traceback.format_exc()}")
                continue

    if not rows:
        raise RuntimeError("No cropped images were produced.")

    with CROP_LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "original_path",
                "output_path",
                "split",
                "true_label",
                "detector_found",
                "detector_conf",
                "fallback_used",
                "crop_width",
                "crop_height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if errors:
        CROP_ERROR_LOG.write_text("\n\n".join(errors), encoding="utf-8")
    elif CROP_ERROR_LOG.exists():
        CROP_ERROR_LOG.unlink()

    return processed, cropped_count, fallback_count


def cast_to_float32(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    # Del suderinamumo su esamu modeliu laikom preprocess kaip projekte: float [0..255].
    return tf.cast(image, tf.float32), label


def discover_class_names(train_dir: Path, val_dir: Path) -> list[str]:
    train_classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    val_classes = sorted([p.name for p in val_dir.iterdir() if p.is_dir()])

    if not train_classes:
        raise RuntimeError(f"No class directories found in {train_dir}")
    if train_classes != val_classes:
        raise RuntimeError(
            "Train/val class folders do not match exactly. "
            f"train={train_classes}, val={val_classes}"
        )
    return train_classes


def collect_labeled_paths(split_dir: Path, class_to_idx: dict[str, int]) -> tuple[list[str], list[int]]:
    paths: list[str] = []
    labels: list[int] = []

    for image_path in sorted(iter_split_images(split_dir)):
        label = infer_label_from_split_path(image_path, split_dir)
        idx = class_to_idx.get(label)
        if idx is None:
            continue
        paths.append(str(image_path))
        labels.append(idx)

    if not paths:
        raise RuntimeError(f"No images found in split: {split_dir}")
    return paths, labels


def load_and_preprocess_for_classifier(path_str: tf.Tensor | bytes, img_size: int) -> np.ndarray:
    """Match app classifier preprocessing: decode -> EXIF orient -> RGB -> resize -> float32 0..255."""
    if hasattr(path_str, "numpy"):
        raw = path_str.numpy()
    else:
        raw = path_str

    if isinstance(raw, np.ndarray):
        raw = raw.item()

    if isinstance(raw, (bytes, bytearray)):
        image_path = Path(raw.decode("utf-8"))
    else:
        image_path = Path(str(raw))

    with Image.open(image_path) as img:
        oriented = ImageOps.exif_transpose(img).convert("RGB")
        resized = oriented.resize((img_size, img_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    return arr


def _tf_load_image(path: tf.Tensor, label: tf.Tensor, img_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.py_function(
        func=lambda p: load_and_preprocess_for_classifier(p, img_size),
        inp=[path],
        Tout=tf.float32,
    )
    image.set_shape((img_size, img_size, 3))
    label = tf.cast(label, tf.int32)
    return image, label


def make_dataset_from_paths(
    paths: list[str],
    labels: list[int],
    img_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: _tf_load_image(p, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(
    cropped_root: Path,
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_dir = cropped_root / "train"
    val_dir = cropped_root / "val"

    for required in (train_dir, val_dir):
        if not required.exists():
            raise FileNotFoundError(f"Required cropped split folder missing: {required}")

    class_names = discover_class_names(train_dir, val_dir)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    train_paths, train_labels = collect_labeled_paths(train_dir, class_to_idx)
    val_paths, val_labels = collect_labeled_paths(val_dir, class_to_idx)

    train_ds = make_dataset_from_paths(
        train_paths,
        train_labels,
        img_size=image_size[0],
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds = make_dataset_from_paths(
        val_paths,
        val_labels,
        img_size=image_size[0],
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    return train_ds, val_ds, class_names


def save_preview_samples(cropped_root: Path, preview_dir: Path, per_class: int) -> None:
    if per_class < 1:
        return

    train_dir = cropped_root / "train"
    val_dir = cropped_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Cropped train/val directories are required for preview generation.")

    class_names = discover_class_names(train_dir, val_dir)

    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        class_preview_dir = preview_dir / class_name
        class_preview_dir.mkdir(parents=True, exist_ok=True)

        candidates: list[Path] = []
        for split_name in ("train", "val"):
            src_dir = cropped_root / split_name / class_name
            if src_dir.exists():
                candidates.extend(sorted(iter_split_images(src_dir)))

        for i, src in enumerate(candidates[:per_class], start=1):
            ext = src.suffix.lower() if src.suffix else ".jpg"
            out_name = f"{i:02d}_{src.stem}{ext}"
            shutil.copy2(src, class_preview_dir / out_name)


def save_training_curves(history: tf.keras.callbacks.History, output_path: Path) -> None:
    if plt is None:
        return

    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)
    if not list(epochs):
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, hist.get("top1", []), label="train_top1")
    axes[0].plot(epochs, hist.get("val_top1", []), label="val_top1")
    axes[0].plot(epochs, hist.get("top3", []), label="train_top3")
    axes[0].plot(epochs, hist.get("val_top3", []), label="val_top3")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(epochs, hist.get("loss", []), label="train_loss")
    axes[1].plot(epochs, hist.get("val_loss", []), label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_model(num_classes: int, img_size: int, learning_rate: float) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image", dtype=tf.float32)
    x = augmentation(inputs)

    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    logits = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name="wheel_mobilenetv3small_cropped")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top1"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )
    return model


def export_tflite_float32(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def export_tflite_fp16(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def train_classifier(
    cropped_root: Path,
    img_size: int,
    batch_size: int,
    epochs: int,
    seed: int,
    learning_rate: float,
) -> dict[str, float]:
    tf.keras.utils.set_random_seed(seed)

    train_ds, val_ds, class_names = build_datasets(
        cropped_root=cropped_root,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        seed=seed,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    model = build_model(num_classes=len(class_names), img_size=img_size, learning_rate=learning_rate)

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_top1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_top1",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(TRAIN_HISTORY_CSV), append=False),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(FINAL_MODEL_PATH)
    save_training_curves(history, TRAIN_CURVES_PNG)

    val_metrics = model.evaluate(val_ds, return_dict=True, verbose=1)
    with VAL_METRICS_TXT.open("w", encoding="utf-8") as f:
        for key, value in val_metrics.items():
            f.write(f"{key}: {value:.6f}\n")

    val_top1_hist = history.history.get("val_top1", [])
    val_top3_hist = history.history.get("val_top3", [])
    if not val_top1_hist:
        raise RuntimeError("Training history did not contain val_top1.")

    best_epoch_idx = int(np.argmax(val_top1_hist))
    best_epoch = best_epoch_idx + 1
    best_val_top1 = float(val_top1_hist[best_epoch_idx])
    best_val_top3 = float(val_top3_hist[best_epoch_idx]) if val_top3_hist else 0.0

    summary = {
        "class_names": class_names,
        "epochs": epochs,
        "img_size": img_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "history_keys": list(history.history.keys()),
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "best_val_top3": best_val_top3,
        "final_val_metrics": {k: float(v) for k, v in val_metrics.items()},
    }
    TRAIN_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    export_tflite_float32(best_model, BEST_TFLITE_FLOAT32_PATH)
    export_tflite_fp16(best_model, BEST_TFLITE_FP16_PATH)

    return {
        **{k: float(v) for k, v in val_metrics.items()},
        "best_val_top1": best_val_top1,
        "best_val_top3": best_val_top3,
        "best_epoch": float(best_epoch),
    }


def main() -> None:
    args = parse_args()

    source_root = Path(args.source_root)
    cropped_root = Path(args.cropped_root)

    if not source_root.exists():
        raise FileNotFoundError(f"Source split root not found: {source_root}")
    if not DETECTOR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Detector model not found: {DETECTOR_MODEL_PATH}")

    print("Source split root:", source_root)
    print("Cropped split root:", cropped_root)
    print("Detector model:", DETECTOR_MODEL_PATH)

    detector = DetectorRunner(DETECTOR_MODEL_PATH)

    if args.skip_crop_generation:
        print("\nSkipping crop generation and reusing existing cropped dataset.")
    else:
        processed, cropped_count, fallback_count = generate_cropped_dataset(
            source_root=source_root,
            cropped_root=cropped_root,
            detector=detector,
        )

        if processed == 0:
            raise RuntimeError("No images were processed. Cannot start training.")

        print("\nCrop generation summary")
        print("Processed images:", processed)
        print("Detected+cropped images:", cropped_count)
        print("Fallback images:", fallback_count)
        print("Crop log CSV:", CROP_LOG_CSV)
        if CROP_ERROR_LOG.exists():
            print("Crop errors log:", CROP_ERROR_LOG)

    save_preview_samples(cropped_root=cropped_root, preview_dir=PREVIEW_DIR, per_class=args.preview_per_class)
    print("Preview crops:", PREVIEW_DIR)

    metrics = train_classifier(
        cropped_root=cropped_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        learning_rate=args.learning_rate,
    )

    print("\nTraining completed")
    print("Best model:", BEST_MODEL_PATH)
    print("Final model:", FINAL_MODEL_PATH)
    print("Labels:", LABELS_PATH)
    print("Float32 TFLite:", BEST_TFLITE_FLOAT32_PATH)
    print("FP16 TFLite:", BEST_TFLITE_FP16_PATH)
    print("Training history:", TRAIN_HISTORY_CSV)
    print("Training curves:", TRAIN_CURVES_PNG)
    print("Training summary:", TRAIN_SUMMARY_JSON)
    print("Final validation metrics:")
    for name, value in metrics.items():
        if name == "best_epoch":
            print(f"  {name}: {int(value)}")
        else:
            print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
