from __future__ import annotations

import csv
import hashlib
import math
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Lokalus benchmark setas, is kurio darau palyginimo testus
BENCHMARK_ROOT = Path(r"C:\Users\vilja\Desktop\testing")

# Naudojam tuos pacius assets kaip appse, kad rezultatai butu kuo arciau realaus flow.
PROJECT_ROOT = Path(__file__).resolve().parent
CLASSIFIER_MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "wheel_classifier_cropped_best_float32_v2.tflite"
DETECTOR_MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "best_float16.tflite"
LABELS_PATH = PROJECT_ROOT / "assets" / "labels" / "labels.txt"

RESULTS_CSV = PROJECT_ROOT / "benchmark_results.csv"
SUMMARY_TXT = PROJECT_ROOT / "benchmark_summary.txt"
PER_CLASS_CSV = PROJECT_ROOT / "benchmark_per_class.csv"
ERROR_LOG = PROJECT_ROOT / "benchmark_errors.log"
OUTPUTS_DIR = PROJECT_ROOT / "benchmark_outputs"

OLD_WRONG_DIR = OUTPUTS_DIR / "old_wrong"
NEW_WRONG_DIR = OUTPUTS_DIR / "new_wrong"
IMPROVED_DIR = OUTPUTS_DIR / "improved"
WORSENED_DIR = OUTPUTS_DIR / "worsened"

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ClassificationResult:
    top1_label: str
    top1_conf: float
    top3_labels: list[str]
    top3_confs: list[float]


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


def load_labels(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in: {path}")
    return labels


def create_interpreter(model_path: Path) -> tf.lite.Interpreter:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def format_topk(labels: list[str], confs: list[float]) -> str:
    return ";".join(f"{label}|{conf:.6f}" for label, conf in zip(labels, confs))


def list_images(root: Path) -> list[Path]:
    images: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            images.append(p)
    images.sort()
    return images


def safe_copy(image_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()[:8]
    out_name = f"{image_path.parent.name}__{image_path.stem}__{digest}{image_path.suffix.lower()}"
    shutil.copy2(image_path, target_dir / out_name)


class ClassifierRunner:
    """Classifier behavior mirrored from Dart/Python project code."""

    def __init__(self, model_path: Path, labels: list[str]) -> None:
        self.labels = labels
        self.interpreter = create_interpreter(model_path)

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        shape = self.input_details["shape"].tolist()
        if len(shape) != 4:
            raise ValueError(f"Unexpected classifier input shape: {shape}")

        self.input_h = int(shape[1])
        self.input_w = int(shape[2])

    def classify_pil(self, image: Image.Image) -> ClassificationResult:
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        resized = oriented.resize((self.input_w, self.input_h), Image.Resampling.BILINEAR)

        # Laikom ta pati preprocess kaip appse: float32 RGB [0..255], be papildomo scaling.
        inp = np.asarray(resized, dtype=np.float32)
        inp = np.expand_dims(inp, axis=0).astype(self.input_details["dtype"])

        self.interpreter.set_tensor(self.input_details["index"], inp)
        self.interpreter.invoke()

        logits = self.interpreter.get_tensor(self.output_details["index"])
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        probs = softmax(logits)

        top3_idx = np.argsort(probs)[::-1][:3]
        top3_labels = [self.labels[i] if i < len(self.labels) else f"class_{i}" for i in top3_idx]
        top3_confs = [float(probs[i]) for i in top3_idx]

        return ClassificationResult(
            top1_label=top3_labels[0],
            top1_conf=top3_confs[0],
            top3_labels=top3_labels,
            top3_confs=top3_confs,
        )

    def classify_path(self, image_path: Path) -> ClassificationResult:
        with Image.open(image_path) as img:
            return self.classify_pil(img)


class DetectorRunner:
    """Detector + NMS behavior minimally ported from WheelDetectorService in Dart."""

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

        self.interpreter = create_interpreter(model_path)
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

        # Cia atkartotas Dart elgesys: piešiam su round(), bet reverse transformui paliekam float pad reikšmes.
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

            out = self.interpreter.get_tensor(self.output_details["index"])  # Expected [1,5,8400] or [1,8400,5]
            out = np.asarray(out, dtype=np.float32)

            candidates = self._parse_candidates(out, scale, pad_x, pad_y, original_w, original_h)
            kept = self._nms(candidates)
            return kept[0] if kept else None


def crop_detected_wheel(image_path: Path, detection: DetectionResult, padding_ratio: float = 0.08) -> Image.Image:
    """Crop logic ported from WheelCropService in Dart (minimal, behavior-focused)."""
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


def infer_true_label(image_path: Path, benchmark_root: Path) -> str:
    # Ground truth imam is pirmo aplanko po benchmark root.
    rel_parts = image_path.relative_to(benchmark_root).parts
    if len(rel_parts) < 2:
        return "<root>"
    return rel_parts[0]


def run_benchmark() -> None:
    labels = load_labels(LABELS_PATH)
    classifier = ClassifierRunner(CLASSIFIER_MODEL_PATH, labels)
    detector = DetectorRunner(DETECTOR_MODEL_PATH)

    images = list_images(BENCHMARK_ROOT)
    if not images:
        raise RuntimeError(f"No images found in {BENCHMARK_ROOT} with supported suffixes: {sorted(SUPPORTED_SUFFIXES)}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OLD_WRONG_DIR.mkdir(parents=True, exist_ok=True)
    NEW_WRONG_DIR.mkdir(parents=True, exist_ok=True)
    IMPROVED_DIR.mkdir(parents=True, exist_ok=True)
    WORSENED_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    errors: list[str] = []

    for idx, image_path in enumerate(images, start=1):
        true_label = infer_true_label(image_path, BENCHMARK_ROOT)
        print(f"[{idx}/{len(images)}] class={true_label} image={image_path}")

        try:
            old_res = classifier.classify_path(image_path)

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
                    cropped = crop_detected_wheel(image_path, det)
                    new_res = classifier.classify_pil(cropped)
                except Exception as crop_or_cls_err:
                    fallback_used = True
                    errors.append(
                        f"CROP_OR_NEW_CLASSIFY_FAIL | {image_path} | {crop_or_cls_err}\n{traceback.format_exc()}"
                    )
                    new_res = classifier.classify_path(image_path)
            else:
                fallback_used = True
                new_res = classifier.classify_path(image_path)

            old_top1_correct = old_res.top1_label == true_label
            new_top1_correct = new_res.top1_label == true_label
            old_top3_correct = true_label in old_res.top3_labels
            new_top3_correct = true_label in new_res.top3_labels

            if not old_top1_correct and new_top1_correct:
                change = "improved"
            elif old_top1_correct and new_top1_correct:
                change = "same_correct"
            elif not old_top1_correct and not new_top1_correct:
                change = "same_wrong"
            else:
                change = "worsened"

            row = {
                "image_path": str(image_path),
                "true_label": true_label,
                "old_top1": old_res.top1_label,
                "old_top1_conf": f"{old_res.top1_conf:.6f}",
                "old_top3": format_topk(old_res.top3_labels, old_res.top3_confs),
                "old_top3_confs": ";".join(f"{c:.6f}" for c in old_res.top3_confs),
                "new_top1": new_res.top1_label,
                "new_top1_conf": f"{new_res.top1_conf:.6f}",
                "new_top3": format_topk(new_res.top3_labels, new_res.top3_confs),
                "new_top3_confs": ";".join(f"{c:.6f}" for c in new_res.top3_confs),
                "detector_found": detector_found,
                "detector_conf": detector_conf,
                "fallback_used": fallback_used,
                "old_top1_correct": old_top1_correct,
                "new_top1_correct": new_top1_correct,
                "old_top3_correct": old_top3_correct,
                "new_top3_correct": new_top3_correct,
                "change": change,
            }
            rows.append(row)

            # Issiskaidom klaidas i atskirus aplankus, kad greiciau perziureti ranka.
            if not old_top1_correct:
                safe_copy(image_path, OLD_WRONG_DIR)
            if not new_top1_correct:
                safe_copy(image_path, NEW_WRONG_DIR)
            if change == "improved":
                safe_copy(image_path, IMPROVED_DIR)
            if change == "worsened":
                safe_copy(image_path, WORSENED_DIR)

        except Exception as e:
            errors.append(f"IMAGE_FAIL | {image_path} | {e}\n{traceback.format_exc()}")
            continue

    if not rows:
        raise RuntimeError("Benchmark finished but no valid rows were produced.")

    write_results_csv(rows, RESULTS_CSV)
    write_per_class_csv(rows, PER_CLASS_CSV)
    write_summary(rows, SUMMARY_TXT)

    if errors:
        ERROR_LOG.write_text("\n\n".join(errors), encoding="utf-8")
    elif ERROR_LOG.exists():
        ERROR_LOG.unlink()

    print("\nBenchmark completed.")
    print(f"Rows: {len(rows)}")
    print(f"Results CSV: {RESULTS_CSV}")
    print(f"Per-class CSV: {PER_CLASS_CSV}")
    print(f"Summary TXT: {SUMMARY_TXT}")
    if errors:
        print(f"Errors logged: {ERROR_LOG}")


def _safe_acc(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def write_results_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "image_path",
        "true_label",
        "old_top1",
        "old_top1_conf",
        "old_top3",
        "old_top3_confs",
        "new_top1",
        "new_top1_conf",
        "new_top3",
        "new_top3_confs",
        "detector_found",
        "detector_conf",
        "fallback_used",
        "old_top1_correct",
        "new_top1_correct",
        "old_top3_correct",
        "new_top3_correct",
        "change",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, object]], out_path: Path) -> None:
    total = len(rows)
    classes = sorted({str(r["true_label"]) for r in rows})

    old_top1_ok = sum(bool(r["old_top1_correct"]) for r in rows)
    new_top1_ok = sum(bool(r["new_top1_correct"]) for r in rows)
    old_top3_ok = sum(bool(r["old_top3_correct"]) for r in rows)
    new_top3_ok = sum(bool(r["new_top3_correct"]) for r in rows)

    improved = sum(str(r["change"]) == "improved" for r in rows)
    worsened = sum(str(r["change"]) == "worsened" for r in rows)
    fallbacks = sum(bool(r["fallback_used"]) for r in rows)

    lines = [
        f"total_image_count: {total}",
        f"class_count: {len(classes)}",
        f"old_top1_accuracy: {_safe_acc(old_top1_ok, total):.6f}",
        f"new_top1_accuracy: {_safe_acc(new_top1_ok, total):.6f}",
        f"old_top3_accuracy: {_safe_acc(old_top3_ok, total):.6f}",
        f"new_top3_accuracy: {_safe_acc(new_top3_ok, total):.6f}",
        f"number_improved: {improved}",
        f"number_worsened: {worsened}",
        f"number_detector_fallbacks: {fallbacks}",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_class_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        cls = str(row["true_label"])
        grouped.setdefault(cls, []).append(row)

    fieldnames = [
        "class_name",
        "image_count",
        "old_top1_accuracy",
        "new_top1_accuracy",
        "old_top3_accuracy",
        "new_top3_accuracy",
        "improved_count",
        "worsened_count",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cls in sorted(grouped):
            items = grouped[cls]
            n = len(items)
            old_top1_ok = sum(bool(r["old_top1_correct"]) for r in items)
            new_top1_ok = sum(bool(r["new_top1_correct"]) for r in items)
            old_top3_ok = sum(bool(r["old_top3_correct"]) for r in items)
            new_top3_ok = sum(bool(r["new_top3_correct"]) for r in items)
            improved = sum(str(r["change"]) == "improved" for r in items)
            worsened = sum(str(r["change"]) == "worsened" for r in items)

            writer.writerow(
                {
                    "class_name": cls,
                    "image_count": n,
                    "old_top1_accuracy": f"{_safe_acc(old_top1_ok, n):.6f}",
                    "new_top1_accuracy": f"{_safe_acc(new_top1_ok, n):.6f}",
                    "old_top3_accuracy": f"{_safe_acc(old_top3_ok, n):.6f}",
                    "new_top3_accuracy": f"{_safe_acc(new_top3_ok, n):.6f}",
                    "improved_count": improved,
                    "worsened_count": worsened,
                }
            )


def main() -> None:
    print("Benchmark root:", BENCHMARK_ROOT)
    print("Classifier model:", CLASSIFIER_MODEL_PATH)
    print("Detector model:", DETECTOR_MODEL_PATH)
    print("Labels:", LABELS_PATH)
    run_benchmark()


if __name__ == "__main__":
    main()
