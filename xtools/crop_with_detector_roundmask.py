from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf
Interpreter = tf.lite.Interpreter

import numpy as np
from PIL import Image, ImageOps

try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except Exception:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}

# Same kas ir lib/services/wheel_detector_service.dart
APP_DEFAULT_DETECTOR_MODEL = Path("assets/models/0421best_float16.tflite")
APP_DEFAULT_INPUT_SIZE = 640
APP_DEFAULT_CONFIDENCE_THRESHOLD = 0.25
APP_DEFAULT_NMS_THRESHOLD = 0.45

# Same kas ir lib/core/constants/app_constants.dart
APP_DETECTOR_MIN_SCORE_THRESHOLD = 0.60
APP_DETECTOR_MIN_BBOX_AREA_RATIO = 0.0035
APP_DETECTOR_MAX_BBOX_AREA_RATIO = 1.00
APP_DETECTOR_MAX_ASPECT_RATIO = 3.0
APP_DETECTOR_HARD_MIN_SCORE_THRESHOLD = 0.35
APP_DETECTOR_HARD_MIN_BBOX_AREA_RATIO = 0.002
APP_DETECTOR_HARD_MAX_ASPECT_RATIO = 5.0

APP_ENABLE_CENTERED_RIM_ONLY_FALLBACK = False
APP_RIM_ONLY_FALLBACK_CENTERED_CROP_RATIO = 0.86
APP_RIM_ONLY_FALLBACK_CENTER_ENERGY_MIN_RATIO = 0.52

APP_RETRIEVAL_CROP_PADDING_RATIO = 0.08
APP_RETRIEVAL_CROP_TIGHTEN_RATIO = 1.00
APP_RETRIEVAL_CROP_ENFORCE_SQUARE = True

APP_ENABLE_RETRIEVAL_ELLIPSE_MASK = True
APP_RETRIEVAL_ELLIPSE_MASK_INSET_RATIO = 0.07
APP_RETRIEVAL_ELLIPSE_MASK_FEATHER = 0.05
APP_RETRIEVAL_ELLIPSE_MASK_USE_CIRCLE_FALLBACK = False


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


@dataclass
class LetterboxPrep:
    image: np.ndarray
    scale: float
    pad_x: float
    pad_y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop class-folder dataset using Flutter-matching TFLite detector pipeline."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root dataset folder with class subfolders")
    parser.add_argument("--output-root", type=Path, required=True, help="Output cropped dataset root")
    parser.add_argument(
        "--model",
        type=Path,
        default=APP_DEFAULT_DETECTOR_MODEL,
        help="Detector .tflite model path (default mirrors app: assets/models/best_float16.tflite)",
    )
    parser.add_argument("--input-size", type=int, default=APP_DEFAULT_INPUT_SIZE, help="Detector input size (app default 640)")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=APP_DEFAULT_CONFIDENCE_THRESHOLD,
        help="Detector raw candidate threshold (app default 0.25)",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=APP_DEFAULT_NMS_THRESHOLD,
        help="Detector NMS IoU threshold (app default 0.45)",
    )

    parser.add_argument(
        "--detector-gates-mode",
        choices=["app", "off"],
        default="app",
        help="Apply app-level detector acceptance gates (hard + borderline + centered fallback)",
    )

    parser.add_argument("--padding-ratio", type=float, default=APP_RETRIEVAL_CROP_PADDING_RATIO, help="Crop padding ratio")
    parser.add_argument("--tighten-ratio", type=float, default=APP_RETRIEVAL_CROP_TIGHTEN_RATIO, help="Crop tighten ratio")
    parser.add_argument(
        "--enforce-square",
        dest="enforce_square",
        action="store_true",
        default=APP_RETRIEVAL_CROP_ENFORCE_SQUARE,
        help="Force square crop (app retrieval default true)",
    )
    parser.add_argument("--no-enforce-square", dest="enforce_square", action="store_false", help="Disable square crop")

    parser.add_argument(
        "--apply-ellipse-mask",
        dest="apply_ellipse_mask",
        action="store_true",
        default=APP_ENABLE_RETRIEVAL_ELLIPSE_MASK,
        help="Apply app-style ellipse mask (default true)",
    )
    parser.add_argument("--no-ellipse-mask", dest="apply_ellipse_mask", action="store_false", help="Disable ellipse mask")
    parser.add_argument("--ellipse-inset", type=float, default=APP_RETRIEVAL_ELLIPSE_MASK_INSET_RATIO, help="Ellipse mask inset ratio")
    parser.add_argument("--ellipse-feather", type=float, default=APP_RETRIEVAL_ELLIPSE_MASK_FEATHER, help="Ellipse mask feather")
    parser.add_argument(
        "--ellipse-circle-fallback",
        dest="ellipse_circle_fallback",
        action="store_true",
        default=APP_RETRIEVAL_ELLIPSE_MASK_USE_CIRCLE_FALLBACK,
        help="Force circle instead of ellipse (app default false)",
    )

    parser.add_argument("--skip-no-detection", action="store_true", help="Skip images with no valid app-like detection/crop")
    parser.add_argument("--copy-no-detection", action="store_true", help="Copy original image when no valid detection/crop")

    parser.add_argument("--save-debug-boxes", action="store_true", help="Save debug image with selected final detection")
    parser.add_argument("--save-failure-log", action="store_true", help="Save per-image failure/action CSV")
    parser.add_argument("--jpg-quality", type=int, default=95, help="Output JPG quality (app mask path uses 95)")
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def clampf(val: float, low: float, high: float) -> float:
    return max(low, min(val, high))


def load_oriented_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        oriented = ImageOps.exif_transpose(im)
        return np.asarray(oriented.convert("RGB"), dtype=np.uint8)


def letterbox(source_rgb: np.ndarray, size: int) -> LetterboxPrep:
    src_h, src_w = source_rgb.shape[:2]
    scale = min(size / float(src_w), size / float(src_h))

    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))

    resized = np.asarray(
        Image.fromarray(source_rgb).resize((resized_w, resized_h), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - resized_w) / 2.0
    pad_y = (size - resized_h) / 2.0
    dst_x = int(round(pad_x))
    dst_y = int(round(pad_y))
    canvas[dst_y:dst_y + resized_h, dst_x:dst_x + resized_w] = resized

    return LetterboxPrep(image=canvas, scale=scale, pad_x=pad_x, pad_y=pad_y)


def iou(a: DetectionResult, b: DetectionResult) -> float:
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


def nms(boxes: list[DetectionResult], iou_threshold: float) -> list[DetectionResult]:
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    selected: list[DetectionResult] = []
    while sorted_boxes:
        current = sorted_boxes.pop(0)
        selected.append(current)
        sorted_boxes = [b for b in sorted_boxes if iou(current, b) <= iou_threshold]
    return selected


def _try_add_candidate(
    out: list[DetectionResult],
    cx: float,
    cy: float,
    w: float,
    h: float,
    score: float,
    prep: LetterboxPrep,
    original_width: int,
    original_height: int,
    input_size: int,
    confidence_threshold: float,
) -> None:
    if (not math.isfinite(score)) or score < confidence_threshold:
        return

    normalized = (abs(cx) <= 1.5 and abs(cy) <= 1.5 and abs(w) <= 1.5 and abs(h) <= 1.5)

    box_cx = cx
    box_cy = cy
    box_w = w
    box_h = h
    if normalized:
        box_cx *= input_size
        box_cy *= input_size
        box_w *= input_size
        box_h *= input_size

    x1_model = box_cx - box_w / 2.0
    y1_model = box_cy - box_h / 2.0
    x2_model = box_cx + box_w / 2.0
    y2_model = box_cy + box_h / 2.0

    left = (x1_model - prep.pad_x) / prep.scale
    top = (y1_model - prep.pad_y) / prep.scale
    right = (x2_model - prep.pad_x) / prep.scale
    bottom = (y2_model - prep.pad_y) / prep.scale

    left = clampf(left, 0.0, float(original_width))
    top = clampf(top, 0.0, float(original_height))
    right = clampf(right, 0.0, float(original_width))
    bottom = clampf(bottom, 0.0, float(original_height))

    if (right - left) < 2.0 or (bottom - top) < 2.0:
        return

    out.append(DetectionResult(left=left, top=top, right=right, bottom=bottom, score=score))


def parse_candidates(
    out_tensor: np.ndarray,
    prep: LetterboxPrep,
    original_width: int,
    original_height: int,
    input_size: int,
    confidence_threshold: float,
) -> list[DetectionResult]:
    # Flutter supports two layouts: [1,5,N] and [1,N,5].
    shape = list(out_tensor.shape)
    if len(shape) != 3:
        raise RuntimeError(f"Unexpected detector output shape: {shape}")

    out0 = out_tensor[0]
    candidates: list[DetectionResult] = []

    if shape[1] == 5:
        count = shape[2]
        for i in range(count):
            _try_add_candidate(
                candidates,
                cx=float(out0[0, i]),
                cy=float(out0[1, i]),
                w=float(out0[2, i]),
                h=float(out0[3, i]),
                score=float(out0[4, i]),
                prep=prep,
                original_width=original_width,
                original_height=original_height,
                input_size=input_size,
                confidence_threshold=confidence_threshold,
            )
    elif shape[2] == 5:
        count = shape[1]
        for i in range(count):
            _try_add_candidate(
                candidates,
                cx=float(out0[i, 0]),
                cy=float(out0[i, 1]),
                w=float(out0[i, 2]),
                h=float(out0[i, 3]),
                score=float(out0[i, 4]),
                prep=prep,
                original_width=original_width,
                original_height=original_height,
                input_size=input_size,
                confidence_threshold=confidence_threshold,
            )
    else:
        raise RuntimeError(f"Unsupported detector output shape: {shape}")

    candidates.sort(key=lambda d: d.score, reverse=True)
    return candidates


def detect_best(
    interpreter: Interpreter,
    image_rgb: np.ndarray,
    input_size: int,
    confidence_threshold: float,
    nms_threshold: float,
) -> DetectionResult | None:
    prep = letterbox(image_rgb, input_size)

    # Flutter sends normalized RGB [0..1] float32 with NHWC [1,640,640,3].
    inp = prep.image.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details["index"])

    candidates = parse_candidates(
        out_tensor=out,
        prep=prep,
        original_width=image_rgb.shape[1],
        original_height=image_rgb.shape[0],
        input_size=input_size,
        confidence_threshold=confidence_threshold,
    )

    kept = nms(candidates, nms_threshold)
    return kept[0] if kept else None


def appears_dominant_centered_object(image_rgb: np.ndarray) -> bool:
    resized = np.asarray(
        Image.fromarray(image_rgb).resize((192, 192), Image.Resampling.BILINEAR),
        dtype=np.int16,
    )
    center_min = int(round(192 * 0.25))
    center_max = int(round(192 * 0.75))

    total_energy = 0.0
    center_energy = 0.0
    for y in range(1, 191):
        for x in range(1, 191):
            p = resized[y, x]
            px = resized[y, x + 1]
            py = resized[y + 1, x]

            dx = abs(int(p[0]) - int(px[0])) + abs(int(p[1]) - int(px[1])) + abs(int(p[2]) - int(px[2]))
            dy = abs(int(p[0]) - int(py[0])) + abs(int(p[1]) - int(py[1])) + abs(int(p[2]) - int(py[2]))
            e = float(dx + dy)

            total_energy += e
            if center_min <= x <= center_max and center_min <= y <= center_max:
                center_energy += e

    if total_energy <= 1e-6:
        return False
    center_ratio = center_energy / total_energy
    return center_ratio >= APP_RIM_ONLY_FALLBACK_CENTER_ENERGY_MIN_RATIO


def centered_rim_only_fallback_crop(image_rgb: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    max_crop_side = min(w, h)
    crop_size = int(
        clampf(
            round(max_crop_side * APP_RIM_ONLY_FALLBACK_CENTERED_CROP_RATIO),
            32,
            max_crop_side,
        )
    )

    left = int(round((w - crop_size) / 2.0))
    top = int(round((h - crop_size) / 2.0))
    return image_rgb[top:top + crop_size, left:left + crop_size].copy()


def crop_detected_wheel(
    image_rgb: np.ndarray,
    detection: DetectionResult,
    padding_ratio: float,
    tighten_ratio: float,
    enforce_square: bool,
) -> np.ndarray:
    # Mirrors WheelCropService.cropDetectedWheel in Flutter.
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
    side = max(width, height) if enforce_square else width
    side_y = side if enforce_square else height

    if tighten_ratio > 0 and math.isfinite(tighten_ratio):
        side *= tighten_ratio
        side_y *= tighten_ratio
    if side < 1.0:
        side = 1.0
    if side_y < 1.0:
        side_y = 1.0

    h, w = image_rgb.shape[:2]
    if enforce_square:
        side = min(side, min(float(w), float(h)))
        side_y = side
    else:
        side = min(side, float(w))
        side_y = min(side_y, float(h))

    crop_left = cx - side / 2.0
    crop_top = cy - side_y / 2.0

    if crop_left < 0:
        crop_left = 0
    if crop_top < 0:
        crop_top = 0
    if crop_left + side > w:
        crop_left = w - side
    if crop_top + side_y > h:
        crop_top = h - side_y

    x = int(np.clip(round(crop_left), 0, w - 1))
    y = int(np.clip(round(crop_top), 0, h - 1))
    s_w = int(np.clip(round(side), 1, w - x))
    s_h = int(np.clip(round(side_y), 1, h - y))

    return image_rgb[y:y + s_h, x:x + s_w].copy()


def apply_optional_ellipse_mask(
    crop_rgb: np.ndarray,
    enabled: bool,
    inset: float,
    feather: float,
    use_circle: bool,
) -> np.ndarray:
    if not enabled:
        return crop_rgb

    h, w = crop_rgb.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    inset = clampf(inset, 0.0, 0.45)
    feather = clampf(feather, 0.0, 0.4)

    rx_base = w * 0.5 * (1.0 - inset)
    ry_base = h * 0.5 * (1.0 - inset)
    if use_circle:
        rx = min(rx_base, ry_base)
        ry = min(rx_base, ry_base)
    else:
        rx = rx_base
        ry = ry_base

    out = crop_rgb.astype(np.float32).copy()
    inner = clampf(1.0 - feather, 0.0, 1.0)

    for y in range(h):
        ny = (y - cy) / (1.0 if ry <= 1e-6 else ry)
        for x in range(w):
            nx = (x - cx) / (1.0 if rx <= 1e-6 else rx)
            d = math.sqrt(nx * nx + ny * ny)

            if d <= inner:
                keep = 1.0
            elif d >= 1.0:
                keep = 0.0
            else:
                t = (d - inner) / max(1e-6, (1.0 - inner))
                keep = 1.0 - t

            if keep >= 0.999:
                continue
            out[y, x, 0] *= keep
            out[y, x, 1] *= keep
            out[y, x, 2] *= keep

    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def save_rgb_image(path: Path, image_rgb: np.ndarray, jpg_quality: int) -> None:
    import cv2

    out_path = path.with_suffix(".jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Input image is RGB (from Pillow/Numpy pipeline); OpenCV expects BGR.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(
        str(out_path),
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)],
    )
    if not ok:
        raise RuntimeError(f"Failed to write JPEG output: {out_path}")


def save_debug_box(path: Path, image_rgb: np.ndarray, box: DetectionResult) -> None:
    import cv2

    dbg = image_rgb[:, :, ::-1].copy()  # RGB -> BGR for cv2 drawing
    cv2.rectangle(
        dbg,
        (int(round(box.left)), int(round(box.top))),
        (int(round(box.right)), int(round(box.bottom))),
        (0, 255, 0),
        2,
    )
    out_path = path.with_suffix(".jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(out_path),
        dbg,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not ok:
        raise RuntimeError(f"Failed to write debug JPEG output: {out_path}")


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def main() -> None:
    args = parse_args()
    forced_output_ext = ".jpg"

    if args.skip_no_detection and args.copy_no_detection:
        raise ValueError("Choose only one of --skip-no-detection or --copy-no-detection")

    if not args.model.exists():
        raise FileNotFoundError(f"Detector model not found: {args.model}")

    interpreter = Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()

    images = list_images(args.input_root)
    if not images:
        raise FileNotFoundError(f"No images found in {args.input_root}")

    saved = 0
    skipped = 0
    copied = 0
    failed = 0
    failure_rows: list[dict[str, str]] = []

    for img_path in images:
        rel_path = img_path.relative_to(args.input_root)
        out_path = (args.output_root / rel_path).with_suffix(forced_output_ext)

        try:
            source_rgb = load_oriented_rgb(img_path)
            best = detect_best(
                interpreter=interpreter,
                image_rgb=source_rgb,
                input_size=args.input_size,
                confidence_threshold=args.confidence_threshold,
                nms_threshold=args.nms_threshold,
            )

            crop_rgb: np.ndarray | None = None
            reason = ""
            rejection_reason = ""
            raw_best_score: float | None = None
            raw_bbox_area_ratio: float | None = None
            raw_aspect_ratio: float | None = None

            if best is None:
                reason = "detector_no_box"
                rejection_reason = "no_boxes_after_nms"
            else:
                box_w = best.width
                box_h = best.height
                image_area = float(source_rgb.shape[0] * source_rgb.shape[1])
                box_area = box_w * box_h
                area_ratio = (box_area / image_area) if image_area > 0 else 0.0
                long_side = max(box_w, box_h)
                short_side = min(box_w, box_h)
                aspect_ratio = (long_side / short_side) if short_side > 0 else 0.0

                raw_best_score = best.score
                raw_bbox_area_ratio = area_ratio
                raw_aspect_ratio = aspect_ratio

                if args.detector_gates_mode == "app":
                    hard_score_reject = best.score < APP_DETECTOR_HARD_MIN_SCORE_THRESHOLD
                    hard_tiny_reject = area_ratio < APP_DETECTOR_HARD_MIN_BBOX_AREA_RATIO
                    hard_aspect_reject = aspect_ratio > APP_DETECTOR_HARD_MAX_ASPECT_RATIO

                    if hard_score_reject or hard_tiny_reject or hard_aspect_reject:
                        reason = "detector_hard_reject"
                        if hard_score_reject:
                            rejection_reason = "score_below_hard_min"
                        elif hard_tiny_reject:
                            rejection_reason = "bbox_area_too_small"
                        else:
                            rejection_reason = "aspect_too_large"
                    else:
                        borderline_score = best.score < APP_DETECTOR_MIN_SCORE_THRESHOLD
                        borderline_area_small = area_ratio < APP_DETECTOR_MIN_BBOX_AREA_RATIO
                        borderline_area_large = area_ratio > APP_DETECTOR_MAX_BBOX_AREA_RATIO
                        borderline_area = borderline_area_small or borderline_area_large
                        borderline_aspect = aspect_ratio > APP_DETECTOR_MAX_ASPECT_RATIO

                        if borderline_score or borderline_area or borderline_aspect:
                            if borderline_score:
                                rejection_reason = "score_below_soft_min"
                            elif borderline_area_small:
                                rejection_reason = "bbox_area_too_small"
                            elif borderline_area_large:
                                rejection_reason = "bbox_area_too_large"
                            else:
                                rejection_reason = "aspect_too_large"

                            if APP_ENABLE_CENTERED_RIM_ONLY_FALLBACK and appears_dominant_centered_object(source_rgb):
                                crop_rgb = centered_rim_only_fallback_crop(source_rgb)
                                reason = "centered_rim_only_fallback"
                                failure_rows.append(
                                    {
                                        "image": str(rel_path),
                                        "status": "used_centered_fallback",
                                        "reason": reason,
                                        "raw_best_score": _fmt_float(raw_best_score),
                                        "raw_bbox_area_ratio": _fmt_float(raw_bbox_area_ratio),
                                        "raw_aspect_ratio": _fmt_float(raw_aspect_ratio),
                                        "rejection_reason": rejection_reason,
                                    }
                                )
                            else:
                                reason = "detector_borderline_reject"
                        else:
                            crop_rgb = crop_detected_wheel(
                                image_rgb=source_rgb,
                                detection=best,
                                padding_ratio=args.padding_ratio,
                                tighten_ratio=args.tighten_ratio,
                                enforce_square=args.enforce_square,
                            )
                            reason = "detector_crop"
                else:
                    crop_rgb = crop_detected_wheel(
                        image_rgb=source_rgb,
                        detection=best,
                        padding_ratio=args.padding_ratio,
                        tighten_ratio=args.tighten_ratio,
                        enforce_square=args.enforce_square,
                    )
                    reason = "detector_crop"

            if crop_rgb is None:
                if args.copy_no_detection:
                    save_rgb_image(out_path, source_rgb, jpg_quality=args.jpg_quality)
                    copied += 1
                    action = "copied_original"
                elif args.skip_no_detection:
                    skipped += 1
                    action = "skipped"
                else:
                    save_rgb_image(out_path, source_rgb, jpg_quality=args.jpg_quality)
                    copied += 1
                    action = "copied_original"

                failure_rows.append({"image": str(rel_path), "status": action, "reason": reason})
                failure_rows[-1]["raw_best_score"] = _fmt_float(raw_best_score)
                failure_rows[-1]["raw_bbox_area_ratio"] = _fmt_float(raw_bbox_area_ratio)
                failure_rows[-1]["raw_aspect_ratio"] = _fmt_float(raw_aspect_ratio)
                failure_rows[-1]["rejection_reason"] = rejection_reason
                continue

            masked = apply_optional_ellipse_mask(
                crop_rgb=crop_rgb,
                enabled=args.apply_ellipse_mask,
                inset=args.ellipse_inset,
                feather=args.ellipse_feather,
                use_circle=args.ellipse_circle_fallback,
            )

            save_rgb_image(out_path, masked, jpg_quality=args.jpg_quality)
            saved += 1

            if args.save_debug_boxes and best is not None:
                debug_path = (args.output_root / "_debug_boxes" / rel_path).with_suffix(".jpg")
                save_debug_box(debug_path, source_rgb, best)

        except Exception as exc:
            failed += 1
            failure_rows.append(
                {
                    "image": str(rel_path),
                    "status": "failed",
                    "reason": str(exc),
                    "raw_best_score": "",
                    "raw_bbox_area_ratio": "",
                    "raw_aspect_ratio": "",
                    "rejection_reason": "",
                }
            )
            print(f"[WARN] Failed on {img_path}: {exc}")

    if args.save_failure_log:
        log_path = args.output_root / "_pipeline_failures.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
                    "status",
                    "reason",
                    "raw_best_score",
                    "raw_bbox_area_ratio",
                    "raw_aspect_ratio",
                    "rejection_reason",
                ],
            )
            writer.writeheader()
            writer.writerows(failure_rows)
        print(f"Failure log: {log_path}")

    print("\nDone.")
    print(f"Saved cropped images: {saved}")
    print(f"Copied originals (no detection fallback): {copied}")
    print(f"Skipped (no detection): {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()