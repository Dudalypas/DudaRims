from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path(r"C:\Users\vilja\Desktop\Training_Mixed_V1")
DEFAULT_EMBEDDING_MODEL = PROJECT_ROOT / "assets" / "models" / "wheel_embedding_cropped_float32.tflite"
DEFAULT_REFERENCE_JSON = PROJECT_ROOT / "assets" / "data" / "wheel_reference_embeddings.json"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "trained_cropped_classifier" / "retrieval_eval_summary.json"
DEFAULT_REPORT_CSV = PROJECT_ROOT / "trained_cropped_classifier" / "retrieval_eval_per_class.csv"
DEFAULT_EXPERIMENT_CSV = PROJECT_ROOT / "trained_cropped_classifier" / "retrieval_eval_experiments.csv"
DEFAULT_TOP1_COMPARE_CSV = PROJECT_ROOT / "trained_cropped_classifier" / "retrieval_eval_top1_compare.csv"
DEFAULT_DETECTOR_MODEL = PROJECT_ROOT / "assets" / "models" / "best_float16.tflite"

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
TOP_KS = (1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate embedding retrieval modes with Recall@k on val/test splits."
    )
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--model", type=str, default=str(DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--reference-json", type=str, default=str(DEFAULT_REFERENCE_JSON))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument(
        "--retrieval-modes",
        nargs="+",
        default=["centroid", "multi_max", "multi_topn_avg"],
        choices=["centroid", "multi_max", "multi_topn_avg"],
    )
    parser.add_argument(
        "--topn-values",
        nargs="+",
        type=int,
        default=[3],
        help="Used only for multi_topn_avg mode.",
    )
    parser.add_argument(
        "--crop-profiles",
        nargs="+",
        default=["no_crop:false:0.04:0.94:true", "det_crop:true:0.04:0.94:true"],
        help=(
            "List of profile specs in format: name:use_detector_crop:padding:tighten:enforce_square. "
            "Example: det_tight:true:0.02:0.90:true"
        ),
    )
    parser.add_argument("--detector-model", type=str, default=str(DEFAULT_DETECTOR_MODEL))
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--report-json", type=str, default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-csv", type=str, default=str(DEFAULT_REPORT_CSV))
    parser.add_argument("--experiments-csv", type=str, default=str(DEFAULT_EXPERIMENT_CSV))
    parser.add_argument("--top1-compare-csv", type=str, default=str(DEFAULT_TOP1_COMPARE_CSV))
    return parser.parse_args()


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


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


def parse_crop_profile(spec: str) -> dict[str, object]:
    parts = spec.split(":")
    if len(parts) != 5:
        raise ValueError(
            f"Invalid crop profile '{spec}'. Expected format name:use_detector_crop:padding:tighten:enforce_square"
        )
    name, use_det, pad, tight, square = parts
    return {
        "name": name,
        "use_detector_crop": use_det.lower() == "true",
        "padding_ratio": float(pad),
        "tighten_ratio": float(tight),
        "enforce_square": square.lower() == "true",
    }


def preprocess_image(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        oriented = ImageOps.exif_transpose(img).convert("RGB")
        resized = oriented.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def preprocess_image_object(image: Image.Image, size: int) -> np.ndarray:
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def iter_class_images(split_dir: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    for class_dir in class_dirs:
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_SUFFIXES:
                samples.append((img_path, class_dir.name))
    return samples


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
    return l2_normalize(np.asarray(out, dtype=np.float32).reshape(-1))


def embed_image_array(
    interpreter: tf.lite.Interpreter,
    input_details: dict,
    output_details: dict,
    image_arr: np.ndarray,
) -> np.ndarray:
    inp = np.expand_dims(image_arr, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details["index"])
    return l2_normalize(np.asarray(out, dtype=np.float32).reshape(-1))


def load_references(reference_json: Path) -> dict[str, dict[str, object]]:
    if not reference_json.exists():
        raise FileNotFoundError(f"Reference JSON not found: {reference_json}")

    payload = json.loads(reference_json.read_text(encoding="utf-8"))
    classes = payload.get("classes", [])
    if not isinstance(classes, list):
        raise ValueError("Invalid reference JSON: classes must be a list")

    refs: dict[str, dict[str, object]] = {}
    for entry in classes:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", "")).strip()
        if not label:
            continue

        centroid = None
        centroid_raw = entry.get("centroid_embedding", entry.get("embedding"))
        if isinstance(centroid_raw, list):
            centroid = l2_normalize(np.asarray(centroid_raw, dtype=np.float32).reshape(-1))

        vectors: list[np.ndarray] = []
        refs_raw = entry.get("references")
        if isinstance(refs_raw, list):
            for r in refs_raw:
                if not isinstance(r, dict):
                    continue
                emb = r.get("embedding")
                if not isinstance(emb, list):
                    continue
                vectors.append(l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1)))

        # Backward compatibility with centroid-only files.
        if not vectors and centroid is not None:
            vectors = [centroid]

        if not vectors:
            continue

        refs[label] = {
            "centroid": centroid if centroid is not None else l2_normalize(np.mean(np.stack(vectors), axis=0)),
            "vectors": vectors,
        }

    if not refs:
        raise RuntimeError("No valid references loaded from JSON")
    return refs


def score_labels(
    query: np.ndarray,
    refs: dict[str, dict[str, object]],
    retrieval_mode: str,
    topn: int,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for label, payload in refs.items():
        if retrieval_mode == "centroid":
            centroid = payload["centroid"]
            score = float(np.dot(query, centroid))
        else:
            vectors: list[np.ndarray] = payload["vectors"]
            sims = [float(np.dot(query, v)) for v in vectors]
            if not sims:
                score = -1.0
            elif retrieval_mode == "multi_max":
                score = max(sims)
            elif retrieval_mode == "multi_topn_avg":
                sims.sort(reverse=True)
                n = max(1, min(topn, len(sims)))
                score = float(np.mean(sims[:n]))
            else:
                raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")
        scored.append((label, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def compute_recall_at_k(ranked_labels: list[str], true_label: str, k: int) -> int:
    top = ranked_labels[:k]
    return 1 if true_label in top else 0


def evaluate_split(
    split_name: str,
    split_dir: Path,
    interpreter: tf.lite.Interpreter,
    input_details: dict,
    output_details: dict,
    refs: dict[str, dict[str, object]],
    img_size: int,
    retrieval_mode: str,
    topn: int,
    crop_profile: dict[str, object],
    detector: DetectorCropRunner | None,
    limit_per_class: int,
) -> dict[str, object]:
    samples = iter_class_images(split_dir)
    if not samples:
        raise RuntimeError(f"No images found in split: {split_dir}")

    if limit_per_class > 0:
        grouped: dict[str, list[tuple[Path, str]]] = defaultdict(list)
        for s in samples:
            grouped[s[1]].append(s)
        samples = []
        for _, rows in grouped.items():
            samples.extend(rows[:limit_per_class])

    totals = {k: 0 for k in TOP_KS}
    per_class = defaultdict(lambda: {"count": 0, **{f"r@{k}": 0 for k in TOP_KS}})
    top1_cmp = defaultdict(int)
    detector_failures = 0

    for image_path, true_label in samples:
        if crop_profile["use_detector_crop"]:
            with Image.open(image_path) as raw:
                oriented = ImageOps.exif_transpose(raw).convert("RGB")
            box = None if detector is None else detector.detect_best(oriented)
            if box is None:
                detector_failures += 1
                refined = oriented
            else:
                refined = refined_crop(
                    oriented=oriented,
                    box=box,
                    padding_ratio=float(crop_profile["padding_ratio"]),
                    tighten_ratio=float(crop_profile["tighten_ratio"]),
                    enforce_square=bool(crop_profile["enforce_square"]),
                )
            arr = preprocess_image_object(refined, img_size)
            query = embed_image_array(interpreter, input_details, output_details, arr)
        else:
            query = embed_image(
                interpreter=interpreter,
                input_details=input_details,
                output_details=output_details,
                image_path=image_path,
                size=img_size,
            )

        ranked_scores = score_labels(query, refs, retrieval_mode=retrieval_mode, topn=topn)
        ranked = [label for label, _ in ranked_scores]
        if ranked:
            top1_cmp[(true_label, ranked[0])] += 1

        class_row = per_class[true_label]
        class_row["count"] += 1

        for k in TOP_KS:
            hit = compute_recall_at_k(ranked, true_label, k)
            totals[k] += hit
            class_row[f"r@{k}"] += hit

    n = len(samples)
    recall = {f"recall@{k}": totals[k] / n for k in TOP_KS}

    per_class_rows = []
    for label, row in sorted(per_class.items()):
        count = int(row["count"])
        per_class_rows.append(
            {
                "split": split_name,
                "label": label,
                "count": count,
                **{
                    f"recall@{k}": (row[f"r@{k}"] / count if count else 0.0)
                    for k in TOP_KS
                },
            }
        )

    return {
        "split": split_name,
        "num_samples": n,
        "retrieval_mode": retrieval_mode,
        "topn": topn,
        "crop_profile": crop_profile,
        "detector_failures": detector_failures,
        **recall,
        "per_class": per_class_rows,
        "top1_compare": [{"true_label": t, "pred_label": p, "count": c} for (t, p), c in sorted(top1_cmp.items())],
    }


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    model_path = Path(args.model)
    reference_json = Path(args.reference_json)
    report_json = Path(args.report_json)
    report_csv = Path(args.report_csv)
    experiments_csv = Path(args.experiments_csv)
    top1_compare_csv = Path(args.top1_compare_csv)

    if not model_path.exists():
        raise FileNotFoundError(f"Embedding model not found: {model_path}")

    refs = load_references(reference_json)
    interpreter, input_details, output_details = create_interpreter(model_path)

    crop_profiles = [parse_crop_profile(spec) for spec in args.crop_profiles]
    detector = None
    if any(profile["use_detector_crop"] for profile in crop_profiles):
        detector_model_path = Path(args.detector_model)
        if not detector_model_path.exists():
            raise FileNotFoundError(f"Detector model not found: {detector_model_path}")
        detector = DetectorCropRunner(detector_model_path)

    split_reports: list[dict[str, object]] = []
    all_per_class_rows: list[dict[str, object]] = []
    experiment_rows: list[dict[str, object]] = []
    top1_compare_rows: list[dict[str, object]] = []

    for split in args.splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split not found: {split_dir}")

        for mode in args.retrieval_modes:
            topn_values = args.topn_values if mode == "multi_topn_avg" else [1]

            for topn in topn_values:
                for crop_profile in crop_profiles:
                    report = evaluate_split(
                        split_name=split,
                        split_dir=split_dir,
                        interpreter=interpreter,
                        input_details=input_details,
                        output_details=output_details,
                        refs=refs,
                        img_size=args.img_size,
                        retrieval_mode=mode,
                        topn=topn,
                        crop_profile=crop_profile,
                        detector=detector,
                        limit_per_class=args.limit_per_class,
                    )
                    split_reports.append(report)

                    profile_name = str(crop_profile["name"])
                    print(
                        f"[{split}] mode={mode} topn={topn} crop={profile_name} samples={report['num_samples']} "
                        f"R@1={report['recall@1']:.4f} R@3={report['recall@3']:.4f} R@5={report['recall@5']:.4f}"
                    )

                    for row in report["per_class"]:
                        all_per_class_rows.append(
                            {
                                **row,
                                "retrieval_mode": mode,
                                "topn": topn,
                                "crop_profile": profile_name,
                            }
                        )

                    experiment_rows.append(
                        {
                            "split": split,
                            "retrieval_mode": mode,
                            "topn": topn,
                            "crop_profile": profile_name,
                            "num_samples": report["num_samples"],
                            "detector_failures": report["detector_failures"],
                            "recall@1": report["recall@1"],
                            "recall@3": report["recall@3"],
                            "recall@5": report["recall@5"],
                        }
                    )

                    for row in report["top1_compare"]:
                        top1_compare_rows.append(
                            {
                                "split": split,
                                "retrieval_mode": mode,
                                "topn": topn,
                                "crop_profile": profile_name,
                                **row,
                            }
                        )

    summary = {
        "model": str(model_path),
        "reference_json": str(reference_json),
        "dataset_root": str(dataset_root),
        "img_size": args.img_size,
        "retrieval_modes": args.retrieval_modes,
        "topn_values": args.topn_values,
        "crop_profiles": crop_profiles,
        "preprocessing": {
            "exif_transpose": True,
            "resize": [args.img_size, args.img_size],
            "channel_order": "RGB",
            "dtype": "float32",
            "input_scale": "0..255",
            "l2_normalize_embedding": True,
            "similarity_metric": "cosine",
        },
        "splits": split_reports,
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "retrieval_mode",
                "topn",
                "crop_profile",
                "label",
                "count",
                "recall@1",
                "recall@3",
                "recall@5",
            ],
        )
        writer.writeheader()
        writer.writerows(all_per_class_rows)

    experiments_csv.parent.mkdir(parents=True, exist_ok=True)
    with experiments_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "retrieval_mode",
                "topn",
                "crop_profile",
                "num_samples",
                "detector_failures",
                "recall@1",
                "recall@3",
                "recall@5",
            ],
        )
        writer.writeheader()
        writer.writerows(experiment_rows)

    top1_compare_csv.parent.mkdir(parents=True, exist_ok=True)
    with top1_compare_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "retrieval_mode",
                "topn",
                "crop_profile",
                "true_label",
                "pred_label",
                "count",
            ],
        )
        writer.writeheader()
        writer.writerows(top1_compare_rows)

    print("Saved summary:", report_json)
    print("Saved per-class report:", report_csv)
    print("Saved experiment grid:", experiments_csv)
    print("Saved top1 compare:", top1_compare_csv)


if __name__ == "__main__":
    main()
