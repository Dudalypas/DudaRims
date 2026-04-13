from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError(
        "matplotlib is required for confusion matrix images. Install with: pip install matplotlib"
    ) from exc

CROPPED_SPLIT_ROOT = Path(r"C:\Users\vilja\Desktop\WHEEL22_ready")
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "trained_cropped_classifier"

MODEL_PATH = ARTIFACTS_DIR / "best.keras"
LABELS_PATH = ARTIFACTS_DIR / "labels.txt"
EVAL_OUTPUT_DIR = ARTIFACTS_DIR / "evaluation"

CLASSIFICATION_REPORT_TXT = EVAL_OUTPUT_DIR / "classification_report.txt"
PER_CLASS_METRICS_CSV = EVAL_OUTPUT_DIR / "per_class_metrics.csv"
CONFUSION_MATRIX_PNG = EVAL_OUTPUT_DIR / "confusion_matrix.png"
CONFUSION_MATRIX_NORMALIZED_PNG = EVAL_OUTPUT_DIR / "confusion_matrix_normalized.png"
ERROR_LOG = EVAL_OUTPUT_DIR / "evaluation_errors.log"

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cropped classifier on val split.")
    parser.add_argument("--cropped-root", type=str, default=str(CROPPED_SPLIT_ROOT))
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--labels-path", type=str, default=str(LABELS_PATH))
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            paths.append(p)
    paths.sort()
    return paths


def preprocess_image(image_path: Path, target_h: int, target_w: int) -> np.ndarray:
    with Image.open(image_path) as img:
        oriented = ImageOps.exif_transpose(img).convert("RGB")
        resized = oriented.resize((target_w, target_h), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def save_confusion_matrix_image(cm: np.ndarray, class_names: list[str], output_path: Path, title: str) -> None:
    fig_size = max(8, len(class_names) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            display = f"{value:.2f}" if isinstance(value, float) else str(value)
            ax.text(
                j,
                i,
                display,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=7,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate(
    cropped_root: Path,
    model_path: Path,
    labels_path: Path,
) -> None:
    val_dir = cropped_root / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation split not found: {val_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    class_names = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not class_names:
        raise RuntimeError(f"No labels found in {labels_path}")

    model = tf.keras.models.load_model(model_path)

    input_shape = model.input_shape
    if not isinstance(input_shape, tuple) or len(input_shape) != 4:
        raise RuntimeError(f"Unexpected model input shape: {input_shape}")
    target_h = int(input_shape[1])
    target_w = int(input_shape[2])

    images = list_images(val_dir)
    if not images:
        raise RuntimeError(f"No validation images found in {val_dir}")

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    top1_ok = 0
    top3_ok = 0
    total = 0
    errors: list[str] = []

    per_class_total = np.zeros(len(class_names), dtype=np.int64)
    per_class_top1 = np.zeros(len(class_names), dtype=np.int64)
    per_class_top3 = np.zeros(len(class_names), dtype=np.int64)

    for idx, image_path in enumerate(images, start=1):
        rel = image_path.relative_to(val_dir)
        if len(rel.parts) < 2:
            errors.append(f"INVALID_PATH | {image_path} | missing class folder")
            continue

        true_label = rel.parts[0]
        true_idx = class_to_idx.get(true_label)
        if true_idx is None:
            errors.append(f"UNKNOWN_LABEL | {image_path} | label={true_label}")
            continue

        try:
            print(f"[{idx}/{len(images)}] class={true_label} image={image_path}")
            batch = preprocess_image(image_path, target_h=target_h, target_w=target_w)
            logits = model(batch, training=False).numpy()[0]
            probs = softmax(np.asarray(logits, dtype=np.float64))

            top3_idx = np.argsort(probs)[::-1][:3]
            pred_idx = int(top3_idx[0])

            cm[true_idx, pred_idx] += 1
            total += 1
            per_class_total[true_idx] += 1

            if pred_idx == true_idx:
                top1_ok += 1
                per_class_top1[true_idx] += 1
            if true_idx in top3_idx:
                top3_ok += 1
                per_class_top3[true_idx] += 1
        except Exception as exc:
            errors.append(f"IMAGE_FAIL | {image_path} | {exc}")
            continue

    if total == 0:
        raise RuntimeError("No images were successfully evaluated.")

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    row_sum = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm,
        row_sum,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sum != 0,
    )

    save_confusion_matrix_image(cm, class_names, CONFUSION_MATRIX_PNG, "Confusion Matrix")
    save_confusion_matrix_image(cm_normalized, class_names, CONFUSION_MATRIX_NORMALIZED_PNG, "Normalized Confusion Matrix")

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)

    with PER_CLASS_METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_name",
                "support",
                "top1_accuracy",
                "top3_accuracy",
                "precision",
                "recall",
                "f1_score",
            ],
        )
        writer.writeheader()
        for i, class_name in enumerate(class_names):
            support = int(per_class_total[i])
            top1_acc = float(per_class_top1[i] / support) if support else 0.0
            top3_acc = float(per_class_top3[i] / support) if support else 0.0
            writer.writerow(
                {
                    "class_name": class_name,
                    "support": support,
                    "top1_accuracy": f"{top1_acc:.6f}",
                    "top3_accuracy": f"{top3_acc:.6f}",
                    "precision": f"{precision[i]:.6f}",
                    "recall": f"{recall[i]:.6f}",
                    "f1_score": f"{f1[i]:.6f}",
                }
            )

    with CLASSIFICATION_REPORT_TXT.open("w", encoding="utf-8") as f:
        f.write(f"Total evaluated images: {total}\n")
        f.write(f"Top-1 accuracy: {top1_ok / total:.6f}\n")
        f.write(f"Top-3 accuracy: {top3_ok / total:.6f}\n\n")
        f.write("Per-class metrics\n")
        f.write("class_name,support,top1_accuracy,top3_accuracy,precision,recall,f1_score\n")
        for i, class_name in enumerate(class_names):
            support = int(per_class_total[i])
            top1_acc = float(per_class_top1[i] / support) if support else 0.0
            top3_acc = float(per_class_top3[i] / support) if support else 0.0
            f.write(
                f"{class_name},{support},{top1_acc:.6f},{top3_acc:.6f},{precision[i]:.6f},{recall[i]:.6f},{f1[i]:.6f}\n"
            )

    if errors:
        ERROR_LOG.write_text("\n".join(errors), encoding="utf-8")
    elif ERROR_LOG.exists():
        ERROR_LOG.unlink()

    print("\nEvaluation completed")
    print(f"Top-1 accuracy: {top1_ok / total:.6f}")
    print(f"Top-3 accuracy: {top3_ok / total:.6f}")
    print("Per-class top-1 accuracy:")
    for i, class_name in enumerate(class_names):
        support = int(per_class_total[i])
        acc = float(per_class_top1[i] / support) if support else 0.0
        print(f"  {class_name}: {acc:.6f} (n={support})")

    print("\nSaved outputs:")
    print("Classification report:", CLASSIFICATION_REPORT_TXT)
    print("Per-class metrics CSV:", PER_CLASS_METRICS_CSV)
    print("Confusion matrix:", CONFUSION_MATRIX_PNG)
    print("Normalized confusion matrix:", CONFUSION_MATRIX_NORMALIZED_PNG)
    if ERROR_LOG.exists():
        print("Evaluation errors log:", ERROR_LOG)


def main() -> None:
    args = parse_args()
    evaluate(
        cropped_root=Path(args.cropped_root),
        model_path=Path(args.model_path),
        labels_path=Path(args.labels_path),
    )


if __name__ == "__main__":
    main()
