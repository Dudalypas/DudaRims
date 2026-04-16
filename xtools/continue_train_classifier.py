from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError(
        "matplotlib is required for confusion matrix outputs. Install with: pip install matplotlib"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(r"C:\Users\vilja\Desktop\Training_split_cropped")

# Requested base model identifier in this project is a TFLite export.
# Fine-tuning cannot run from TFLite directly, so we verify this file exists
# and then continue from its matching trainable Keras checkpoint.
BASE_TFLITE_PATH = PROJECT_ROOT / "trained_cropped_classifier" / "wheel_classifier_v4.tflite"
BASE_KERAS_PATH = PROJECT_ROOT / "trained_cropped_classifier" / "best copy.keras"
BASE_LABELS_PATH = PROJECT_ROOT / "trained_cropped_classifier" / "labels.txt"

OUTPUT_DIR = PROJECT_ROOT / "continued_training_output"
BEST_CONTINUED_MODEL_PATH = OUTPUT_DIR / "continued_best.keras"
FINAL_CONTINUED_MODEL_PATH = OUTPUT_DIR / "continued_final.keras"
CONTINUED_BEST_TFLITE_PATH = OUTPUT_DIR / "continued_best_float32.tflite"
CONTINUED_LABELS_PATH = OUTPUT_DIR / "labels.txt"
TRAIN_HISTORY_CSV = OUTPUT_DIR / "training_history.csv"
TRAIN_SUMMARY_JSON = OUTPUT_DIR / "training_summary.json"
TRAIN_SUMMARY_TXT = OUTPUT_DIR / "training_summary.txt"
TRAIN_CURVES_PNG = OUTPUT_DIR / "training_curves.png"

TEST_METRICS_TXT = OUTPUT_DIR / "test_metrics.txt"
PER_CLASS_METRICS_CSV = OUTPUT_DIR / "per_class_metrics.csv"
CONFUSION_MATRIX_PNG = OUTPUT_DIR / "confusion_matrix.png"
CONFUSION_MATRIX_NORMALIZED_PNG = OUTPUT_DIR / "confusion_matrix_normalized.png"
CLASSIFICATION_REPORT_TXT = OUTPUT_DIR / "classification_report.txt"

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".tiff", ".tif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue-training wheel classifier from existing best model.")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--base-tflite", type=str, default=str(BASE_TFLITE_PATH))
    parser.add_argument("--base-keras", type=str, default=str(BASE_KERAS_PATH))
    parser.add_argument("--base-labels", type=str, default=str(BASE_LABELS_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    items: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            items.append(p)
    items.sort()
    return items


def infer_label_from_split_path(image_path: Path, split_dir: Path) -> str:
    rel = image_path.relative_to(split_dir)
    if len(rel.parts) < 2:
        raise ValueError(f"Image is not inside class folder: {image_path}")
    return rel.parts[0]


def discover_class_names(train_dir: Path, val_dir: Path, test_dir: Path) -> list[str]:
    train_classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    val_classes = sorted([p.name for p in val_dir.iterdir() if p.is_dir()])
    test_classes = sorted([p.name for p in test_dir.iterdir() if p.is_dir()])

    if not train_classes:
        raise RuntimeError(f"No class directories found in {train_dir}")
    if train_classes != val_classes or train_classes != test_classes:
        raise RuntimeError(
            "Class folder mismatch across train/val/test. "
            f"train={train_classes}, val={val_classes}, test={test_classes}"
        )
    return train_classes


def load_and_preprocess_for_classifier(path_obj: tf.Tensor | bytes, img_size: int) -> np.ndarray:
    """Match current classifier preprocessing: EXIF -> RGB -> resize -> float32 0..255."""
    if hasattr(path_obj, "numpy"):
        raw = path_obj.numpy()
    else:
        raw = path_obj

    if isinstance(raw, np.ndarray):
        raw = raw.item()

    if isinstance(raw, (bytes, bytearray)):
        image_path = Path(raw.decode("utf-8"))
    else:
        image_path = Path(str(raw))

    with Image.open(image_path) as img:
        oriented = ImageOps.exif_transpose(img).convert("RGB")
        resized = oriented.resize((img_size, img_size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _tf_load_image(path: tf.Tensor, label: tf.Tensor, img_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.py_function(
        func=lambda p: load_and_preprocess_for_classifier(p, img_size),
        inp=[path],
        Tout=tf.float32,
    )
    image.set_shape((img_size, img_size, 3))
    return image, tf.cast(label, tf.int32)


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
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def collect_labeled_paths(split_dir: Path, class_to_idx: dict[str, int]) -> tuple[list[str], list[int]]:
    paths: list[str] = []
    labels: list[int] = []

    for image_path in list_images(split_dir):
        label = infer_label_from_split_path(image_path, split_dir)
        idx = class_to_idx.get(label)
        if idx is None:
            continue
        paths.append(str(image_path))
        labels.append(idx)

    if not paths:
        raise RuntimeError(f"No images found in split: {split_dir}")
    return paths, labels


def build_datasets(
    dataset_root: Path,
    class_names: list[str],
    img_size: int,
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    class_to_idx = {name: i for i, name in enumerate(class_names)}

    train_paths, train_labels = collect_labeled_paths(train_dir, class_to_idx)
    val_paths, val_labels = collect_labeled_paths(val_dir, class_to_idx)
    test_paths, test_labels = collect_labeled_paths(test_dir, class_to_idx)

    train_ds = make_dataset_from_paths(train_paths, train_labels, img_size, batch_size, True, seed)
    val_ds = make_dataset_from_paths(val_paths, val_labels, img_size, batch_size, False, seed)
    test_ds = make_dataset_from_paths(test_paths, test_labels, img_size, batch_size, False, seed)
    return train_ds, val_ds, test_ds


def read_labels(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise RuntimeError(f"No labels found in {labels_path}")
    return labels


def export_tflite_float32(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def save_confusion_matrix_image(cm: np.ndarray, class_names: list[str], out_path: Path, title: str) -> None:
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
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def evaluate_on_test(
    model: tf.keras.Model,
    class_names: list[str],
    dataset_root: Path,
    img_size: int,
    output_dir: Path,
) -> dict[str, float]:
    test_dir = dataset_root / "test"
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    images = list_images(test_dir)
    if not images:
        raise RuntimeError(f"No test images found in {test_dir}")

    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    per_class_total = np.zeros(len(class_names), dtype=np.int64)
    per_class_top1 = np.zeros(len(class_names), dtype=np.int64)
    per_class_top3 = np.zeros(len(class_names), dtype=np.int64)

    top1_ok = 0
    top3_ok = 0
    total = 0

    for idx, image_path in enumerate(images, start=1):
        true_label = infer_label_from_split_path(image_path, test_dir)
        true_idx = class_to_idx.get(true_label)
        if true_idx is None:
            continue

        print(f"[test {idx}/{len(images)}] class={true_label} image={image_path}")

        arr = load_and_preprocess_for_classifier(str(image_path).encode("utf-8"), img_size)
        batch = np.expand_dims(arr, axis=0)
        logits = model(batch, training=False).numpy()[0]
        probs = softmax(np.asarray(logits, dtype=np.float64))

        top3_idx = np.argsort(probs)[::-1][:3]
        pred_idx = int(top3_idx[0])

        cm[true_idx, pred_idx] += 1
        per_class_total[true_idx] += 1
        total += 1

        if pred_idx == true_idx:
            top1_ok += 1
            per_class_top1[true_idx] += 1
        if true_idx in top3_idx:
            top3_ok += 1
            per_class_top3[true_idx] += 1

    if total == 0:
        raise RuntimeError("No test images were evaluated.")

    row_sum = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64), where=row_sum != 0)

    save_confusion_matrix_image(cm, class_names, output_dir / "confusion_matrix.png", "Test Confusion Matrix")
    save_confusion_matrix_image(
        cm_normalized,
        class_names,
        output_dir / "confusion_matrix_normalized.png",
        "Test Normalized Confusion Matrix",
    )

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)

    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
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

    with (output_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
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

    with (output_dir / "test_metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"top1_accuracy: {top1_ok / total:.6f}\n")
        f.write(f"top3_accuracy: {top3_ok / total:.6f}\n")
        f.write(f"image_count: {total}\n")

    return {
        "top1_accuracy": float(top1_ok / total),
        "top3_accuracy": float(top3_ok / total),
        "image_count": float(total),
    }


def save_training_curves(history: tf.keras.callbacks.History, out_path: Path) -> None:
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
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    base_tflite = Path(args.base_tflite)
    base_keras = Path(args.base_keras)
    base_labels = Path(args.base_labels)
    output_dir = Path(args.output_dir)

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    for split in (train_dir, val_dir, test_dir):
        if not split.exists():
            raise FileNotFoundError(f"Required split directory not found: {split}")

    if not base_tflite.exists():
        raise FileNotFoundError(
            f"Requested base TFLite model not found: {base_tflite}. "
            "Cannot verify the requested wheel_classifier_cropped_best_float32 artifact."
        )

    if not base_keras.exists():
        raise RuntimeError(
            "Cannot continue training from the requested base because only TFLite is available. "
            "TFLite is an inference format and does not contain a trainable optimizer/graph state. "
            f"Expected trainable Keras model at: {base_keras}"
        )

    # Strict class-order verification across dataset and labels.
    dataset_class_names = discover_class_names(train_dir, val_dir, test_dir)
    labels = read_labels(base_labels)

    if labels != dataset_class_names:
        raise RuntimeError(
            "Class mismatch between dataset and base model labels. "
            f"labels={labels}, dataset={dataset_class_names}"
        )

    print("Starting model path (requested):", base_tflite)
    print("Training continuation source model:", base_keras)
    print("Dataset root:", dataset_root)
    print("Class count:", len(dataset_class_names))

    model = tf.keras.models.load_model(base_keras)

    # Verify output dimension compatibility with class count.
    output_units = int(model.output_shape[-1])
    if output_units != len(dataset_class_names):
        raise RuntimeError(
            "Loaded model output class count does not match dataset/labels. "
            f"model_output_units={output_units}, expected={len(dataset_class_names)}"
        )

    # Fine-tuning uses lower LR than initial training.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top1"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )

    train_ds, val_ds, test_ds = build_datasets(
        dataset_root=dataset_root,
        class_names=dataset_class_names,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    CONTINUED_LABELS_PATH.write_text("\n".join(dataset_class_names) + "\n", encoding="utf-8")

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_CONTINUED_MODEL_PATH),
            monitor="val_top1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_top1",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(TRAIN_HISTORY_CSV), append=False),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(FINAL_CONTINUED_MODEL_PATH)
    save_training_curves(history, TRAIN_CURVES_PNG)

    val_metrics = model.evaluate(val_ds, return_dict=True, verbose=1)

    val_top1_hist = history.history.get("val_top1", [])
    val_top3_hist = history.history.get("val_top3", [])
    if not val_top1_hist:
        raise RuntimeError("Training history did not contain val_top1.")

    best_epoch_idx = int(np.argmax(val_top1_hist))
    best_epoch = best_epoch_idx + 1
    best_val_top1 = float(val_top1_hist[best_epoch_idx])
    best_val_top3 = float(val_top3_hist[best_epoch_idx]) if val_top3_hist else 0.0

    best_model = tf.keras.models.load_model(BEST_CONTINUED_MODEL_PATH)
    export_tflite_float32(best_model, CONTINUED_BEST_TFLITE_PATH)

    test_metrics_from_keras = model.evaluate(test_ds, return_dict=True, verbose=1)
    manual_test_metrics = evaluate_on_test(
        model=model,
        class_names=dataset_class_names,
        dataset_root=dataset_root,
        img_size=args.img_size,
        output_dir=output_dir,
    )

    summary = {
        "requested_base_tflite": str(base_tflite),
        "continued_from_keras": str(base_keras),
        "dataset_root": str(dataset_root),
        "class_count": len(dataset_class_names),
        "class_names": dataset_class_names,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "best_val_top3": best_val_top3,
        "final_val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics_keras": {k: float(v) for k, v in test_metrics_from_keras.items()},
        "test_metrics_manual": manual_test_metrics,
    }
    TRAIN_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with TRAIN_SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write(f"starting_model_tflite: {base_tflite}\n")
        f.write(f"continued_from_keras: {base_keras}\n")
        f.write(f"class_count: {len(dataset_class_names)}\n")
        f.write(f"best_val_top1: {best_val_top1:.6f}\n")
        f.write(f"best_val_top3: {best_val_top3:.6f}\n")
        f.write(f"best_epoch: {best_epoch}\n")

    print("\nContinue-training completed")
    print("Starting model path (requested):", base_tflite)
    print("Continued-from Keras model:", base_keras)
    print("Class count:", len(dataset_class_names))
    print(f"Best val top-1: {best_val_top1:.6f}")
    print(f"Best val top-3: {best_val_top3:.6f}")
    print(f"Best epoch: {best_epoch}")
    print("Best continued model:", BEST_CONTINUED_MODEL_PATH)
    print("Final continued model:", FINAL_CONTINUED_MODEL_PATH)
    print("Continued float32 TFLite:", CONTINUED_BEST_TFLITE_PATH)
    print("Labels:", CONTINUED_LABELS_PATH)
    print("Training history:", TRAIN_HISTORY_CSV)
    print("Training summary:", TRAIN_SUMMARY_JSON)
    print("Training curves:", TRAIN_CURVES_PNG)
    print("Test metrics:", TEST_METRICS_TXT)
    print("Per-class metrics:", PER_CLASS_METRICS_CSV)
    print("Confusion matrix:", CONFUSION_MATRIX_PNG)
    print("Normalized confusion matrix:", CONFUSION_MATRIX_NORMALIZED_PNG)
    print("Classification report:", CLASSIFICATION_REPORT_TXT)


if __name__ == "__main__":
    main()
