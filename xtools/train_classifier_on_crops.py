from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

DEFAULT_DATASET_ROOT = Path(r"C:\Users\vilja\Desktop\PROD_TRAIN_CROPPED")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "trained_cropped_classifier"

BEST_MODEL_PATH = "best.keras"
FINAL_MODEL_PATH = "final.keras"
TRAIN_HISTORY_CSV = "training_history.csv"
TRAIN_HISTORY_JSON = "training_history.json"
TRAIN_SUMMARY_JSON = "training_summary.json"
LABELS_PATH = "labels.txt"
BEST_TFLITE_FLOAT32_PATH = "classifier_best_float32.tflite"
BEST_TFLITE_FP16_PATH = "classifier_best_fp16.tflite"


@dataclass
class SplitDatasets:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    class_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train wheel classifier from already processed train/val/test crops."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable light augmentation (flip + small rotation).",
    )
    return parser.parse_args()


def validate_dataset_root(dataset_root: Path) -> tuple[Path, Path, Path]:
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    for required in (train_dir, val_dir, test_dir):
        if not required.exists() or not required.is_dir():
            raise FileNotFoundError(f"Required split folder missing: {required}")

    return train_dir, val_dir, test_dir


def _normalize(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def _load_split(
    split_dir: Path,
    img_size: int,
    batch_size: int,
    seed: int,
    class_names: list[str] | None,
    shuffle: bool,
) -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        interpolation="bilinear",
        shuffle=shuffle,
        seed=seed,
        class_names=class_names,
    )
    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(
    dataset_root: Path,
    img_size: int,
    batch_size: int,
    seed: int,
) -> SplitDatasets:
    train_dir, val_dir, test_dir = validate_dataset_root(dataset_root)

    raw_train = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        interpolation="bilinear",
        shuffle=True,
        seed=seed,
    )
    class_names = list(raw_train.class_names)

    train_ds = raw_train.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

    val_ds = _load_split(
        split_dir=val_dir,
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
        class_names=class_names,
        shuffle=False,
    )
    test_ds = _load_split(
        split_dir=test_dir,
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
        class_names=class_names,
        shuffle=False,
    )

    return SplitDatasets(train=train_ds, val=val_ds, test=test_ds, class_names=class_names)


def build_model(
    num_classes: int,
    img_size: int,
    learning_rate: float,
    dropout_rate: float,
    use_augmentation: bool,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")

    x = inputs
    if use_augmentation:
        aug = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.03),
            ],
            name="light_augmentation",
        )
        x = aug(x)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_head")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_head")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="wheel_classifier_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
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


def save_history_csv(history: tf.keras.callbacks.History, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = history.history
    keys = list(rows.keys())
    n = len(rows[keys[0]]) if keys else 0

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", *keys])
        writer.writeheader()
        for i in range(n):
            row = {"epoch": i + 1}
            for k in keys:
                row[k] = rows[k][i]
            writer.writerow(row)


def train(args: argparse.Namespace) -> None:
    tf.keras.utils.set_random_seed(args.seed)

    datasets = build_datasets(
        dataset_root=args.dataset_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    artifacts_dir: Path = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = artifacts_dir / BEST_MODEL_PATH
    final_model_path = artifacts_dir / FINAL_MODEL_PATH
    history_csv_path = artifacts_dir / TRAIN_HISTORY_CSV
    history_json_path = artifacts_dir / TRAIN_HISTORY_JSON
    summary_json_path = artifacts_dir / TRAIN_SUMMARY_JSON
    labels_path = artifacts_dir / LABELS_PATH
    best_tflite_float32_path = artifacts_dir / BEST_TFLITE_FLOAT32_PATH
    best_tflite_fp16_path = artifacts_dir / BEST_TFLITE_FP16_PATH

    labels_path.write_text("\n".join(datasets.class_names) + "\n", encoding="utf-8")

    model = build_model(
        num_classes=len(datasets.class_names),
        img_size=args.img_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout,
        use_augmentation=not args.no_augmentation,
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        datasets.train,
        validation_data=datasets.val,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(final_model_path)
    save_history_csv(history, history_csv_path)
    history_json_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    val_metrics = model.evaluate(datasets.val, return_dict=True, verbose=1)
    test_metrics = model.evaluate(datasets.test, return_dict=True, verbose=1)

    best_model = tf.keras.models.load_model(best_model_path)
    export_tflite_float32(best_model, best_tflite_float32_path)
    export_tflite_fp16(best_model, best_tflite_fp16_path)

    summary = {
        "dataset_root": str(args.dataset_root),
        "class_names": datasets.class_names,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "learning_rate": args.learning_rate,
        "augmentation_enabled": not args.no_augmentation,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "best_tflite_float32": str(best_tflite_float32_path),
        "best_tflite_fp16": str(best_tflite_fp16_path),
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTraining complete")
    print("Best model:", best_model_path)
    print("Final model:", final_model_path)
    print("Labels:", labels_path)
    print("History CSV:", history_csv_path)
    print("History JSON:", history_json_path)
    print("Summary JSON:", summary_json_path)
    print("TFLite float32:", best_tflite_float32_path)
    print("TFLite fp16:", best_tflite_fp16_path)


# Easy export snippet for later use in notebooks/scripts:
# best_model = tf.keras.models.load_model("trained_cropped_classifier/best.keras")
# export_tflite_float32(best_model, Path("trained_cropped_classifier/classifier_best_float32.tflite"))
# export_tflite_fp16(best_model, Path("trained_cropped_classifier/classifier_best_fp16.tflite"))


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
