import argparse
from pathlib import Path

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wheel classifier and export TFLite models")
    parser.add_argument("--dataset-dir", type=str, default=r"C:\Users\vilja\Desktop\WHEEL22_ready")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def cast_to_float32(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    # IMPORTANT: MobileNetV3 default preprocessing in Keras expects float images in [0, 255].
    # Do not scale to [0, 1] or [-1, 1] for this setup.
    return tf.cast(image, tf.float32), label


def build_datasets(
    dataset_dir: Path,
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"

    for split_dir in (train_dir, val_dir, test_dir):
        if not split_dir.exists():
            raise FileNotFoundError(f"Required split directory not found: {split_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    train_ds = train_ds.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes: int, img_size: int) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image", dtype=tf.float32)
    x = data_augmentation(inputs)

    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="wheel_mobilenetv3small")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top1"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )
    return model


def write_labels(labels_path: Path, class_names: list[str]) -> None:
    labels_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")


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


def print_tflite_io_details(tflite_path: Path) -> None:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nFloat32 TFLite sanity-check")
    print("Input tensor details:")
    for details in input_details:
        print(details)

    print("Output tensor details:")
    for details in output_details:
        print(details)


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    image_size = (args.img_size, args.img_size)
    train_ds, val_ds, test_ds, class_names = build_datasets(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    labels_path = dataset_dir / "labels.txt"
    best_model_path = dataset_dir / "best.keras"
    final_model_path = dataset_dir / "wheel_classifier.keras"
    best_tflite_float32_path = dataset_dir / "wheel_classifier_best_float32.tflite"
    best_tflite_fp16_path = dataset_dir / "wheel_classifier_best_fp16.tflite"

    write_labels(labels_path, class_names)
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Saved labels to: {labels_path}")

    model = build_model(num_classes=len(class_names), img_size=args.img_size)
    model.optimizer.learning_rate.assign(args.learning_rate)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_top1",
            mode="max",
            save_best_only=True,
            verbose=1,
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(final_model_path)
    print(f"Saved final model to: {final_model_path}")

    test_results = model.evaluate(test_ds, return_dict=True, verbose=1)
    print("Test results:")
    for metric_name, metric_value in test_results.items():
        print(f"  {metric_name}: {metric_value:.6f}")

    best_model = tf.keras.models.load_model(best_model_path)
    export_tflite_float32(best_model, best_tflite_float32_path)
    print(f"Exported float32 TFLite: {best_tflite_float32_path}")

    export_tflite_fp16(best_model, best_tflite_fp16_path)
    print(f"Exported fp16 TFLite: {best_tflite_fp16_path}")

    print_tflite_io_details(best_tflite_float32_path)


if __name__ == "__main__":
    main()