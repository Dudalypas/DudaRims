from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_MODEL = PROJECT_ROOT / "assets" / "models" / "best.keras"
DEFAULT_OUTPUT_MODEL = PROJECT_ROOT / "assets" / "models" / "wheel_embedding_cropped_float32.tflite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an embedding extractor from a trained Keras model"
    )
    parser.add_argument("--source-model", type=str, default=str(DEFAULT_SOURCE_MODEL))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_MODEL))
    parser.add_argument(
        "--penultimate-layer",
        type=str,
        default="",
        help="Layer name to use as embedding source",
    )
    parser.add_argument(
        "--export-fp16",
        action="store_true",
        help="Also export an fp16 variant",
    )
    return parser.parse_args()


def build_embedding_model(base_model: tf.keras.Model, penultimate_layer: str | None) -> tf.keras.Model:
    if penultimate_layer:
        try:
            embedding_source = base_model.get_layer(penultimate_layer).output
        except ValueError as exc:
            raise ValueError(f"Layer '{penultimate_layer}' not found in source model.") from exc
    else:
        if len(base_model.layers) < 2:
            raise ValueError("Source model does not have enough layers for penultimate extraction.")
        embedding_source = base_model.layers[-2].output

    embedding = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(tf.cast(x, tf.float32), axis=-1),
        name="l2_embedding",
    )(embedding_source)

    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=embedding,
        name=f"{base_model.name}_embedding",
    )
    return model


def export_float32_tflite(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)


def export_fp16_tflite(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)


def print_tflite_io(model_path: Path) -> None:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_tensor = interpreter.get_input_details()[0]
    output_tensor = interpreter.get_output_details()[0]

    print("[convert] model:", model_path)
    print("[convert] input:", input_tensor["shape"], input_tensor["dtype"])
    print("[convert] output:", output_tensor["shape"], output_tensor["dtype"])


def main() -> None:
    args = parse_args()

    source_model_path = Path(args.source_model)
    output_path = Path(args.output)

    if not source_model_path.exists():
        raise FileNotFoundError(f"Source model not found: {source_model_path}")

    source_model = tf.keras.models.load_model(source_model_path)
    embedding_model = build_embedding_model(
        source_model,
        args.penultimate_layer.strip() or None,
    )

    export_float32_tflite(embedding_model, output_path)
    print("[convert] float32:", output_path)
    print_tflite_io(output_path)

    if args.export_fp16:
        fp16_out = output_path.with_name(output_path.stem + "_fp16.tflite")
        export_fp16_tflite(embedding_model, fp16_out)
        print("[convert] fp16:", fp16_out)
        print_tflite_io(fp16_out)


if __name__ == "__main__":
    main()
