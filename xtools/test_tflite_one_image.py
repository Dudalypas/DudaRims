from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Update this image path before running.
IMAGE_PATH = Path(r"C:\Users\vilja\Desktop\nivalis2.png")
DATASET_ROOT = Path(r"C:\Users\vilja\Desktop\WHEEL22_ready")
TFLITE_MODEL_PATH = DATASET_ROOT / "wheel_classifier_best_float32.tflite"
LABELS_PATH = DATASET_ROOT / "labels.txt"
IMG_SIZE = (224, 224)


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.txt not found: {labels_path}")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in: {labels_path}")
    return labels


def preprocess_image(image_path: Path, img_size: tuple[int, int]) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = img.resize(img_size)

    # IMPORTANT: Keep float32 pixels in [0, 255] for MobileNetV3 default preprocessing behavior.
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def main() -> None:
    labels = load_labels(LABELS_PATH)
    input_batch = preprocess_image(IMAGE_PATH, IMG_SIZE)

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details["index"], input_batch.astype(input_details["dtype"]))
    interpreter.invoke()

    logits = interpreter.get_tensor(output_details["index"])[0]
    probs = softmax(logits)

    top5_indices = np.argsort(probs)[::-1][:5]
    print("Top-5 predictions:")
    for rank, idx in enumerate(top5_indices, start=1):
        label = labels[idx] if idx < len(labels) else f"<missing-label-{idx}>"
        print(f"{rank}. {label}: {probs[idx]:.6f}")


if __name__ == "__main__":
    main()
