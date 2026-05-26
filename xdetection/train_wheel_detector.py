from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "Wheel_Detection"
DATA_YAML = DATASET_ROOT / "data.yaml"
PROJECT_DIR = DATASET_ROOT / "runs"
RUN_NAME = "wheel_det"


def ensure_required_structure() -> None:
    required = [
        DATASET_ROOT / "train" / "images",
        DATASET_ROOT / "train" / "labels",
        DATASET_ROOT / "val" / "images",
        DATASET_ROOT / "val" / "labels",
        DATASET_ROOT / "test" / "images",
        DATASET_ROOT / "test" / "labels",
        DATA_YAML,
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        print("[train] missing dataset paths:")
        for p in missing:
            print(f"  - {p}")
        sys.exit(1)


def verify_data_yaml() -> None:
    with DATA_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    expected_paths = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
    }

    bad_paths = [k for k, v in expected_paths.items() if data.get(k) != v]
    if bad_paths:
        print("[train] data.yaml split paths:")
        for k, v in expected_paths.items():
            print(f"  {k}: {v}")
        print("[train] fix data.yaml first")
        sys.exit(1)

    names = data.get("names", [])
    if isinstance(names, list):
        if names != ["wheel"]:
            print(f"[train] names: {names}, expected ['wheel']")
    elif isinstance(names, dict):
        name_values = [str(v) for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
        if name_values != ["wheel"]:
            print(f"[train] names: {name_values}, expected ['wheel']")


def print_torch_info() -> None:
    print(f"[train] torch: {torch.__version__}")
    print(f"[train] cuda runtime: {torch.version.cuda}")
    print(f"[train] cuda available: {torch.cuda.is_available()}")
    print(f"[train] cuda devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[train] gpu: {torch.cuda.get_device_name(0)}")
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[train] gpu mem: {total_gb:.2f} GB")
    else:
        print("[train] cuda unavailable")


def choose_device() -> int | str:
    if torch.cuda.is_available():
        return 0
    return "cpu"


def pick_safe_batch_size(device: int | str) -> int:
    if device == "cpu":
        return 4

    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_gb >= 10:
        return 16
    if total_gb >= 6:
        return 8
    return 4


def pick_nano_model_name() -> str:
    # Pirmiausia bandom naujesnius nano checkpointus, tada fallback
    candidates = ["yolo11n.pt", "yolov8n.pt"]

    for name in candidates:
        try:
            YOLO(name)
            print(f"[train] model: {name}")
            return name
        except Exception as e:
            print(f"[train] skip {name}: {e}")

    raise RuntimeError("[train] no supported nano model")


def train_model() -> Path:
    ensure_required_structure()
    verify_data_yaml()
    print_torch_info()

    device = choose_device()
    batch_size = pick_safe_batch_size(device)
    model_name = pick_nano_model_name()
    model = YOLO(model_name)

    if device == "cpu":
        print("[train] cpu mode")
    else:
        print(f"[train] gpu batch={batch_size}")

    results = model.train(
        data=str(DATA_YAML),
        imgsz=640,
        epochs=80,
        patience=20,
        batch=batch_size,
        workers=0,          # Maziau problemu windowsu aplinkoje
        device=device,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    save_dir = Path(getattr(results, "save_dir", PROJECT_DIR / RUN_NAME))
    best_weights = save_dir / "weights" / "best.pt"

    if not best_weights.exists():
        fallback = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
        if fallback.exists():
            best_weights = fallback
        else:
            raise FileNotFoundError(f"Could not find best weights at {best_weights}")

    print(f"[train] best: {best_weights}")
    return best_weights


def validate_best(best_weights: Path) -> None:
    device = 0 if torch.cuda.is_available() else "cpu"
    print("[train] validate")
    model = YOLO(str(best_weights))
    model.val(
        data=str(DATA_YAML),
        imgsz=640,
        batch=8 if device == 0 else 4,
        workers=0,
        device=device,
    )


def export_tflite(best_weights: Path) -> None:
    print("[train] export tflite")
    model = YOLO(str(best_weights))
    exported = model.export(format="tflite", imgsz=640)

    if isinstance(exported, (list, tuple)):
        for path in exported:
            print(f"[train] export: {path}")
    else:
        print(f"[train] export: {exported}")


def main() -> int:
    try:
        best_weights = train_model()
        validate_best(best_weights)
        export_tflite(best_weights)
        print("[train] done")
        return 0
    except KeyboardInterrupt:
        print("\n[train] interrupted")
        return 130
    except Exception as e:
        print(f"[train] error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())