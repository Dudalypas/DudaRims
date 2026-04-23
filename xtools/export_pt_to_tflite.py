from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a YOLO .pt model to TFLite using Ultralytics."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to source .pt model",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export float16 TFLite when supported",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export int8 TFLite (requires dataset/calibration in some cases)",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Include NMS in exported model when supported",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic shapes when supported",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Export device, usually "cpu"',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = YOLO(str(args.model))

    export_kwargs = {
        "format": "tflite",
        "imgsz": args.imgsz,
        "device": args.device,
    }

    if args.half:
        export_kwargs["half"] = True
    if args.int8:
        export_kwargs["int8"] = True
    if args.nms:
        export_kwargs["nms"] = True
    if args.dynamic:
        export_kwargs["dynamic"] = True

    print("Export args:")
    for k, v in export_kwargs.items():
        print(f"  {k}={v}")

    exported_path = model.export(**export_kwargs)
    print(f"\nExport complete: {exported_path}")


if __name__ == "__main__":
    main()