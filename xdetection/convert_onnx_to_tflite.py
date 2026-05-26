from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX_PATH = PROJECT_ROOT / "Wheel_Detection" / "runs" / "wheel_det" / "weights" / "best.onnx"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "Wheel_Detection" / "runs" / "wheel_det" / "weights" / "tflite_export"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a wheel detector ONNX model to TFLite")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH, help="Path to the ONNX model")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT_DIR, help="Output directory for conversion files")
    return parser.parse_args()


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Paleidzia komanda ir grizta su uzfiksuota isvestimi"""
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def find_tflite_files(base_dir: Path) -> list[Path]:
    """Randa konversijos TFLite failus"""
    return sorted(base_dir.rglob("*.tflite"))


def main() -> int:
    args = parse_args()
    onnx_path = Path(args.onnx)
    export_dir = Path(args.output_dir)

    # Tikrinam fiksuota ONNX ivesti
    if not onnx_path.exists():
        print(f"[ERROR] ONNX file not found: {onnx_path}")
        return 1

    # Sukuriam isvesties aplanka ir laikom visus tarpinius failus jame
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"[convert] out: {export_dir}")

    # Naudojam onnx2tf CLI, nes taip stabiliau konvertuoja
    onnx2tf_exe = shutil.which("onnx2tf")
    if onnx2tf_exe is None:
        print("[convert] missing onnx2tf")
        print("[convert] install setup_onnx2tf_env.ps1")
        return 1

    cmd = [
        onnx2tf_exe,
        "-i",
        str(onnx_path),
        "-o",
        str(export_dir),
    ]

    print("[convert] onnx2tf")
    print(f"[convert] cmd: {' '.join(cmd)}")

    result = run_command(cmd)

    if result.stdout:
        print("[convert] stdout")
        print(result.stdout)

    if result.returncode != 0:
        print("[convert] failed")
        if result.stderr:
            print("[convert] stderr")
            print(result.stderr)
        else:
            print("[convert] failed without stderr")
        return result.returncode

    tflite_files = find_tflite_files(export_dir)
    if not tflite_files:
        print("[convert] no tflite output")
        print(f"[convert] checked: {export_dir}")
        return 1

    # Isvedam sugeneruotus TFLite failus greitam pasirinkimui
    print("[convert] tflite files")
    for p in tflite_files:
        print(f"  - {p}")

    print(f"[convert] tflite: {tflite_files[0]}")
    print(f"[convert] out: {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
