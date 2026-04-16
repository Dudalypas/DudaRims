from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("fitment_checker")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None


def _parse_bolt_and_pcd(value: Any) -> tuple[int | None, float | None]:
    if value is None:
        return None, None
    raw = str(value).strip().lower().replace(" ", "")
    if "x" not in raw:
        return None, None
    left, right = raw.split("x", 1)
    try:
        bolt_count = int(float(left))
    except ValueError:
        bolt_count = None
    try:
        pcd = float(right.replace(",", "."))
    except ValueError:
        pcd = None
    return bolt_count, pcd


def _upgrade_result(current: str, new: str) -> str:
    order = {"fits": 0, "caution": 1, "does_not_fit": 2}
    return new if order[new] > order[current] else current


def check_fitment(wheel: dict[str, Any], vehicle: dict[str, Any]) -> dict[str, Any]:
    result = "fits"
    reasons: list[str] = []

    wheel_pcd_str = wheel.get("pcd")
    vehicle_pcd_str = vehicle.get("pcd")
    wheel_bolt, wheel_pcd = _parse_bolt_and_pcd(wheel_pcd_str)
    vehicle_bolt, vehicle_pcd = _parse_bolt_and_pcd(vehicle_pcd_str)

    if wheel_pcd is None or vehicle_pcd is None:
        result = _upgrade_result(result, "caution")
        reasons.append("Missing PCD information for wheel or vehicle.")
    else:
        if abs(wheel_pcd - vehicle_pcd) > 0.1:
            result = _upgrade_result(result, "does_not_fit")
            reasons.append(f"PCD mismatch: wheel {wheel_pcd_str} vs vehicle {vehicle_pcd_str}.")

    if wheel_bolt is not None and vehicle_bolt is not None and wheel_bolt != vehicle_bolt:
        result = _upgrade_result(result, "does_not_fit")
        reasons.append(f"Bolt-count mismatch: wheel {wheel_bolt} vs vehicle {vehicle_bolt}.")

    wheel_cb = _to_float(wheel.get("cb"))
    vehicle_cb = _to_float(vehicle.get("center_bore_mm") or vehicle.get("cb"))
    if wheel_cb is None or vehicle_cb is None:
        result = _upgrade_result(result, "caution")
        reasons.append("Missing center bore value.")
    else:
        if wheel_cb < vehicle_cb:
            result = _upgrade_result(result, "does_not_fit")
            reasons.append(f"Wheel CB {wheel_cb} is smaller than vehicle requirement {vehicle_cb}.")
        elif wheel_cb > vehicle_cb:
            result = _upgrade_result(result, "caution")
            reasons.append(f"Wheel CB {wheel_cb} is larger than vehicle {vehicle_cb}; hub-centric rings may be required.")

    wheel_diameter = _to_float(wheel.get("diameter_in"))
    diameter_min = _to_float(vehicle.get("diameter_min_in"))
    diameter_max = _to_float(vehicle.get("diameter_max_in"))
    if wheel_diameter is None or diameter_min is None or diameter_max is None:
        result = _upgrade_result(result, "caution")
        reasons.append("Missing diameter range data.")
    else:
        if wheel_diameter < diameter_min or wheel_diameter > diameter_max:
            result = _upgrade_result(result, "does_not_fit")
            reasons.append(f"Wheel diameter {wheel_diameter} is outside allowed range {diameter_min}-{diameter_max}.")

    wheel_width = _to_float(wheel.get("width_j"))
    width_min = _to_float(vehicle.get("width_min_j"))
    width_max = _to_float(vehicle.get("width_max_j"))
    if wheel_width is None or width_min is None or width_max is None:
        result = _upgrade_result(result, "caution")
        reasons.append("Missing width range data.")
    else:
        if wheel_width < width_min - 0.3 or wheel_width > width_max + 0.3:
            result = _upgrade_result(result, "does_not_fit")
            reasons.append(f"Wheel width {wheel_width}J is clearly outside allowed range {width_min}-{width_max}J.")
        elif wheel_width < width_min or wheel_width > width_max:
            result = _upgrade_result(result, "caution")
            reasons.append(f"Wheel width {wheel_width}J is slightly outside range {width_min}-{width_max}J.")

    wheel_et = _to_float(wheel.get("et"))
    et_min = _to_float(vehicle.get("et_min"))
    et_max = _to_float(vehicle.get("et_max"))
    if wheel_et is None or et_min is None or et_max is None:
        result = _upgrade_result(result, "caution")
        reasons.append("Missing ET range data.")
    else:
        if wheel_et < et_min - 5 or wheel_et > et_max + 5:
            result = _upgrade_result(result, "does_not_fit")
            reasons.append(f"ET {wheel_et} is clearly outside allowed range {et_min}-{et_max}.")
        elif wheel_et < et_min or wheel_et > et_max:
            result = _upgrade_result(result, "caution")
            reasons.append(f"ET {wheel_et} is slightly outside range {et_min}-{et_max}.")

    if not reasons:
        reasons.append("All checked fields are within expected compatibility ranges.")

    return {"result": result, "reasons": reasons}


def build_fitment_examples(
    wheels_path: Path,
    models_path: Path,
    out_path: Path,
    limit: int,
) -> None:
    wheels_df = pd.read_csv(wheels_path)
    models_df = pd.read_csv(models_path)

    examples: list[dict[str, Any]] = []
    for _, wheel_row in wheels_df.head(limit).iterrows():
        wheel = wheel_row.to_dict()
        selected = models_df.head(min(limit, len(models_df)))
        for _, vehicle_row in selected.iterrows():
            vehicle = vehicle_row.to_dict()
            outcome = check_fitment(wheel=wheel, vehicle=vehicle)
            examples.append(
                {
                    "wheel": {
                        "wheel_class_name": wheel.get("wheel_class_name"),
                        "official_wheel_name": wheel.get("official_wheel_name"),
                        "diameter_in": wheel.get("diameter_in"),
                        "width_j": wheel.get("width_j"),
                        "et": wheel.get("et"),
                        "pcd": wheel.get("pcd"),
                        "cb": wheel.get("cb"),
                    },
                    "vehicle": {
                        "model": vehicle.get("model"),
                        "generation": vehicle.get("generation"),
                        "pcd": vehicle.get("pcd"),
                        "center_bore_mm": vehicle.get("center_bore_mm"),
                        "diameter_min_in": vehicle.get("diameter_min_in"),
                        "diameter_max_in": vehicle.get("diameter_max_in"),
                        "width_min_j": vehicle.get("width_min_j"),
                        "width_max_j": vehicle.get("width_max_j"),
                        "et_min": vehicle.get("et_min"),
                        "et_max": vehicle.get("et_max"),
                    },
                    "fitment": outcome,
                }
            )

    out_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Saved %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run wheel-to-model compatibility checks and generate sample outputs.")
    parser.add_argument("--wheels", default="wheels.csv", help="Path to wheels CSV.")
    parser.add_argument("--models", default="skoda_models.csv", help="Path to models CSV.")
    parser.add_argument("--output", default="fitment_examples.json", help="Output path for fitment example results.")
    parser.add_argument("--limit", type=int, default=5, help="How many wheel rows to evaluate.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    wheels_path = Path(args.wheels)
    models_path = Path(args.models)
    out_path = Path(args.output)

    if not wheels_path.exists() or not models_path.exists():
        raise FileNotFoundError("Required inputs not found. Run scrapers and merge first.")

    build_fitment_examples(
        wheels_path=wheels_path,
        models_path=models_path,
        out_path=out_path,
        limit=max(1, args.limit),
    )


if __name__ == "__main__":
    main()