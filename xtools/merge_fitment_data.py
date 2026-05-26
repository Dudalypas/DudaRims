from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("merge_fitment_data")

WHEEL_SCHEMA = [
    "wheel_class_name",
    "official_wheel_name",
    "brand",
    "color_variant",
    "diameter_in",
    "width_j",
    "et",
    "pcd",
    "cb",
    "bolt_count",
    "designed_for_models",
    "oem_part_code",
    "review_reason",
    "needs_manual_review",
]

MODEL_SCHEMA = [
    "brand",
    "model",
    "generation",
    "year_from",
    "year_to",
    "pcd",
    "cb",
    "bolt_count",
    "thread_size",
    "center_bore_mm",
    "diameter_min_in",
    "diameter_max_in",
    "width_min_j",
    "width_max_j",
    "et_min",
    "et_max",
    "review_reason",
    "needs_manual_review",
]

WHEEL_CRITICAL_FIELDS = ["diameter_in", "width_j", "et", "pcd", "cb", "bolt_count"]
MODEL_CRITICAL_FIELDS = [
    "pcd",
    "center_bore_mm",
    "bolt_count",
    "diameter_min_in",
    "diameter_max_in",
    "width_min_j",
    "width_max_j",
    "et_min",
    "et_max",
]



def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")



def ensure_schema(df: pd.DataFrame, columns: list[str], dataset_name: str) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing columns: {missing}")
    return df[columns].copy()



def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None



def to_int(value: Any) -> int | None:
    parsed = to_float(value)
    if parsed is None:
        return None
    if pd.isna(parsed):
        return None
    return int(round(parsed))



def to_bool_int(value: Any) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 0 if float(value) == 0 else 1
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "taip"}:
        return 1
    if text in {"0", "false", "no", "n", "ne", "", "none", "null"}:
        return 0
    return 0



def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None



def _merge_reason(parts: list[str]) -> str | None:
    unique = [p for p in dict.fromkeys(p for p in parts if p)]
    return " | ".join(unique) if unique else None



def normalize_wheels(wheels_df: pd.DataFrame) -> pd.DataFrame:
    float_cols = ["diameter_in", "width_j", "et", "cb"]
    int_cols = ["bolt_count"]

    for col in float_cols:
        wheels_df[col] = wheels_df[col].apply(to_float)
    for col in int_cols:
        wheels_df[col] = wheels_df[col].apply(to_int)

    text_cols = [
        "wheel_class_name",
        "official_wheel_name",
        "brand",
        "color_variant",
        "pcd",
        "designed_for_models",
        "oem_part_code",
        "review_reason",
    ]
    for col in text_cols:
        wheels_df[col] = wheels_df[col].apply(normalize_text)

    wheels_df["needs_manual_review"] = wheels_df["needs_manual_review"].apply(to_bool_int)

    for col in ["official_wheel_name", "oem_part_code", "brand"]:
        wheels_df[col] = wheels_df[col].fillna("")

    grouped = wheels_df.groupby(["official_wheel_name", "oem_part_code", "brand"], dropna=False)
    merged_rows: list[dict[str, Any]] = []

    for _, group in grouped:
        row: dict[str, Any] = {}
        reasons: list[str] = []

        for col in ["wheel_class_name", "official_wheel_name", "brand", "color_variant", "pcd", "designed_for_models", "oem_part_code"]:
            values = [v for v in group[col].tolist() if normalize_text(v)]
            row[col] = values[0] if values else None

        for col in ["diameter_in", "width_j", "et", "cb"]:
            values = [to_float(v) for v in group[col].tolist() if to_float(v) is not None]
            row[col] = values[0] if values else None
            if len(set(values)) > 1:
                reasons.append(f"conflicting {col} values")

        bolt_values = [to_int(v) for v in group["bolt_count"].tolist() if to_int(v) is not None]
        row["bolt_count"] = bolt_values[0] if bolt_values else None
        if len(set(bolt_values)) > 1:
            reasons.append("conflicting bolt_count values")

        prior_reasons = [normalize_text(v) for v in group["review_reason"].tolist() if normalize_text(v)]
        reasons.extend(prior_reasons)

        missing_critical = [field for field in WHEEL_CRITICAL_FIELDS if row.get(field) in (None, "")]
        if missing_critical:
            reasons.append("missing critical fields")

        needs_manual = bool(group["needs_manual_review"].astype(int).max()) or bool(missing_critical) or any(
            r.startswith("conflicting ") for r in reasons
        )

        row["review_reason"] = _merge_reason(reasons)
        row["needs_manual_review"] = 1 if needs_manual else 0
        merged_rows.append(row)

    result = pd.DataFrame(merged_rows)
    for col in WHEEL_SCHEMA:
        if col not in result.columns:
            result[col] = None
    return result[WHEEL_SCHEMA]



def normalize_models(models_df: pd.DataFrame) -> pd.DataFrame:
    float_cols = [
        "cb",
        "center_bore_mm",
        "diameter_min_in",
        "diameter_max_in",
        "width_min_j",
        "width_max_j",
        "et_min",
        "et_max",
    ]
    int_cols = ["bolt_count", "year_from", "year_to"]

    for col in float_cols:
        models_df[col] = models_df[col].apply(to_float)
    for col in int_cols:
        models_df[col] = models_df[col].apply(to_int)

    text_cols = ["brand", "model", "generation", "pcd", "thread_size", "review_reason"]
    for col in text_cols:
        models_df[col] = models_df[col].apply(normalize_text)

    models_df["needs_manual_review"] = models_df["needs_manual_review"].apply(to_bool_int)

    for col in ["brand", "model", "generation"]:
        models_df[col] = models_df[col].fillna("")

    grouped = models_df.groupby(["brand", "model", "generation", "year_from", "year_to"], dropna=False)
    merged_rows: list[dict[str, Any]] = []

    for _, group in grouped:
        row: dict[str, Any] = {}
        reasons: list[str] = []

        for col in ["brand", "model", "generation", "pcd", "thread_size"]:
            values = [v for v in group[col].tolist() if normalize_text(v)]
            row[col] = values[0] if values else None
            if col in {"pcd", "thread_size"} and len(set(values)) > 1:
                reasons.append(f"conflicting {col} values")

        year_from_values = [to_int(v) for v in group["year_from"].tolist() if to_int(v) is not None]
        year_to_values = [to_int(v) for v in group["year_to"].tolist() if to_int(v) is not None]
        row["year_from"] = min(year_from_values) if year_from_values else None
        row["year_to"] = max(year_to_values) if year_to_values else None

        numeric_pick_fields = ["cb", "center_bore_mm", "bolt_count"]
        for col in numeric_pick_fields:
            values = [to_float(v) for v in group[col].tolist() if to_float(v) is not None]
            if col == "bolt_count":
                ivals = [to_int(v) for v in group[col].tolist() if to_int(v) is not None]
                row[col] = ivals[0] if ivals else None
                if len(set(ivals)) > 1:
                    reasons.append("conflicting bolt_count values")
            else:
                row[col] = values[0] if values else None
                if len(set(values)) > 1:
                    reasons.append(f"conflicting {col} values")

        row["diameter_min_in"] = min([to_float(v) for v in group["diameter_min_in"].tolist() if to_float(v) is not None], default=None)
        row["diameter_max_in"] = max([to_float(v) for v in group["diameter_max_in"].tolist() if to_float(v) is not None], default=None)
        row["width_min_j"] = min([to_float(v) for v in group["width_min_j"].tolist() if to_float(v) is not None], default=None)
        row["width_max_j"] = max([to_float(v) for v in group["width_max_j"].tolist() if to_float(v) is not None], default=None)
        row["et_min"] = min([to_float(v) for v in group["et_min"].tolist() if to_float(v) is not None], default=None)
        row["et_max"] = max([to_float(v) for v in group["et_max"].tolist() if to_float(v) is not None], default=None)

        prior_reasons = [normalize_text(v) for v in group["review_reason"].tolist() if normalize_text(v)]
        reasons.extend(prior_reasons)

        missing_critical = [field for field in MODEL_CRITICAL_FIELDS if row.get(field) in (None, "")]
        if missing_critical:
            reasons.append("missing critical fields")

        needs_manual = bool(group["needs_manual_review"].astype(int).max()) or bool(missing_critical) or any(
            r.startswith("conflicting ") for r in reasons
        )

        row["review_reason"] = _merge_reason(reasons)
        row["needs_manual_review"] = 1 if needs_manual else 0
        merged_rows.append(row)

    result = pd.DataFrame(merged_rows)
    for col in MODEL_SCHEMA:
        if col not in result.columns:
            result[col] = None
    return result[MODEL_SCHEMA]



def save_json(path: Path, df: pd.DataFrame) -> None:
    path.write_text(
        json.dumps(df.where(pd.notna(df), None).to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge canonical wheel and model fitment datasets.")
    parser.add_argument("--wheels", default="wheels.csv", help="Path to wheels CSV.")
    parser.add_argument("--models", default="skoda_models.csv", help="Path to models CSV.")
    parser.add_argument("--output-dir", default=".", help="Output directory.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wheels_df = pd.read_csv(args.wheels)
    models_df = pd.read_csv(args.models)

    wheels_df = ensure_schema(wheels_df, WHEEL_SCHEMA, "wheels")
    models_df = ensure_schema(models_df, MODEL_SCHEMA, "skoda_models")

    wheels_df = normalize_wheels(wheels_df)
    models_df = normalize_models(models_df)

    wheels_df.to_csv(output_dir / "wheels.csv", index=False, encoding="utf-8")
    models_df.to_csv(output_dir / "skoda_models.csv", index=False, encoding="utf-8")
    save_json(output_dir / "wheels.json", wheels_df)
    save_json(output_dir / "skoda_models.json", models_df)

    LOGGER.info("Merged outputs refreshed in %s", output_dir)


if __name__ == "__main__":
    main()
