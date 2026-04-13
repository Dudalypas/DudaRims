from __future__ import annotations

import argparse
import json
import logging
import re
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
    "generation",
    "year_from",
    "year_to",
    "oem_part_code",
    "notes",
    "source_type",
    "source_page",
    "extraction_confidence",
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
    "notes",
    "source_type",
    "source_page",
    "extraction_confidence",
    "needs_manual_review",
]

MAPPING_SCHEMA = [
    "wheel_class_name",
    "official_wheel_name",
    "variant_type",
    "mapping_confidence",
    "needs_manual_review",
]

MANUAL_REVIEW_SCHEMA = [
    "dataset",
    "issue_type",
    "key",
    "details",
    "source_page",
    "recommended_action",
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
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None


def to_int(value: Any) -> int | None:
    parsed = to_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def normalize_wheels(wheels_df: pd.DataFrame) -> pd.DataFrame:
    float_cols = ["diameter_in", "width_j", "et", "cb", "extraction_confidence"]
    int_cols = ["bolt_count", "year_from", "year_to"]
    bool_cols = ["needs_manual_review"]

    for col in float_cols:
        wheels_df[col] = wheels_df[col].apply(to_float)
    for col in int_cols:
        wheels_df[col] = wheels_df[col].apply(to_int)
    for col in bool_cols:
        wheels_df[col] = wheels_df[col].astype(bool)

    wheels_df["pcd"] = wheels_df["pcd"].astype(str).replace("nan", "").str.strip()
    wheels_df["official_wheel_name"] = wheels_df["official_wheel_name"].astype(str).replace("nan", "").str.strip()

    dedupe_keys = ["official_wheel_name", "oem_part_code", "source_page"]
    wheels_df = wheels_df.drop_duplicates(subset=dedupe_keys, keep="first")
    return wheels_df


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
        "extraction_confidence",
    ]
    int_cols = ["bolt_count", "year_from", "year_to"]

    for col in float_cols:
        models_df[col] = models_df[col].apply(to_float)
    for col in int_cols:
        models_df[col] = models_df[col].apply(to_int)

    models_df["needs_manual_review"] = models_df["needs_manual_review"].astype(bool)
    models_df["pcd"] = models_df["pcd"].astype(str).replace("nan", "").str.strip()
    models_df["model"] = models_df["model"].astype(str).replace("nan", "").str.strip()
    models_df["generation"] = models_df["generation"].astype(str).replace("nan", "").str.strip()

    dedupe_keys = ["model", "generation", "source_page"]
    models_df = models_df.drop_duplicates(subset=dedupe_keys, keep="first")
    return models_df


def _source_from_row(row: pd.Series) -> str:
    source_page = str(row.get("source_page", "") or "").lower()
    source_type = str(row.get("source_type", "") or "").lower()
    if "wheel-size.com" in source_page or "wheelsize" in source_type:
        return "wheelsize"
    if "wheelfitment.eu" in source_page or "wheelfitment" in source_type:
        return "wheelfitment"
    return "other"


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _normalize_key_text(value: Any) -> str:
    if _is_empty(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def _pick_field_value(group: pd.DataFrame, field: str, preferred_source: str) -> Any:
    preferred = group[group["_source"] == preferred_source]
    for _, row in preferred.iterrows():
        if not _is_empty(row.get(field)):
            return row.get(field)
    for _, row in group.iterrows():
        if not _is_empty(row.get(field)):
            return row.get(field)
    return None


def _field_values_for_conflict(group: pd.DataFrame, field: str) -> list[str]:
    values = []
    for _, row in group.iterrows():
        val = row.get(field)
        if _is_empty(val):
            continue
        values.append(str(val).strip())
    normalized_unique = sorted({_normalize_key_text(v) for v in values if _normalize_key_text(v)})
    return normalized_unique


def merge_model_sources(models_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if models_df.empty:
        return models_df.copy(), pd.DataFrame(columns=MANUAL_REVIEW_SCHEMA)

    work = models_df.copy()
    work["_source"] = work.apply(_source_from_row, axis=1)
    work["_model_key"] = work["model"].fillna("").astype(str).str.strip().str.lower()
    work["_gen_key"] = work["generation"].fillna("").astype(str).str.strip().str.lower()
    work["_brand_key"] = work["brand"].fillna("").astype(str).str.strip().str.lower()

    merge_keys = ["_brand_key", "_model_key", "_gen_key", "year_from", "year_to"]
    merged_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []

    for _, group in work.groupby(merge_keys, dropna=False):
        exemplar = group.iloc[0].to_dict()

        merged: dict[str, Any] = {col: exemplar.get(col) for col in MODEL_SCHEMA}
        merged["brand"] = exemplar.get("brand")
        merged["model"] = exemplar.get("model")
        merged["generation"] = exemplar.get("generation")
        merged["year_from"] = exemplar.get("year_from")
        merged["year_to"] = exemplar.get("year_to")

        # Prefer wheel-size for fitment ranges.
        for field in ["diameter_min_in", "diameter_max_in", "width_min_j", "width_max_j", "et_min", "et_max"]:
            merged[field] = _pick_field_value(group, field, preferred_source="wheelsize")

        # Prefer wheelfitment for hub and bolt specs.
        for field in ["pcd", "cb", "center_bore_mm", "bolt_count", "thread_size"]:
            merged[field] = _pick_field_value(group, field, preferred_source="wheelfitment")

        notes = [str(v).strip() for v in group["notes"].dropna().tolist() if str(v).strip()]
        merged["notes"] = " | ".join(dict.fromkeys(notes)) if notes else None
        merged["source_type"] = "merged_secondary"
        merged["source_page"] = " | ".join(sorted({str(v) for v in group["source_page"].dropna().tolist() if str(v).strip()}))

        conf_vals = group["extraction_confidence"].dropna().tolist()
        merged["extraction_confidence"] = max(conf_vals) if conf_vals else 0.5

        critical_missing = any(
            _is_empty(merged.get(f))
            for f in ["pcd", "center_bore_mm", "diameter_min_in", "diameter_max_in", "width_min_j", "width_max_j"]
        )
        merged["needs_manual_review"] = bool(critical_missing)

        key = f"{merged.get('brand', '')}:{merged.get('model', '')}:{merged.get('generation', '')}:{merged.get('year_from', '')}-{merged.get('year_to', '')}"
        for field in ["pcd", "cb", "center_bore_mm", "diameter_min_in", "diameter_max_in", "width_min_j", "width_max_j", "et_min", "et_max"]:
            uniq = _field_values_for_conflict(group, field)
            if len(uniq) > 1:
                source_pages = " | ".join(sorted({str(v) for v in group["source_page"].dropna().tolist() if str(v).strip()}))
                conflict_rows.append(
                    {
                        "dataset": "models",
                        "issue_type": f"conflicting_{field}",
                        "key": key,
                        "details": f"Conflicting values for {field}: {uniq}",
                        "source_page": source_pages,
                        "recommended_action": "Verify source priority rules and confirm against trusted references.",
                    }
                )

        merged_rows.append(merged)

    merged_df = pd.DataFrame(merged_rows)
    merged_df = merged_df[MODEL_SCHEMA]
    conflicts_df = pd.DataFrame(conflict_rows, columns=MANUAL_REVIEW_SCHEMA)
    return merged_df, conflicts_df


def detect_wheel_conflicts(wheels_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = wheels_df[wheels_df["official_wheel_name"] != ""].groupby("official_wheel_name", dropna=False)
    for wheel_name, group in grouped:
        unique_et = sorted({x for x in group["et"].dropna().tolist()})
        if len(unique_et) > 1:
            rows.append(
                {
                    "dataset": "wheels",
                    "issue_type": "conflicting_et_values",
                    "key": wheel_name,
                    "details": f"ET values: {unique_et}",
                    "source_page": " | ".join(sorted({str(x) for x in group["source_page"].dropna().tolist()})),
                    "recommended_action": "Verify official accessory page and keep official value if available.",
                }
            )
    return pd.DataFrame(rows, columns=MANUAL_REVIEW_SCHEMA)


def detect_model_conflicts(models_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = models_df[models_df["model"] != ""].groupby(["brand", "model"], dropna=False)
    for (brand, model), group in grouped:
        generations = sorted({x for x in group["generation"].dropna().tolist() if str(x).strip()})
        if len(generations) > 1 and any(not x.startswith(tuple(generations[:1])) for x in generations[1:]):
            rows.append(
                {
                    "dataset": "models",
                    "issue_type": "conflicting_model_generations",
                    "key": f"{brand}:{model}",
                    "details": f"Generations: {generations}",
                    "source_page": " | ".join(sorted({str(x) for x in group["source_page"].dropna().tolist()})),
                    "recommended_action": "Confirm generation naming in official brochures or OEM docs.",
                }
            )
    return pd.DataFrame(rows, columns=MANUAL_REVIEW_SCHEMA)


def detect_missing_critical_fields(wheels_df: pd.DataFrame, models_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    wheel_missing = wheels_df[
        wheels_df[["diameter_in", "width_j", "pcd", "cb"]]
        .isna()
        .any(axis=1)
        | (wheels_df["pcd"].astype(str).str.strip() == "")
    ]
    for _, row in wheel_missing.iterrows():
        rows.append(
            {
                "dataset": "wheels",
                "issue_type": "missing_critical_fields",
                "key": row.get("official_wheel_name", ""),
                "details": "Missing one of: diameter_in, width_j, pcd, cb",
                "source_page": row.get("source_page", ""),
                "recommended_action": "Add value from official page or keep blank and review manually.",
            }
        )

    model_missing = models_df[
        models_df[["pcd", "center_bore_mm", "thread_size", "diameter_min_in", "diameter_max_in", "width_min_j", "width_max_j"]].isna().any(axis=1)
        | (models_df["pcd"].astype(str).str.strip() == "")
        | (models_df["thread_size"].astype(str).str.strip() == "")
    ]
    for _, row in model_missing.iterrows():
        rows.append(
            {
                "dataset": "models",
                "issue_type": "missing_critical_fields",
                "key": row.get("model", ""),
                "details": "Missing one of: pcd, center_bore_mm, thread_size, diameter range, width range",
                "source_page": row.get("source_page", ""),
                "recommended_action": "Validate against multiple references; do not guess values.",
            }
        )

    return pd.DataFrame(rows, columns=MANUAL_REVIEW_SCHEMA)


def detect_ambiguous_mappings(mapping_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ambiguous = mapping_df[(mapping_df["needs_manual_review"] == True) | (mapping_df["mapping_confidence"] < 0.8)]
    for _, row in ambiguous.iterrows():
        rows.append(
            {
                "dataset": "mapping",
                "issue_type": "ambiguous_wheel_name_mapping",
                "key": row.get("wheel_class_name", ""),
                "details": f"Mapped to '{row.get('official_wheel_name', '')}' with confidence {row.get('mapping_confidence', '')}",
                "source_page": "",
                "recommended_action": "Confirm with visual check and official catalog naming.",
            }
        )
    return pd.DataFrame(rows, columns=MANUAL_REVIEW_SCHEMA)


def save_json(path: Path, df: pd.DataFrame) -> None:
    path.write_text(json.dumps(df.where(pd.notna(df), None).to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge wheel and model data, validate schemas, and generate manual review tasks.")
    parser.add_argument("--wheels", default="wheels.csv", help="Path to wheels CSV.")
    parser.add_argument("--models", default="skoda_models.csv", help="Path to models CSV.")
    parser.add_argument("--mapping", default="wheel_class_mapping.csv", help="Path to wheel class mapping CSV.")
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
    models_df, model_source_conflicts_df = merge_model_sources(models_df)

    wheels_df.to_csv(output_dir / "wheels.csv", index=False, encoding="utf-8")
    models_df.to_csv(output_dir / "skoda_models.csv", index=False, encoding="utf-8")
    save_json(output_dir / "wheels.json", wheels_df)
    save_json(output_dir / "skoda_models.json", models_df)

    review_frames = [
        detect_wheel_conflicts(wheels_df),
        detect_model_conflicts(models_df),
        detect_missing_critical_fields(wheels_df, models_df),
        model_source_conflicts_df,
    ]

    if Path(args.mapping).exists():
        mapping_df = pd.read_csv(args.mapping)
        mapping_df = ensure_schema(mapping_df, MAPPING_SCHEMA, "wheel_class_mapping")
        review_frames.append(detect_ambiguous_mappings(mapping_df))

    manual_review_df = pd.concat(review_frames, ignore_index=True).drop_duplicates()
    manual_review_df = manual_review_df[MANUAL_REVIEW_SCHEMA]
    manual_review_df.to_csv(output_dir / "manual_review.csv", index=False, encoding="utf-8")

    conflicts_df = manual_review_df[manual_review_df["issue_type"].str.contains("conflicting", na=False)].copy()
    conflicts_df.to_csv(output_dir / "source_conflicts.csv", index=False, encoding="utf-8")

    LOGGER.info("Saved %s", output_dir / "manual_review.csv")
    LOGGER.info("Saved %s", output_dir / "source_conflicts.csv")
    LOGGER.info("Merged outputs refreshed in %s", output_dir)


if __name__ == "__main__":
    main()