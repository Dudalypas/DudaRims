from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CARS_JSON = "C:/Users/vilja/flutter_bandymas/assets/data/skoda_models.json"
DEFAULT_WHEELS_JSON = "C:/Users/vilja/flutter_bandymas/assets/data/wheels.json"
DEFAULT_DB_PATH = "C:/Users/vilja/flutter_bandymas/assets/data/fitment.sqlite3"


CAR_FIELDS = [
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

# Intentionally excludes generation/year_from/year_to for wheels.
WHEEL_FIELDS = [
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
    "notes",
    "source_type",
    "source_page",
    "extraction_confidence",
    "needs_manual_review",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import skoda_models.json and wheels.json into SQLite."
    )
    parser.add_argument("--cars-json", type=Path, default=DEFAULT_CARS_JSON)
    parser.add_argument("--wheels-json", type=Path, default=DEFAULT_WHEELS_JSON)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing rows. By default existing table rows are replaced.",
    )
    return parser.parse_args()


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8-sig"))

    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        for key in ("items", "rows", "data", "models", "wheels"):
            val = raw.get(key)
            if isinstance(val, list):
                rows = val
                break
        else:
            raise ValueError(f"Unsupported JSON object structure in {path}")
    else:
        raise ValueError(f"Unsupported JSON root type in {path}: {type(raw)!r}")

    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} in {path} is not a JSON object")
        out.append(row)
    return out


def to_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text != "" else None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(round(float(text.replace(",", "."))))
        except ValueError:
            return None


def to_bool_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 0 if float(value) == 0 else 1
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    return None


def normalize_car_row(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        to_text(row.get("brand")),
        to_text(row.get("model")),
        to_text(row.get("generation")),
        to_int(row.get("year_from")),
        to_int(row.get("year_to")),
        to_text(row.get("pcd")),
        to_float(row.get("cb")),
        to_int(row.get("bolt_count")),
        to_text(row.get("thread_size")),
        to_float(row.get("center_bore_mm")),
        to_float(row.get("diameter_min_in")),
        to_float(row.get("diameter_max_in")),
        to_float(row.get("width_min_j")),
        to_float(row.get("width_max_j")),
        to_float(row.get("et_min")),
        to_float(row.get("et_max")),
        to_text(row.get("notes")),
        to_text(row.get("source_type")),
        to_text(row.get("source_page")),
        to_float(row.get("extraction_confidence")),
        to_bool_int(row.get("needs_manual_review")),
    )


def normalize_wheel_row(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        to_text(row.get("wheel_class_name")),
        to_text(row.get("official_wheel_name")),
        to_text(row.get("brand")),
        to_text(row.get("color_variant")),
        to_float(row.get("diameter_in")),
        to_float(row.get("width_j")),
        to_float(row.get("et")),
        to_text(row.get("pcd")),
        to_float(row.get("cb")),
        to_int(row.get("bolt_count")),
        to_text(row.get("designed_for_models")),
        to_text(row.get("oem_part_code")),
        to_text(row.get("notes")),
        to_text(row.get("source_type")),
        to_text(row.get("source_page")),
        to_float(row.get("extraction_confidence")),
        to_bool_int(row.get("needs_manual_review")),
    )


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS car_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT,
            model TEXT,
            generation TEXT,
            year_from INTEGER,
            year_to INTEGER,
            pcd TEXT,
            cb REAL,
            bolt_count INTEGER,
            thread_size TEXT,
            center_bore_mm REAL,
            diameter_min_in REAL,
            diameter_max_in REAL,
            width_min_j REAL,
            width_max_j REAL,
            et_min REAL,
            et_max REAL,
            notes TEXT,
            source_type TEXT,
            source_page TEXT,
            extraction_confidence REAL,
            needs_manual_review INTEGER
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS wheel_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wheel_class_name TEXT,
            official_wheel_name TEXT,
            brand TEXT,
            color_variant TEXT,
            diameter_in REAL,
            width_j REAL,
            et REAL,
            pcd TEXT,
            cb REAL,
            bolt_count INTEGER,
            designed_for_models TEXT,
            oem_part_code TEXT,
            notes TEXT,
            source_type TEXT,
            source_page TEXT,
            extraction_confidence REAL,
            needs_manual_review INTEGER
        )
        """
    )

    conn.execute("CREATE INDEX IF NOT EXISTS idx_car_models_brand ON car_models(brand)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_car_models_model ON car_models(model)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_car_models_pcd ON car_models(pcd)")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_wheel_models_class_name ON wheel_models(wheel_class_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_wheel_models_brand ON wheel_models(brand)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_wheel_models_pcd ON wheel_models(pcd)")


def insert_rows(
    conn: sqlite3.Connection,
    cars: list[dict[str, Any]],
    wheels: list[dict[str, Any]],
    append: bool,
) -> tuple[int, int]:
    if not append:
        conn.execute("DELETE FROM car_models")
        conn.execute("DELETE FROM wheel_models")

    car_values = [normalize_car_row(r) for r in cars]
    wheel_values = [normalize_wheel_row(r) for r in wheels]

    conn.executemany(
        """
        INSERT INTO car_models (
            brand, model, generation, year_from, year_to,
            pcd, cb, bolt_count, thread_size, center_bore_mm,
            diameter_min_in, diameter_max_in, width_min_j, width_max_j,
            et_min, et_max, notes, source_type, source_page,
            extraction_confidence, needs_manual_review
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?
        )
        """,
        car_values,
    )

    conn.executemany(
        """
        INSERT INTO wheel_models (
            wheel_class_name, official_wheel_name, brand, color_variant,
            diameter_in, width_j, et, pcd, cb, bolt_count,
            designed_for_models, oem_part_code, notes,
            source_type, source_page, extraction_confidence, needs_manual_review
        ) VALUES (
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?
        )
        """,
        wheel_values,
    )

    return len(car_values), len(wheel_values)


def main() -> None:
    args = parse_args()

    cars = load_json_rows(args.cars_json)
    wheels = load_json_rows(args.wheels_json)

    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(args.db_path) as conn:
        create_schema(conn)
        car_count, wheel_count = insert_rows(conn, cars, wheels, append=args.append)
        conn.commit()

    print(f"SQLite DB: {args.db_path}")
    print(f"Imported car_models rows: {car_count}")
    print(f"Imported wheel_models rows: {wheel_count}")


if __name__ == "__main__":
    main()
