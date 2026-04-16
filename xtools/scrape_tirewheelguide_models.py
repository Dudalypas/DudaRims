from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger("scrape_tirewheelguide_models")

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

URL_BRAND_MODEL_YEAR_PATTERN = re.compile(
    r"/sizes/(?P<brand>[^/]+)/(?P<model>[^/]+)/(?P<year>19\d{2}|20\d{2})/?",
    re.IGNORECASE,
)
YEAR_RANGE_PATTERN = re.compile(r"(?P<from>19\d{2}|20\d{2})\s*(?:-|to|–|—)\s*(?P<to>19\d{2}|20\d{2})", re.IGNORECASE)
GENERATION_HEADER_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\s*(?:-|to|–|—)\s*(19\d{2}|20\d{2})\b")
MODIFICATION_HEADER_PATTERN = re.compile(r"^(19\d{2}|20\d{2})\s+([A-Za-z0-9-]+)\s+(.+)$")
PCD_PATTERN = re.compile(r"\b(?P<bolt>\d+)\s*[xX]\s*(?P<diameter>\d+(?:[\.,]\d+)?)\b")
CB_PATTERN = re.compile(r"(?P<cb>\d+(?:[\.,]\d+)?)\s*mm\b", re.IGNORECASE)
THREAD_PATTERN = re.compile(r"\bM\s*(?P<dia>\d+)\s*[xX]\s*(?P<pitch>\d+(?:[\.,]\d+)?)\b", re.IGNORECASE)
RIM_WIDTH_FIRST_PATTERN = re.compile(
    r"(?P<width>\d+(?:[\.,]\d+)?)\s*J\s*[xX]\s*(?P<diameter>\d+(?:[\.,]\d+)?)\s*(?:ET\s*(?P<et>[+-]?\d+(?:[\.,]\d+)?))?",
    re.IGNORECASE,
)
RIM_DIAMETER_FIRST_PATTERN = re.compile(
    r"(?P<diameter>\d+(?:[\.,]\d+)?)\s*[xX]\s*(?P<width>\d+(?:[\.,]\d+)?)\s*(?:J)?\s*(?:ET\s*(?P<et>[+-]?\d+(?:[\.,]\d+)?))?",
    re.IGNORECASE,
)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def normalize_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(value)).strip()


def to_float(value: Any) -> float | None:
    raw = normalize_text(value)
    if not raw:
        return None
    raw = raw.replace(",", ".")
    match = re.search(r"[+-]?\d+(?:\.\d+)?", raw)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def slug_to_name(value: str) -> str:
    clean = re.sub(r"[-_]+", " ", normalize_text(value))
    return normalize_space(clean).title()


def build_session(user_agent: str, retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def fetch_html(session: requests.Session, source_url: str, timeout: int) -> str:
    response = session.get(source_url, timeout=timeout)
    response.raise_for_status()
    return response.text


def safe_slug(value: str, fallback: str = "page") -> str:
    text = normalize_space(value).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or fallback


def extract_url_slug_parts(source_url: str) -> tuple[str, str]:
    match = URL_BRAND_MODEL_YEAR_PATTERN.search(urlparse(source_url).path)
    if not match:
        return "model", "unknown"
    model_slug = safe_slug(match.group("model"), fallback="model")
    year_slug = safe_slug(match.group("year"), fallback="unknown")
    return model_slug, year_slug


def save_raw_html(raw_html_dir: Path, source_url: str, html: str) -> Path:
    raw_html_dir.mkdir(parents=True, exist_ok=True)
    model_slug, year_slug = extract_url_slug_parts(source_url)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{model_slug}_{year_slug}_{timestamp}.html"
    output_path = raw_html_dir / filename
    output_path.write_text(html, encoding="utf-8")
    return output_path


def parse_page_identity(soup: BeautifulSoup, source_url: str) -> dict[str, Any]:
    canonical_tag = soup.find("link", rel=lambda value: value and "canonical" in str(value).lower())
    canonical = normalize_text(canonical_tag.get("href")) if canonical_tag else ""
    canonical = canonical or source_url

    title_text = normalize_space(soup.title.get_text(" ", strip=True) if soup.title else "")
    h1_tag = soup.find("h1")
    h1_text = normalize_space(h1_tag.get_text(" ", strip=True) if h1_tag else "")

    url_for_parse = canonical or source_url
    url_match = URL_BRAND_MODEL_YEAR_PATTERN.search(urlparse(url_for_parse).path)
    brand = slug_to_name(url_match.group("brand")) if url_match else ""
    model = slug_to_name(url_match.group("model")) if url_match else ""
    page_year = int(url_match.group("year")) if url_match else None

    if not brand and h1_text:
        heading_match = re.search(r"\b(19\d{2}|20\d{2})\s+([A-Za-z0-9-]+)\s+(.+?)\s+tire\s+and\s+wheel\s+sizes\b", h1_text, re.IGNORECASE)
        if heading_match:
            page_year = page_year or int(heading_match.group(1))
            brand = slug_to_name(heading_match.group(2))
            model = slug_to_name(heading_match.group(3))

    return {
        "canonical": canonical,
        "title": title_text,
        "h1": h1_text,
        "brand": brand,
        "model": model,
        "page_year": page_year,
    }


def parse_year_range(value: str) -> tuple[int | None, int | None]:
    text = normalize_space(value)
    match = YEAR_RANGE_PATTERN.search(text)
    if not match:
        return None, None
    return int(match.group("from")), int(match.group("to"))


def parse_generation_header(header_text: str, brand: str, model: str, heading_id: str) -> tuple[str, int | None, int | None] | None:
    clean = normalize_space(header_text)
    year_from, year_to = parse_year_range(clean)
    if year_from is None or year_to is None:
        return None

    # Avoid selecting modification headings that start with a single year.
    if MODIFICATION_HEADER_PATTERN.match(clean):
        return None

    if brand and clean.lower().startswith(brand.lower()):
        clean = normalize_space(clean[len(brand) :])
    if model and clean.lower().startswith(model.lower()):
        clean = normalize_space(clean[len(model) :])

    clean = normalize_space(re.sub(r"\b(19\d{2}|20\d{2})\s*(?:-|to|–|—)\s*(19\d{2}|20\d{2})\b", "", clean, flags=re.IGNORECASE))

    if not clean:
        clean = generation_from_id(heading_id)

    clean = normalize_space(clean)
    return clean, year_from, year_to


def generation_from_id(heading_id: str) -> str:
    raw = normalize_text(heading_id)
    if not raw:
        return ""
    parts = [p for p in raw.split("-") if p]
    while parts and re.fullmatch(r"19\d{2}|20\d{2}", parts[-1]):
        parts.pop()
    formatted: list[str] = []
    for token in parts:
        if re.fullmatch(r"\d+[A-Za-z]+", token):
            formatted.append(f"({token.upper()})")
        elif re.fullmatch(r"[Mm][Kk]\d+", token):
            formatted.append(token.capitalize())
        else:
            formatted.append(token.upper() if len(token) <= 3 else token.capitalize())
    return normalize_space(" ".join(formatted))


def normalize_pcd(value: str | None) -> str | None:
    raw = normalize_text(value)
    if not raw:
        return None
    match = PCD_PATTERN.search(raw)
    if not match:
        return None
    bolt = int(match.group("bolt"))
    diameter = match.group("diameter").replace(",", ".")
    return f"{bolt}x{diameter}"


def derive_bolt_count(pcd: str | None) -> int | None:
    if not pcd:
        return None
    match = re.match(r"(\d+)\s*[xX]", pcd)
    return int(match.group(1)) if match else None


def normalize_thread_size(value: str | None) -> str | None:
    raw = normalize_text(value)
    if not raw:
        return None
    match = THREAD_PATTERN.search(raw)
    if not match:
        return None
    dia = match.group("dia")
    pitch = match.group("pitch").replace(",", ".")
    return f"M{dia} x {pitch}"


def parse_center_bore(value: str | None) -> float | None:
    raw = normalize_text(value)
    if not raw:
        return None
    match = CB_PATTERN.search(raw)
    if not match:
        return to_float(raw)
    return to_float(match.group("cb"))


def parse_modification_name(mod_header_text: str, brand: str, model: str) -> str:
    text = normalize_space(mod_header_text)
    pattern = re.compile(rf"^(19\d{{2}}|20\d{{2}})\s+{re.escape(brand)}\s+{re.escape(model)}\s+", re.IGNORECASE)
    return normalize_space(pattern.sub("", text))


def parse_rim_spec_values(value: str) -> tuple[float | None, float | None, float | None]:
    text = normalize_space(value)
    if not text:
        return None, None, None

    match = RIM_WIDTH_FIRST_PATTERN.search(text)
    if match:
        width = to_float(match.group("width"))
        diameter = to_float(match.group("diameter"))
        et = to_float(match.group("et"))
        return diameter, width, et

    match = RIM_DIAMETER_FIRST_PATTERN.search(text)
    if match:
        diameter = to_float(match.group("diameter"))
        width = to_float(match.group("width"))
        et = to_float(match.group("et"))
        return diameter, width, et

    return None, None, None


def extract_rim_cell_candidates(rim_cell: Tag) -> list[str]:
    candidates: list[str] = []
    visible = normalize_space(rim_cell.get_text(" ", strip=True))
    if visible:
        candidates.append(visible)

    title = normalize_space(rim_cell.get("title"))
    if title:
        candidates.append(title)

    for child in rim_cell.find_all(True):
        title_value = normalize_space(child.get("title"))
        if title_value:
            candidates.append(title_value)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def parse_rim_table(table: Tag) -> list[dict[str, float | None]]:
    rows = table.find_all("tr")
    if not rows:
        return []

    header_cells = rows[0].find_all(["th", "td"])
    headers = [normalize_space(cell.get_text(" ", strip=True)).lower() for cell in header_cells]
    rim_col_idx = None
    for idx, header in enumerate(headers):
        if "rim size" in header and "bolt pattern" in header:
            rim_col_idx = idx
            break
    if rim_col_idx is None:
        return []

    parsed_rows: list[dict[str, float | None]] = []
    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) <= rim_col_idx:
            continue
        rim_cell = cells[rim_col_idx]
        candidates = extract_rim_cell_candidates(rim_cell)

        best: tuple[float | None, float | None, float | None] = (None, None, None)
        for candidate in candidates:
            parsed = parse_rim_spec_values(candidate)
            if parsed[0] is not None and parsed[1] is not None:
                best = parsed
                break

        if best[0] is None or best[1] is None:
            continue

        parsed_rows.append({"diameter_in": best[0], "width_j": best[1], "et": best[2]})

    return parsed_rows


def most_common_non_null(values: list[Any]) -> tuple[Any, list[str]]:
    normalized_values: list[str] = []
    original_map: dict[str, Any] = {}
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        key = normalize_space(str(value)).lower()
        if not key:
            continue
        normalized_values.append(key)
        original_map.setdefault(key, value)

    if not normalized_values:
        return None, []

    counts = Counter(normalized_values)
    winner_key, _ = counts.most_common(1)[0]
    conflict_values = [str(original_map[k]) for k in counts.keys() if k != winner_key]
    return original_map[winner_key], sorted(set(conflict_values))


def collect_generation_nodes(start_h2: Tag, next_generation_h2: Tag | None) -> list[Tag]:
    nodes: list[Tag] = []
    for node in start_h2.find_all_next(True):
        if next_generation_h2 is not None and node is next_generation_h2:
            break
        nodes.append(node)
    return nodes


def parse_generation_block(
    generation_h2: Tag,
    next_generation_h2: Tag | None,
    brand: str,
    model: str,
    source_url: str,
) -> dict[str, Any] | None:
    generation_info = parse_generation_header(
        header_text=generation_h2.get_text(" ", strip=True),
        brand=brand,
        model=model,
        heading_id=normalize_text(generation_h2.get("id")),
    )
    if generation_info is None:
        return None

    generation, year_from, year_to = generation_info
    generation_nodes = collect_generation_nodes(generation_h2, next_generation_h2)

    modifications: list[dict[str, Any]] = []
    current_mod: dict[str, Any] | None = None
    tables_parsed = 0
    total_rim_rows = 0

    for node in generation_nodes:
        if node.name == "h2":
            header_text = normalize_space(node.get_text(" ", strip=True))
            if MODIFICATION_HEADER_PATTERN.match(header_text):
                current_mod = {
                    "header": header_text,
                    "name": parse_modification_name(header_text, brand=brand, model=model),
                    "pcd": None,
                    "cb": None,
                    "thread_size": None,
                    "tables": 0,
                    "rim_rows": [],
                }
                modifications.append(current_mod)
            continue

        if current_mod is None:
            continue

        if node.name == "p":
            line = normalize_space(node.get_text(" ", strip=True))
            line_lower = line.lower()
            if "center bore" in line_lower:
                current_mod["cb"] = parse_center_bore(line)
            elif line_lower.startswith("pcd") or " bolt pattern" in line_lower:
                current_mod["pcd"] = normalize_pcd(line)
            elif "thread size" in line_lower:
                current_mod["thread_size"] = normalize_thread_size(line)
            continue

        if node.name == "table":
            rim_rows = parse_rim_table(node)
            if not rim_rows:
                continue
            current_mod["tables"] += 1
            tables_parsed += 1
            total_rim_rows += len(rim_rows)
            current_mod["rim_rows"].extend(rim_rows)

    if not modifications:
        LOGGER.warning("No modification sections found under generation '%s' (%s)", generation, source_url)
        return None

    diameter_values: list[float] = []
    width_values: list[float] = []
    et_values: list[float] = []
    pcd_values: list[str] = []
    cb_values: list[float] = []
    thread_values: list[str] = []

    for mod in modifications:
        if mod.get("pcd"):
            pcd_values.append(str(mod["pcd"]))
        if mod.get("cb") is not None:
            cb_values.append(float(mod["cb"]))
        if mod.get("thread_size"):
            thread_values.append(str(mod["thread_size"]))

        for rim_row in mod.get("rim_rows", []):
            if rim_row.get("diameter_in") is not None:
                diameter_values.append(float(rim_row["diameter_in"]))
            if rim_row.get("width_j") is not None:
                width_values.append(float(rim_row["width_j"]))
            if rim_row.get("et") is not None:
                et_values.append(float(rim_row["et"]))

    pcd, pcd_conflicts = most_common_non_null(pcd_values)
    cb, cb_conflicts = most_common_non_null(cb_values)
    thread_size, thread_conflicts = most_common_non_null(thread_values)
    bolt_count = derive_bolt_count(pcd)

    diameter_min = min(diameter_values) if diameter_values else None
    diameter_max = max(diameter_values) if diameter_values else None
    width_min = min(width_values) if width_values else None
    width_max = max(width_values) if width_values else None
    et_min = min(et_values) if et_values else None
    et_max = max(et_values) if et_values else None

    unique_mod_names = [m.get("name", "") for m in modifications if normalize_text(m.get("name"))]
    unique_mod_names = list(dict.fromkeys(unique_mod_names))

    notes_parts = [
        f"aggregated_from_modifications={len(modifications)}",
        f"modification_names={'|'.join(unique_mod_names)}",
        f"tables_parsed={tables_parsed}",
        f"rim_rows_parsed={total_rim_rows}",
    ]

    conflicts_found = False
    if pcd_conflicts:
        conflicts_found = True
        notes_parts.append(f"pcd_conflict={pcd}|{'|'.join(pcd_conflicts)}")
    if cb_conflicts:
        conflicts_found = True
        notes_parts.append(f"cb_conflict={cb}|{'|'.join(cb_conflicts)}")
    if thread_conflicts:
        conflicts_found = True
        notes_parts.append(f"thread_size_conflict={thread_size}|{'|'.join(thread_conflicts)}")

    missing_key_fields = any(
        value is None or value == ""
        for value in [pcd, cb, thread_size, diameter_min, diameter_max, width_min, width_max, et_min, et_max]
    )

    needs_manual_review = bool(conflicts_found or missing_key_fields)
    if not needs_manual_review:
        extraction_confidence = 0.9
    elif missing_key_fields:
        extraction_confidence = 0.6
    else:
        extraction_confidence = 0.75

    LOGGER.info("Generation parsed: %s (%s-%s)", generation, year_from, year_to)
    LOGGER.info("Generation '%s' modification sections: %d", generation, len(modifications))
    LOGGER.info("Generation '%s' tables parsed: %d", generation, tables_parsed)
    LOGGER.info("Generation '%s' total rim rows parsed: %d", generation, total_rim_rows)
    LOGGER.info(
        "Generation '%s' aggregated ranges -> diameter_min_in=%s, diameter_max_in=%s, width_min_j=%s, width_max_j=%s, et_min=%s, et_max=%s",
        generation,
        diameter_min,
        diameter_max,
        width_min,
        width_max,
        et_min,
        et_max,
    )
    if pcd_conflicts:
        LOGGER.warning("Generation '%s' PCD conflict: selected=%s, alternatives=%s", generation, pcd, pcd_conflicts)
    if cb_conflicts:
        LOGGER.warning("Generation '%s' CB conflict: selected=%s, alternatives=%s", generation, cb, cb_conflicts)
    if thread_conflicts:
        LOGGER.warning(
            "Generation '%s' Thread Size conflict: selected=%s, alternatives=%s",
            generation,
            thread_size,
            thread_conflicts,
        )

    return {
        "brand": brand,
        "model": model,
        "generation": generation,
        "year_from": year_from,
        "year_to": year_to,
        "pcd": pcd,
        "cb": cb,
        "bolt_count": bolt_count,
        "thread_size": thread_size,
        "center_bore_mm": cb,
        "diameter_min_in": diameter_min,
        "diameter_max_in": diameter_max,
        "width_min_j": width_min,
        "width_max_j": width_max,
        "et_min": et_min,
        "et_max": et_max,
        "notes": " | ".join(notes_parts),
        "source_type": "tirewheelguide",
        "source_page": source_url,
        "extraction_confidence": extraction_confidence,
        "needs_manual_review": needs_manual_review,
    }


def parse_source_page(session: requests.Session, source_url: str, timeout: int, fallback_brand: str = "", fallback_model: str = "") -> list[dict[str, Any]]:
    LOGGER.info("Parsing source URL: %s", source_url)
    html = fetch_html(session=session, source_url=source_url, timeout=timeout)
    return parse_source_page_html(
        html=html,
        source_url=source_url,
        fallback_brand=fallback_brand,
        fallback_model=fallback_model,
    )


def parse_source_page_html(html: str, source_url: str, fallback_brand: str = "", fallback_model: str = "") -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")

    page_identity = parse_page_identity(soup=soup, source_url=source_url)
    brand = page_identity.get("brand") or fallback_brand or "Skoda"
    model = page_identity.get("model") or fallback_model

    generation_h2s: list[Tag] = []
    for h2 in soup.find_all("h2"):
        parsed = parse_generation_header(
            header_text=h2.get_text(" ", strip=True),
            brand=brand,
            model=model,
            heading_id=normalize_text(h2.get("id")),
        )
        if parsed is not None:
            generation_h2s.append(h2)

    LOGGER.info("Generation blocks found on %s: %d", source_url, len(generation_h2s))

    rows: list[dict[str, Any]] = []
    for idx, generation_h2 in enumerate(generation_h2s):
        next_h2 = generation_h2s[idx + 1] if idx + 1 < len(generation_h2s) else None
        row = parse_generation_block(
            generation_h2=generation_h2,
            next_generation_h2=next_h2,
            brand=brand,
            model=model,
            source_url=source_url,
        )
        if row:
            rows.append(row)

    return rows


def load_sources_from_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    expected_columns = {"brand", "model", "source_url"}
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        source_url = normalize_text(row.get("source_url"))
        if not source_url:
            continue
        rows.append(
            {
                "brand": normalize_text(row.get("brand")),
                "model": normalize_text(row.get("model")),
                "source_url": source_url,
            }
        )
    return rows


def save_json(path: Path, df: pd.DataFrame) -> None:
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def load_existing_output(output_dir: Path) -> pd.DataFrame:
    csv_path = output_dir / "skoda_models.csv"
    json_path = output_dir / "skoda_models.json"

    if csv_path.exists():
        LOGGER.info("Loading existing dataset from CSV: %s", csv_path)
        return pd.read_csv(csv_path)

    if json_path.exists():
        LOGGER.info("Loading existing dataset from JSON: %s", json_path)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Existing JSON output has invalid format: {json_path}")
        return pd.DataFrame(data)

    return pd.DataFrame(columns=MODEL_SCHEMA)


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col in MODEL_SCHEMA:
        if col not in df.columns:
            df[col] = None

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
    int_cols = ["year_from", "year_to", "bolt_count"]

    for col in float_cols:
        df[col] = df[col].apply(to_float)
    for col in int_cols:
        df[col] = df[col].apply(lambda value: int(value) if value is not None and str(value) != "" else None)

    df["needs_manual_review"] = df["needs_manual_review"].astype(bool)
    df["brand"] = df["brand"].astype(str).replace("nan", "").str.strip()
    df["model"] = df["model"].astype(str).replace("nan", "").str.strip()
    df["generation"] = df["generation"].astype(str).replace("nan", "").str.strip()
    df["pcd"] = df["pcd"].astype(str).replace("nan", "").str.strip()
    df["thread_size"] = df["thread_size"].astype(str).replace("nan", "").str.strip()

    df = df[MODEL_SCHEMA].copy()
    dedupe_keys = ["brand", "model", "generation", "year_from", "year_to", "source_page"]
    return df.drop_duplicates(subset=dedupe_keys, keep="first")


def _merge_numeric_min(series: pd.Series) -> float | None:
    values = [to_float(v) for v in series.tolist() if to_float(v) is not None]
    return min(values) if values else None


def _merge_numeric_max(series: pd.Series) -> float | None:
    values = [to_float(v) for v in series.tolist() if to_float(v) is not None]
    return max(values) if values else None


def _merge_sources(series: pd.Series) -> str:
    sources: list[str] = []
    for raw in series.tolist():
        text = normalize_text(raw)
        if not text:
            continue
        parts = [normalize_space(p) for p in text.split("|")]
        for part in parts:
            if part and part not in sources:
                sources.append(part)
    return " | ".join(sources)


def _merge_notes(series: pd.Series, merged_url_count: int, extra_notes: list[str], merged_existing: bool = False) -> str:
    parts: list[str] = []
    seen: set[str] = set()

    for raw in series.tolist():
        text = normalize_space(normalize_text(raw))
        if not text:
            continue
        text = normalize_space(re.sub(r"(?:\|\s*)?merged_from_urls=\d+", "", text, flags=re.IGNORECASE))
        text = normalize_space(re.sub(r"(?:\|\s*)?merged_existing=true", "", text, flags=re.IGNORECASE))
        text = normalize_space(re.sub(r"\s*\|\s*", " | ", text)).strip("|").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        parts.append(text)

    merged_note = f"merged_from_urls={merged_url_count}"
    if merged_note not in seen:
        parts.append(merged_note)

    if merged_existing and "merged_existing=true" not in seen:
        parts.append("merged_existing=true")

    for note in extra_notes:
        if note and note not in seen:
            parts.append(note)

    return " | ".join(parts)


def _merge_categorical_field(values: list[Any], field_name: str) -> tuple[Any, bool, str | None]:
    winner, conflicts = most_common_non_null(values)
    has_conflict = bool(conflicts)
    conflict_note = None
    if has_conflict:
        conflict_note = f"dedupe_{field_name}_conflict={winner}|{'|'.join(conflicts)}"
    return winner, has_conflict, conflict_note


def consolidate_generation_rows(df: pd.DataFrame, verbose: bool = False, merged_existing: bool = False) -> pd.DataFrame:
    if df.empty:
        return df

    key_cols = ["brand", "model", "generation", "year_from", "year_to"]
    grouped = df.groupby(key_cols, dropna=False, sort=False)
    merged_rows: list[dict[str, Any]] = []

    for key, group in grouped:
        key_str = " | ".join(str(item) for item in key)
        merged: dict[str, Any] = {
            "brand": key[0],
            "model": key[1],
            "generation": key[2],
            "year_from": key[3],
            "year_to": key[4],
            "source_type": "tirewheelguide",
        }

        conflict_notes: list[str] = []
        has_conflicts = False

        for field in ["pcd", "cb", "bolt_count", "thread_size", "center_bore_mm"]:
            winner, conflict, conflict_note = _merge_categorical_field(group[field].tolist(), field)
            merged[field] = winner
            if conflict:
                has_conflicts = True
                if conflict_note:
                    conflict_notes.append(conflict_note)
                if verbose:
                    LOGGER.warning("Dedupe conflict for key '%s' field '%s': %s", key_str, field, conflict_note)

        merged["diameter_min_in"] = _merge_numeric_min(group["diameter_min_in"])
        merged["diameter_max_in"] = _merge_numeric_max(group["diameter_max_in"])
        merged["width_min_j"] = _merge_numeric_min(group["width_min_j"])
        merged["width_max_j"] = _merge_numeric_max(group["width_max_j"])
        merged["et_min"] = _merge_numeric_min(group["et_min"])
        merged["et_max"] = _merge_numeric_max(group["et_max"])

        merged["source_page"] = _merge_sources(group["source_page"])
        merged_url_count = len([p for p in merged["source_page"].split(" | ") if normalize_text(p)])

        base_manual_review = bool(group["needs_manual_review"].fillna(False).astype(bool).any())
        missing_keys = any(
            merged.get(field) in (None, "")
            for field in [
                "pcd",
                "cb",
                "thread_size",
                "diameter_min_in",
                "diameter_max_in",
                "width_min_j",
                "width_max_j",
                "et_min",
                "et_max",
            ]
        )

        needs_manual_review = bool(base_manual_review or has_conflicts or missing_keys)
        merged["needs_manual_review"] = needs_manual_review

        if has_conflicts or missing_keys:
            merged["extraction_confidence"] = 0.7 if not missing_keys else 0.55
        else:
            merged["extraction_confidence"] = 0.9

        merged["notes"] = _merge_notes(
            group["notes"],
            merged_url_count=merged_url_count,
            extra_notes=conflict_notes,
            merged_existing=merged_existing,
        )

        if len(group) > 1 and verbose:
            LOGGER.info("Merged duplicate generation key '%s' from %d rows", key_str, len(group))

        merged_rows.append(merged)

    merged_df = pd.DataFrame(merged_rows)
    for col in MODEL_SCHEMA:
        if col not in merged_df.columns:
            merged_df[col] = None
    return merged_df[MODEL_SCHEMA].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape TireWheelGuide fitment pages into generation-level Skoda model records.")
    parser.add_argument("--source-url", action="append", default=[], help="TireWheelGuide source URL. Repeatable.")
    parser.add_argument("--pairs-csv", default="", help="CSV input with columns: brand,model,source_url")
    parser.add_argument("--input-csv", default="", help="Alias of --pairs-csv, same columns: brand,model,source_url")
    parser.add_argument("--output-dir", default=".", help="Output directory for skoda_models.csv/json")
    parser.add_argument("--save-raw-html", action="store_true", help="Save fetched source HTML pages to disk")
    parser.add_argument("--raw-html-dir", default="raw_html/tirewheelguide", help="Directory for saving raw HTML files")
    parser.add_argument("--timeout", type=int, default=25, help="Request timeout in seconds")
    parser.add_argument("--user-agent", default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--merge-existing", action="store_true", help="Merge newly scraped rows with existing skoda_models output in --output-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    source_inputs: list[dict[str, str]] = []
    source_inputs.extend({"brand": "", "model": "", "source_url": normalize_text(url)} for url in args.source_url if normalize_text(url))

    csv_path = normalize_text(args.pairs_csv) or normalize_text(args.input_csv)
    if csv_path:
        source_inputs.extend(load_sources_from_csv(Path(csv_path)))

    if not source_inputs:
        raise ValueError("No sources provided. Use --source-url and/or --pairs-csv/--input-csv.")

    deduped_sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in source_inputs:
        source_url = normalize_text(item.get("source_url"))
        if not source_url:
            continue
        if source_url in seen_urls:
            continue
        seen_urls.add(source_url)
        deduped_sources.append(item)

    session = build_session(user_agent=args.user_agent)
    raw_html_dir = Path(args.raw_html_dir)

    LOGGER.info("Source URLs provided: %d", len(source_inputs))
    LOGGER.info("Source URLs after de-duplication: %d", len(deduped_sources))

    all_rows: list[dict[str, Any]] = []
    for source in deduped_sources:
        try:
            source_url = source["source_url"]
            html = fetch_html(session=session, source_url=source_url, timeout=args.timeout)

            if args.save_raw_html:
                saved_path = save_raw_html(raw_html_dir=raw_html_dir, source_url=source_url, html=html)
                LOGGER.info("Raw HTML saved: %s", saved_path)

            rows = parse_source_page_html(html=html, source_url=source_url, fallback_brand=source.get("brand", ""), fallback_model=source.get("model", ""))
            all_rows.extend(rows)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to parse %s: %s", source.get("source_url"), exc)

    new_rows_df = pd.DataFrame(all_rows)
    new_rows_df = normalize_schema(new_rows_df) if not new_rows_df.empty else pd.DataFrame(columns=MODEL_SCHEMA)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_rows_df = pd.DataFrame(columns=MODEL_SCHEMA)
    if args.merge_existing:
        existing_rows_df = load_existing_output(output_dir)
        existing_rows_df = normalize_schema(existing_rows_df) if not existing_rows_df.empty else existing_rows_df

    LOGGER.info("New generation rows scraped: %d", len(new_rows_df))
    LOGGER.info("Existing generation rows loaded: %d", len(existing_rows_df))

    models_df = pd.concat([existing_rows_df, new_rows_df], ignore_index=True)
    before_dedupe = len(models_df)
    models_df = consolidate_generation_rows(models_df, verbose=args.verbose, merged_existing=args.merge_existing) if not models_df.empty else models_df
    models_df = normalize_schema(models_df) if not models_df.empty else models_df
    after_dedupe = len(models_df)

    LOGGER.info("Generation rows before dedupe: %d", before_dedupe)
    LOGGER.info("Generation rows after dedupe: %d", after_dedupe)
    LOGGER.info("Final rows after merge+dedupe: %d", after_dedupe)
    csv_out = output_dir / "skoda_models.csv"
    json_out = output_dir / "skoda_models.json"

    models_df.to_csv(csv_out, index=False, encoding="utf-8")
    save_json(json_out, models_df)

    LOGGER.info("Rows written: %d", len(models_df))
    LOGGER.info("CSV saved to: %s", csv_out)
    LOGGER.info("JSON saved to: %s", json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
