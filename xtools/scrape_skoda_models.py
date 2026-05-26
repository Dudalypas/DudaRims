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
    "review_reason",
    "needs_manual_review",
]

URL_BRAND_MODEL_YEAR_PATTERN = re.compile(
    r"/sizes/(?P<brand>[^/]+)/(?P<model>[^/]+)/(?P<year>19\d{2}|20\d{2})/?",
    re.IGNORECASE,
)
YEAR_RANGE_PATTERN = re.compile(r"(?P<from>19\d{2}|20\d{2})\s*(?:-|to|–|—)\s*(?P<to>19\d{2}|20\d{2})", re.IGNORECASE)
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
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")



def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text



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



def to_int(value: Any) -> int | None:
    val = to_float(value)
    return int(round(val)) if val is not None else None



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
    return 0



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



def load_input_rows(source_urls: list[str], pairs_csv: str | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for url in source_urls:
        if url and url.strip():
            rows.append({"brand": "", "model": "", "source_url": url.strip()})
    if pairs_csv:
        rows.extend(load_sources_from_csv(Path(pairs_csv)))

    seen: set[str] = set()
    dedup: list[dict[str, str]] = []
    for row in rows:
        source_url = normalize_text(row.get("source_url"))
        if not source_url or source_url in seen:
            continue
        seen.add(source_url)
        dedup.append(row)
    return dedup



def parse_page_identity(soup: BeautifulSoup, source_url: str) -> tuple[str, str]:
    canonical_tag = soup.find("link", rel=lambda value: value and "canonical" in str(value).lower())
    canonical = normalize_text(canonical_tag.get("href")) if canonical_tag else ""
    canonical = canonical or source_url

    url_match = URL_BRAND_MODEL_YEAR_PATTERN.search(urlparse(canonical).path)
    brand = slug_to_name(url_match.group("brand")) if url_match else ""
    model = slug_to_name(url_match.group("model")) if url_match else ""

    return brand, model



def parse_year_range(value: str) -> tuple[int | None, int | None]:
    text = normalize_space(value)
    match = YEAR_RANGE_PATTERN.search(text)
    if not match:
        return None, None
    return int(match.group("from")), int(match.group("to"))



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



def parse_generation_header(header_text: str, brand: str, model: str, heading_id: str) -> tuple[str, int | None, int | None] | None:
    clean = normalize_space(header_text)
    year_from, year_to = parse_year_range(clean)
    if year_from is None or year_to is None:
        return None

    if MODIFICATION_HEADER_PATTERN.match(clean):
        return None

    if brand and clean.lower().startswith(brand.lower()):
        clean = normalize_space(clean[len(brand) :])
    if model and clean.lower().startswith(model.lower()):
        clean = normalize_space(clean[len(model) :])

    clean = normalize_space(re.sub(r"\b(19\d{2}|20\d{2})\s*(?:-|to|–|—)\s*(19\d{2}|20\d{2})\b", "", clean, flags=re.IGNORECASE))
    if not clean:
        clean = generation_from_id(heading_id)

    return normalize_space(clean), year_from, year_to



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
        return to_float(match.group("diameter")), to_float(match.group("width")), to_float(match.group("et"))

    match = RIM_DIAMETER_FIRST_PATTERN.search(text)
    if match:
        return to_float(match.group("diameter")), to_float(match.group("width")), to_float(match.group("et"))

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
        candidates = extract_rim_cell_candidates(cells[rim_col_idx])

        best = (None, None, None)
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
        if value is None or (isinstance(value, str) and not value.strip()):
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



def parse_generation_block(generation_h2: Tag, next_generation_h2: Tag | None, brand: str, model: str) -> dict[str, Any] | None:
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

    for node in generation_nodes:
        if node.name == "h2":
            header_text = normalize_space(node.get_text(" ", strip=True))
            if MODIFICATION_HEADER_PATTERN.match(header_text):
                current_mod = {
                    "name": parse_modification_name(header_text, brand=brand, model=model),
                    "pcd": None,
                    "cb": None,
                    "thread_size": None,
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
            if rim_rows:
                current_mod["rim_rows"].extend(rim_rows)

    if not modifications:
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

    review_reasons: list[str] = []
    if pcd_conflicts:
        review_reasons.append("conflicting PCD values")
    if cb_conflicts:
        review_reasons.append("conflicting CB values")
    if thread_conflicts:
        review_reasons.append("conflicting thread_size values")
    if pcd is None or (isinstance(pcd, str) and not pcd.strip()):
        review_reasons.append("missing PCD")
    if cb is None:
        review_reasons.append("missing CB")
    if diameter_min is None or diameter_max is None:
        review_reasons.append("missing diameter range")
    if width_min is None or width_max is None:
        review_reasons.append("missing width range")
    if et_min is None or et_max is None:
        review_reasons.append("missing ET range")

    needs_manual_review = 1 if review_reasons else 0

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
        "review_reason": " | ".join(sorted(set(review_reasons))) if review_reasons else None,
        "needs_manual_review": needs_manual_review,
    }



def parse_html_rows(html: str, source_url: str, fallback_brand: str = "", fallback_model: str = "") -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    parsed_brand, parsed_model = parse_page_identity(soup=soup, source_url=source_url)
    brand = parsed_brand or fallback_brand or "Skoda"
    model = parsed_model or fallback_model

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

    rows: list[dict[str, Any]] = []
    for idx, generation_h2 in enumerate(generation_h2s):
        next_h2 = generation_h2s[idx + 1] if idx + 1 < len(generation_h2s) else None
        row = parse_generation_block(generation_h2=generation_h2, next_generation_h2=next_h2, brand=brand, model=model)
        if row:
            rows.append(row)

    return rows



def save_json(path: Path, df: pd.DataFrame) -> None:
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")



def load_existing_output(output_dir: Path) -> pd.DataFrame:
    csv_path = output_dir / "skoda_models.csv"
    json_path = output_dir / "skoda_models.json"

    if csv_path.exists():
        return pd.read_csv(csv_path)

    if json_path.exists():
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
    ]
    int_cols = ["year_from", "year_to", "bolt_count"]

    for col in float_cols:
        df[col] = df[col].apply(to_float)
    for col in int_cols:
        df[col] = df[col].apply(to_int)

    df["needs_manual_review"] = df["needs_manual_review"].apply(to_bool_int)
    df["review_reason"] = df["review_reason"].apply(normalize_text)
    df["brand"] = df["brand"].astype(str).replace("nan", "").str.strip()
    df["model"] = df["model"].astype(str).replace("nan", "").str.strip()
    df["generation"] = df["generation"].astype(str).replace("nan", "").str.strip()
    df["pcd"] = df["pcd"].astype(str).replace("nan", "").str.strip()
    df["thread_size"] = df["thread_size"].astype(str).replace("nan", "").str.strip()

    dedupe_keys = ["brand", "model", "generation", "year_from", "year_to"]
    return df[MODEL_SCHEMA].drop_duplicates(subset=dedupe_keys, keep="first")



def consolidate_generation_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    key_cols = ["brand", "model", "generation", "year_from", "year_to"]
    grouped = df.groupby(key_cols, dropna=False, sort=False)
    merged_rows: list[dict[str, Any]] = []

    for key, group in grouped:
        merged: dict[str, Any] = {
            "brand": key[0],
            "model": key[1],
            "generation": key[2],
            "year_from": key[3],
            "year_to": key[4],
        }

        reasons: list[str] = []

        for field in ["pcd", "thread_size"]:
            winner, conflicts = most_common_non_null(group[field].tolist())
            merged[field] = winner
            if conflicts:
                reasons.append(f"conflicting {field} values")

        for field in ["cb", "center_bore_mm", "bolt_count"]:
            values = [to_float(v) for v in group[field].tolist() if to_float(v) is not None]
            if field == "bolt_count":
                ivals = [to_int(v) for v in group[field].tolist() if to_int(v) is not None]
                merged[field] = ivals[0] if ivals else None
                if len(set(ivals)) > 1:
                    reasons.append("conflicting bolt_count values")
            else:
                merged[field] = values[0] if values else None
                if len(set(values)) > 1:
                    reasons.append(f"conflicting {field} values")

        merged["diameter_min_in"] = min([to_float(v) for v in group["diameter_min_in"].tolist() if to_float(v) is not None], default=None)
        merged["diameter_max_in"] = max([to_float(v) for v in group["diameter_max_in"].tolist() if to_float(v) is not None], default=None)
        merged["width_min_j"] = min([to_float(v) for v in group["width_min_j"].tolist() if to_float(v) is not None], default=None)
        merged["width_max_j"] = max([to_float(v) for v in group["width_max_j"].tolist() if to_float(v) is not None], default=None)
        merged["et_min"] = min([to_float(v) for v in group["et_min"].tolist() if to_float(v) is not None], default=None)
        merged["et_max"] = max([to_float(v) for v in group["et_max"].tolist() if to_float(v) is not None], default=None)

        existing_reasons = [normalize_space(normalize_text(x)) for x in group["review_reason"].tolist() if normalize_text(x)]
        reasons.extend(existing_reasons)

        if merged.get("pcd") in (None, ""):
            reasons.append("missing PCD")
        if merged.get("center_bore_mm") is None:
            reasons.append("missing CB")
        if merged.get("diameter_min_in") is None or merged.get("diameter_max_in") is None:
            reasons.append("missing diameter range")
        if merged.get("width_min_j") is None or merged.get("width_max_j") is None:
            reasons.append("missing width range")
        if merged.get("et_min") is None or merged.get("et_max") is None:
            reasons.append("missing ET range")

        reason_text = " | ".join(dict.fromkeys([r for r in reasons if r])) if reasons else ""
        merged["review_reason"] = reason_text
        merged["needs_manual_review"] = 1 if reason_text else 0

        merged_rows.append(merged)

    merged_df = pd.DataFrame(merged_rows)
    for col in MODEL_SCHEMA:
        if col not in merged_df.columns:
            merged_df[col] = None
    return merged_df[MODEL_SCHEMA].copy()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape TireWheelGuide pages into canonical Skoda fitment records")
    parser.add_argument("--source-url", action="append", default=[], help="TireWheelGuide URL, can be repeated")
    parser.add_argument("--pairs-csv", default="", help="CSV with brand,model,source_url")
    parser.add_argument("--input-csv", default="", help="Alias for --pairs-csv")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--save-raw-html", action="store_true", help="Save raw HTML")
    parser.add_argument("--raw-html-dir", default="raw_html/tirewheelguide", help="Raw HTML directory")
    parser.add_argument("--timeout", type=int, default=25, help="Request timeout in seconds")
    parser.add_argument("--user-agent", default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--merge-existing", action="store_true", help="Merge with existing output")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    deduped_sources = load_input_rows(args.source_url or [], args.pairs_csv or args.input_csv)
    if not deduped_sources:
        raise ValueError("No sources provided. Use --source-url and/or --pairs-csv/--input-csv.")

    session = build_session(user_agent=args.user_agent)
    raw_html_dir = Path(args.raw_html_dir)

    all_rows: list[dict[str, Any]] = []
    processed_pages = 0
    failed_pages = 0

    for source in deduped_sources:
        source_url = source.get("source_url")
        try:
            html = fetch_html(session=session, source_url=source_url, timeout=args.timeout)
        except Exception:
            failed_pages += 1
            continue

        processed_pages += 1
        if args.save_raw_html:
            try:
                save_raw_html(raw_html_dir=raw_html_dir, source_url=source_url, html=html)
            except Exception:
                pass

        try:
            rows = parse_html_rows(
                html=html,
                source_url=source_url,
                fallback_brand=source.get("brand", ""),
                fallback_model=source.get("model", ""),
            )
            all_rows.extend(rows)
        except Exception:
            failed_pages += 1

    new_rows_df = pd.DataFrame(all_rows)
    new_rows_df = normalize_schema(new_rows_df) if not new_rows_df.empty else pd.DataFrame(columns=MODEL_SCHEMA)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_rows_df = pd.DataFrame(columns=MODEL_SCHEMA)
    if args.merge_existing:
        existing_rows_df = load_existing_output(output_dir)
        existing_rows_df = normalize_schema(existing_rows_df) if not existing_rows_df.empty else existing_rows_df

    models_df = pd.concat([existing_rows_df, new_rows_df], ignore_index=True)
    models_df = consolidate_generation_rows(models_df) if not models_df.empty else models_df
    models_df = normalize_schema(models_df) if not models_df.empty else models_df

    csv_out = output_dir / "skoda_models.csv"
    json_out = output_dir / "skoda_models.json"

    models_df.to_csv(csv_out, index=False, encoding="utf-8")
    save_json(json_out, models_df)

    manual_review_count = int(models_df[models_df["needs_manual_review"] == 1].shape[0]) if not models_df.empty else 0
    print(f"[scrape] processed: {processed_pages}")
    print(f"[scrape] failed: {failed_pages}")
    print(f"[scrape] rows: {len(models_df)}")
    print(f"[scrape] manual: {manual_review_count}")
    print(f"[scrape] csv: {csv_out}")
    print(f"[scrape] json: {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
