from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger("scrape_skoda_models")

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

YEAR_RANGE_PATTERN = re.compile(r"(?P<from>19\d{2}|20\d{2})\s*(?:-|to|–|—)\s*(?P<to>19\d{2}|20\d{2}|present)", re.IGNORECASE)
RANGE_PATTERN = re.compile(r"(?P<low>[+-]?\d+(?:[\.,]\d+)?)\s*(?:-|to|–|—)\s*(?P<high>[+-]?\d+(?:[\.,]\d+)?)")
RIM_PATTERN = re.compile(
    r"(?P<width>\d+(?:[\.,]\d+)?)\s*J\s*[xX]\s*(?P<diameter>\d+(?:[\.,]\d+)?)\s*(?:ET\s*(?P<et>[+-]?\d+(?:[\.,]\d+)?))?",
    re.IGNORECASE,
)
RIM_ROW_FALLBACK_PATTERN = re.compile(
    r"(?P<width>\d+(?:[\.,]\d+)?)Jx(?P<diameter>\d+)(?:\s*ET(?P<et>[+-]?\d+))?",
    re.IGNORECASE,
)
WHEELSIZE_YEAR_PATH_PATTERN = re.compile(r"(?P<from>19\d{2}|20\d{2})(?:[-_](?P<to>19\d{2}|20\d{2}))?")
STRICT_PCD_PATTERN = re.compile(r"\b\d+\s*[xX]\s*\d+(?:[\.,]\d+)?\b")
STRICT_CB_PATTERN = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
MOUNTING_THREAD_PATTERN = re.compile(r"(\d{2})\s*[xX]\s*(\d+(?:[\.,]\d+)?)")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


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
            "Accept-Language": "en-US,en;q=0.8,lt;q=0.6",
            "Connection": "keep-alive",
        }
    )
    return session


def fetch_page(
    session: requests.Session,
    url: str,
    timeout: int,
    delay_seconds: float,
    save_raw_html: bool,
    raw_html_dir: Path,
) -> str | None:
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        html = response.text
        if save_raw_html:
            raw_html_dir.mkdir(parents=True, exist_ok=True)
            host = re.sub(r"[^A-Za-z0-9._-]+", "_", urlparse(url).netloc)
            path = re.sub(r"[^A-Za-z0-9._-]+", "_", urlparse(url).path)
            out_file = raw_html_dir / f"models_{host}_{path}_{int(time.time() * 1000)}.html"
            out_file.write_text(html, encoding="utf-8")
        time.sleep(delay_seconds)
        return html
    except requests.RequestException as exc:
        LOGGER.warning("Failed to fetch %s: %s", url, exc)
        return None


def normalize_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def to_float(value: Any) -> float | None:
    raw = normalize_text(value)
    if not raw:
        return None
    raw = raw.replace(",", ".")
    m = re.search(r"[+-]?\d+(?:\.\d+)?", raw)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_range(value: str) -> tuple[float | None, float | None]:
    raw = normalize_text(value)
    if not raw:
        return None, None
    match = RANGE_PATTERN.search(raw)
    if match:
        return to_float(match.group("low")), to_float(match.group("high"))
    single = to_float(raw)
    return single, single


def parse_year_range(value: str) -> tuple[int | None, int | None]:
    raw = normalize_text(value)
    if not raw:
        return None, None
    match = YEAR_RANGE_PATTERN.search(raw)
    if not match:
        return None, None
    year_from = int(match.group("from"))
    year_to_raw = match.group("to").lower()
    year_to = None if year_to_raw == "present" else int(year_to_raw)
    return year_from, year_to


def normalize_pcd(value: str | None) -> str | None:
    if not value:
        return None
    m = STRICT_PCD_PATTERN.search(value)
    if not m:
        return None
    raw = m.group(0).replace(" ", "")
    left, right = raw.lower().split("x", 1)
    right = right.replace(",", ".")
    return f"{int(float(left))}x{right}"


def derive_bolt_count_from_pcd(pcd: str | None) -> int | None:
    if not pcd:
        return None
    m = re.match(r"(\d+)\s*[xX]", pcd)
    if not m:
        return None
    return int(m.group(1))


def prettify_slug(value: str) -> str:
    clean = re.sub(r"[-_]+", " ", value.strip())
    clean = re.sub(r"\s+", " ", clean)
    return clean.title() if clean else ""


def _cleanup_generation_text(raw_generation: str) -> str:
    generation = raw_generation.strip()
    generation = re.sub(r"\s*\[\d{4}\s*\.\.\s*\d{4}\]\s*$", "", generation).strip()
    generation = re.sub(r"\s+(Europe|EU|EUDM|USDM|UK|Asia|Australia|Worldwide)\s*$", "", generation, flags=re.IGNORECASE).strip()
    return generation


def _humanize_generation_slug(generation_slug: str) -> str:
    tokens = [t for t in generation_slug.split("-") if t]
    if not tokens:
        return ""

    parts: list[str] = []
    for token in tokens:
        low = token.lower()
        if re.fullmatch(r"mk\d+", low):
            parts.append(low.capitalize())
            continue
        if re.fullmatch(r"[a-z]\d+", low):
            parts.append(low.upper())
            continue
        if re.fullmatch(r"\d+[a-z]+", low):
            parts.append(f"({low.upper()})")
            continue
        parts.append(token.upper() if len(token) <= 3 else token.capitalize())

    return " ".join(parts).strip()


def _parse_generation_token(token: str) -> tuple[str | None, int | None, int | None]:
    clean = unquote(token).strip("/")
    bits = [b for b in clean.split("-") if b]
    if len(bits) < 2:
        return None, None, None

    year_from: int | None = None
    year_to: int | None = None
    generation_bits = bits[:]
    if len(bits) >= 2 and re.fullmatch(r"\d{4}", bits[-1]) and re.fullmatch(r"\d{4}", bits[-2]):
        year_from = int(bits[-2])
        year_to = int(bits[-1])
        generation_bits = bits[:-2]
    elif re.fullmatch(r"\d{4}", bits[-1]):
        year_from = int(bits[-1])
        generation_bits = bits[:-1]

    generation = _humanize_generation_slug("-".join(generation_bits)) if generation_bits else None
    generation = _cleanup_generation_text(generation) if generation else None
    return generation, year_from, year_to


def parse_wheelsize_metadata_from_url(source_page: str) -> dict[str, Any]:
    parsed = urlparse(source_page)
    parts = [p for p in parsed.path.split("/") if p]
    lower_parts = [p.lower() for p in parts]

    model: str | None = None
    generation: str | None = None
    year_from: int | None = None
    year_to: int | None = None

    if "skoda" in lower_parts:
        skoda_idx = lower_parts.index("skoda")
        trailing = parts[skoda_idx + 1 :]
        if trailing:
            model = prettify_slug(unquote(trailing[0])) or None

        for token in trailing[1:]:
            gen_from_token, year_from_token, year_to_token = _parse_generation_token(token)
            if gen_from_token and generation is None:
                generation = gen_from_token
            if year_from is None and year_from_token is not None:
                year_from = year_from_token
            if year_to is None and year_to_token is not None:
                year_to = year_to_token
            if generation is not None and year_from is not None:
                break

            year_match = WHEELSIZE_YEAR_PATH_PATTERN.search(token)
            if year_match and year_from is None:
                year_from = int(year_match.group("from"))
                to_part = year_match.group("to")
                year_to = int(to_part) if to_part else None

    return {
        "model": model,
        "generation": generation,
        "year_from": year_from,
        "year_to": year_to,
    }


def parse_wheelsize_generation_from_meta(soup: BeautifulSoup, model: str | None) -> str | None:
    meta = soup.find("meta", attrs={"property": "car:title"})
    if not meta:
        meta = soup.find("meta", attrs={"name": "car:title"})
    if not meta:
        return None

    content = normalize_text(meta.get("content"))
    if not content:
        return None

    text = content
    text = re.sub(r"^Skoda\s+", "", text, flags=re.IGNORECASE).strip()
    if model and text.lower().startswith(model.lower()):
        text = text[len(model) :].strip()

    generation = _cleanup_generation_text(text)
    return generation or None


def parse_wheelsize_generation_from_header(soup: BeautifulSoup, model: str | None) -> str | None:
    headers = soup.select("h2[id^='generation-']")
    for h2 in headers:
        h2_text = h2.get_text(" ", strip=True)
        if model and model.lower() not in h2_text.lower():
            continue

        span = h2.find("span")
        raw_generation = span.get_text(" ", strip=True) if span else h2_text
        generation = re.sub(r"\s*\[\d{4}\s*\.\.\s*\d{4}\]\s*$", "", raw_generation).strip()

        if model and generation.lower().startswith(model.lower()):
            generation = generation[len(model) :].strip()
        generation = re.sub(r"^Skoda\s+", "", generation, flags=re.IGNORECASE).strip()
        if generation:
            return generation

    return None


def parse_rim_value(value: str) -> tuple[float | None, float | None, float | None]:
    raw = normalize_text(value)
    if not raw:
        return None, None, None
    match = RIM_PATTERN.search(raw)
    if not match:
        return None, None, None
    width = to_float(match.group("width"))
    diameter = to_float(match.group("diameter"))
    et = to_float(match.group("et"))
    return width, diameter, et


def parse_rim_value_fallback_from_row_text(value: str) -> tuple[float | None, float | None, float | None]:
    raw = normalize_text(value)
    if not raw:
        return None, None, None
    match = RIM_ROW_FALLBACK_PATTERN.search(raw)
    if not match:
        return None, None, None
    width = to_float(match.group("width"))
    diameter = to_float(match.group("diameter"))
    et = to_float(match.group("et"))
    return width, diameter, et


def parse_tables_with_bs4(soup: BeautifulSoup) -> list[pd.DataFrame]:
    def normalize_cell_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()

    def unique_headers(headers: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        output: list[str] = []
        for idx, header in enumerate(headers):
            base = header or f"col_{idx}"
            count = seen.get(base, 0)
            seen[base] = count + 1
            output.append(base if count == 0 else f"{base}_{count + 1}")
        return output

    def build_table_grid(table_tag: Any) -> list[list[str]]:
        tr_nodes = table_tag.find_all("tr")
        if not tr_nodes:
            return []

        # Determine maximum logical column count considering colspans.
        max_cols = 0
        for tr in tr_nodes:
            logical_cols = 0
            for cell in tr.find_all(["td", "th"]):
                colspan = int(cell.get("colspan", 1) or 1)
                logical_cols += max(1, colspan)
            max_cols = max(max_cols, logical_cols)
        if max_cols == 0:
            return []

        grid: list[list[str]] = []
        # col_idx -> (rows_left, value)
        pending_rowspans: dict[int, tuple[int, str]] = {}

        for tr in tr_nodes:
            row_values = [""] * max_cols

            # Apply pending rowspan values from previous rows.
            for col_idx in range(max_cols):
                if col_idx in pending_rowspans:
                    rows_left, value = pending_rowspans[col_idx]
                    row_values[col_idx] = value
                    if rows_left <= 1:
                        del pending_rowspans[col_idx]
                    else:
                        pending_rowspans[col_idx] = (rows_left - 1, value)

            col_ptr = 0
            cells = tr.find_all(["td", "th"])
            for cell in cells:
                while col_ptr < max_cols and row_values[col_ptr] != "":
                    col_ptr += 1
                if col_ptr >= max_cols:
                    break

                text = normalize_cell_text(cell.get_text(" ", strip=True))
                colspan = int(cell.get("colspan", 1) or 1)
                rowspan = int(cell.get("rowspan", 1) or 1)

                span_width = max(1, colspan)
                span_height = max(1, rowspan)
                for off in range(span_width):
                    target = col_ptr + off
                    if target >= max_cols:
                        break
                    row_values[target] = text
                    if span_height > 1:
                        pending_rowspans[target] = (span_height - 1, text)

                col_ptr += span_width

            if any(v.strip() for v in row_values):
                grid.append(row_values)

        return grid

    def pick_header_row(grid: list[list[str]]) -> int | None:
        best_idx: int | None = None
        best_score = -1
        for idx, row in enumerate(grid):
            low = [str(v).lower() for v in row]
            score = 0
            if any("rim" in v for v in low):
                score += 2
            if any("tire" in v or "tyre" in v for v in low):
                score += 2
            if any("offset" in v for v in low):
                score += 1
            if any(v.strip() for v in row):
                score += 1
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None or best_score < 2:
            return None
        return best_idx

    tables: list[pd.DataFrame] = []
    for table in soup.find_all("table"):
        grid = build_table_grid(table)
        if not grid:
            continue

        header_idx = pick_header_row(grid)
        if header_idx is None:
            continue

        headers = unique_headers([normalize_cell_text(x) for x in grid[header_idx]])

        data_rows: list[list[str]] = []
        for row in grid[header_idx + 1 :]:
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            if len(row) > len(headers):
                row = row[: len(headers)]
            if any(str(v).strip() for v in row):
                data_rows.append([normalize_cell_text(str(v)) for v in row])

        if data_rows:
            tables.append(pd.DataFrame(data_rows, columns=headers))
    return tables


def is_usable_wheelsize_table(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    rim_col = None
    for col in df.columns:
        if "rim" in str(col).lower():
            rim_col = str(col)
            break
    if rim_col is None:
        return False
    rim_values = df[rim_col].astype(str).str.strip()
    non_empty = rim_values[(rim_values != "") & (rim_values.str.lower() != "nan")]
    return (len(non_empty) / len(df)) >= 0.3 if len(df) else False


def parse_wheelsize_ranges(soup: BeautifulSoup, source_page: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    valid_table_count = 0
    parsed_tables = parse_tables_with_bs4(soup)
    LOGGER.info("Wheel-Size total HTML tables parsed for %s: %d", source_page, len(parsed_tables))

    # Conservative dedupe: only treat tables as duplicates when full normalized parseable rim rows are identical.
    seen_parseable_row_signatures: set[str] = set()

    for table_idx, df in enumerate(parsed_tables, start=1):
        column_names = [str(c) for c in df.columns]
        LOGGER.info("Wheel-Size table %d columns: %s", table_idx, column_names)

        normalized_cols = {str(c).lower().strip(): c for c in df.columns}

        def pick(*tokens: str) -> str | None:
            for low, orig in normalized_cols.items():
                if all(t in low for t in tokens):
                    return str(orig)
            return None

        rim_col = pick("rim")
        diameter_col = pick("diameter")
        width_col = pick("width")
        et_col = pick("et") or pick("offset")
        offset_range_col = pick("offset", "range")

        if rim_col is None:
            LOGGER.info("Wheel-Size table %d rejected: missing Rim column", table_idx)
            continue

        table_rows: list[dict[str, Any]] = []
        parseable_row_keys: list[str] = []
        rim_regex_match_count = 0
        offset_only_count = 0
        fallback_used = False

        for _, row in df.iterrows():
            rim_text = normalize_text(row.get(rim_col, ""))
            w_rim, d_rim, et_rim = parse_rim_value(rim_text)

            # Some Wheel-Size tables have an empty Rim column but still include rim spec in other cells.
            if d_rim is None and w_rim is None:
                full_row_text = " ".join(normalize_text(v) for v in row.tolist() if normalize_text(v))
                w_fb, d_fb, et_fb = parse_rim_value_fallback_from_row_text(full_row_text)
                if w_fb is not None and d_fb is not None:
                    w_rim, d_rim, et_rim = w_fb, d_fb, et_fb
                    fallback_used = True

            diameter_text = normalize_text(row.get(diameter_col, "")) if diameter_col else ""
            width_text = normalize_text(row.get(width_col, "")) if width_col else ""
            et_text = normalize_text(row.get(et_col, "")) if et_col else ""
            offset_range_text = normalize_text(row.get(offset_range_col, "")) if offset_range_col else ""

            d_min, d_max = parse_range(diameter_text)
            w_min, w_max = parse_range(width_text)
            et_min, et_max = parse_range(offset_range_text or et_text)

            if d_rim is not None:
                d_min = d_min if d_min is not None else d_rim
                d_max = d_max if d_max is not None else d_rim
            if w_rim is not None:
                w_min = w_min if w_min is not None else w_rim
                w_max = w_max if w_max is not None else w_rim
            if et_rim is not None and et_min is None and et_max is None:
                et_min = et_rim
                et_max = et_rim

            # A row is rim-parsed only if both width and diameter are successfully parsed.
            rim_matched = w_rim is not None and d_rim is not None
            if rim_matched:
                rim_regex_match_count += 1
            elif et_min is not None or et_max is not None:
                offset_only_count += 1

            if d_min is None and d_max is None and w_min is None and w_max is None and et_min is None and et_max is None:
                continue

            parsed_row = {
                "diameter_min_in": d_min,
                "diameter_max_in": d_max,
                "width_min_j": w_min,
                "width_max_j": w_max,
                "et_min": et_min,
                "et_max": et_max,
            }
            table_rows.append(parsed_row)

            parseable_row_keys.append(
                "|".join(
                    [
                        str(d_min),
                        str(d_max),
                        str(w_min),
                        str(w_max),
                        str(et_min),
                        str(et_max),
                    ]
                )
            )

        if not table_rows:
            LOGGER.info("Wheel-Size table %d rejected: zero parseable rim rows", table_idx)
            continue

        if rim_regex_match_count == 0:
            LOGGER.info(
                "Wheel-Size table %d rejected: no successful rim regex matches (offset_only_rows=%d)",
                table_idx,
                offset_only_count,
            )
            continue

        dedupe_signature = "\n".join(sorted(parseable_row_keys))
        if dedupe_signature in seen_parseable_row_signatures:
            LOGGER.info("Wheel-Size table %d rejected: duplicate parseable rim rows", table_idx)
            continue
        seen_parseable_row_signatures.add(dedupe_signature)

        valid_table_count += 1
        rows.extend(table_rows)

        table_diameters: list[float] = []
        table_widths: list[float] = []
        table_et_mins: list[float] = []
        table_et_maxs: list[float] = []
        for parsed_row in table_rows:
            for key in ("diameter_min_in", "diameter_max_in"):
                value = to_float(parsed_row.get(key))
                if value is not None:
                    table_diameters.append(value)
            for key in ("width_min_j", "width_max_j"):
                value = to_float(parsed_row.get(key))
                if value is not None:
                    table_widths.append(value)
            vmin = to_float(parsed_row.get("et_min"))
            vmax = to_float(parsed_row.get("et_max"))
            if vmin is not None:
                table_et_mins.append(vmin)
            if vmax is not None:
                table_et_maxs.append(vmax)

        LOGGER.info(
            "Wheel-Size table %d accepted: rim_regex_rows=%d, offset_only_rows=%d, fallback_used=%s, diameter_min_in=%s, diameter_max_in=%s, width_min_j=%s, width_max_j=%s, et_min=%s, et_max=%s",
            table_idx,
            rim_regex_match_count,
            offset_only_count,
            fallback_used,
            min(table_diameters) if table_diameters else None,
            max(table_diameters) if table_diameters else None,
            min(table_widths) if table_widths else None,
            max(table_widths) if table_widths else None,
            min(table_et_mins) if table_et_mins else None,
            max(table_et_maxs) if table_et_maxs else None,
        )

    diameters: list[float] = []
    widths: list[float] = []
    et_mins: list[float] = []
    et_maxs: list[float] = []
    for row in rows:
        for key in ("diameter_min_in", "diameter_max_in"):
            v = to_float(row.get(key))
            if v is not None:
                diameters.append(v)
        for key in ("width_min_j", "width_max_j"):
            v = to_float(row.get(key))
            if v is not None:
                widths.append(v)
        vmin = to_float(row.get("et_min"))
        vmax = to_float(row.get("et_max"))
        if vmin is not None:
            et_mins.append(vmin)
        if vmax is not None:
            et_maxs.append(vmax)

    aggregated = {
        "diameter_min_in": min(diameters) if diameters else None,
        "diameter_max_in": max(diameters) if diameters else None,
        "width_min_j": min(widths) if widths else None,
        "width_max_j": max(widths) if widths else None,
        "et_min": min(et_mins) if et_mins else None,
        "et_max": max(et_maxs) if et_maxs else None,
        "valid_table_count": valid_table_count,
        "aggregated_from_multiple_variants": valid_table_count > 1,
        "source_page": source_page,
    }

    LOGGER.info("Wheel-Size valid tables used for %s: %d", source_page, valid_table_count)
    LOGGER.info(
        "Wheel-Size aggregated ranges for %s -> diameter_min_in=%s, diameter_max_in=%s, width_min_j=%s, width_max_j=%s, et_min=%s, et_max=%s",
        source_page,
        aggregated["diameter_min_in"],
        aggregated["diameter_max_in"],
        aggregated["width_min_j"],
        aggregated["width_max_j"],
        aggregated["et_min"],
        aggregated["et_max"],
    )

    return aggregated


def parse_wheelfitment_page(soup: BeautifulSoup, source_page: str) -> dict[str, Any]:
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    normalized = re.sub(r"\s+", " ", text)

    title = ""
    if soup.title:
        title = soup.title.get_text(" ", strip=True)

    model = None
    generation = None
    year_from = None
    year_to = None

    if title:
        # Example: Skoda Octavia (1997 - 2005) Wheel Fitment
        title_match = re.search(r"Skoda\s+([^\(]+)\((\d{4})\s*-\s*(\d{4})\)", title, flags=re.IGNORECASE)
        if title_match:
            model = title_match.group(1).strip()
            year_from = int(title_match.group(2))
            year_to = int(title_match.group(3))

    if model is None:
        path_text = unquote(urlparse(source_page).path)
        # /car/Skoda/Octavia (1997 - 2005).html
        path_match = re.search(r"/Skoda/([^/\(]+)\s*\((\d{4})\s*-\s*(\d{4})\)", path_text, flags=re.IGNORECASE)
        if path_match:
            model = path_match.group(1).strip()
            year_from = int(path_match.group(2))
            year_to = int(path_match.group(3))

    pcd_match = re.search(r"PCD\s*:\s*([^\n\r<]+)", normalized, flags=re.IGNORECASE)
    cb_match = re.search(r"Center\s*bore\s*:\s*([^\n\r<]+)", normalized, flags=re.IGNORECASE)
    mounting_match = re.search(r"Mounting\s*:\s*([^\n\r<]+)", normalized, flags=re.IGNORECASE)
    offset_match = re.search(r"Offset\s*:\s*([+-]?\d+(?:[\.,]\d+)?)", normalized, flags=re.IGNORECASE)

    pcd = normalize_pcd(pcd_match.group(1)) if pcd_match else None

    cb = None
    if cb_match:
        cb_raw = cb_match.group(1)
        cb_num = STRICT_CB_PATTERN.search(cb_raw)
        if cb_num:
            cb = to_float(cb_num.group(0))

    thread_size = None
    if mounting_match:
        mount_raw = mounting_match.group(1)
        mount_match = MOUNTING_THREAD_PATTERN.search(mount_raw)
        if mount_match:
            thread_size = f"M{mount_match.group(1)} x {mount_match.group(2).replace(',', '.')}"

    default_offset = to_float(offset_match.group(1)) if offset_match else None

    return {
        "model": model,
        "generation": generation,
        "year_from": year_from,
        "year_to": year_to,
        "pcd": pcd,
        "cb": cb,
        "thread_size": thread_size,
        "bolt_count": derive_bolt_count_from_pcd(pcd),
        "default_offset": default_offset,
        "source_page": source_page,
    }


def merge_pair(
    pair_meta: dict[str, Any],
    wheelsize_data: dict[str, Any],
    wheelfitment_data: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    conflicts: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []

    model = wheelsize_data.get("model") or pair_meta.get("model") or wheelfitment_data.get("model")
    generation = wheelsize_data.get("generation") or pair_meta.get("generation")
    year_from = wheelsize_data.get("year_from") or pair_meta.get("year_from") or wheelfitment_data.get("year_from")
    year_to = wheelsize_data.get("year_to") or pair_meta.get("year_to") or wheelfitment_data.get("year_to")

    if wheelfitment_data.get("model") and model and normalize_text(wheelfitment_data.get("model")).lower() != normalize_text(model).lower():
        conflicts.append(
            {
                "dataset": "models",
                "issue_type": "conflicting_model_name",
                "key": f"{model}:{generation}:{year_from}-{year_to}",
                "details": f"Wheel-Size model '{model}' vs Wheelfitment model '{wheelfitment_data.get('model')}'",
                "source_page": f"{wheelsize_data.get('source_page')} | {wheelfitment_data.get('source_page')}",
                "recommended_action": "Confirm model naming and URL pairing.",
            }
        )

    notes_parts: list[str] = []
    if wheelfitment_data.get("default_offset") is not None:
        notes_parts.append(f"wheelfitment_default_offset={wheelfitment_data.get('default_offset')}")
    if wheelsize_data.get("aggregated_from_multiple_variants"):
        notes_parts.append("aggregated_from_multiple_variants=true")

    row: dict[str, Any] = {
        "brand": pair_meta.get("brand") or "Skoda",
        "model": model,
        "generation": generation,
        "year_from": year_from,
        "year_to": year_to,
        "pcd": wheelfitment_data.get("pcd"),
        "cb": wheelfitment_data.get("cb"),
        "bolt_count": wheelfitment_data.get("bolt_count"),
        "thread_size": wheelfitment_data.get("thread_size"),
        "center_bore_mm": wheelfitment_data.get("cb"),
        "diameter_min_in": wheelsize_data.get("diameter_min_in"),
        "diameter_max_in": wheelsize_data.get("diameter_max_in"),
        "width_min_j": wheelsize_data.get("width_min_j"),
        "width_max_j": wheelsize_data.get("width_max_j"),
        "et_min": wheelsize_data.get("et_min"),
        "et_max": wheelsize_data.get("et_max"),
        "notes": " | ".join(notes_parts) if notes_parts else None,
        "source_type": "merged_wheelsize_wheelfitment",
        "source_page": f"{wheelsize_data.get('source_page', '')} | {wheelfitment_data.get('source_page', '')}",
        "extraction_confidence": 0.85,
        "needs_manual_review": False,
    }

    critical_missing = ["pcd", "cb", "thread_size", "diameter_min_in", "diameter_max_in", "width_min_j", "width_max_j"]
    if any(row.get(k) in (None, "") for k in critical_missing):
        row["needs_manual_review"] = True
        row["extraction_confidence"] = 0.55
        reviews.append(
            {
                "dataset": "models",
                "issue_type": "missing_critical_fields",
                "key": f"{row.get('brand')}:{row.get('model')}:{row.get('generation')}:{row.get('year_from')}-{row.get('year_to')}",
                "details": "Missing one of: pcd, cb, thread_size, diameter range, width range",
                "source_page": row.get("source_page", ""),
                "recommended_action": "Verify both source URLs and update parser mappings.",
            }
        )

    return row, conflicts, reviews


def ensure_schema(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]


def load_pairs_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    required = ["wheel_size_url", "wheelfitment_url"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"pairs CSV missing columns: {missing}")

    pairs: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        ws = normalize_text(row.get("wheel_size_url"))
        wf = normalize_text(row.get("wheelfitment_url"))
        if not ws or not wf:
            continue
        pairs.append(
            {
                "brand": normalize_text(row.get("brand")) or "Skoda",
                "model": normalize_text(row.get("model")) or None,
                "generation": normalize_text(row.get("generation")) or None,
                "year_from": int(row.get("year_from")) if pd.notna(row.get("year_from")) else None,
                "year_to": int(row.get("year_to")) if pd.notna(row.get("year_to")) else None,
                "wheel_size_url": ws,
                "wheelfitment_url": wf,
            }
        )
    return pairs


def load_pairs_from_cli(wheel_size_urls: list[str], wheelfitment_urls: list[str]) -> list[dict[str, Any]]:
    ws = [normalize_text(x) for x in wheel_size_urls if normalize_text(x)]
    wf = [normalize_text(x) for x in wheelfitment_urls if normalize_text(x)]
    if not ws and not wf:
        return []
    if len(ws) != len(wf):
        raise ValueError("--wheel-size-url and --wheelfitment-url counts must match.")
    return [
        {
            "brand": "Skoda",
            "model": None,
            "generation": None,
            "year_from": None,
            "year_to": None,
            "wheel_size_url": w,
            "wheelfitment_url": f,
        }
        for w, f in zip(ws, wf)
    ]


def save_outputs(models_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_csv = output_dir / "skoda_models.csv"
    models_json = output_dir / "skoda_models.json"
    models_df.to_csv(models_csv, index=False, encoding="utf-8")
    models_json.write_text(
        json.dumps(models_df.where(pd.notna(models_df), None).to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.info("Saved %s", models_csv)
    LOGGER.info("Saved %s", models_json)


def save_review_outputs(conflicts: list[dict[str, Any]], reviews: list[dict[str, Any]], output_dir: Path) -> None:
    cols = ["dataset", "issue_type", "key", "details", "source_page", "recommended_action"]
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(conflicts, columns=cols).to_csv(output_dir / "source_conflicts.csv", index=False, encoding="utf-8")
    pd.DataFrame(reviews, columns=cols).to_csv(output_dir / "manual_review.csv", index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Skoda model fitment using URL pairs (Wheel-Size + wheelfitment.eu).")
    parser.add_argument("--pairs-csv", default="", help="CSV with columns: brand,model,generation,year_from,year_to,wheel_size_url,wheelfitment_url")
    parser.add_argument("--wheel-size-url", action="append", default=[], help="Wheel-Size URL. Repeatable.")
    parser.add_argument("--wheelfitment-url", action="append", default=[], help="wheelfitment.eu URL. Repeatable.")
    parser.add_argument("--output-dir", default=".", help="Output directory for model files.")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout in seconds.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds.")
    parser.add_argument("--save-raw-html", action="store_true", help="Save raw HTML snapshots for debugging.")
    parser.add_argument("--raw-html-dir", default="raw_html/models", help="Raw HTML output directory.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    parser.add_argument(
        "--user-agent",
        default="BachelorThesisSkodaWheelResearchBot/1.0 (+contact: local-research)",
        help="HTTP user-agent header.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    output_dir = Path(args.output_dir)
    raw_html_dir = Path(args.raw_html_dir)
    session = build_session(user_agent=args.user_agent)

    pairs: list[dict[str, Any]] = []
    if args.pairs_csv:
        pairs.extend(load_pairs_from_csv(Path(args.pairs_csv)))
    pairs.extend(load_pairs_from_cli(args.wheel_size_url, args.wheelfitment_url))

    if not pairs:
        raise ValueError("No source pairs provided. Use --pairs-csv or matching --wheel-size-url/--wheelfitment-url pairs.")

    records: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []

    for pair in pairs:
        ws_url = pair["wheel_size_url"]
        wf_url = pair["wheelfitment_url"]

        ws_html = fetch_page(
            session=session,
            url=ws_url,
            timeout=args.timeout,
            delay_seconds=args.delay,
            save_raw_html=args.save_raw_html,
            raw_html_dir=raw_html_dir,
        )
        wf_html = fetch_page(
            session=session,
            url=wf_url,
            timeout=args.timeout,
            delay_seconds=args.delay,
            save_raw_html=args.save_raw_html,
            raw_html_dir=raw_html_dir,
        )

        if not ws_html or not wf_html:
            reviews.append(
                {
                    "dataset": "models",
                    "issue_type": "source_fetch_failed",
                    "key": f"{pair.get('brand')}:{pair.get('model')}:{pair.get('generation')}",
                    "details": "Failed to fetch one or both pair URLs.",
                    "source_page": f"{ws_url} | {wf_url}",
                    "recommended_action": "Retry fetch and check network/source availability.",
                }
            )
            continue

        ws_soup = BeautifulSoup(ws_html, "html.parser")
        wf_soup = BeautifulSoup(wf_html, "html.parser")

        ws_meta = parse_wheelsize_metadata_from_url(ws_url)
        if not ws_meta.get("generation"):
            ws_generation = parse_wheelsize_generation_from_meta(ws_soup, model=ws_meta.get("model"))
            if ws_generation:
                ws_meta["generation"] = ws_generation
        if not ws_meta.get("generation"):
            ws_generation = parse_wheelsize_generation_from_header(ws_soup, model=ws_meta.get("model"))
            if ws_generation:
                ws_meta["generation"] = ws_generation
        ws_ranges = parse_wheelsize_ranges(ws_soup, source_page=ws_url)
        wheelsize_data = {**ws_meta, **ws_ranges}
        wheelfitment_data = parse_wheelfitment_page(wf_soup, source_page=wf_url)

        merged, row_conflicts, row_reviews = merge_pair(pair, wheelsize_data, wheelfitment_data)
        records.append(merged)
        conflicts.extend(row_conflicts)
        reviews.extend(row_reviews)

    models_df = ensure_schema(pd.DataFrame(records), MODEL_SCHEMA)
    save_outputs(models_df=models_df, output_dir=output_dir)
    save_review_outputs(conflicts=conflicts, reviews=reviews, output_dir=output_dir)


if __name__ == "__main__":
    main()
