from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger("scrape_skoda_wheels")

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

MAPPING_SCHEMA = [
    "wheel_class_name",
    "official_wheel_name",
    "variant_type",
    "mapping_confidence",
    "needs_manual_review",
]

DEFAULT_WHEEL_CLASSES = [
    "Alaris",
    "Alcatras",
    "Braga",
    "Braga_Diamond_Cut_Black",
    "Castor",
    "Denom",
    "Gemini",
    "Hawk",
    "Ilias",
    "Mytikas",
    "Mytikas_Black",
    "Nivalis",
    "Ratikon",
    "Steel_Wheel",
    "Triton",
    "Trius",
    "Trius_Black",
    "Turbine",
    "Vega",
    "Velorum",
    "Xtreme",
]

DIMENSION_PATTERN = re.compile(
    r"(?P<width>\d+(?:[\.,]\d+)?)\s*J\s*[xX]\s*(?P<diameter>\d+(?:[\.,]\d+)?)\s*(?:ET\s*(?P<et>[+-]?\d+(?:[\.,]\d+)?))?",
    flags=re.IGNORECASE,
)


PCD_PATTERN = re.compile(r"\b(?P<bolt>\d+)\s*[xX]\s*(?P<diameter>\d+(?:[\.,]\d+)?)\b")
YEAR_RANGE_PATTERN = re.compile(r"(?P<from>19\d{2}|20\d{2})\s*(?:-|to|–|—)\s*(?P<to>19\d{2}|20\d{2}|present)", re.IGNORECASE)
OEM_PATTERN = re.compile(r"\b([0-9A-Z]{2,}-[0-9A-Z-]{2,}|[0-9A-Z]{6,})\b")


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


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    return cleaned.strip("_") or "page"


def snapshot_html(html: str, url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    host = sanitize_name(urlparse(url).netloc)
    path = sanitize_name(urlparse(url).path)
    stamp = int(time.time() * 1000)
    out_file = out_dir / f"{host}_{path}_{stamp}.html"
    out_file.write_text(html, encoding="utf-8")


def fetch_page(
    session: requests.Session,
    url: str,
    timeout: int,
    delay_seconds: float,
    save_raw_html: bool,
    raw_html_dir: Path,
) -> str | None:
    try:
        LOGGER.debug("Fetching %s", url)
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        html = response.text
        if save_raw_html:
            snapshot_html(html, url, raw_html_dir)
        time.sleep(delay_seconds)
        return html
    except requests.RequestException as exc:
        LOGGER.warning("Failed to fetch %s: %s", url, exc)
        return None


def extract_links_from_listing(html: str, listing_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    found: set[str] = set()
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()
        if not href:
            continue
        url = urljoin(listing_url, href)
        text = " ".join(anchor.stripped_strings).lower()
        if "wheel" in text or "rim" in text or re.search(r"wheel|rim|alloy", url, flags=re.IGNORECASE):
            found.add(url)
    return sorted(found)


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def parse_dimensions(text: str) -> dict[str, float | None]:
    match = DIMENSION_PATTERN.search(text)
    if not match:
        return {"width_j": None, "diameter_in": None, "et": None}
    return {
        "width_j": parse_float(match.group("width")),
        "diameter_in": parse_float(match.group("diameter")),
        "et": parse_float(match.group("et")),
    }


def normalize_wheel_style_name(raw_name: str, source_url: str) -> str:
    text = raw_name or ""
    text = re.sub(r"\|\s*Skoda-Parts\.com.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[0-9A-Z]{8,}\b", "", text)
    text = re.sub(r"\b(aluminium|alloy|steel)\s*(rim|wheel)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\brim\b|\bwheel\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[ŠS]koda\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{1,2}\"\b", "", text)
    text = re.sub(r"\b\d{1,2}\s*inch\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" -|_")

    if text:
        # Jei pavadinimas triuksmingas, imam paskutini prasminga tokena(us) kaip style label.
        tokens = [tok for tok in re.split(r"\s+", text) if tok]
        if tokens:
            return " ".join(tokens[-2:]) if len(tokens) > 1 and tokens[-2].isalpha() and tokens[-1].isalpha() else tokens[-1]

    path = urlparse(source_url).path.lower()
    slug = path.rsplit("/", 1)[-1].replace(".html", "")
    match = re.search(r"-\d{1,2}-(?P<style>[a-z0-9-]+)-skoda-", slug)
    if match:
        return match.group("style").replace("-", " ").upper()
    return ""


def extract_description_text(soup: BeautifulSoup) -> str:
    item_desc = soup.select_one('[itemprop="description"]')
    if item_desc:
        return " ".join(item_desc.stripped_strings)

    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        return str(meta_desc.get("content")).strip()

    return ""


def extract_pcd(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"spacing\s*:\s*(\d+\s*[xX]\s*\d+(?:[\.,]\d+)?)", text, flags=re.IGNORECASE)
    if not match:
        match = PCD_PATTERN.search(text)
        if not match:
            return None
        return f"{int(match.group('bolt'))}x{match.group('diameter').replace(',', '.')}"

    raw = match.group(1)
    parsed = PCD_PATTERN.search(raw)
    if not parsed:
        return None
    return f"{int(parsed.group('bolt'))}x{parsed.group('diameter').replace(',', '.')}"


def extract_color_variant(text: str) -> str | None:
    if not text:
        return None
    lower = text.lower()
    candidates = [
        r"brilliant\s+silver\s+metallic\s+finish",
        r"silver\s+metallic\s+finish",
        r"silver\s+metallic",
        r"black\s+metallic\s+finish",
        r"black\s+metallic",
        r"anthracite",
    ]
    for pattern in candidates:
        match = re.search(pattern, lower)
        if match:
            value = re.sub(r"\s+finish$", "", match.group(0)).strip()
            if "silver metallic" in value:
                return "silver metallic"
            if "black metallic" in value:
                return "black metallic"
            return re.sub(r"\s+", " ", value)
    return None


def extract_notes(text: str) -> str | None:
    if not text:
        return None
    notes: list[str] = []
    for pattern in [r"note\s*::\s*([^\.\n]+)", r"note\s*:\s*([^\.\n]+)", r"without\s+center\s+cover"]:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(1) if match.lastindex else match.group(0)
            value = re.sub(r"^without", "without", value.strip(), flags=re.IGNORECASE)
            value = re.sub(r"\s+", " ", value).strip(" .")
            if value and value.lower() not in [n.lower() for n in notes]:
                notes.append(value)
    return " | ".join(notes) if notes else None


def extract_designed_models(soup: BeautifulSoup) -> str | None:
    options = soup.select("div.suitablefor .selectors select option")
    models: list[str] = []
    for option in options:
        value = (option.get("value") or "").strip()
        text = " ".join(option.stripped_strings)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if value in {"", "0"}:
            continue
        if text.lower() == "all models":
            continue
        if text.lower() not in [m.lower() for m in models]:
            models.append(text)

    return " | ".join(models) if models else None


def extract_oem_code_from_page(soup: BeautifulSoup, full_text: str) -> str | None:
    part_input = soup.find("input", attrs={"name": "partnr"})
    if part_input and part_input.get("value"):
        return str(part_input.get("value")).strip().upper()
    return extract_oem_code(full_text)


def extract_year_range(text: str) -> tuple[int | None, int | None]:
    match = YEAR_RANGE_PATTERN.search(text)
    if not match:
        return None, None
    year_from = int(match.group("from"))
    year_to_raw = match.group("to").lower()
    year_to = None if year_to_raw == "present" else int(year_to_raw)
    return year_from, year_to


def extract_oem_code(text: str) -> str | None:
    for item in OEM_PATTERN.findall(text):
        if any(ch.isdigit() for ch in item):
            return item
    return None


def find_value_after_label(lines: list[str], label_tokens: tuple[str, ...]) -> str | None:
    for line in lines:
        lower = line.lower()
        if any(token in lower for token in label_tokens):
            parts = re.split(r":|\-|\u2013|\u2014", line, maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()
    return None


def parse_product_page(url: str, html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.select_one("h1 .title") or soup.find("h1") or soup.find("title"))
    page_title = title.get_text(" ", strip=True) if title else ""
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
    full_text = "\n".join(lines)

    description_text = extract_description_text(soup)
    official_wheel_name = normalize_wheel_style_name(page_title, url)

    dim_text = description_text or find_value_after_label(lines, ("dimension", "rim", "size", "j x", "jx")) or page_title
    dims = parse_dimensions(dim_text)

    model_text = extract_designed_models(soup)
    generation_text = find_value_after_label(lines, ("generation",))
    notes_text = extract_notes(description_text) or find_value_after_label(lines, ("restriction", "note", "warning", "only"))

    pcd = extract_pcd(description_text)
    bolt_count = int(pcd.split("x", 1)[0]) if pcd else None
    cb = parse_float(find_value_after_label(lines, ("center bore", "centre bore", "cb", "dia")))

    # Metu nespeliojam is laisvo teksto, imam tik aiskiai nurodyta generation range.
    year_from, year_to = extract_year_range(generation_text) if generation_text else (None, None)
    oem_code = extract_oem_code_from_page(soup, full_text)

    color_variant = extract_color_variant(description_text) or find_color_variant(page_title)

    wheel_class_name = official_wheel_name.title() if official_wheel_name else ""

    record: dict[str, Any] = {
        "wheel_class_name": wheel_class_name,
        "official_wheel_name": official_wheel_name,
        "brand": "Skoda",
        "color_variant": color_variant,
        "diameter_in": dims["diameter_in"],
        "width_j": dims["width_j"],
        "et": dims["et"],
        "pcd": pcd,
        "cb": cb,
        "bolt_count": bolt_count,
        "designed_for_models": model_text,
        "generation": None,
        "year_from": year_from,
        "year_to": year_to,
        "oem_part_code": oem_code,
        "notes": notes_text,
        "source_type": "skoda_parts_reseller",
        "source_page": url,
        "extraction_confidence": 0.8,
        "needs_manual_review": False,
    }

    critical_missing = [
        record["official_wheel_name"],
        record["diameter_in"],
        record["width_j"],
        record["et"],
        record["pcd"],
        record["oem_part_code"],
        record["designed_for_models"],
    ]
    if any(value in (None, "") for value in critical_missing):
        record["needs_manual_review"] = True
        record["extraction_confidence"] = 0.55
    else:
        record["needs_manual_review"] = False
        record["extraction_confidence"] = 0.95
    return record


def find_color_variant(name: str) -> str | None:
    lower = name.lower()
    variant_words = ["black metallic", "silver metallic", "diamond cut", "anthracite", "glossy", "matt", "steel", "black", "silver"]
    found = [w for w in variant_words if w in lower]
    if not found:
        return None
    return found[0]


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def infer_variant_type(class_name: str) -> str:
    tokens = class_name.split("_")
    if len(tokens) <= 1:
        return "base"
    return "finish_variant"


def build_class_mapping(wheel_classes: list[str], official_names: list[str]) -> list[dict[str, Any]]:
    normalized_official = {normalize_name(name): name for name in official_names if name}
    rows: list[dict[str, Any]] = []

    for class_name in wheel_classes:
        parts = class_name.split("_")
        base = parts[0]
        base_norm = normalize_name(base)
        exact_norm = normalize_name(class_name)

        official_name: str | None = None
        confidence = 0.45
        review = True

        if exact_norm in normalized_official:
            official_name = normalized_official[exact_norm]
            confidence = 0.95
            review = False
        elif base_norm in normalized_official:
            official_name = normalized_official[base_norm]
            confidence = 0.8
            review = len(parts) > 1
        else:
            for key, value in normalized_official.items():
                if base_norm and base_norm in key:
                    official_name = value
                    confidence = 0.65
                    review = True
                    break

        rows.append(
            {
                "wheel_class_name": class_name,
                "official_wheel_name": official_name or base,
                "variant_type": infer_variant_type(class_name),
                "mapping_confidence": confidence,
                "needs_manual_review": review,
            }
        )
    return rows


def ensure_schema(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]


def save_outputs(wheels_df: pd.DataFrame, mapping_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    wheels_csv = output_dir / "wheels.csv"
    wheels_json = output_dir / "wheels.json"
    mapping_csv = output_dir / "wheel_class_mapping.csv"

    wheels_df.to_csv(wheels_csv, index=False, encoding="utf-8")
    wheels_json.write_text(
        json.dumps(wheels_df.where(pd.notna(wheels_df), None).to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    mapping_df.to_csv(mapping_csv, index=False, encoding="utf-8")

    LOGGER.info("Saved %s", wheels_csv)
    LOGGER.info("Saved %s", wheels_json)
    LOGGER.info("Saved %s", mapping_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape official Skoda wheel pages into normalized wheels dataset.")
    parser.add_argument("--listing-url", action="append", default=[], help="Official listing page URL. Repeatable.")
    parser.add_argument("--product-url", action="append", default=[], help="Direct official product page URL. Repeatable.")
    parser.add_argument("--output-dir", default=".", help="Output directory for wheels files.")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout in seconds.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds.")
    parser.add_argument("--save-raw-html", action="store_true", help="Save raw HTML snapshots for debugging.")
    parser.add_argument("--raw-html-dir", default="raw_html/wheels", help="Raw HTML output directory.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    parser.add_argument(
        "--wheel-classes",
        default=",".join(DEFAULT_WHEEL_CLASSES),
        help="Comma-separated ML class names. Defaults to thesis dataset classes.",
    )
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
    wheel_classes = [x.strip() for x in str(args.wheel_classes).split(",") if x.strip()]

    session = build_session(user_agent=args.user_agent)

    listing_urls: list[str] = list(dict.fromkeys(args.listing_url))
    direct_product_urls: list[str] = list(dict.fromkeys(args.product_url))

    discovered_product_urls: set[str] = set(direct_product_urls)
    for listing_url in listing_urls:
        html = fetch_page(
            session=session,
            url=listing_url,
            timeout=args.timeout,
            delay_seconds=args.delay,
            save_raw_html=args.save_raw_html,
            raw_html_dir=raw_html_dir,
        )
        if not html:
            continue
        links = extract_links_from_listing(html, listing_url)
        LOGGER.info("Discovered %d candidate wheel links from %s", len(links), listing_url)
        discovered_product_urls.update(links)

    records: list[dict[str, Any]] = []
    for product_url in sorted(discovered_product_urls):
        html = fetch_page(
            session=session,
            url=product_url,
            timeout=args.timeout,
            delay_seconds=args.delay,
            save_raw_html=args.save_raw_html,
            raw_html_dir=raw_html_dir,
        )
        if not html:
            continue
        records.append(parse_product_page(product_url, html))

    wheels_df = pd.DataFrame(records)
    wheels_df = ensure_schema(wheels_df, WHEEL_SCHEMA)

    mapping_rows = build_class_mapping(wheel_classes=wheel_classes, official_names=wheels_df["official_wheel_name"].dropna().astype(str).tolist())
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df = ensure_schema(mapping_df, MAPPING_SCHEMA)

    name_to_class = {
        normalize_name(str(row["official_wheel_name"])): str(row["wheel_class_name"]) for _, row in mapping_df.iterrows() if row["official_wheel_name"]
    }

    wheel_class_values: list[str] = []
    for _, row in wheels_df.iterrows():
        normalized = normalize_name(str(row["official_wheel_name"]))
        wheel_class_values.append(name_to_class.get(normalized, ""))
    wheels_df["wheel_class_name"] = wheel_class_values

    save_outputs(wheels_df=wheels_df, mapping_df=mapping_df, output_dir=output_dir)


if __name__ == "__main__":
    main()