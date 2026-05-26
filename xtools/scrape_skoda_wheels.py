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
    "oem_part_code",
    "review_reason",
    "needs_manual_review",
]

DIMENSION_PATTERN = re.compile(
    r"(?P<width>\d+(?:[\.,]\d+)?)\s*J\s*[xX]\s*(?P<diameter>\d+(?:[\.,]\d+)?)\s*(?:ET\s*(?P<et>[+-]?\d+(?:[\.,]\d+)?))?",
    flags=re.IGNORECASE,
)
PCD_PATTERN = re.compile(r"\b(?P<bolt>\d+)\s*[xX]\s*(?P<diameter>\d+(?:[\.,]\d+)?)\b")
OEM_PATTERN = re.compile(r"\b([0-9A-Z]{2,}-[0-9A-Z-]{2,}|[0-9A-Z]{6,})\b")



def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
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
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        html = response.text
        if save_raw_html:
            snapshot_html(html, url, raw_html_dir)
        time.sleep(delay_seconds)
        return html
    except requests.RequestException:
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
        tokens = [tok for tok in re.split(r"\s+", text) if tok]
        if tokens:
            if len(tokens) > 1 and tokens[-2].isalpha() and tokens[-1].isalpha():
                return " ".join(tokens[-2:])
            return tokens[-1]

    path = urlparse(source_url).path.lower()
    slug = path.rsplit("/", 1)[-1].replace(".html", "")
    match = re.search(r"-\d{1,2}-(?P<style>[a-z0-9-]+)-skoda-", slug)
    if match:
        return match.group("style").replace("-", " ").upper()
    return ""



def normalize_class_name(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip())



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



COLOR_PATTERNS: list[tuple[str, str]] = [
    (r"brushed\s+metallic\s+anthracite", "Brushed Metallic Anthracite"),
    (r"brushed\s+metallic\s+silver", "Brushed Metallic Silver"),
    (r"black\s+metallic\s+red", "Black Metallic Red"),
    (r"anthracite\s+metallic", "Anthracite Metallic"),
    (r"black\s+metallic", "Black Metallic"),
    (r"silver\s+metallic", "Silver Metallic"),
    (r"brilliant\s+silver", "Brilliant Silver"),
    (r"silver\s+brilliant", "Silver Brilliant"),
    (r"metallic\s+silver", "Metallic Silver"),
    (r"glossy\s+black", "Glossy Black"),
    (r"black\s+matt", "Black Matt"),
    (r"anthracite", "Anthracite"),
    (r"black", "Black"),
]


def extract_color_variant(text: str) -> str | None:
    if not text:
        return None

    normalized = re.sub(r"\s+", " ", text.lower()).strip()

    matches: list[tuple[int, str]] = []

    for pattern, canonical_name in COLOR_PATTERNS:
        for match in re.finditer(pattern, normalized):
            matches.append((match.start(), canonical_name))

    if not matches:
        return None

    # Paliekam source tvarka, saliname dublikatus
    ordered_unique: list[str] = []
    seen: set[str] = set()

    for _, color in sorted(matches, key=lambda item: item[0]):
        if color not in seen:
            ordered_unique.append(color)
            seen.add(color)

    return ", ".join(ordered_unique)



def extract_designed_models(soup: BeautifulSoup) -> str | None:
    options = soup.select("div.suitablefor .selectors select option")
    models: list[str] = []
    for option in options:
        value = (option.get("value") or "").strip()
        text = " ".join(option.stripped_strings)
        text = re.sub(r"\s+", " ", text).strip()
        if not text or value in {"", "0"} or text.lower() == "all models":
            continue
        if text.lower() not in [m.lower() for m in models]:
            models.append(text)

    return " | ".join(models) if models else None



def extract_oem_code_from_page(soup: BeautifulSoup, full_text: str) -> str | None:
    part_input = soup.find("input", attrs={"name": "partnr"})
    if part_input and part_input.get("value"):
        return str(part_input.get("value")).strip().upper()
    return extract_oem_code(full_text)



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

    pcd = extract_pcd(description_text)
    bolt_count = int(pcd.split("x", 1)[0]) if pcd else None
    cb = parse_float(find_value_after_label(lines, ("center bore", "centre bore", "cb", "dia")))

    oem_code = extract_oem_code_from_page(soup, full_text)
    color_variant = extract_color_variant(description_text)
    designed_for_models = extract_designed_models(soup)

    wheel_class_name = normalize_class_name(official_wheel_name) if official_wheel_name else ""

    reasons: list[str] = []
    if not official_wheel_name:
        reasons.append("missing wheel name")
    if dims["diameter_in"] is None:
        reasons.append("missing diameter")
    if dims["width_j"] is None:
        reasons.append("missing width")
    if dims["et"] is None:
        reasons.append("missing ET")
    if not pcd:
        reasons.append("missing PCD")
    if cb is None:
        reasons.append("missing CB")
    if bolt_count is None:
        reasons.append("missing bolt_count")
    if not oem_code:
        reasons.append("missing OEM code")
    if not designed_for_models:
        reasons.append("missing designed_for_models")

    return {
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
        "designed_for_models": designed_for_models,
        "oem_part_code": oem_code,
        "review_reason": " | ".join(reasons) if reasons else None,
        "needs_manual_review": 1 if reasons else 0,
    }



def ensure_schema(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]



def save_outputs(wheels_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    wheels_csv = output_dir / "wheels.csv"
    wheels_json = output_dir / "wheels.json"

    wheels_df.to_csv(wheels_csv, index=False, encoding="utf-8")
    wheels_json.write_text(
        json.dumps(wheels_df.where(pd.notna(wheels_df), None).to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Skoda wheel pages")
    parser.add_argument("--listing-url", action="append", default=[], help="Listing URL, can be repeated")
    parser.add_argument("--product-url", action="append", default=[], help="Product URL, can be repeated")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout in seconds")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument("--save-raw-html", action="store_true", help="Save raw HTML")
    parser.add_argument("--raw-html-dir", default="raw_html/wheels", help="Raw HTML directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    parser.add_argument(
        "--user-agent",
        default="DudaRimsResearchBot/1.0 (+contact: local-research)",
        help="HTTP user-agent header",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    output_dir = Path(args.output_dir)
    raw_html_dir = Path(args.raw_html_dir)

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
        discovered_product_urls.update(links)

    records: list[dict[str, Any]] = []
    fetched_count = 0
    failed_fetches = 0
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
            failed_fetches += 1
            continue
        fetched_count += 1
        try:
            records.append(parse_product_page(product_url, html))
        except Exception:
            failed_fetches += 1

    wheels_df = pd.DataFrame(records)
    wheels_df = ensure_schema(wheels_df, WHEEL_SCHEMA)

    # Normalizuoja rankini perziuros flaga i 0/1 reiksmes
    wheels_df["needs_manual_review"] = wheels_df["needs_manual_review"].fillna(0).astype(int).clip(lower=0, upper=1)

    save_outputs(wheels_df=wheels_df, output_dir=output_dir)

    discovered_count = len(discovered_product_urls)
    exported_rows = len(wheels_df)
    manual_review_count = int(wheels_df[wheels_df.get("needs_manual_review") == 1].shape[0]) if exported_rows else 0
    print(f"[scrape] discovered: {discovered_count}")
    print(f"[scrape] fetched: {fetched_count}")
    print(f"[scrape] failed: {failed_fetches}")
    print(f"[scrape] rows: {exported_rows}")
    print(f"[scrape] manual: {manual_review_count}")


if __name__ == "__main__":
    main()
