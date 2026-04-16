from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import Browser, Page, Playwright, sync_playwright

LOGGER = logging.getLogger("scrape_skoda_models_playwright")

ENRICH_SCHEMA = ["source_page", "pcd", "cb", "thread_size", "bolt_count"]

STRICT_PCD_PATTERN = re.compile(r"\b\d+\s*[xX]\s*\d+(?:[\.,]\d+)?\b")
STRICT_CB_PATTERN = re.compile(r"\b\d+(?:[\.,]\d+)?\s*mm\b", re.IGNORECASE)
STRICT_THREAD_PATTERN = re.compile(r"\bM\d+\s*[xX]\s*\d+(?:[\.,]\d+)?\b", re.IGNORECASE)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def normalize_pcd(value: str | None) -> str | None:
    if not value:
        return None
    match = STRICT_PCD_PATTERN.search(value)
    if not match:
        return None
    raw = match.group(0).replace(" ", "")
    left, right = raw.lower().split("x", 1)
    right = right.replace(",", ".")
    return f"{int(float(left))}x{right}"


def normalize_cb(value: str | None) -> float | None:
    if not value:
        return None
    match = STRICT_CB_PATTERN.search(value)
    if not match:
        return None
    num_match = re.search(r"\d+(?:[\.,]\d+)?", match.group(0))
    if not num_match:
        return None
    return float(num_match.group(0).replace(",", "."))


def normalize_thread_size(value: str | None) -> str | None:
    if not value:
        return None
    match = STRICT_THREAD_PATTERN.search(value)
    if not match:
        return None
    raw = match.group(0).upper().replace(",", ".")
    return re.sub(r"\s*[xX]\s*", " x ", raw).strip()


def derive_bolt_count_from_pcd(pcd: str | None) -> int | None:
    if not pcd:
        return None
    m = re.match(r"(\d+)\s*[xX]", pcd)
    if not m:
        return None
    return int(m.group(1))


def _text_snippet(value: str, max_len: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 3] + "..."


def _extract_value_near_label_node(label_node: Any, value_pattern: re.Pattern[str], field_name: str) -> str | None:
    label_text = label_node.get_text(" ", strip=True) if hasattr(label_node, "get_text") else str(label_node)
    LOGGER.debug("Found %s label node <%s>: %s", field_name, getattr(label_node, "name", "?"), _text_snippet(label_text))

    search_chunks: list[str] = []

    for sibling in label_node.next_siblings:
        if hasattr(sibling, "get_text"):
            text = sibling.get_text(" ", strip=True)
        else:
            text = str(sibling).strip()
        if text:
            search_chunks.append(text)

    parent = getattr(label_node, "parent", None)
    if parent is not None and hasattr(parent, "get_text"):
        search_chunks.append(parent.get_text(" ", strip=True))

    ancestor = parent
    for _ in range(3):
        ancestor = getattr(ancestor, "parent", None) if ancestor is not None else None
        if ancestor is None or not hasattr(ancestor, "get_text"):
            continue
        block_text = ancestor.get_text(" ", strip=True)
        if block_text and len(block_text) <= 500:
            search_chunks.append(block_text)

    for chunk in search_chunks:
        m = value_pattern.search(chunk)
        if m:
            extracted = m.group(0).strip()
            LOGGER.debug("Extracted local %s value: %s", field_name, extracted)
            return extracted

    LOGGER.debug("Extracted local %s value: <none>", field_name)
    return None


def _find_label_nodes(soup: BeautifulSoup, label_pattern: re.Pattern[str]) -> list[Any]:
    nodes: list[Any] = []
    seen: set[int] = set()

    for tag in soup.find_all(["span", "div", "td", "th", "strong", "label"]):
        text = tag.get_text(" ", strip=True)
        if text and label_pattern.search(text):
            key = id(tag)
            if key not in seen:
                seen.add(key)
                nodes.append(tag)

    for text_node in soup.find_all(string=label_pattern):
        node = getattr(text_node, "parent", None)
        if node is None:
            continue
        key = id(node)
        if key not in seen:
            seen.add(key)
            nodes.append(node)

    return nodes


def _nearest_container_block(node: Any) -> Any:
    container_tags = {"tr", "td", "li", "div", "section", "article", "dl"}
    current = node
    for _ in range(6):
        if current is None:
            break
        name = getattr(current, "name", "")
        if name in container_tags:
            return current
        current = getattr(current, "parent", None)
    return node


def _save_debug_block(debug_dir: Path, filename: str, block_node: Any, label_text: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_file = debug_dir / filename
    outer_html = ""
    if hasattr(block_node, "prettify"):
        outer_html = block_node.prettify()
    else:
        outer_html = str(block_node)
    content = "\n".join(
        [
            "<!-- label -->",
            f"<!-- {label_text} -->",
            "<!-- block -->",
            outer_html,
        ]
    )
    out_file.write_text(content, encoding="utf-8")
    LOGGER.debug("Saved debug block snippet: %s", out_file)


def extract_field_with_debug(
    soup: BeautifulSoup,
    label_pattern: re.Pattern[str],
    value_pattern: re.Pattern[str],
    field_name: str,
    debug_block_file: str,
    debug_dir: Path,
) -> str | None:
    label_nodes = _find_label_nodes(soup=soup, label_pattern=label_pattern)
    LOGGER.debug("Found %d matching label nodes for %s", len(label_nodes), field_name)

    if not label_nodes:
        _save_debug_block(debug_dir, debug_block_file, block_node="<!-- no label node found -->", label_text=f"{field_name}:none")
        return None

    chosen_block_node: Any = None
    chosen_label_text = ""
    chosen_value: str | None = None

    for idx, label_node in enumerate(label_nodes, start=1):
        label_text = label_node.get_text(" ", strip=True) if hasattr(label_node, "get_text") else str(label_node)
        block_node = _nearest_container_block(label_node)
        block_text = block_node.get_text(" ", strip=True) if hasattr(block_node, "get_text") else str(block_node)
        candidate_match = value_pattern.search(block_text)
        candidate_value = candidate_match.group(0).strip() if candidate_match else None

        LOGGER.debug("%s label #%d text: %s", field_name, idx, _text_snippet(label_text))
        LOGGER.debug("%s label #%d block outer HTML:\n%s", field_name, idx, str(block_node)[:2000])
        LOGGER.debug("%s label #%d local candidate value: %s", field_name, idx, candidate_value if candidate_value else "<none>")

        if chosen_block_node is None:
            chosen_block_node = block_node
            chosen_label_text = label_text

        if candidate_value and chosen_value is None:
            chosen_value = candidate_value
            chosen_block_node = block_node
            chosen_label_text = label_text

    if chosen_block_node is not None:
        _save_debug_block(debug_dir, debug_block_file, block_node=chosen_block_node, label_text=chosen_label_text)

    return chosen_value


def extract_labeled_value_local(
    soup: BeautifulSoup,
    label_pattern: re.Pattern[str],
    value_pattern: re.Pattern[str],
    field_name: str,
) -> str | None:
    for tag in soup.find_all(["span", "div", "td", "th", "strong", "label"]):
        text = tag.get_text(" ", strip=True)
        if text and label_pattern.search(text):
            value = _extract_value_near_label_node(tag, value_pattern=value_pattern, field_name=field_name)
            if value:
                return value

    for text_node in soup.find_all(string=label_pattern):
        node = getattr(text_node, "parent", None)
        if node is None:
            continue
        value = _extract_value_near_label_node(node, value_pattern=value_pattern, field_name=field_name)
        if value:
            return value

    LOGGER.debug("Label node for %s not found", field_name)
    return None


def parse_rendered_html(source_page: str, html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    debug_dir = Path("debug_playwright_blocks")

    pcd_raw = extract_field_with_debug(
        soup=soup,
        label_pattern=re.compile(r"Bolt\s+Pattern\s*\(PCD\)|Bolt\s+Pattern|PCD", re.IGNORECASE),
        value_pattern=STRICT_PCD_PATTERN,
        field_name="pcd",
        debug_block_file="pcd_block.html",
        debug_dir=debug_dir,
    )
    cb_raw = extract_field_with_debug(
        soup=soup,
        label_pattern=re.compile(r"Center\s+Bore\s*/\s*Hub\s+Bore|Center\s+Bore|Hub\s+Bore|CB", re.IGNORECASE),
        value_pattern=STRICT_CB_PATTERN,
        field_name="cb",
        debug_block_file="cb_block.html",
        debug_dir=debug_dir,
    )
    thread_raw = extract_field_with_debug(
        soup=soup,
        label_pattern=re.compile(r"Thread\s+Size", re.IGNORECASE),
        value_pattern=STRICT_THREAD_PATTERN,
        field_name="thread_size",
        debug_block_file="thread_block.html",
        debug_dir=debug_dir,
    )

    pcd = normalize_pcd(pcd_raw)
    cb = normalize_cb(cb_raw)
    thread_size = normalize_thread_size(thread_raw)
    bolt_count = derive_bolt_count_from_pcd(pcd)

    record = {
        "source_page": source_page,
        "pcd": pcd,
        "cb": cb,
        "thread_size": thread_size,
        "bolt_count": bolt_count,
    }
    LOGGER.info("Extracted values for %s -> %s", source_page, record)
    return record


def make_cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def get_cache_paths(cache_dir: Path, source_url: str) -> tuple[Path, Path]:
    key = make_cache_key(source_url)
    parsed_cache = cache_dir / "parsed" / f"{key}.json"
    rendered_cache = cache_dir / "rendered_html" / f"{key}.html"
    return parsed_cache, rendered_cache


def load_parsed_cache(parsed_cache: Path) -> dict[str, Any] | None:
    if not parsed_cache.exists():
        return None
    try:
        return json.loads(parsed_cache.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to load parsed cache %s: %s", parsed_cache, exc)
        return None


def save_parsed_cache(parsed_cache: Path, record: dict[str, Any]) -> None:
    parsed_cache.parent.mkdir(parents=True, exist_ok=True)
    parsed_cache.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def load_rendered_cache(rendered_cache: Path) -> str | None:
    if not rendered_cache.exists():
        return None
    try:
        return rendered_cache.read_text(encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("Failed to load rendered HTML cache %s: %s", rendered_cache, exc)
        return None


def save_rendered_cache(rendered_cache: Path, html: str) -> None:
    rendered_cache.parent.mkdir(parents=True, exist_ok=True)
    rendered_cache.write_text(html, encoding="utf-8")


def fetch_rendered_html(playwright: Playwright, url: str) -> str:
    browser: Browser = playwright.chromium.launch(headless=True)
    LOGGER.info("Playwright launched for %s", url)
    try:
        page: Page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_load_state("networkidle", timeout=30000)
        page.wait_for_timeout(1200)
        return page.content()
    finally:
        browser.close()


def process_url(
    source_url: str,
    cache_dir: Path,
    save_rendered_html: bool,
    playwright: Playwright,
) -> dict[str, Any]:
    parsed_cache, rendered_cache = get_cache_paths(cache_dir, source_url)

    parsed = load_parsed_cache(parsed_cache)
    if parsed:
        LOGGER.info("Parsed cache hit for %s", source_url)
        return parsed
    LOGGER.info("Parsed cache miss for %s", source_url)

    html = load_rendered_cache(rendered_cache)
    if html is not None:
        LOGGER.info("Rendered HTML cache hit for %s", source_url)
    else:
        LOGGER.info("Rendered HTML cache miss for %s", source_url)
        html = fetch_rendered_html(playwright, source_url)
        if save_rendered_html:
            save_rendered_cache(rendered_cache, html)

    record = parse_rendered_html(source_url, html)
    save_parsed_cache(parsed_cache, record)
    return record


def save_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    for col in ENRICH_SCHEMA:
        if col not in df.columns:
            df[col] = None
    df = df[ENRICH_SCHEMA]

    out_csv = output_dir / "skoda_models_enriched.csv"
    out_json = output_dir / "skoda_models_enriched.json"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(
        json.dumps(df.where(pd.notna(df), None).to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    LOGGER.info("Saved %s", out_csv)
    LOGGER.info("Saved %s", out_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich model fitment with JS-rendered fields using Playwright.")
    parser.add_argument("--source-url", action="append", required=True, help="Source URL. Repeatable.")
    parser.add_argument("--output-dir", default=".", help="Output directory.")
    parser.add_argument("--cache-dir", default="fitment_data/cache_playwright", help="Cache directory.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    parser.add_argument("--save-rendered-html", action="store_true", help="Persist rendered HTML in cache.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    source_urls = list(dict.fromkeys(args.source_url))

    records: list[dict[str, Any]] = []
    with sync_playwright() as playwright:
        for source_url in source_urls:
            try:
                record = process_url(
                    source_url=source_url,
                    cache_dir=cache_dir,
                    save_rendered_html=args.save_rendered_html,
                    playwright=playwright,
                )
                records.append(record)
            except Exception as exc:
                LOGGER.exception("Failed to enrich %s: %s", source_url, exc)
                records.append(
                    {
                        "source_page": source_url,
                        "pcd": None,
                        "cb": None,
                        "thread_size": None,
                        "bolt_count": None,
                    }
                )

    save_outputs(records=records, output_dir=output_dir)


if __name__ == "__main__":
    main()