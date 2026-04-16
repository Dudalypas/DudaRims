from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

URL = "https://www.wheel-size.com/size/skoda/rapid/mk2-2019-2024/"
OUT_DIR = Path("network_spy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEYWORDS = [
    "5x100",
    "57.1",
    "m14",
    "bolt pattern",
    "pcd",
    "center bore",
    "hub bore",
    "thread size",
    "rapid",
    "mk2",
]

URL_HINTS = [
    "api",
    "fitment",
    "spec",
    "vehicle",
    "wheel",
    "size",
    "trim",
    "data",
    "json",
]


def safe_name(url: str, idx: int) -> str:
    return f"{idx:04d}_{quote(url, safe='')[:180]}"


def looks_interesting(url: str, body: str, content_type: str) -> bool:
    blob = f"{url}\n{content_type}\n{body}".lower()
    if any(k in blob for k in KEYWORDS):
        return True
    if any(h in url.lower() for h in URL_HINTS):
        return True
    return False


def main() -> None:
    hits: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        def on_response(response) -> None:
            try:
                url = response.url
                status = response.status
                headers = response.headers
                content_type = headers.get("content-type", "")

                if status >= 400:
                    return

                if not any(t in content_type.lower() for t in ["json", "javascript", "html", "text", "xml"]):
                    return

                body = response.text()
                if not looks_interesting(url, body, content_type):
                    return

                idx = len(hits) + 1
                stem = safe_name(url, idx)

                meta = {
                    "url": url,
                    "status": status,
                    "content_type": content_type,
                }

                (OUT_DIR / f"{stem}.meta.json").write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (OUT_DIR / f"{stem}.txt").write_text(body, encoding="utf-8", errors="ignore")

                hits.append(meta)
                print(f"[HIT] {status} | {content_type} | {url}")

            except Exception as exc:
                print(f"[WARN] response parse failed: {exc}")

        page.on("response", on_response)

        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(8000)

        # Extra browser-side probes
        probes = {
            "window_keys": """
                () => Object.keys(window)
                    .filter(k => /wheel|fit|spec|data|rapid|skoda/i.test(k))
                    .slice(0, 200)
            """,
            "local_storage": "() => ({...localStorage})",
            "session_storage": "() => ({...sessionStorage})",
            "full_html": "() => document.documentElement.outerHTML",
        }

        for name, js in probes.items():
            try:
                result = page.evaluate(js)
                path = OUT_DIR / f"probe_{name}.json"
                if isinstance(result, str):
                    path = OUT_DIR / f"probe_{name}.txt"
                    path.write_text(result, encoding='utf-8', errors='ignore')
                else:
                    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[PROBE] saved {path}")
            except Exception as exc:
                print(f"[WARN] probe {name} failed: {exc}")

        browser.close()

    print(f"\nSaved {len(hits)} matching responses into: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()