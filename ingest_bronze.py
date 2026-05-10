from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).parent
SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"
BRONZE_PATH = PROJECT_ROOT / "lakehouse" / "bronze" / "raw_documents.parquet"


def load_sources(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def fetch_page(url: str) -> str:
    headers = {
        "User-Agent": "databricks-mslearn-semantic-search-demo/1.0 (+local learning project)"
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove site layout and executable elements so chunks focus on document content.
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    main_content = soup.find("main") or soup.body or soup
    text = main_content.get_text(separator="\n")
    return clean_text(text)


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)


def main() -> None:
    sources = load_sources(str(SOURCES_PATH))
    rows: list[dict[str, Any]] = []
    success_count = 0

    BRONZE_PATH.parent.mkdir(parents=True, exist_ok=True)

    for source in sources:
        print(f"Fetching: {source['title']} - {source['url']}")
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            raw_html = fetch_page(source["url"])
            raw_text = extract_main_text(raw_html)
            fetch_status = "success"
            error_message = ""
            success_count += 1
            print(f"  Success: extracted {len(raw_text):,} characters")
        except Exception as exc:
            raw_html = ""
            raw_text = ""
            fetch_status = "failed"
            error_message = str(exc)
            print(f"  Failed: {error_message}")

        rows.append(
            {
                "source_id": source["source_id"],
                "title": source["title"],
                "url": source["url"],
                "category": source["category"],
                "status": source["status"],
                "access_level": source["access_level"],
                "raw_html": raw_html,
                "raw_text": raw_text,
                "fetch_status": fetch_status,
                "error_message": error_message,
                "ingestion_timestamp": timestamp,
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(BRONZE_PATH, index=False)

    print(f"\nSaved Bronze documents to: {BRONZE_PATH}")
    print(f"Successfully fetched {success_count} of {len(sources)} documents.")


if __name__ == "__main__":
    main()
