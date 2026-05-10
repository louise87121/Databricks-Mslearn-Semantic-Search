from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent
BRONZE_PATH = PROJECT_ROOT / "lakehouse" / "bronze" / "raw_documents.parquet"
SILVER_PATH = PROJECT_ROOT / "lakehouse" / "silver" / "document_chunks.parquet"


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def make_chunk_id(source_id: str, chunk_index: int) -> str:
    return f"{source_id}__chunk_{chunk_index:04d}"


def main() -> None:
    if not BRONZE_PATH.exists():
        print("Bronze file not found. Run `python ingest_bronze.py` first.")
        return

    df = pd.read_parquet(BRONZE_PATH)
    eligible = df[
        (df["status"] == "active")
        & (df["fetch_status"] == "success")
        & (df["raw_text"].fillna("").str.strip() != "")
    ].copy()

    rows: list[dict[str, object]] = []
    created_timestamp = datetime.now(timezone.utc).isoformat()

    for _, doc in eligible.iterrows():
        chunks = chunk_text(str(doc["raw_text"]))
        for chunk_index, text in enumerate(chunks):
            rows.append(
                {
                    "chunk_id": make_chunk_id(str(doc["source_id"]), chunk_index),
                    "source_id": doc["source_id"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "category": doc["category"],
                    "status": doc["status"],
                    "access_level": doc["access_level"],
                    "chunk_index": chunk_index,
                    "chunk_text": text,
                    "created_timestamp": created_timestamp,
                }
            )

    SILVER_PATH.parent.mkdir(parents=True, exist_ok=True)
    silver_df = pd.DataFrame(rows)
    silver_df.to_parquet(SILVER_PATH, index=False)

    print(f"Saved Silver chunks to: {SILVER_PATH}")
    print(f"Created {len(silver_df)} chunks.")


if __name__ == "__main__":
    main()
