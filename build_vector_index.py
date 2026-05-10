from __future__ import annotations

from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).parent
SILVER_PATH = PROJECT_ROOT / "lakehouse" / "silver" / "document_chunks.parquet"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "mslearn_databricks_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_index(df: pd.DataFrame) -> None:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    documents = df["chunk_text"].astype(str).tolist()
    ids = df["chunk_id"].astype(str).tolist()
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    metadatas = [
        {
            "source_id": str(row["source_id"]),
            "title": str(row["title"]),
            "url": str(row["url"]),
            "category": str(row["category"]),
            "status": str(row["status"]),
            "access_level": str(row["access_level"]),
            "chunk_index": int(row["chunk_index"]),
        }
        for _, row in df.iterrows()
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Indexed {len(ids)} chunks into ChromaDB collection `{COLLECTION_NAME}`.")


def main() -> None:
    if not SILVER_PATH.exists():
        print("Silver file not found. Run `python build_silver_chunks.py` first.")
        return

    df = load_chunks(str(SILVER_PATH))
    if df.empty:
        print("No chunks found in Silver file. Re-run Bronze and Silver steps first.")
        return

    build_index(df)


if __name__ == "__main__":
    main()
