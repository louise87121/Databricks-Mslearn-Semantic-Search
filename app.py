from __future__ import annotations

from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "mslearn_databricks_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

load_dotenv(PROJECT_ROOT / ".env")


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(COLLECTION_NAME)


def _query_collection(collection, query_embedding: list[float], top_k: int) -> dict:
    where_filter = {"$and": [{"status": "active"}, {"access_level": "public"}]}

    try:
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        # Some ChromaDB versions are strict about compound where filters.
        # Retrieve extra chunks, then enforce metadata filtering in Python.
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k * 4, top_k),
            include=["documents", "metadatas", "distances"],
        )


def retrieve_context(query: str, top_k: int) -> list[dict]:
    model = load_embedding_model()
    collection = load_chroma_collection()
    query_embedding = model.encode(query).tolist()

    results = _query_collection(collection, query_embedding, top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    contexts: list[dict] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        if metadata.get("status") != "active" or metadata.get("access_level") != "public":
            continue
        contexts.append(
            {
                "content": document,
                "source_id": metadata.get("source_id", ""),
                "title": metadata.get("title", ""),
                "url": metadata.get("url", ""),
                "category": metadata.get("category", ""),
                "distance": distance,
            }
        )
        if len(contexts) >= top_k:
            break

    return contexts


def render_sidebar() -> None:
    st.sidebar.header("Databricks Concept Mapping")
    st.sidebar.markdown(
        """
- Bronze Parquet = Bronze Delta table concept
- Silver chunks = Silver Delta table concept
- ChromaDB = Mosaic AI Vector Search concept
- Metadata filtering = Unity Catalog governance concept
- Source URL = Lineage / citation concept
"""
    )


def main() -> None:
    st.set_page_config(
        page_title="Databricks Microsoft Learn 語意搜尋系統",
        page_icon="🔎",
        layout="wide",
    )
    render_sidebar()

    st.title("Databricks Microsoft Learn 語意搜尋系統")
    st.write(
        "這是一個不用 API 的 Databricks 文件語意搜尋練習：系統會讀取 Microsoft Learn 上的 Azure Databricks 文件，"
        "透過 Bronze / Silver / Vector Index 的流程建立知識庫。使用者輸入問題後，系統會用向量搜尋找出最相關的文件段落，"
        "並顯示來源引用。"
    )

    query = st.text_input("請輸入搜尋問題", placeholder="例如：Mosaic AI Vector Search 是什麼？")
    top_k = st.slider("檢索文件片段數量 top_k", min_value=1, max_value=10, value=5)
    submitted = st.button("搜尋", type="primary")

    if submitted:
        if not query.strip():
            st.warning("請先輸入搜尋問題。")
            return

        try:
            with st.spinner("正在進行語意搜尋..."):
                contexts = retrieve_context(query, top_k)
        except Exception:
            st.warning(
                "找不到向量資料庫，請先依序執行 ingest_bronze.py、build_silver_chunks.py、build_vector_index.py。"
            )
            return

        st.subheader("最相關文件段落")
        st.caption("Distance 越低通常代表語意越接近。")

        if not contexts:
            st.info("沒有找到符合治理條件的文件段落。")
            return

        for index, context in enumerate(contexts, start=1):
            with st.expander(f"{index}. {context['title']}"):
                st.write(f"**Title:** {context['title']}")
                st.write(f"**Category:** {context['category']}")
                st.write(f"**URL:** {context['url']}")
                st.write(f"**Distance score:** {context['distance']:.4f}")
                st.write(f"**Source ID:** {context['source_id']}")
                st.text(context["content"])


if __name__ == "__main__":
    main()
