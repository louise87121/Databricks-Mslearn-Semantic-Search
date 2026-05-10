# Databricks Microsoft Learn Semantic Search

## Project Overview

This is a local semantic search app for Azure Databricks documentation from Microsoft Learn. It lets users enter a question, retrieve the most relevant documentation chunks, and inspect the original source links.

這是一個本機執行的 Microsoft Learn Azure Databricks 文件語意搜尋工具。使用者可以輸入問題，系統會找出最相關的文件片段，並顯示來源連結。

The app focuses on:

- Documentation search
- Semantic retrieval with embeddings
- Local ChromaDB vector index
- Metadata filtering
- Source citation

## Architecture Flow

```text
Microsoft Learn documentation
→ Clean text
→ Searchable chunks
→ Embeddings
→ Vector index
→ User question
→ Relevant documentation chunks
→ Source links
```

## Main Files

```text
app.py                  Streamlit search app
data/sources.json       Microsoft Learn source list
chroma_db/              Local ChromaDB vector index
requirements.txt        Python dependencies
```

The helper scripts below are only needed when rebuilding the local index:

```text
ingest_bronze.py
build_silver_chunks.py
build_vector_index.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```powershell
.venv\Scripts\activate
```

## Run the App

If `chroma_db/` already exists:

```bash
streamlit run app.py
```

If the vector index is missing, rebuild it first:

```bash
python ingest_bronze.py
python build_silver_chunks.py
python build_vector_index.py
streamlit run app.py
```

## Example Questions

- Mosaic AI Vector Search 是什麼？
- Databricks 的 RAG 流程包含哪些步驟？
- Unity Catalog 在 Databricks 中負責什麼？
- Databricks 如何查詢 vector search index？
- 為什麼 metadata filtering 對企業知識庫很重要？
- 為什麼 source citation 對企業 AI 很重要？

## Streamlit Community Cloud

Use Python 3.11 in Streamlit Community Cloud advanced settings.

The local vector index is stored in `chroma_db/`. If `chroma_db/` is ignored by git, the deployed app will not include the prebuilt index. For deployment, either include a prepared index or add a startup/build strategy that creates the index before search.

## Learning Goals

- Understand how documentation can be split into searchable chunks.
- Understand why embeddings support semantic search.
- Understand how a vector index retrieves related documentation.
- Understand why metadata filtering and source citation matter.
