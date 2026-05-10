# databricks-mslearn-semantic-search-demo

## A. Project Overview

這是一個可在本機執行、適合初學者的 semantic search prototype，使用 Microsoft Learn Azure Databricks 文件作為知識來源。它示範如何把線上文件轉成可搜尋的知識庫，並用 embeddings 與本機向量資料庫找出語意最接近的文件段落。

這個專案特別強調：

- 不需要 OpenAI API。
- 不需要付費 API。
- 不需要 Azure credentials。
- 不需要 Databricks workspace。

## B. Why this project is Databricks-focused

這個專案以 Azure Databricks 的 Microsoft Learn 文件作為來源資料，並使用 Lakehouse-inspired 的資料流程來組織非結構化文件。雖然它在本機執行，但每個元件都對應到 Databricks 常見的資料工程、向量搜尋與治理概念。

| Local demo | Databricks concept |
|---|---|
| `lakehouse/bronze/raw_documents.parquet` | Bronze Delta table concept |
| `lakehouse/silver/document_chunks.parquet` | Silver Delta table concept |
| ChromaDB collection | Mosaic AI Vector Search concept |
| Metadata columns | Unity Catalog metadata and governance concept |
| `status` / `access_level` filtering | Governance and permission filtering concept |
| Source citation | Lineage and traceability concept |
| Streamlit app | Simple semantic search interface |

## C. Architecture Flow

Microsoft Learn Azure Databricks URLs
→ Bronze raw ingestion
→ Silver cleaned chunks
→ Embedding generation
→ Vector index
→ User search query
→ Semantic retrieval
→ Relevant document chunks
→ Source citation

## D. Difference Between This Project and Full RAG

這個專案實作 RAG 的 retrieval 部分，不實作 generation 部分。

Full RAG:

```text
User question → retrieval → LLM generation → answer
```

This project:

```text
User question → retrieval → relevant source chunks
```

這是刻意設計的，因為本專案避免使用付費 API。它仍然能教會 RAG 最重要的基礎：document ingestion、chunking、embeddings、vector search、metadata filtering 與 citation。

## E. Folder Structure

```text
databricks-mslearn-semantic-search-demo/
├── app.py
├── ingest_bronze.py
├── build_silver_chunks.py
├── build_vector_index.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── sources.json
├── lakehouse/
│   ├── bronze/
│   │   └── raw_documents.parquet
│   └── silver/
│       └── document_chunks.parquet
└── chroma_db/
```

注意：`raw_documents.parquet`、`document_chunks.parquet` 與 `chroma_db/` 內容會在執行 pipeline 後產生。

## F. Installation Steps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows 啟用虛擬環境：

```powershell
.venv\Scripts\activate
```

## G. How to Run

請依序執行：

```bash
python ingest_bronze.py
python build_silver_chunks.py
python build_vector_index.py
streamlit run app.py
```

## G.1 Streamlit Community Cloud Deployment

When deploying to Streamlit Community Cloud, choose Python 3.11 in the app's Advanced settings. This project uses ChromaDB, sentence-transformers, and protobuf-dependent packages that should not be deployed on Python 3.14.

If the app was already deployed with a newer Python version, delete and redeploy the app with Python 3.11. Changing `requirements.txt` alone will not change the Python runtime for an existing Streamlit Community Cloud app.

## H. Example Search Queries

- Databricks 的 RAG 是什麼？
- Mosaic AI Vector Search 是什麼？
- Vector Search index 是從什麼資料建立的？
- Unity Catalog 在 Databricks 中負責什麼？
- Delta table 和 Vector Search 有什麼關係？
- Databricks 如何查詢 vector search index？
- 為什麼 RAG 適合企業內部文件問答？

## I. Learning Goals

完成這個專案後，你應該能理解：

- Document ingestion 如何運作。
- 為什麼需要 chunking。
- 為什麼需要 embeddings。
- 為什麼 vector retrieval 有用。
- Bronze/Silver 思維如何協助組織非結構化文件 pipeline。
- Metadata filtering 如何代表簡單的資料治理。
- 為什麼 source citation 對企業 AI 很重要。
- 這個專案未來如何延伸成完整 RAG system。

## J. Data Governance Concepts Demonstrated

這個 demo 用簡單方式展示資料治理概念：

- Metadata：每筆文件與 chunk 保留 `title`、`url`、`category`、`status`、`access_level`。
- Status filtering：只有 `active` 文件會被索引或檢索。
- Access filtering：只有 `public` 文件會被回傳。
- Citation：每個檢索結果都包含來源文件。
- Traceability：每個 chunk 保留 `source_id`、`title`、`url`、`chunk_index`。
- Conceptual Unity Catalog mapping：metadata、permissions、lineage 可對應到 Unity Catalog 的治理概念。

## K. Resume Bullets

English:
Built a Databricks-focused Microsoft Learn semantic search prototype using a Lakehouse-inspired Bronze/Silver pipeline, sentence-transformer embeddings, local ChromaDB vector indexing, metadata filtering, and source citation.

Chinese:
建立 Databricks 文件導向的 Microsoft Learn 語意搜尋系統，使用 Bronze/Silver 文件處理流程、sentence-transformer embeddings、本機 ChromaDB 向量索引、Metadata 過濾與來源引用，展示企業知識庫檢索與資料治理概念。

## L. Future Extension

這個專案未來可以延伸成完整 RAG assistant，加入：

- OpenAI API or Azure OpenAI
- Databricks Mosaic AI Model Serving
- Databricks Delta tables
- Unity Catalog permissions and lineage
- Mosaic AI Vector Search
