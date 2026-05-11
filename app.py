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
CUSTOM_PROMPT_OPTION = "Custom input"
MAX_RETURNED_CHUNKS = 5
RETRIEVAL_OVERFETCH_MULTIPLIER = 4
PROMPT_GROUPS = {
    "Vector Search": {
        "概念理解": [
            "Mosaic AI Vector Search 是什麼？",
            "Databricks 中的 vector search index 是用來解決什麼問題？",
        ],
        "操作流程": [
            "Databricks 如何查詢 vector search index？",
            "Vector Search index 是從什麼資料建立的？",
        ],
        "架構關係": [
            "Vector Search 在 RAG 架構中扮演什麼角色？",
            "向量索引如何幫助找到語意相近的文件片段？",
        ],
    },
    "RAG 與文件檢索": {
        "概念理解": [
            "Databricks 的 RAG 流程包含哪些步驟？",
            "為什麼 RAG 適合企業內部文件問答？",
        ],
        "資料流程": [
            "文件從原始網頁到可搜尋知識庫需要經過哪些處理？",
            "為什麼文件需要先切成 chunks 再建立 embeddings？",
        ],
        "結果解讀": [
            "語意搜尋結果中的 distance score 代表什麼？",
            "如何判斷檢索結果是否和問題相關？",
        ],
    },
    "Data Governance": {
        "概念理解": [
            "Unity Catalog 在 Databricks 中負責什麼？",
            "為什麼 metadata filtering 對企業知識庫很重要？",
        ],
        "權限與篩選": [
            "如何用 status 和 access_level 控制可檢索文件？",
            "metadata filtering 如何代表簡單的資料治理？",
        ],
        "來源追蹤": [
            "為什麼 source citation 對企業 AI 很重要？",
            "文件檢索系統如何保留來源 URL 和 lineage？",
        ],
    },
}

load_dotenv(PROJECT_ROOT / ".env")


class VectorDatabaseNotReadyError(RuntimeError):
    pass


def bilingual_text(english: str, chinese: str) -> None:
    st.write(english)
    st.caption(chinese)


def render_global_styles() -> None:
    st.markdown(
        """
<style>
[data-testid="stAppViewContainer"] .block-container {
    max-width: 1080px;
    padding-top: 1.75rem;
    padding-bottom: 2rem;
}
[data-testid="stVerticalBlock"] {
    gap: 0.75rem;
}
.field-label {
    margin: 1rem 0 0.35rem;
    font-weight: 700;
}
.field-label .zh,
.section-zh,
.zh-small {
    display: block;
    color: #6b7280;
    font-size: 0.86rem;
    font-weight: 500;
    line-height: 1.45;
}
.workflow {
    margin: 0.5rem 0 1rem;
    padding: 1rem 1.15rem;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background: #f9fafb;
    line-height: 1.65;
}
.workflow .zh {
    color: #6b7280;
    font-size: 0.86rem;
}
.architecture-section {
    margin-top: 1.2rem;
}
.architecture-note {
    line-height: 1.65;
}
.search-section {
    margin-top: 1.35rem;
}
.search-section h3 {
    margin-bottom: 0;
}
.search-section .zh {
    display: block;
    margin-top: 0.15rem;
    color: #6b7280;
    font-size: 0.86rem;
    font-weight: 500;
    line-height: 1.45;
}
[data-testid="stAlert"] {
    margin: 0.75rem 0;
}
</style>
""",
        unsafe_allow_html=True,
    )


def bilingual_heading(english: str, chinese: str, level: int = 3) -> None:
    if level == 1:
        st.title(english)
    elif level == 2:
        st.header(english)
    else:
        st.subheader(english)
    st.caption(chinese)


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def readable_document_paragraphs(text: str, max_chars: int = 520) -> list[str]:
    normalized = " ".join(text.split())
    sentences = []
    start = 0

    for index, character in enumerate(normalized):
        if character in ".!?" and (index == len(normalized) - 1 or normalized[index + 1] == " "):
            sentences.append(normalized[start : index + 1].strip())
            start = index + 1

    remaining = normalized[start:].strip()
    if remaining:
        sentences.append(remaining)

    paragraphs: list[str] = []
    current = ""
    for sentence in sentences:
        next_paragraph = f"{current} {sentence}".strip()
        if current and len(next_paragraph) > max_chars:
            paragraphs.append(current)
            current = sentence
        else:
            current = next_paragraph

    if current:
        paragraphs.append(current)

    return paragraphs


def render_readable_document_chunk(text: str) -> None:
    for paragraph in readable_document_paragraphs(text):
        st.write(paragraph)


def document_dedupe_key(document: str) -> str:
    return " ".join(document.split()).casefold()


def paragraph_label(number: int) -> str:
    labels = {
        1: "一",
        2: "二",
        3: "三",
        4: "四",
        5: "五",
    }
    return f"段落{labels.get(number, number)}"


def load_chroma_collection():
    if not CHROMA_PATH.exists():
        raise VectorDatabaseNotReadyError(
            "The chroma_db folder does not exist. Run the ingestion and index build scripts first."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection_names = [
        collection.name if hasattr(collection, "name") else str(collection)
        for collection in client.list_collections()
    ]
    if COLLECTION_NAME not in collection_names:
        raise VectorDatabaseNotReadyError(
            f"Collection `{COLLECTION_NAME}` was not found in chroma_db. Run build_vector_index.py to create it."
        )

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
    collection = load_chroma_collection()
    model = load_embedding_model()
    query_embedding = model.encode(query).tolist()
    result_limit = min(top_k, MAX_RETURNED_CHUNKS)
    retrieval_count = max(result_limit * RETRIEVAL_OVERFETCH_MULTIPLIER, result_limit)

    results = _query_collection(collection, query_embedding, retrieval_count)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    contexts: list[dict] = []
    seen_documents: set[str] = set()
    for document, metadata, distance in zip(documents, metadatas, distances):
        if metadata.get("status") != "active" or metadata.get("access_level") != "public":
            continue
        dedupe_key = document_dedupe_key(document)
        if dedupe_key in seen_documents:
            continue
        seen_documents.add(dedupe_key)
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
        if len(contexts) >= result_limit:
            break

    return contexts


def render_sidebar() -> None:
    st.sidebar.header("Search App")
    st.sidebar.caption("文件搜尋工具")
    page = st.sidebar.radio(
        "View",
        ["Search", "Architecture"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    st.sidebar.markdown(
        """
- Source: Microsoft Learn Azure Databricks docs  
  <span class="zh-small">來源：Microsoft Learn Azure Databricks 文件</span>
- Search method: embeddings + local vector index  
  <span class="zh-small">搜尋方式：embeddings + 本機向量索引</span>
- Output: relevant chunks + source URLs  
  <span class="zh-small">輸出：相關文件片段與來源連結</span>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.divider()
    st.sidebar.subheader("How It Works")
    st.sidebar.caption("運作方式")
    st.sidebar.markdown(
        """
1. Choose a guided prompt  
   <span class="zh-small">選擇建議問題</span>
2. Adjust `top_k`  
   <span class="zh-small">調整回傳片段數量</span>
3. Review source chunks  
   <span class="zh-small">查看文件片段與來源</span>
""",
        unsafe_allow_html=True,
    )

    try:
        collection = load_chroma_collection()
        st.sidebar.success(f"Index ready: {collection.count()} chunks")
        st.sidebar.caption("索引已就緒")
    except VectorDatabaseNotReadyError:
        st.sidebar.warning("Index not ready")
        st.sidebar.caption("索引尚未建立")

    return page


def render_search_controls() -> tuple[str, int, bool]:
    bilingual_heading("Search Official Documentation Chunks", "搜尋官方文件片段")
    bilingual_text(
        "Use the prompt filters to build a question, then retrieve relevant Microsoft Learn documentation chunks with source links.",
        "使用 prompt 篩選器建立問題，系統會找出相關的 Microsoft Learn 文件片段並附上來源連結。",
    )

    st.markdown(
        """
<div class="search-section">
    <h3>Guided Prompt Builder</h3>
    <span class="zh">引導式 Prompt 建立器</span>
</div>
""",
        unsafe_allow_html=True,
    )

    topic_column, intent_column = st.columns(2)
    with topic_column:
        st.markdown(
            "<div class='field-label'><strong>Topic</strong><br><span class='zh-small'>想問的主題</span></div>",
            unsafe_allow_html=True,
        )
        selected_topic = st.selectbox(
            "Topic",
            list(PROMPT_GROUPS),
            help="Choose whether to explore Vector Search, RAG, or governance.\n\n先決定要探索 Vector Search、RAG 或治理。",
            label_visibility="collapsed",
        )

    intent_options = list(PROMPT_GROUPS[selected_topic])
    with intent_column:
        st.markdown(
            "<div class='field-label'><strong>Question type</strong><br><span class='zh-small'>想了解什麼</span></div>",
            unsafe_allow_html=True,
        )
        selected_intent = st.selectbox(
            "Question type",
            intent_options,
            help="Choose the kind of question, such as concept, workflow, architecture, governance, or extension.\n\n依主題選擇概念、流程、架構、治理或延伸應用等問法。",
            label_visibility="collapsed",
        )

    prompt_options = PROMPT_GROUPS[selected_topic][selected_intent]
    st.markdown(
        "<div class='field-label'><strong>Suggested prompt</strong><br><span class='zh-small'>建議 prompt</span></div>",
        unsafe_allow_html=True,
    )
    selected_prompt = st.selectbox(
        "Suggested prompt",
        [CUSTOM_PROMPT_OPTION, *prompt_options],
        help="Selecting a prompt automatically fills the search box for quick testing.\n\n選擇後會自動填入搜尋框，方便快速測試語意搜尋效果。",
        label_visibility="collapsed",
    )

    if "query" not in st.session_state:
        st.session_state.query = ""
    if "selected_prompt" not in st.session_state:
        st.session_state.selected_prompt = CUSTOM_PROMPT_OPTION

    prompt_selection = (selected_topic, selected_intent, selected_prompt)
    if (
        selected_prompt != CUSTOM_PROMPT_OPTION
        and prompt_selection != st.session_state.selected_prompt
    ):
        st.session_state.query = selected_prompt

    st.session_state.selected_prompt = prompt_selection

    st.markdown(
        """
<div class="search-section">
    <h3>Search Query</h3>
    <span class="zh">搜尋問題</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='field-label'><strong>Question</strong><br><span class='zh-small'>請輸入或修改你想查詢的問題</span></div>",
        unsafe_allow_html=True,
    )
    query = st.text_input("Search question", key="query", placeholder="例如：Mosaic AI Vector Search 是什麼？", label_visibility="collapsed")

    st.markdown(
        """
<div class="search-section">
    <h3>Result Size</h3>
    <span class="zh">回傳結果數量</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='field-label'><strong>Returned chunks (top_k)</strong><br><span class='zh-small'>回傳文件片段數量</span></div>",
        unsafe_allow_html=True,
    )
    top_k = st.slider(
        "Returned chunks (top_k)",
        min_value=1,
        max_value=MAX_RETURNED_CHUNKS,
        value=5,
        help="Controls how many relevant documentation chunks are returned.\n\n控制每次搜尋最多回傳幾個相關文件片段。",
        label_visibility="collapsed",
    )

    bilingual_text(
        f"Returns up to {top_k} unique chunks. Duplicate chunks are hidden.",
        f"最多回傳 {top_k} 個不重複片段。重複的片段不會顯示。",
    )
    submitted = st.button("Search", type="primary")

    return query, top_k, submitted


def render_project_overview() -> None:
    bilingual_heading("Project Architecture", "專案架構")
    bilingual_text(
        "This prototype turns Azure Databricks documentation from Microsoft Learn into a searchable local knowledge base. It focuses on document search, semantic retrieval, metadata filtering, and source citation.",
        "這個 prototype 會把 Microsoft Learn 上的 Azure Databricks 文件轉成可搜尋的本機知識庫，重點是文件搜尋、語意檢索、metadata filtering 與來源引用。",
    )

    st.divider()
    st.subheader("Workflow")
    st.caption("架構流程")
    st.markdown(
        """
<div class="workflow">
Microsoft Learn documentation <span class="zh">Microsoft Learn 文件</span>
→ Clean text <span class="zh">清理文字</span>
→ Searchable chunks <span class="zh">可搜尋文件片段</span>
→ Embeddings
→ Vector index <span class="zh">向量索引</span>
→ User question <span class="zh">使用者問題</span>
→ Relevant chunks <span class="zh">相關片段</span>
→ Source links <span class="zh">來源連結</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="architecture-section"></div>', unsafe_allow_html=True)
    st.subheader("Main Components")
    st.caption("主要元件")
    st.dataframe(
        [
            {"Component": "data/sources.json", "中文說明": "文件來源清單"},
            {"Component": "ChromaDB vector index", "中文說明": "本機向量索引"},
            {"Component": "SentenceTransformer", "中文說明": "文字轉 embedding 的模型"},
            {"Component": "Metadata filters", "中文說明": "控制可檢索內容"},
            {"Component": "Source URL", "中文說明": "結果來源連結"},
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.markdown('<div class="architecture-section"></div>', unsafe_allow_html=True)
    usage_column, scope_column = st.columns(2)
    with usage_column:
        st.subheader("How to Use")
        st.caption("如何使用")
        st.markdown(
            """
1. **Select a topic**  
   <span class="zh-small">選擇主題</span>
2. **Pick or edit a prompt**  
   <span class="zh-small">選擇或改寫 prompt</span>
3. **Adjust returned chunks**  
   <span class="zh-small">調整回傳片段數量</span>
4. **Review chunks and URLs**  
   <span class="zh-small">查看片段與來源連結</span>
""",
            unsafe_allow_html=True,
        )

    with scope_column:
        st.subheader("Scope")
        st.caption("範圍")
        st.markdown(
            """
<div class="architecture-note">
This app performs semantic search over trusted documentation. It returns relevant chunks, metadata, distance scores, and source URLs for review, but does not generate final answers.<br>
<span class="zh-small">這個 app 會針對可信任文件做語意搜尋，回傳相關片段、metadata、distance score 與來源 URL 供使用者檢查，但不會自動產生完整回答。</span>
</div>
""",
            unsafe_allow_html=True,
        )


def render_search_results(query: str, top_k: int) -> None:
    if not query.strip():
        st.warning("Please enter a search question first.\n\n請先輸入搜尋問題。")
        return

    try:
        with st.spinner("Running semantic search...\n\n正在進行語意搜尋..."):
            contexts = retrieve_context(query, top_k)
    except VectorDatabaseNotReadyError as exc:
        st.warning(
            f"{exc}\n\n"
            "Local setup: rebuild the local vector index, then restart the app.\n\n"
            "Streamlit Cloud note: `chroma_db/` is ignored by `.gitignore`, so the deployed app will not include a prebuilt vector index unless you change the deployment strategy.\n\n"
            "本機使用：請先重新建立本機向量索引，再重新啟動 app。\n\n"
            "Streamlit Cloud 注意事項：目前 `.gitignore` 會忽略 `chroma_db/`，因此部署版不會自動帶入已建立好的向量索引。"
        )
        return
    except Exception:
        st.error("Search failed because of an unexpected retrieval error.\n\n搜尋因為非預期的檢索錯誤而失敗。")
        return

    bilingual_heading("Most Relevant Documentation Chunks", "最相關文件片段")
    bilingual_text("Lower distance usually means stronger semantic similarity.", "Distance 越低通常代表語意越接近。")

    if not contexts:
        st.info("No documentation chunks matched the governance filters.\n\n沒有找到符合治理條件的文件片段。")
        return

    title_counts: dict[str, int] = {}
    for context in contexts:
        title_counts[context["title"]] = title_counts.get(context["title"], 0) + 1

    title_seen: dict[str, int] = {}
    for index, context in enumerate(contexts, start=1):
        title = context["title"]
        title_seen[title] = title_seen.get(title, 0) + 1
        expander_title = f"{index}. {title}"
        if title_counts[title] > 1:
            expander_title = f"{expander_title} - {paragraph_label(title_seen[title])}"

        with st.expander(expander_title):
            st.write(f"**Title:** {context['title']}")
            if title_counts[title] > 1:
                st.write(f"**Paragraph:** {paragraph_label(title_seen[title])}")
            st.write(f"**Category:** {context['category']}")
            st.write(f"**URL:** {context['url']}")
            st.write(f"**Distance score:** {context['distance']:.4f}")
            st.write(f"**Source ID:** {context['source_id']}")

            st.markdown("**Readable source text:**")
            render_readable_document_chunk(context["content"])


def main() -> None:
    st.set_page_config(
        page_title="Databricks Microsoft Learn Semantic Search",
        page_icon="🔎",
        layout="wide",
    )
    render_global_styles()
    page = render_sidebar()
    bilingual_heading(
        "Databricks Microsoft Learn Semantic Search",
        "Databricks Microsoft Learn 語意搜尋系統",
        level=1,
    )
    st.write(
        "這是一個不用 API 的 Databricks 文件語意搜尋練習：系統會讀取 Microsoft Learn 上的 Azure Databricks 文件，"
        "將文件切成可搜尋片段並建立本機向量索引。使用者輸入問題後，系統會用向量搜尋找出最相關的文件段落，"
        "並顯示來源引用。"
    )

    if page == "Architecture":
        render_project_overview()
    else:
        query, top_k, submitted = render_search_controls()
        if submitted:
            render_search_results(query, top_k)


if __name__ == "__main__":
    main()
