"""
Streamlit frontend for local RAG assistant testing.

Run:
    streamlit run frontend.py

Prerequisites:
    FastAPI backend running at http://127.0.0.1:8000
    python -m uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import contextmanager

import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"
QUERY_URL = f"{API_BASE_URL}/query"
HEALTH_URL = f"{API_BASE_URL}/health"
DOCUMENTS_URL = f"{API_BASE_URL}/documents"
UPLOAD_URL = f"{API_BASE_URL}/upload"
QUERY_TIMEOUT_SECONDS = 300
UPLOAD_TIMEOUT_SECONDS = 300
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_UPLOAD_TYPES = ["pdf", "docx", "txt", "md"]

SAMPLE_QUESTIONS = [
    "What is the purpose of the document grading node?",
    "What happens when no relevant documents are found?",
    "Summarize the assignment architecture in 5 bullet points.",
    "How do I deploy Kubernetes operators?",
]

CONFIDENCE_LABELS = {
    "high": "🟢 High",
    "medium": "🟡 Medium",
    "low": "🔴 Low",
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer, header {visibility: hidden;}
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.65rem 0.85rem;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: #0f172a !important;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            color: #334155 !important;
            font-weight: 600;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #1d4ed8 !important;
        }
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #1d4ed8 100%);
            border-radius: 16px;
            padding: 1.75rem 2rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.15);
        }
        .hero-kicker {
            color: #bfdbfe !important;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }
        .hero-title {
            color: #ffffff !important;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.2;
            margin: 0 0 0.5rem 0;
        }
        .hero-subtitle {
            color: #dbeafe !important;
            font-size: 1rem;
            line-height: 1.55;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    if "pending_uploads" not in st.session_state:
        st.session_state.pending_uploads = {}
    if "documents_cache" not in st.session_state:
        st.session_state.documents_cache = []
    if "last_query_result" not in st.session_state:
        st.session_state.last_query_result = None


def _check_api_health() -> tuple[bool, str]:
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        response.raise_for_status()
        if response.json().get("status") == "healthy":
            return True, "Online"
        return False, "Unexpected response"
    except requests.exceptions.ConnectionError:
        return False, "Offline"
    except (requests.exceptions.RequestException, ValueError, TypeError):
        return False, "Unavailable"


def _fetch_documents() -> list[dict] | None:
    try:
        response = requests.get(DOCUMENTS_URL, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not reach the API to fetch indexed documents.")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to fetch documents: {exc}")
        return None
    if not isinstance(payload, list):
        st.error("Invalid documents response from API.")
        return None
    return payload


def _upload_file(filename: str, content: bytes) -> dict | None:
    try:
        response = requests.post(
            UPLOAD_URL,
            files={"file": (filename, content)},
            timeout=UPLOAD_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach the API while uploading `{filename}`.")
        return None
    except requests.exceptions.Timeout:
        st.error(f"Upload timed out for `{filename}`.")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Upload failed for `{filename}`: {exc}")
        return None

    if response.status_code == 413:
        st.error(f"`{filename}` exceeds the 10 MB upload limit.")
        return None
    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        st.error(f"Upload failed for `{filename}`: {detail}")
        return None

    try:
        return response.json()
    except ValueError:
        st.error(f"Invalid upload response for `{filename}`.")
        return None


def _submit_query(question: str) -> dict | None:
    try:
        response = requests.post(
            QUERY_URL,
            json={"question": question},
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        st.error("Could not reach the API. Start the backend with `python -m uvicorn app.main:app --reload`")
        return None
    except requests.exceptions.Timeout:
        st.error(f"Request timed out after {QUERY_TIMEOUT_SECONDS} seconds.")
        return None
    except requests.exceptions.HTTPError:
        st.error(f"API returned HTTP {response.status_code}: {response.text}")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")
        return None

    try:
        payload = response.json()
    except ValueError:
        st.error("Invalid response: API did not return valid JSON.")
        return None

    required = ("answer", "confidence", "retry_count", "fallback_used", "citations")
    missing = [field for field in required if field not in payload]
    if missing:
        st.error(f"Invalid response: missing fields {', '.join(missing)}.")
        return None
    if not isinstance(payload["answer"], str):
        st.error("Invalid response: answer must be a string.")
        return None
    return payload


@contextmanager
def _section(title: str, caption: str):
    with st.container(border=True):
        st.subheader(title)
        st.caption(caption)
        yield


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("Express Analytics")
        st.caption("RAG Assistant · Local Demo")

        is_healthy, status_msg = _check_api_health()
        if is_healthy:
            st.success(f"API status: {status_msg}")
        else:
            st.error(f"API status: {status_msg}")

        st.markdown(f"**Backend:** `{API_BASE_URL}`")

        if st.button("Refresh status", use_container_width=True, type="secondary"):
            st.rerun()

        st.divider()
        st.markdown("**Quick links**")
        st.markdown(f"- [Swagger UI]({API_BASE_URL}/docs)")
        st.markdown(f"- [Health check]({HEALTH_URL})")

        st.divider()
        st.markdown("**Supported uploads**")
        st.markdown("PDF · DOCX · TXT · MD")
        st.caption("Maximum size: 10 MB per file")


def _render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Express Analytics Intern Assignment</div>
            <div class="hero-title">RAG Technical Documentation Assistant</div>
            <p class="hero-subtitle">
                Upload documents, index them into ChromaDB, and query a self-corrective
                LangGraph pipeline with grounded answers and citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_stats_row(doc_count: int, is_healthy: bool) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indexed documents", doc_count)
    c2.metric("API status", "Online" if is_healthy else "Offline")
    c3.metric("Pending uploads", len(st.session_state.pending_uploads))
    c4.metric("Supported formats", len(SUPPORTED_UPLOAD_TYPES))


def _render_steps() -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Step 1 — Start API**\n\n`python -m uvicorn app.main:app --reload`")
    with c2:
        st.info("**Step 2 — Upload & ingest**\n\nAdd PDF, DOCX, TXT, or MD files to the vector index.")
    with c3:
        st.info("**Step 3 — Ask questions**\n\nReview answers, citations, confidence, and retry metadata.")


def _render_upload_section() -> None:
    with _section("Upload Documents", "Queue files, then ingest them into the vector store."):
        left, right = st.columns([1.1, 0.9], gap="large")

        with left:
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                type=SUPPORTED_UPLOAD_TYPES,
                accept_multiple_files=True,
                help="Supported: pdf, docx, txt, md. Maximum size: 10 MB per file.",
            )

            if uploaded_files:
                for uploaded in uploaded_files:
                    if uploaded.size and uploaded.size > MAX_UPLOAD_BYTES:
                        st.warning(f"`{uploaded.name}` exceeds the 10 MB limit.")
                        continue
                    st.session_state.pending_uploads[uploaded.name] = uploaded.getvalue()

            if st.button("Ingest selected files", type="primary", use_container_width=True):
                pending_names = sorted(st.session_state.pending_uploads.keys())
                if not pending_names:
                    st.warning("Add at least one file before ingesting.")
                else:
                    with st.spinner("Indexing uploaded documents…"):
                        ingested_any = False
                        for name in list(pending_names):
                            result = _upload_file(name, st.session_state.pending_uploads[name])
                            if result is None:
                                continue
                            status = result.get("status")
                            if status == "success":
                                st.success(
                                    f"**{name}** indexed · "
                                    f"{result.get('chunks_created', 0)} chunks · "
                                    f"{result.get('processing_time_seconds', 0)}s"
                                )
                                st.session_state.pending_uploads.pop(name, None)
                                ingested_any = True
                            elif status == "already_exists":
                                st.warning(f"**{name}** already exists in uploads.")
                                st.session_state.pending_uploads.pop(name, None)
                            else:
                                st.error(result.get("message") or f"Upload failed for {name}.")
                    if ingested_any:
                        refreshed = _fetch_documents()
                        if refreshed is not None:
                            st.session_state.documents_cache = refreshed

        with right:
            st.markdown("##### Pending uploads")
            pending_names = sorted(st.session_state.pending_uploads.keys())
            if not pending_names:
                st.markdown("_No files queued yet._")
            else:
                for name in pending_names:
                    size_kb = len(st.session_state.pending_uploads[name]) / 1024
                    ext = name.rsplit(".", 1)[-1].upper() if "." in name else "FILE"
                    st.markdown(f"- **{name}** · `{ext}` · {size_kb:.1f} KB")


def _render_documents_section() -> None:
    with _section("Indexed Documents", "Documents currently available for retrieval."):
        _, refresh_col = st.columns([4, 1])
        with refresh_col:
            if st.button("Refresh", use_container_width=True):
                refreshed = _fetch_documents()
                if refreshed is not None:
                    st.session_state.documents_cache = refreshed

        documents = st.session_state.documents_cache
        if not documents:
            refreshed = _fetch_documents()
            if refreshed is not None:
                st.session_state.documents_cache = refreshed
                documents = refreshed

        if not documents:
            st.warning("No indexed documents yet. Upload and ingest a file to get started.")
            return

        rows = [
            {
                "Document": doc.get("source_title") or doc.get("filename") or "—",
                "Type": (doc.get("file_type") or "—").upper(),
                "Pages": doc.get("pages") if doc.get("pages") is not None else "—",
                "Chunks": doc.get("chunks") if doc.get("chunks") is not None else "—",
                "Source ID": doc.get("source_id") or "—",
            }
            for doc in documents
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_result(payload: dict) -> None:
    st.markdown("##### Answer")
    with st.container(border=True):
        st.write(payload["answer"])

    confidence = str(payload["confidence"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", CONFIDENCE_LABELS.get(confidence, confidence.title()))
    m2.metric("Retry count", payload["retry_count"])
    m3.metric("Fallback used", "Yes" if payload["fallback_used"] else "No")

    citations = payload.get("citations") or []
    st.markdown("##### Citations")
    if not citations:
        st.caption("No citations returned.")
        return

    for citation in citations:
        title = citation.get("source_title", "Unknown")
        chunk_index = citation.get("chunk_index", "—")
        source_id = citation.get("source_id", "—")
        st.markdown(f"- **{title}** · Chunk `{chunk_index}` · Source ID `{source_id}`")


def _render_chat_section() -> None:
    with _section("Ask the Assistant", "Query the LangGraph RAG pipeline over indexed documentation."):
        st.markdown("##### Sample questions")
        cols = st.columns(2)
        for index, sample in enumerate(SAMPLE_QUESTIONS):
            with cols[index % 2]:
                if st.button(sample, key=f"sample_{index}", use_container_width=True):
                    st.session_state["question"] = sample

        question = st.text_area(
            "Your question",
            value=st.session_state.get("question", ""),
            height=140,
            placeholder="Ask a question about the ingested documentation…",
        )

        if st.button("Ask question", type="primary"):
            trimmed = question.strip()
            if not trimmed:
                st.warning("Please enter a question before submitting.")
            else:
                with st.spinner("Running retrieval, grading, and generation…"):
                    result = _submit_query(trimmed)
                if result is not None:
                    st.session_state.last_query_result = result

        if st.session_state.last_query_result is not None:
            st.divider()
            _render_result(st.session_state.last_query_result)


def main() -> None:
    st.set_page_config(
        page_title="Express Analytics RAG Assistant",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    _init_session_state()
    _render_sidebar()

    _render_hero()

    documents = st.session_state.documents_cache
    if not documents:
        refreshed = _fetch_documents()
        if refreshed is not None:
            st.session_state.documents_cache = refreshed
            documents = refreshed

    is_healthy, _ = _check_api_health()
    _render_stats_row(len(documents), is_healthy)
    _render_steps()

    tab_upload, tab_docs, tab_chat = st.tabs(
        ["Upload & Ingest", "Indexed Documents", "Chat Assistant"]
    )
    with tab_upload:
        _render_upload_section()
    with tab_docs:
        _render_documents_section()
    with tab_chat:
        _render_chat_section()


if __name__ == "__main__":
    main()
