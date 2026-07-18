# Express Analytics RAG Assistant

A self-corrective RAG technical documentation assistant for the Express Analytics AI/ML Engineer Intern assignment. The system ingests local documentation into **ChromaDB**, answers questions through a **LangGraph** workflow, and uses **Ollama** for LLM grading and generation — exposed via **FastAPI** and a **Streamlit** UI.

---

## Project Overview

This project implements retrieval-augmented generation for technical documentation with source citations, bounded query retries, and safe fallback when context is insufficient.

**Stack highlights:**

- **LangGraph** `StateGraph` — query analysis → retrieval → grading → retry → generation → fallback
- **ChromaDB** + `sentence-transformers/all-MiniLM-L6-v2` embeddings
- **Ollama** — structured grading and answer generation (with deterministic fallbacks)
- **FastAPI** — `/query`, `/upload`, `/ingest`, `/documents`, `/feedback`, `/health`
- **Incremental indexing** — corpus re-ingest preserves uploaded documents
- **74 passing tests**

Assignment design write-up (architecture reasoning, chunking/embedding choices, assumptions): **[WRITEUP.md](WRITEUP.md)**

---

## Features

| Area | Capabilities |
|---|---|
| **Ingestion** | MD, TXT, HTML, PDF (PyMuPDF), DOCX (python-docx); chunking with citation metadata |
| **RAG pipeline** | Query rewrite, vector retrieval, LLM grading, retry loop, grounded generation, fallback |
| **Upload** | `POST /upload` → `data/uploads/`; PDF/DOCX/TXT/MD; 10 MB limit; duplicate filename check |
| **Indexing** | Incremental upsert; source-scoped corpus re-index via `replace_documents_for_sources()` |
| **API & UI** | FastAPI + OpenAPI docs; Streamlit frontend for upload, document list, and chat |
| **Feedback** | JSONL-backed `POST /feedback` |

---

## Architecture

**High-level flow:**

```text
User → FastAPI → LangGraph StateGraph → VectorStoreService → ChromaDB
```

**Detailed architecture:**

```text
                    +----------------------+
                    |        User          |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |       FastAPI        |
                    | /query /upload ...   |
                    +----------+-----------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
   +----------------------+          +----------------------+
   |   LangGraph Workflow |          |  Ingestion Pipeline  |
   | query -> retrieve -> |          | load -> chunk        |
   | grade -> generate    |          +----------+-----------+
   +----------+-----------+                     |
              |                                 v
              v                      +----------------------+
   +----------------------+          |   VectorStoreService |
   |    LLMService        |          | upsert / replace     |
   | Ollama grading/gen   |          +----------+-----------+
   +----------------------+                     |
                                      +----------------------+
                                      |       ChromaDB       |
                                      | app/storage/chroma_db
                                      +----------------------+
```

| Path | Role |
|---|---|
| `app/graph/` | LangGraph state, builder, routers, nodes |
| `app/ingestion/` | Loaders, chunking, pipeline |
| `app/services/` | LLM (Ollama), embeddings, vector store |
| `app/api/` | FastAPI routes |
| `frontend.py` | Streamlit UI |

**Upload workflow:**

```text
User Upload → data/uploads/ → Loader → Chunker → Embeddings → ChromaDB → queryable
```

- Formats: PDF, DOCX, TXT, MD · Max 10 MB · Duplicate filename → `already_exists`

**Incremental indexing:**

```text
POST /ingest → replace_documents_for_sources() → refresh corpus source_ids only
                                              → upload source_ids UNTOUCHED
```

- **Problem:** Full index rebuild wiped uploaded documents.
- **Solution:** Source-scoped replacement using deterministic `source_id` (SHA-1 of resolved path).
- **Result:** Uploaded documents survive corpus re-ingestion.

---

## LangGraph Workflow

```text
START → Query Analysis → Retrieval → Grading → Conditional Routing
                                                    |
                        +-------------+-------------+-------------+
                        |             |                           |
                        v             v                           v
                   Generation   Transform Query              Fallback
                        |             |                           |
                       END        Retrieval                      END
                                      |
                                      v
                                  Grading  (retry loop)
```

**Routing logic:**

```python
if len(graded_docs) > 0:        return "generate"
if retry_count < max_retries:   return "transform_query"
return "fallback"
```

| Node | Purpose |
|---|---|
| Query Analysis | Rewrite and classify user queries |
| Retrieval | Fetch top-k chunks from ChromaDB |
| Grading | Filter irrelevant chunks (Ollama + heuristic fallback) |
| Transform Query | Retry with improved query (`retry_count += 1`) |
| Generation | Produce grounded answer with structural citations |
| Fallback | Return safe response when context is insufficient |

**Tech defaults:** `chunk_size=800`, `chunk_overlap=100`, `max_retries=2`, `k=5`, Ollama model `qwen2.5:14b`

---

## Setup & Running

**Requirements:** Python 3.11+ · See `requirements.txt`

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**Ollama (recommended):**

```bash
ollama serve
ollama pull qwen2.5:14b
```

Optional: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `LLM_REQUEST_TIMEOUT_SECONDS` (default 30s).  
Without Ollama, heuristic grading and extractive generation still work.

**Start API:**

```bash
python -m uvicorn app.main:app --reload
```

**Ingest demo corpus** (`data/corpus/`):

```bash
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d "{\"path\":\"data/corpus\"}"
```

**Query:**

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\":\"How does FastAPI validate request bodies?\"}"
```

**Streamlit UI:**

```bash
streamlit run frontend.py
```

Open `http://localhost:8000/docs` (API) · `http://localhost:8501` (UI)

---

## API Overview

Base URL: `http://localhost:8000` · Interactive docs: `/docs`

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Application liveness check |
| POST | `/query` | Run the LangGraph RAG workflow |
| POST | `/upload` | Upload and index a document (`multipart/form-data`) |
| POST | `/ingest` | Ingest/re-index a local corpus directory |
| GET | `/documents` | List indexed documents with metadata |
| POST | `/feedback` | Store answer feedback (`query_id`, `rating` 1–5) |

**Example request** (`POST /query`):

```json
{ "question": "How does FastAPI validate request bodies?" }
```

**Example response:**

```json
{
  "query_id": "b5ef82bf-74a6-4b5d-8bdc-7d6b888748e2",
  "answer": "FastAPI validates request bodies using Pydantic models.",
  "citations": [{ "source_id": "a3c8707fa870016c", "source_title": "request_models", "chunk_index": 0 }],
  "confidence": "high",
  "retry_count": 0,
  "fallback_used": false
}
```

Full API examples and error codes: see [REPORT.md](REPORT.md).

---

## Streamlit UI

`frontend.py` provides a local demo with three tabs:

1. **Upload & Ingest** — file uploader, pending queue, ingest button (`POST /upload`)
2. **Indexed Documents** — table from `GET /documents`
3. **Chat Assistant** — question input, answer, citations, confidence, retry/fallback metadata

Start FastAPI first, then `streamlit run frontend.py`.

---

## Screenshots

### Upload & Ingest

![Upload workflow](docs/screenshots/upload.png)

### Indexed Documents

![Indexed documents table](docs/screenshots/indexed-documents.png)

### Query Result

![Query result with answer and citations](docs/screenshots/query.png)

---

## Testing

### Test results

**74 tests passing** — run with `python -m pytest -q`

**Coverage:** API endpoints · LangGraph workflow · retrieval · grading · generation · upload pipeline · incremental indexing · feedback storage · ingestion (PDF/DOCX)

### Validation summary

- ✅ Successful retrieval and answer generation
- ✅ Retry loop validation (`retry_count` bounded by `max_retries`)
- ✅ Fallback path validation (low confidence, safe message)
- ✅ Upload workflow validation (API + Streamlit)
- ✅ Incremental indexing validation (`tests/test_ingest_incremental.py`)
- ✅ Citations generated from chunk metadata

Detailed validation traces: see [REPORT.md](REPORT.md).

---

## Design Decisions & Tradeoffs

| Decision | Why |
|---|---|
| **ChromaDB** | Local, persistent vector store — no external dependency |
| **all-MiniLM-L6-v2** | Fast local embeddings, strong retrieval baseline |
| **Ollama** | No API keys; full local LLM grading and generation |
| **Heuristic fallback grading** | Graph stays alive when Ollama is unavailable |
| **Extractive fallback generation** | Always returns context-grounded answers |
| **Bounded retries** | `max_retries=2` prevents infinite loops |
| **Structural citations** | Derived from metadata, not LLM free-text |
| **Incremental indexing** | Uploads survive corpus re-ingest |

Architecture reasoning, chunking/embedding strategy, assumptions, and build phasing: [WRITEUP.md](WRITEUP.md).  
Node-level detail, API reference, and validation traces: [REPORT.md](REPORT.md).

---

## Assignment Compliance Checklist

| Requirement | Status | Evidence |
|---|---|---|
| RAG technical documentation assistant | ✅ | `app/ingestion/`, `vector_store.py`, `graph/builder.py`, `api/query.py` |
| LangGraph StateGraph | ✅ | `graph/builder.py`, `state.py`, `router.py` |
| Document ingestion | ✅ | `loaders.py`, `chunking.py`, `pipeline.py` |
| MD / TXT / HTML / PDF / DOCX support | ✅ | `DocumentLoader.SUPPORTED_EXTENSIONS` |
| Chunking with overlap | ✅ | `chunk_size=800`, `chunk_overlap=100` |
| Source metadata | ✅ | `source_id`, `source_title`, `source_path`, `chunk_index` |
| Embeddings | ✅ | `embeddings.py` — `all-MiniLM-L6-v2` |
| Vector retrieval | ✅ | ChromaDB — `vector_store.py` |
| LLM document grading | ✅ | `llm.py` + `grading.py` (Ollama) |
| Answer generation | ✅ | `llm.py` + `generation.py` (Ollama) |
| Rewrite-and-retry | ✅ | `transform_query.py` |
| Bounded retry loop | ✅ | `router.py`, `config.py` |
| Citations | ✅ | `generation.py` |
| Fallback behavior | ✅ | `fallback.py` |
| FastAPI endpoints | ✅ | `/query`, `/upload`, `/ingest`, `/documents`, `/feedback`, `/health` |
| Pydantic models | ✅ | `app/schemas/` |
| Feedback storage | ✅ | `feedback_store.py` |
| Tests | ✅ | **74 passed** |
| Streamlit UI *(bonus)* | ✅ | `frontend.py` |
| Document upload *(extension)* | ✅ | `POST /upload` |
| Incremental indexing *(extension)* | ✅ | `replace_documents_for_sources()` |
| Hallucination check *(bonus)* | ⚠️ | `hallucination_check.py` — implemented, not wired |

---

## Future Improvements

- Wire hallucination check into the main graph after generation
- Implement `seen_chunk_ids` deduplication for more effective retries
- Add cross-encoder reranking after vector retrieval
- Add web-search fallback for out-of-corpus questions
- Add streaming responses for `/query`
- Increase default LLM timeout for larger Ollama models
- Add authentication and rate limiting for deployment

---

## Documentation

| Document | Contents |
|---|---|
| **[WRITEUP.md](WRITEUP.md)** | Assignment write-up — thought process, phased build decisions, architecture & workflow reasoning, chunking/embedding strategy, assumptions, known limitations, future improvements |
| **[REPORT.md](REPORT.md)** | Technical report — node reference, retry/confidence logic, full API examples, validation traces, extended tradeoffs |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Original system design spec and MVP phasing plan |

---

*Built as an internship assignment submission for Express Analytics.*
