# Express Analytics RAG Assistant — Technical Report

This document supplements [README.md](README.md) with detailed design explanations, validation traces, and API reference material requested for the internship assignment.

---

## Table of Contents

1. [Self-Corrective RAG Design](#self-corrective-rag-design)
2. [Node Reference](#node-reference)
3. [Retry Logic & Confidence Scoring](#retry-logic--confidence-scoring)
4. [Ingestion & Vector Search](#ingestion--vector-search)
5. [Ollama Integration](#ollama-integration)
6. [Upload Workflow](#upload-workflow)
7. [Incremental Indexing](#incremental-indexing)
8. [API Reference](#api-reference)
9. [Validation Results](#validation-results)
10. [Assumptions](#assumptions)
11. [Known Limitations](#known-limitations)
12. [Extended Design Tradeoffs](#extended-design-tradeoffs)

---

## Self-Corrective RAG Design

The pipeline follows a self-corrective pattern: retrieve chunks, grade relevance, and either generate an answer, rewrite the query and retry, or return a safe fallback.

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
                                  Grading
```

**Routing predicate** (`app/graph/router.py`):

```python
if len(graded_docs) > 0:
    return "generate"
if retry_count < max_retries:
    return "transform_query"
return "fallback"
```

---

## Node Reference

### Query Analysis (`app/graph/nodes/query_analysis.py`)

Normalizes and rewrites the user question for retrieval. Classifies query type:

- `conceptual` · `how_to` · `troubleshooting` · `api_reference`

Example:

```text
Question: How does FastAPI validate request bodies?
Rewrite:  How does FastAPI validate request bodies using Pydantic models?
```

### Retrieval (`app/graph/nodes/retrieval.py`)

Reads `rewritten_query` and calls `VectorStoreService.similarity_search(query, k=5)`.

Returns chunks as:

```json
{ "content": "...", "metadata": {...}, "score": 0.91 }
```

### LLM Grading (`app/graph/nodes/grading.py`)

Primary path: `LLMService.grade_relevance(question, chunk)` via Ollama structured JSON:

```json
{
  "verdict": "relevant",
  "confidence": 0.88,
  "reason": "The chunk directly explains FastAPI request body validation."
}
```

### Heuristic Fallback Grading

When Ollama is unavailable, times out, or returns malformed JSON, a deterministic heuristic grades each chunk using:

- vector similarity score
- lexical overlap count between question and chunk terms
- overlap ratio (conservative — avoids false positives on generic terms like `FastAPI`)

### Transform Query (`app/graph/nodes/transform_query.py`)

When no relevant chunks are found:

- increments `retry_count`
- replaces `rewritten_query` with a unique alternate formulation
- appends previous query to `previous_queries` if not already present

Example retry sequence:

```text
How do I configure OAuth2 scopes in FastAPI?
Give a broad overview of configure OAuth2 scopes in FastAPI and related concepts.
Documentation about: configure, OAuth2, scopes, FastAPI.
```

### Generation (`app/graph/nodes/generation.py`)

Calls `LLMService.generate_answer(question, context)` with context built exclusively from `graded_docs`. Citations are derived structurally from chunk metadata. Falls back to extractive concatenation when Ollama fails.

### Fallback (`app/graph/nodes/fallback.py`)

Returns a safe, low-confidence message when retries are exhausted or the graph records an error. Sets `fallback_used=True`, `confidence="low"`.

### Hallucination Check *(bonus, not wired)*

`app/graph/nodes/hallucination_check.py` — deterministic lexical groundedness check. Implemented and tested but not connected to the main graph builder.

---

## Retry Logic & Confidence Scoring

### Retry logic

- `max_retries` is runtime config (default `2`), not stored in graph state
- `retry_count` is incremented by `transform_query`, read by `decide_generation_path`
- Loop terminates when `retry_count >= max_retries` or relevant chunks are found

### Confidence scoring (deterministic)

| Level | Condition |
|---|---|
| `low` | Fallback path used |
| `high` | First attempt with multiple relevant graded docs |
| `medium` | Thinner context or answer after retries |

---

## Ingestion & Vector Search

### Supported formats (`app/ingestion/loaders.py`)

| Extension | Format | Loader |
|---|---|---|
| `.md` | Markdown | UTF-8 text |
| `.txt` | Plain text | UTF-8 text |
| `.html`, `.htm` | HTML | Visible text extraction |
| `.pdf` | PDF | PyMuPDF — all pages, `page_count` metadata |
| `.docx` | Word | python-docx — paragraphs in order |

Metadata on all documents: `source_id`, `source_title`, `source_path`, `filename`, `file_type`.

### Chunking

`RecursiveCharacterTextSplitter` — `chunk_size=800`, `chunk_overlap=100`  
Per-chunk metadata: `chunk_index`, `section_heading`

### Embeddings

`sentence-transformers/all-MiniLM-L6-v2` — lazy-loaded, local inference.

### ChromaDB (`app/services/vector_store.py`)

| Method | Purpose |
|---|---|
| `add_documents()` | Incremental upsert |
| `replace_documents_for_sources()` | Source-scoped reindex |
| `similarity_search(query, k=5)` | Semantic retrieval |
| `list_documents()` | Document summaries for API |
| `build_index()` | Full rebuild (scripts/tests only) |

Persistence path: `app/storage/chroma_db`

---

## Ollama Integration

File: `app/services/llm.py`

| Setting | Default |
|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `qwen2.5:14b` |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `30` |

**Grading** — `grade_relevance()` returns `{verdict, confidence, reason}`  
**Generation** — `generate_answer()` returns `{answer}`  
**Temperature** — `0.0` for reproducibility

**Fallbacks when Ollama is down:**

- Grading → heuristic overlap grader
- Generation → extractive answer from graded chunks

---

## Upload Workflow

```text
User Upload (Streamlit or POST /upload)
  → data/uploads/
  → DocumentLoader (PDF / DOCX / TXT / MD)
  → DocumentChunker
  → EmbeddingService
  → VectorStoreService.add_documents()
  → ChromaDB (immediately queryable)
```

| Constraint | Value |
|---|---|
| Max file size | 10 MB |
| Duplicate check | Filename exists in `data/uploads/` → `already_exists` |
| API formats | PDF, DOCX, TXT, MD (HTML via corpus loader only) |

---

## Incremental Indexing

### Problem

`POST /ingest` originally called `build_index()`, deleting the entire Chroma collection. Documents added via `POST /upload` were lost on the next corpus ingest.

### Solution

`POST /ingest` now calls `replace_documents_for_sources()`:

1. Extract `source_id` values from incoming corpus chunks
2. Delete existing chunks matching those `source_id`s only
3. Upsert refreshed corpus chunks

```text
POST /ingest (data/corpus)
        |
        v
replace_documents_for_sources()
        |
        +-- Delete chunks where source_id IN corpus_ids
        +-- Upsert new corpus chunks
        +-- Upload chunks (other source_ids) UNTOUCHED
```

### Deterministic IDs

- `source_id` = first 16 hex chars of `SHA-1(resolved_source_path)`
- Corpus: `data/corpus/foo.md` → one `source_id`
- Upload: `data/uploads/bar.pdf` → different `source_id`
- Chunk ID: `{source_id}:{chunk_index}:{content_hash}` — prevents duplicate chunks on re-index

### Tradeoff

Orphaned chunks from deleted corpus files are not automatically removed.

Test coverage: `tests/test_ingest_incremental.py`, `tests/test_vector_store.py`

---

## API Reference

Base URL: `http://localhost:8000`

### GET /health

```json
{ "status": "healthy" }
```

### POST /query

**Request:**

```json
{ "question": "How does FastAPI validate request bodies?" }
```

**Response:**

```json
{
  "query_id": "b5ef82bf-74a6-4b5d-8bdc-7d6b888748e2",
  "answer": "FastAPI validates request bodies using Pydantic models.",
  "citations": [
    { "source_id": "a3c8707fa870016c", "source_title": "request_models", "chunk_index": 0 }
  ],
  "confidence": "high",
  "retry_count": 0,
  "fallback_used": false
}
```

Errors: `422` empty question · `500` graph failure

### POST /upload

`multipart/form-data` — single file field `file`

**Success:**

```json
{
  "status": "success",
  "filename": "assignment.pdf",
  "chunks_created": 34,
  "processing_time_seconds": 2.418,
  "message": "Upload successful"
}
```

**Duplicate:**

```json
{
  "status": "already_exists",
  "filename": "assignment.pdf",
  "message": "A file with this name already exists in uploads. Skipping ingestion."
}
```

Errors: `400` unsupported/empty · `413` > 10 MB · `500` indexing failure

### POST /ingest

```json
{ "path": "data/corpus" }
```

```json
{ "documents_processed": 5, "chunks_created": 20, "status": "success" }
```

### GET /documents

```json
[
  {
    "source_id": "42ddb2ed27a6ac6f",
    "source_title": "assignment",
    "filename": "assignment.pdf",
    "file_type": "pdf",
    "pages": 6,
    "chunks": 34
  }
]
```

### POST /feedback

```json
{ "query_id": "b5ef82bf-...", "rating": 5, "comment": "Helpful answer" }
```

Persisted to `app/storage/feedback.jsonl`. Errors: `422` rating outside 1–5.

---

## Validation Results

### Positive path

**Question:** `How does FastAPI validate request bodies?`

```text
query_type: conceptual
rewritten_query: How does FastAPI validate request bodies using Pydantic models?
retry_count: 0 | fallback_used: false | confidence: high
```

Answer grounded in `request_models` with structural citations.

**LLM synthesis example:**

**Question:** `Summarize the requirements for the document grading node in 3 bullet points.`

```text
- Use an LLM to evaluate each retrieved chunk as 'relevant' or 'irrelevant'
- Route to a fallback path if all documents are deemed irrelevant
- Filter out irrelevant chunks and proceed with relevant ones
```

### Negative path

**Question:** `How do I configure OAuth2 scopes in FastAPI?`

```text
retry_count: 2 | fallback_used: true | confidence: low
answer: I could not find enough relevant information in the indexed documentation...
```

Heuristic grader rejects unrelated FastAPI chunks sharing only generic terms.

### Retry loop

Unique rewrites observed before `max_retries` exhausted (see Transform Query example above).

### Upload & incremental indexing

1. Upload PDF/DOCX → appears in `GET /documents`
2. Query uploaded content → citations reference uploaded `source_title`
3. Re-ingest corpus → uploaded document remains indexed and searchable

---

## Assumptions

- Indexed corpus is **technical documentation** in **English**
- **Local Ollama** recommended for best LLM quality; fallbacks work without it
- PDFs/DOCX are **text-based** (no OCR for scanned images)
- **Local deployment** target (reviewer laptop), not multi-tenant production
- Single Chroma collection at `app/storage/chroma_db`
- Corpus via `POST /ingest`; ad-hoc files via `POST /upload` or Streamlit

---

## Known Limitations

- **Retrieval-grounded answers only:** The system answers only from retrieved context and does not summarize an entire large document unless the relevant summary information is retrieved. This is an intentional grounding tradeoff to reduce hallucinations.
- **Top-k retrieval scope:** Answers use a bounded set of graded chunks (default `k=5`), not the full index.
- **No OCR:** Scanned/image PDFs are not supported.
- **LLM latency:** `qwen2.5:14b` with 30s timeout may fall back to extractive answers on slower hardware.
- **`query_type` not wired:** Classified but not yet used to adjust retrieval `k` or grading strictness.
- **`seen_chunk_ids` not wired:** Retries may re-fetch similar chunks.

---

## Extended Design Tradeoffs

| Decision | Rationale | Tradeoff |
|---|---|---|
| **ChromaDB** | Lightweight, persistent, local-first | Not optimized for very large corpora |
| **all-MiniLM-L6-v2** | Fast, no API key, good baseline | Lower quality than larger embedding models |
| **Ollama over cloud APIs** | No keys, no cost, fully local | Slower on CPU; model-dependent quality |
| **Deterministic fallback grading** | Graph never hard-fails on LLM outage | Less nuanced than LLM-only grading |
| **Extractive fallback generation** | Always context-grounded | Less fluent than synthesized answers |
| **Bounded retries** | Predictable latency | May miss answers fixable with more retries |
| **Structural citations** | Verifiable, stable across re-index | Does not reflect which chunks LLM actually used |
| **Incremental indexing** | Uploads survive corpus re-ingest | Orphan chunks from deleted files remain |
| **Thin graph nodes** | Business logic in services; nodes as adapters | More files, but testable boundaries |

---

*Express Analytics RAG Assistant — Technical Report*
