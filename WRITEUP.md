# Design Write-Up — Express Analytics RAG Assistant

Assignment deliverable: thought process, architecture reasoning, workflow component choices, assumptions, chunking/embedding strategy, and what I would improve with more time.

This document follows the **same phased build order** used during implementation (each section maps to a directed prompt/decision). For setup and API usage see [README.md](README.md). For node-level traces see [REPORT.md](REPORT.md). Full design spec: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## How this project was built (prompt → decision map)

| Phase | Your direction | Resulting decision |
|------|----------------|-------------------|
| 1 | Analyze PDF; design architecture for max score | Adaptive/Corrective RAG on LangGraph; saved as `ARCHITECTURE.md` |
| 2 | Critical architecture review | v2 fixes: reducers, `confidence`/`error` in state, MVP phasing, `GET /health` |
| 3 | `state.py` only — typed, documented | `GraphState` as core evaluation artifact; every field has producer/consumer |
| 4 | Full skeleton, no business logic | Thin API / graph / service boundaries from day one |
| 5 | Add `health.py`, keep `transform_query.py` | Rewrite-and-retry as a **visible dedicated node**, not hidden in grading |
| 6–7 | Ingestion only; sanity-test metadata | Load → chunk → metadata; no embeddings/graph yet |
| 8 | `embeddings.py` — MiniLM, lazy singleton | Local, key-free embeddings boundary |
| 9–10 | Chroma `vector_store.py`; Phase 6 smoke test | Persistent Chroma; `{content, metadata, score}` contract |
| 11–12 | `query_analysis.py`; natural rewrites | Classify + rewrite for retrieval; avoid keyword stuffing |
| 13 | `retrieval.py` — thin node | Business logic in `VectorStoreService`, not the node |
| 14–15 | `grading.py`; then **LLM grader required by PDF** | Ollama primary grader + heuristic fallback |
| 16 | `router.py` — pure predicates | `decide_generation_path` + bonus `decide_hallucination_path` |
| 17 | `generation.py` — grounded + citations | Context **only** from `graded_docs`; structural citations |
| 18 | `builder.py` — wire full graph | Bounded retry loop; `generation → END` (hallucination loop deferred) |
| 19 | Verify `transform_query` behavior | `retry_count++`, unique rewrites, deduped `previous_queries` |
| 20 | Demo corpus + E2E before FastAPI | `data/corpus/` FastAPI docs; README examples/screenshots |
| 21 | `fallback.py` + `hallucination_check.py` | Safe terminal path; bonus groundedness node (tested, not wired) |
| 22 | Fix duplicate `previous_queries` | Normalized dedup on append |
| 23 | FastAPI layer only | Thin routes; graph invoke; Pydantic schemas |
| 24 | Full pytest suite | 63 → **74** tests across all layers |
| 25 | Feedback → `FeedbackStore` | API → storage → JSONL (not file writes in routes) |
| 26 | Professional README + reviewer audit | Compliance table, validation traces, honest gaps |
| 27 | Implement `LLMService.generate_answer()` | Real Ollama synthesis; extractive fallback preserved |
| 28 | Streamlit `frontend.py` | Local demo without React/DB |
| 29 | PDF + DOCX in loaders | PyMuPDF + python-docx; metadata preserved |
| 30 | Upload workflow + UI sections | `POST /upload`; pending queue; reuse ingestion pipeline |
| 31 | Final assignment audit | Documented ⚠️ items (hallucination not wired, etc.) |
| 32 | Incremental indexing fix | `replace_documents_for_sources()` — uploads survive re-ingest |
| 33 | README polish, screenshots, known limitation | Submission-ready docs; grounding tradeoff stated explicitly |

---

## 1. Architecture design (Prompt 1)

**Your ask:** Read *DS - Intern Assignment.pdf*; extract mandatory requirements, evaluation criteria, deliverables; design complete system architecture (diagram, data flow, folder structure, LangGraph workflow, state schema, retry mechanism); maximize score; no code.

**My reasoning:**

- The PDF weights **LangGraph control flow** and **state schema** highest — not feature count.
- I chose **Adaptive/Corrective RAG** (retrieve → grade → generate *or* rewrite-and-retry *or* fallback) because it matches the assignment’s “self-corrective” wording and the reference architecture.
- **Graph-based orchestration** (not a linear chain) is non-negotiable: conditional edges and bounded loops are the product.
- **FastAPI stays thin** — validate, invoke graph, serialize response. No RAG logic in routes.
- **Bonus features** (hallucination loop, web search, UI) were designed as **optional extensions** of the same pattern so they could be cut under time pressure without breaking core.

Saved as `ARCHITECTURE.md`.

---

## 2. Critical architecture review (Prompt 2)

**Your ask:** Review for missing requirements, state schema bugs, routing issues, FastAPI gaps, retrieval issues, reviewer objections; update architecture.

**Fixes applied (v2):**

| Issue | Fix |
|-------|-----|
| `relevant_count` / `confidence` in prose but not in schema | Added `confidence`, derive from `len(graded_docs)` not a separate counter |
| List fields could accumulate across retries | Explicit **replace vs append** reducers (`retrieved_docs` replace; `previous_queries` append) |
| Retries could re-fetch same chunks | Designed `seen_chunk_ids` (append reducer) — scaffolded, not fully wired in retrieval |
| `max_retries` in state vs config | Caps in **run config** (`Settings`); only `retry_count` in state |
| No error path | `error` field → router → `fallback` |
| API under-specified | `GET /health`, async-safe invoke, feedback validation, error envelope |
| Over-building risk | **Section 9 MVP phasing**: Day 1 core graph; Day 2 afternoon = hallucination loop *only if time* |

This review is why the submission can honestly say some bonus fields exist in schema but are not wired.

---

## 3. State schema (Prompt 3)

**Your ask:** `graph/state.py` only — TypedDict, fields exactly required by workflow, docs for why / who writes / who reads.

**My reasoning:**

- State is a **first-class deliverable** — reviewers probe “what flows between nodes” and “how retries are tracked.”
- `question` stays immutable; `rewritten_query` changes on retry — answers stay tied to user intent.
- `retry_count` incremented only in `transform_query`, never in the router — **routers stay pure**.
- `grading_results` append for auditability; `graded_docs` replace each pass.
- Bonus fields (`is_grounded`, `regen_count`, `web_search_docs`) included so the hallucination loop has a home without refactoring later.

---

## 4. Project skeleton (Prompt 4)

**Your ask:** All folders/files, docstrings only, no business logic, purpose/responsibilities/interfaces per file.

**My reasoning:**

- Skeleton-first forces **clear module boundaries** before implementation drift.
- Every file documents its public contract — reviewers can skim `app/graph/`, `app/services/`, `app/ingestion/` independently.
- Tests, scripts, eval, and data directories included for production-shaped layout.

---

## 5. `health.py` + dedicated `transform_query` (Prompt 5)

**Your ask:** Add `GET /health`; keep `transform_query.py` as its own node for reviewer clarity.

**My reasoning:**

- Health endpoint costs minutes, helps local/CI checks.
- **Rewrite-and-retry is assignment-required** — hiding it inside grading would make the graph harder to grade. A dedicated node makes the retry loop obvious in `builder.py` and diagrams.

Node layout kept as:

```text
query_analysis → retrieval → grading → [router]
                  ↑            ↓
            transform_query ←──┘
generation | fallback
hallucination_check (bonus, file present)
```

---

## 6–7. Ingestion layer (Prompts 6–7)

**Your ask:** Implement `loaders.py`, `chunking.py`, `pipeline.py` only — load, split, metadata, return LangChain `Document`s; **no** LangGraph, embeddings, vector store, FastAPI, LLM. Sanity-test metadata shape.

**My reasoning:**

- **Ingestion is isolated** so chunking/metadata can be unit-tested without the graph.
- One bad file must not fail the run — skip with warnings.
- Deterministic `source_id` (SHA-1 of resolved path) enables stable citations and later incremental re-index.
- Verified metadata: `source_id`, `source_title`, `source_path`, `chunk_index`, `section_heading`.

---

## 8. Embeddings service (Prompt 8)

**Your ask:** `sentence-transformers`, `all-MiniLM-L6-v2`, lazy load, shared singleton, `embed_documents` / `embed_query` / `model_id`; no vector store or graph code.

**My reasoning:**

- **Same model for documents and queries** — required for meaningful Chroma similarity.
- Lazy + shared instance — fast API startup, no reload per request.
- Single service boundary so ingestion, vector store, and future endpoints never import `SentenceTransformer` directly.

---

## 9–10. Vector store (Prompts 9–10)

**Your ask:** ChromaDB persistent; `build_index`, `add_documents`, `similarity_search(k=5)`, `list_documents`, `document_count`; return `{content, metadata, score}`; smoke test with dependency injection query.

**My reasoning:**

- Chroma = local, persistent, no API keys — fits take-home on a reviewer laptop.
- Metadata preserved end-to-end for **structural citations** in generation.
- `similarity_search` shape matches what `retrieval.py` and tests expect — contract before graph wiring.

---

## 11–12. Query analysis (Prompts 11–12)

**Your ask:** Rewrite, expand, classify (`conceptual` | `how_to` | `troubleshooting` | `api_reference`); preserve intent; deterministic where possible. **Then:** prefer natural rewrites over keyword stuffing.

**My reasoning:**

- Original `question` preserved for final answer; `rewritten_query` is what retrieval uses.
- After your feedback, rewrites read like *“How does FastAPI validate request bodies using Pydantic models?”* instead of appended keyword lists — better for reviewers and retrieval.
- `query_type` is populated for future tuning (wider `k` for troubleshooting, etc.) — **classified but not yet wired to retrieval parameters** (documented honestly).

---

## 13. Retrieval (Prompt 13)

**Your ask:** Thin node; read `rewritten_query`, write `retrieved_docs`; delegate to `VectorStoreService`.

**My reasoning:**

- Nodes are **adapters**; vector search logic stays in `VectorStoreService` for reuse by `/ingest`, `/upload`, and tests.
- Default `k=5` balances context size vs precision for technical docs.

---

## 14–15. Document grading (Prompts 14–15)

**Your ask:** Self-corrective core — per-chunk relevant/irrelevant; `graded_docs` + `grading_results`. **Then:** PDF says *“Use an LLM to grade”* — upgrade to Ollama structured output; keep heuristic as fallback.

**My reasoning:**

- Grading is the **self-correction gate** — irrelevant chunks never reach generation.
- Primary path: `LLMService.grade_relevance()` → `{verdict, confidence, reason}` JSON from Ollama.
- Fallback: vector score + lexical overlap when Ollama is down/slow/malformed — graph never hard-fails on grading.
- This directly addresses your concern that a pure heuristic grader would fail a literal reading of the assignment.

---

## 16. Router (Prompt 16)

**Your ask:** Pure functions, no mutation:

```text
Generation:  graded_docs > 0 → generate | retry_count < max_retries → transform_query | else fallback
Hallucination: is_grounded → end | regen_count < max_regen → generate | else fallback
```

**My reasoning:**

- Routing logic is **unit-testable** (`tests/test_graph_routing.py`) separate from side-effecting nodes.
- `max_retries` / `max_regen` from `Settings`, not mutated in routers.
- `decide_hallucination_path` implemented for bonus loop; wired only when builder connects `hallucination_check`.

---

## 17. Generation (Prompt 17)

**Your ask:** Answer only from `graded_docs`; citations from metadata; deterministic `confidence` from doc count, `retry_count`, `fallback_used`; raise if empty `graded_docs` (router’s job to prevent that).

**My reasoning:**

- **Grounding guarantee:** prompt/context built exclusively from graded chunks.
- Citations derived structurally (`source_id`, `source_title`, `chunk_index`) — not LLM free-text.
- Empty `graded_docs` → `GenerationNodeError` — signals wiring bug, not silent degradation.

---

## 18. Graph builder (Prompt 18)

**Your ask:** Wire START → query_analysis → retrieval → grading → conditional → generation | transform_query loop | fallback; `build_graph()`; compile.

**My reasoning:**

- Retry loop: `grading → transform_query → retrieval → grading` until `retry_count >= max_retries` (default 2 → **3 retrieval attempts total**).
- `recursion_limit=25` on invoke as execution-level safety net.
- **Intentional cut:** `generation → END` directly; hallucination loop deferred per MVP phasing (node + router exist, not connected).

---

## 19. Transform query verification (Prompt 19)

**Your ask:** Confirm `transform_query` increments `retry_count`, produces new `rewritten_query`, appends to `previous_queries`, avoids repeats.

**Outcome:** Implemented and later hardened (Prompt 22) with normalized dedup so retry history does not contain duplicate strings.

---

## 20. Demo corpus before FastAPI (Prompt 20)

**Your ask:** `data/corpus/` with 5 FastAPI markdown files; E2E `graph.invoke` for README examples and screenshots.

**Outcome:** Corpus drives all positive-path demos (*“How does FastAPI validate request bodies?”* → citations to `request_models`).

---

## 21. Fallback + hallucination check (Prompt 21)

**Your ask:**

- **fallback:** safe message, `fallback_used=True`, `confidence=low`, never echo `error` to user.
- **hallucination_check:** deterministic groundedness vs `graded_docs`; no LLM; output `is_grounded`.

**My reasoning:**

- Fallback is the **honest terminal** when context is insufficient — no ungrounded guess.
- Hallucination check uses lexical term overlap (75% support ratio) — cheap, reproducible; **implemented and tested but not wired** to avoid destabilizing core before submission (see Prompt 33 / audit).

---

## 22. Transform query dedup (Prompt 22)

**Your ask:** Fix duplicate entries in `previous_queries`; normalize (lowercase, trim) before compare.

**Outcome:** Retries produce genuinely different rewrite sequences in traces.

---

## 23. FastAPI layer (Prompt 23)

**Your ask:** `main.py`, `/query`, `/ingest`, `/documents`, `/feedback`, `/health`; Pydantic schemas; invoke compiled graph on `/query`.

**My reasoning:**

- API maps 1:1 to assignment endpoint table (+ health as pragmatic extra).
- `/query` returns `query_id`, `answer`, `citations`, `confidence`, `retry_count`, `fallback_used` — traceability for reviewers.

---

## 24–25. Tests + feedback storage (Prompts 24–25)

**Your ask:** Production pytest suite across all layers; move feedback persistence to `FeedbackStore` not raw writes in API.

**Outcome:** **74 tests passing**; clean API → storage → `feedback.jsonl` layering.

---

## 26–27. README + reviewer audit + Ollama generation (Prompts 26–27)

**Your ask:** Submission README with requirement mapping, diagrams, validation; critical reviewer pass; implement real `generate_answer()` in `llm.py`.

**My reasoning:**

- Generation was falling back to extractive-only because `generate_answer` was stubbed — fixed with same Ollama stack as grading (JSON answer, temperature 0, timeout, `LLMServiceError` → extractive fallback in `generation.py`).
- Audit documented honest ⚠️ items instead of hiding them.

---

## 28. Streamlit frontend (Prompt 28)

**Your ask:** `frontend.py`, Streamlit only, POST `/query`, show answer/metadata/citations, sample questions, error handling.

**Later prompts:** Polish UI for screenshots; fix invisible text (light theme in `.streamlit/config.toml`).

**Outcome:** Three-tab UI — Upload & Ingest, Indexed Documents, Chat — for local demo without a separate frontend stack.

---

## 29–30. PDF/DOCX + upload workflow (Prompts 29–30)

**Your ask:** PyMuPDF + python-docx in loaders; then upload API + Streamlit flow reusing **existing** ingestion/chunk/embed/store — no duplicate pipeline; 10 MB limit; pending queue → ingest → queryable.

**My reasoning:**

- Format support belongs in **loaders**; chunking unchanged — single pipeline for corpus and uploads.
- `POST /upload` saves to `data/uploads/` then calls same `IngestionPipeline` + `VectorStoreService.add_documents()`.

---

## 31–32. Final audit + incremental indexing (Prompts 31–32)

**Your ask:** Full assignment audit (no changes); then fix **uploads wiped on `POST /ingest`** — preserve uploads, incremental corpus refresh, tests for re-ingest scenario.

**Root cause:** `build_index()` deleted entire Chroma collection.

**Fix:** `replace_documents_for_sources()` — delete/replace only corpus `source_id`s; upload `source_id`s untouched.

**Test:** `tests/test_ingest_incremental.py` — ingest → upload → re-ingest → upload still searchable.

---

## 33. Documentation & submission polish (Prompts 33+)

**Your asks included:**

- Known limitation: answers only from **retrieved context**, not whole-document summarization unless retrieved — intentional anti-hallucination tradeoff.
- Screenshots in `docs/screenshots/`.
- Concise README (~250–350 lines) + `REPORT.md` for depth.
- Analysis: hallucination check implemented but not connected — **recommendation: do not wire before submit** (core complete; risk of false fallbacks).

---

## Chunking strategy (assignment requirement)

Directed in ingestion phase; implemented in `app/ingestion/chunking.py`:

| Choice | Value | Why |
|--------|-------|-----|
| Splitter | `RecursiveCharacterTextSplitter` | Standard for technical prose; respects structure before arbitrary cuts |
| `chunk_size` | **800** characters | ~1–2 paragraphs of docs; precise enough for retrieval, large enough for context |
| `chunk_overlap` | **100** characters | Reduces splits through definitions, lists, and code at boundaries |
| Separators (priority) | `#` headings → `\n\n` → `\n` → `. ` → space | Markdown-aware; `section_heading` from nearest preceding `#` heading |
| Metadata | `source_id`, `source_title`, `source_path`, `chunk_index`, `section_heading` (+ `filename`, `file_type`, `page_count` for PDF) | Citations and `/documents` without re-parsing chunks |
| Formats | `.md`, `.txt`, `.html`, `.htm`, `.pdf`, `.docx` | Assignment + upload extensions |
| Failure mode | Skip bad/empty files with warnings | One corrupt file must not fail batch ingest |

**Chunk IDs:** `{source_id}:{chunk_index}:{content_hash}` — avoids duplicate upserts on re-index.

---

## Embedding strategy (assignment requirement)

Directed in embeddings phase; implemented in `app/services/embeddings.py`:

| Choice | Value | Why |
|--------|-------|-----|
| Library | `sentence-transformers` | Local inference, no API keys, fits take-home |
| Model | **`all-MiniLM-L6-v2`** | Fast CPU/GPU baseline; 384-dim; strong for short English technical Q&A |
| Loading | Lazy on first call | Faster process startup |
| Instance | Shared per model name (thread-safe cache) | Load once per process |
| API | `embed_documents()` + `embed_query()` + `model_id()` | Same model both sides; enables future ingest/query compatibility checks |
| Storage | Chroma persistent at `app/storage/chroma_db` | Vectors + metadata co-located |

**Tradeoff:** MiniLM is weaker than larger embedding models (`bge`, `e5`) on nuanced semantic match — acceptable for local demo; would upgrade with more time.

---

## Assumptions

1. **English technical documentation** (demo corpus: FastAPI docs in `data/corpus/`).
2. **Local deployment** on reviewer machine — not multi-tenant production.
3. **Ollama** (`qwen2.5:14b` default) recommended for grading + generation; heuristics/extractive paths work without it.
4. **Text-based PDF/DOCX** — no OCR for scanned documents.
5. Questions are answerable from **top-k retrieved chunks** (`k=5`), not full-document synthesis in one pass.
6. Corpus via `POST /ingest`; ad-hoc files via `POST /upload` or Streamlit.
7. Single Chroma collection; `source_id` scoped incremental re-ingest (orphan chunks from deleted corpus files may remain).

---

## What I would improve with more time

Prioritized by your own audit and phasing notes:

1. **Wire hallucination check** after generation with bounded regen + tune lexical threshold for Ollama paraphrases.
2. **Implement `seen_chunk_ids`** in retrieval so retries diversify chunks, not just queries.
3. **Wire `query_type`** to retrieval `k` and grading strictness.
4. **Cross-encoder reranking** before LLM grading.
5. **Embedding model version pinning** — warn if ingest vs query models differ.
6. **Golden Q&A eval** (`eval/run_eval.py` scaffold) for regression before demos.
7. **Streaming `/query`** for long Ollama answers.
8. **Production hardening** — auth, rate limits, higher LLM timeout, align `chroma_persist_dir` config with storage path.
9. **OCR** for scanned PDFs.
10. **Auto-prune orphan chunks** when corpus files are removed.

---

## Known limitations (stated for reviewers)

- **Retrieval-grounded answers only** — does not summarize an entire large upload unless relevant chunks are retrieved; intentional grounding tradeoff.
- **Hallucination check, `seen_chunk_ids`, `query_type` tuning** — designed/scaffolded, not fully integrated in the live graph.
- **Top-k bounds context** — very large documents may need multiple targeted questions.
- **Hallucination loop disconnected** — core-first prioritization per `ARCHITECTURE.md` Section 9; node + router + tests exist as bonus evidence.

---

## Current workflow (as shipped)

```text
START → query_analysis → retrieval → grading → decide_generation_path
         ├─ graded_docs > 0     → generation → END
         ├─ retry_count < 2     → transform_query → retrieval (loop)
         └─ else                → fallback → END
```

**Stack:** LangGraph · ChromaDB · Ollama · FastAPI · Streamlit · 74 tests.

---

*Express Analytics — AI/ML Engineer Intern Assignment*
