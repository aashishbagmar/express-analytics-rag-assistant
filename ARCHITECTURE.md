# RAG-Based Technical Documentation Assistant — System Architecture

> Design document for the Express Analytics AI/ML Engineer Intern take-home assignment.
> Prepared as a staff-level architecture proposal, optimized to maximize assignment score.

**Revision note (v2):** Reviewed critically as a senior architecture review pass. Fixed state-schema
inconsistencies (undefined fields referenced in prose), added reducer semantics for list fields,
closed edge cases in routing (empty index, retry de-duplication), hardened the API design
(async-safe graph invocation, health check, ingestion dedup, error envelope), and added an explicit
MVP-vs-bonus phasing plan given the 2-day time budget. See inline notes marked **[v2 fix]**.

---

## 1. Requirement Analysis

**Mandatory requirements (explicitly graded):**
- LangGraph `StateGraph` with 4 named nodes: Query Analysis → Retrieval → Document Grading → Generation
- At least one **conditional edge** driven by grading outcome (relevant → generate; irrelevant → rewrite & retry, with a **retry limit**)
- A well-thought-out **state schema** — called out twice in the PDF as "a core evaluation criterion"
- Document ingestion pipeline: load → chunk → embed → store in a vector DB (ChromaDB/FAISS)
- FastAPI service exposing exactly: `POST /query`, `POST /ingest`, `GET /documents`, `POST /feedback`
- Corpus of 3–5 technical documents included or fetchable via script
- README covering: overview, architecture diagram/description, setup, run instructions, example requests/responses, design decisions & tradeoffs, chunking/embedding rationale, assumptions, "what I'd improve"
- GitHub repo as submission artifact

**Evaluation criteria (inferred from emphasis + rubric language):**
1. Correctness/coherence of the LangGraph workflow and its conditional routing
2. Quality of the state schema (explicit retry counter, typed fields, no ad-hoc dict abuse)
3. Self-corrective behavior actually working (grading catches irrelevant docs, triggers rewrite, bounded retries — no infinite loops)
4. Citation grounding in generated answers
5. API correctness, error handling, validation, status codes
6. Ingestion design justification (chunk size/overlap rationale, not just defaults)
7. Documentation clarity and honest tradeoff discussion ("we value clear thinking... over feature completeness")
8. Bonus points: hallucination/groundedness check, web search fallback, memory, UI — explicitly secondary to core pipeline

**Deliverables checklist:** repo, README, working FastAPI app, corpus (or fetch script), reasoning write-up.

**Architecture requirements:** graph-based orchestration (not a linear chain), vector retrieval, LLM-based grading, grounded generation with citations, bounded self-correction loop.

**Key risk to avoid:** over-building bonus features while under-building the state schema/routing logic that's explicitly weighted highest. The PDF says twice to prioritize the core loop over extras.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Service                             │
│                                                                            │
│   GET  /health ────► liveness check                                     │
│   POST /ingest ───► Ingestion Pipeline ───► Vector Store (Chroma)         │
│   GET  /documents ─► Document Registry (metadata store)                  │
│   POST /feedback ──► Feedback Store (validates query_id, append-only)    │
│   POST /query ─────► LangGraph Compiled Graph (ainvoke, async-safe)      │
│                              │                                            │
└──────────────────────────────┼────────────────────────────────────────────┘
                                ▼
                     ┌────────────────────┐
                     │   LangGraph Runtime │
                     │   (StateGraph,      │
                     │  recursion_limit=N) │
                     └────────────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ Query Analysis │  (rewrite + classify query_type)
                        └───────┬───────┘
                                ▼
                        ┌───────────────┐
              ┌────────►│   Retrieval    │  (vector search, dedup vs seen_chunk_ids)
              │         └───────┬───────┘
              │                 ▼
              │         ┌────────────────┐
              │         │ Document Grading│  (batched LLM relevance judge)
              │         └───────┬────────┘
              │                 ▼
              │      ┌─────────────────────────┐
              │      │ conditional_edge:        │
              │      │ decide_generation_path   │
              │      │  no chunks retrieved  →  │───────► Fallback
              │      │    fallback (fail fast)  │        ("insufficient
              │      │  none relevant &         │         context" /
              │      │    retry_count<MAX  →    │         optional web
              │      │    transform_query ───┐  │         search) → END
              │      │  none relevant &         │  │
              │      │    retry_count==MAX  →   │  │
              │      │    fallback              │  │
              │      │  ≥1 relevant → generate   │  │
              │      └─────────────┬────────────┘  │
              │                    │                │
              │           ┌────────▼────────┐       │
              └───────────┤  transform_query │◄──────┘
                          │ (retry_count+=1) │
                          └──────────────────┘
                                │ (loops back to Retrieval)
                                ▼
                        ┌───────────────┐
                        │   Generation   │  (grounded answer + citations)
                        └───────┬───────┘
                                ▼
                     ┌─────────────────────────┐
                     │ conditional_edge (bonus): │
                     │ decide_hallucination_path │
                     │  grounded → END           │
                     │  not grounded &            │
                     │   regen_count<MAX →        │
                     │   generate (self-loop)     │
                     │  not grounded & exhausted →│
                     │   fallback (low-confidence) │
                     └─────────────────────────┘
                                │
                             END (answer, sources,
                                  confidence, retry_count,
                                  fallback_used, query_id)
```

**Component layers:**
- **API layer** (FastAPI): thin — validates input, invokes graph/ingestion, shapes response. No business logic here.
- **Orchestration layer** (LangGraph): owns all control flow, retries, routing.
- **Domain services**: `Retriever` (vector store wrapper), `Grader` (LLM judge), `Generator` (LLM answerer), `QueryAnalyzer`, `Ingestor`. Each is a plain, testable class the graph nodes call into — keeps nodes as thin adapters over real logic (important for code quality scoring and unit testing).
- **Persistence**: Chroma (or FAISS) for vectors + a lightweight metadata store (SQLite/JSON) for document registry and feedback, since `/documents` and `/feedback` need durable, queryable records independent of the vector index.

---

## 3. Data Flow

**Ingestion flow (offline / `/ingest`):**
1. Load raw docs (markdown/text/HTML from disk or URL fetch)
2. Normalize → split into semantically-aware chunks (heading-aware for markdown, recursive char splitter fallback)
3. Attach metadata to each chunk: `source_id`, `source_title`, `section_heading`, `chunk_index`, `char_span`
4. Embed chunks (batch) → upsert into vector store with metadata
5. Register document in the metadata store (id, filename/URL, ingestion timestamp, chunk count) so `GET /documents` doesn't need to query the vector store directly

**Query flow (online / `/query`):**
1. API receives `{question, session_id?}` → generates `query_id` → builds initial `GraphState` (with `max_retries`/`max_regen` passed as run config, not baked into persisted state) → invokes compiled graph via `ainvoke` (async-safe; see API section) with an explicit `recursion_limit` as a hard execution-level safety net
2. **Query Analysis**: rewrites/expands question, classifies `query_type` (`conceptual|how_to|troubleshooting|api_reference`). **[v2 fix]** This field must have a concrete downstream effect to justify its existence in state — it adjusts retrieval `k` (e.g., wider `k` for troubleshooting) and grading strictness (e.g., stricter relevance threshold for `api_reference`). If not implemented in the first pass, document it honestly in the README as a "planned but not wired up" field rather than leaving it silently unused.
3. **Retrieval**: similarity search (optionally hybrid with metadata filter by query type) → top-k chunks + scores + metadata **replace** `retrieved_docs` in state (not appended — each retrieval attempt is a fresh snapshot). **[v2 fix]** Chunk IDs already seen in prior attempts (`seen_chunk_ids`) are excluded or down-weighted so a retry with a similar rewritten query doesn't just re-fetch the same irrelevant chunks. **[v2 fix]** If retrieval returns zero chunks (e.g., empty/uninitialized index), this is distinguished from "chunks retrieved but graded irrelevant" and routed straight to `fallback` — no point burning a retry on an empty index.
4. **Document Grading**: **[v2 fix]** a single batched LLM call grades all k chunks at once via structured output (list of verdicts), not one LLM call per chunk — same self-corrective behavior, ~k× fewer LLM calls and lower latency, which matters once retries multiply the call count. Result populates `graded_docs` (filtered relevant list, **replaced** each pass) and `grading_results` (**appended** across the whole run for traceability); `relevant_count` is derived from `len(graded_docs)`, not a separately tracked field.
5. **Conditional edge** inspects `graded_docs`, `retrieved_docs`, and `retry_count`:
   - zero chunks retrieved → **Fallback** (fail fast, distinct from irrelevance)
   - relevant found → **Generation**
   - none relevant, `retry_count < MAX_RETRIES` → **transform_query** increments `retry_count`, generates a genuinely different reformulation (not appended blindly — checked against `previous_queries`), loops back to **Retrieval**
   - none relevant, retries exhausted → **Fallback** node (graceful "insufficient context" response, or bonus web search into `web_search_docs`)
6. **Generation**: builds a grounded prompt from `graded_docs` (falling back to `web_search_docs` if that's how fallback was resolved), produces `answer` + `citations` mapped to `source_id`
7. (Bonus) **Hallucination Grader**: checks `answer` against `graded_docs`; if unsupported and `regen_count < MAX_REGEN`, loops back to `generate` once with a stricter grounding instruction; if exhausted, proceeds to `fallback` with a low-confidence disclaimer rather than silently returning an ungrounded answer
8. Graph terminates → API serializes `{query_id, answer, sources[], query_type, retry_count, fallback_used, confidence, error}` back to client. **[v2 fix]** `confidence` and `error` are now explicit, defined state fields (see Section 6) rather than values that appeared in the response with no defined origin.

**Feedback flow:** `/feedback` writes `{query_id, rating, comment}` to the feedback store, after validating `query_id` exists in a short-lived query log (in-memory LRU or the same metadata store) — **[v2 fix]** rejecting feedback for unknown `query_id`s with a `404`, and treating duplicate submissions for the same `query_id` as an update rather than a silent double-count.

---

## 4. Folder Structure

```
express-analytics-rag-assistant/
├── README.md
├── requirements.txt / pyproject.toml
├── .env.example
├── data/
│   ├── corpus/                     # 3-5 source docs (md/html/txt)
│   └── fetch_corpus.py             # optional: script to pull docs from URLs
├── app/
│   ├── main.py                     # FastAPI app factory, router registration
│   ├── api/
│   │   ├── routes_health.py        # GET /health
│   │   ├── routes_query.py         # POST /query
│   │   ├── routes_ingest.py        # POST /ingest
│   │   ├── routes_documents.py     # GET /documents
│   │   └── routes_feedback.py      # POST /feedback
│   ├── schemas/                    # Pydantic request/response models
│   │   ├── query.py
│   │   ├── ingest.py
│   │   └── feedback.py
│   ├── graph/
│   │   ├── state.py                # GraphState TypedDict/Pydantic schema
│   │   ├── builder.py               # StateGraph construction, edges, compile()
│   │   ├── nodes/
│   │   │   ├── query_analysis.py
│   │   │   ├── retrieval.py
│   │   │   ├── grading.py
│   │   │   ├── generation.py
│   │   │   ├── fallback.py
│   │   │   └── hallucination_check.py   # bonus
│   │   └── router.py               # conditional edge functions
│   ├── services/
│   │   ├── embeddings.py
│   │   ├── vector_store.py         # Chroma/FAISS wrapper
│   │   ├── llm.py                  # provider-agnostic LLM client
│   │   ├── web_search.py           # bonus: Tavily/Serper fallback
│   │   └── document_registry.py    # metadata store for /documents
│   ├── ingestion/
│   │   ├── loaders.py               # file/URL loaders
│   │   ├── chunking.py              # splitting strategy
│   │   └── pipeline.py              # orchestrates load→chunk→embed→store
│   ├── storage/
│   │   ├── chroma_db/               # persisted vector index (gitignored)
│   │   └── feedback_store.py        # SQLite/JSON append log
│   └── core/
│       ├── config.py                # settings via env vars
│       └── logging.py
├── tests/
│   ├── test_chunking.py
│   ├── test_graph_routing.py         # asserts conditional-edge behavior + retry cap
│   ├── test_state_schema.py
│   └── test_api_endpoints.py
├── eval/
│   ├── golden_qa.json                # small hand-labeled Q&A set over the corpus
│   └── run_eval.py                   # runs golden set through /query, reports hit-rate
└── scripts/
    └── run_ingestion.py             # standalone CLI entrypoint for ingestion
```

**[v2 fix]** Added `eval/` — a small (8–15 item) golden Q&A set with expected source documents is cheap
to build and is disproportionately convincing evidence that the pipeline actually retrieves and answers
correctly, versus only demonstrating that the graph *runs*.

This separation matters for scoring: `graph/` isolates the LangGraph-specific orchestration the rubric cares about most, `services/` shows clean dependency boundaries (swap LLM/vector store without touching graph logic), and `tests/test_graph_routing.py` directly demonstrates the retry/routing logic works — a cheap, high-signal artifact for reviewers.

---

## 5. LangGraph Workflow

**Nodes:**
- `query_analysis` — entry point
- `retrieve`
- `grade_documents`
- `transform_query` (the "rewrite" step, kept as its own node rather than folded into retrieval, so it's visible in the graph diagram and independently testable)
- `generate`
- `hallucination_check` (bonus)
- `fallback` (terminal "insufficient context" / optional web-search-then-generate)

**Edges:**
- `START → query_analysis` (normal edge)
- `query_analysis → retrieve` (normal edge)
- `retrieve → grade_documents` (normal edge)
- `grade_documents → {conditional edge: decide_generation_path}`
  - all irrelevant AND `retry_count < MAX_RETRIES` → `transform_query`
  - all irrelevant AND `retry_count >= MAX_RETRIES` → `fallback`
  - at least one relevant → `generate`
- `transform_query → retrieve` (closes the self-correction loop)
- `generate → {conditional edge: decide_hallucination_path}` (bonus)
  - grounded → `END`
  - not grounded AND `regen_count < MAX_REGEN` → `generate` (regenerate with stricter grounding instruction)
  - not grounded AND exhausted → `fallback` (return answer with a low-confidence disclaimer, never silently fail)
- `fallback → END`

**[v2 fix] Execution-level safety net:** in addition to the logical `retry_count`/`regen_count` caps
enforced by the routers, the compiled graph is invoked with an explicit `recursion_limit` (e.g., 25,
comfortably above the worst-case path length of ~`3 + 2*MAX_RETRIES + MAX_REGEN` steps). This guards
against bugs in the counter logic itself causing a runaway graph — belt-and-suspenders, and cheap to
state explicitly in the README as evidence of defensive design.

**[v2 fix] Retry semantics, stated precisely:** `MAX_RETRIES = 2` means retrieval is attempted at most
**3 times total** (1 initial + 2 rewrites). This exact phrasing is used consistently in the README and
tests to avoid the classic off-by-one ambiguity between "2 retries" and "2 attempts."

This is deliberately the **Adaptive/Corrective RAG pattern** the PDF explicitly references as "closest to this assignment's architecture" — using it directly (with the added hallucination loop as a differentiator) signals to the reviewer that the design is grounded in the literature they pointed to, without copying a tutorial verbatim.

---

## 6. State Schema

This is the highest-leverage artifact for the score, so it should be explicit, typed, and minimal-but-complete (a `TypedDict` or Pydantic model, not a loose dict). **[v2 fix]** Every field below is one actually referenced consistently in Sections 3, 5, and 7 — the v1 draft had two fields (`relevant_count`, `confidence`) used in prose but missing from the schema itself; that inconsistency is fixed here.

- `query_id: str` — generated at graph start, returned to caller, used to correlate `/feedback` submissions and to validate feedback references a real query
- `question: str` — original user question, immutable, always kept for final citation/answer grounding
- `rewritten_query: str` — current query used for retrieval; **replaced** on each rewrite (not appended)
- `previous_queries: list[str]` — **append**-only history of tried rewrites, so `transform_query` can check it and avoid regenerating a near-duplicate failed query
- `query_type: Literal["conceptual","how_to","troubleshooting","api_reference"]` — must have a real downstream consumer (retrieval `k` / grading strictness) or be documented as not-yet-wired; never left decorative
- `retrieved_docs: list[DocChunk]` — raw retrieval output for the *current* attempt only; **replaced** each retrieval, not accumulated, so grading always operates on one clean batch
- `seen_chunk_ids: set[str]` — **[v2 fix]** **append**-accumulated across attempts; used by `retrieve` to exclude/down-rank chunks already seen in a prior attempt on this same query, so retries actually diversify results instead of re-fetching the same irrelevant chunks
- `graded_docs: list[DocChunk]` — subset of `retrieved_docs` marked relevant this attempt; **replaced** each pass; `relevant_count` is a derived value (`len(graded_docs)`), not a separately stored field, eliminating the risk of the two going out of sync
- `grading_results: list[GradeResult]` — **append**-accumulated `{chunk_id, verdict, rationale, attempt_number}` across the whole run — kept for traceability/debugging and for the response's transparency
- `retry_count: int` — increments each time grading fails and a rewrite is triggered; **this is the field the PDF explicitly asks about**. Starts at 0; capped by `max_retries` passed in via run config
- `regen_count: int` — bounded counter for the hallucination-repair loop, mirroring `retry_count`'s design (bonus)
- `answer: str | None` — generated answer, `None` until `generate` runs
- `citations: list[Citation]` — `{source_id, source_title, chunk_index}` mapped from `graded_docs` (or `web_search_docs`) actually used in the answer
- `is_grounded: bool | None` — hallucination-check verdict (bonus)
- `confidence: Literal["high","medium","low"] | None` — **[v2 fix]** newly defined; derived deterministically — `"high"` if generated on the first attempt with `is_grounded=True`, `"medium"` if it took retries/regeneration but still grounded, `"low"` if served via `fallback`. This gives the response field referenced in Section 3 an actual, auditable origin.
- `fallback_used: bool` — flags whether the response came from the "insufficient context"/web-search path, surfaced to the client for transparency
- `web_search_docs: list[DocChunk] | None` — **[v2 fix]** newly defined; populated only by the bonus fallback's web-search branch, kept separate from `graded_docs` so it's clear in traces which path actually supplied the generation context
- `error: str | None` — **[v2 fix]** newly defined; set by any node that catches an exception (embedding call failure, LLM timeout, vector store error) instead of letting it propagate as an unhandled 500; routes to `fallback` with a clear "temporarily unable to answer" message and is surfaced in the API's error envelope

**Reducer strategy (must be stated explicitly, or LangGraph's default merge behavior will silently corrupt loop semantics):**
- **Replace-on-write** (each node overwrites): `rewritten_query`, `retrieved_docs`, `graded_docs`, `answer`, `is_grounded`, `confidence`, `fallback_used`, `error`
- **Append-only** (`Annotated[list, operator.add]` or equivalent reducer): `previous_queries`, `seen_chunk_ids`, `grading_results`
- Scalars that only ever increment (`retry_count`, `regen_count`) use a simple integer add reducer or are just reassigned `+1` by the one node responsible for incrementing them (`transform_query`, `generate`'s regen branch) — never incremented inside a router, since routers must stay pure/side-effect-free.

**Config vs. state:** `max_retries` and `max_regen` are **run-time configuration** passed into `graph.ainvoke(state, config={...})`, not persisted fields inside the checkpointed state — this avoids mixing static, per-request-overridable limits into the same object that LangGraph checkpoints/serializes on every superstep. They are echoed into the final response for transparency, but their source of truth is the invocation config, not the state graph.

Design rationale worth stating in the README: separating `retrieved_docs` from `graded_docs` (rather than mutating in place), and separating replace-semantics fields from append-semantics fields, preserves an audit trail across the graph and makes each node a pure function of state — this is exactly the "think carefully about state schema" signal the rubric is fishing for, and pre-empting the reducer-semantics question head-on is a strong signal of real (not superficial) LangGraph fluency.

---

## 7. Retry Mechanism

Two independent, symmetric bounded loops, both driven by counters *inside the state* (not global variables — critical for LangGraph's checkpointing/concurrency correctness):

**Retrieval retry loop** (self-corrective core requirement):
- `MAX_RETRIES` (e.g., 2) passed as run config at graph invocation (see Section 6 config-vs-state note), meaning **at most 3 total retrieval attempts** (1 initial + 2 rewrites) — stated precisely to avoid off-by-one ambiguity
- On each `grade_documents → transform_query` transition, `retry_count += 1` inside the `transform_query` node itself (not in the router — routers should stay side-effect-free predicate functions)
- `transform_query` uses `previous_queries` to generate a genuinely different reformulation (e.g., broaden scope, extract key entities, switch phrasing) rather than naively re-asking the same question — prevents pointless identical loops
- **[v2 fix]** `retrieve` additionally excludes/down-ranks chunk IDs already present in `seen_chunk_ids`, so even if the rewritten query is semantically close to the original, the retry is structurally forced to surface *different* candidates rather than silently re-grading the same irrelevant chunks and wasting the retry budget
- **[v2 fix]** Router (`decide_generation_path`) is a pure function with three branches, not two: reads `retrieved_docs` (empty → `"fallback"` immediately, no point retrying an empty index), `graded_docs`/`retry_count` (relevant → `"generate"`; none relevant and under cap → `"transform_query"`; none relevant and at cap → `"fallback"`) — returns a string label, LangGraph maps labels to nodes via `add_conditional_edges`
- Hard ceiling guarantees graph termination — no possibility of infinite loop even under LLM grading flakiness
- **[v2 fix]** As a second, independent safety net, the compiled graph is invoked with an explicit `recursion_limit` (e.g., 25) — protects against a bug in the counter logic itself, not just against the logical case the counters are designed to catch

**Hallucination/regeneration retry loop** (bonus, same pattern):
- `MAX_REGEN` (e.g., 1) — deliberately smaller, since regenerating with the same context rarely helps twice; after exhaustion, degrade gracefully rather than loop forever
- Same shape: counter in state, increment in the node, pure predicate in the router

**Why counters-in-state instead of alternatives:** LangGraph nodes are meant to be pure transformations over state; putting retry counters in external mutable variables breaks replayability/checkpointing and makes the graph non-deterministic under LangGraph's built-in persistence. Keeping all control-flow data in state also makes the entire retry history inspectable in the final response (`retry_count`, `previous_queries`) — useful both for debugging and for demonstrating self-correction actually happened, which is good evidence for the reviewer.

---

## 8. API Design Fixes **[v2, new]**

The original doc mentioned the API only as a thin pass-through layer; a critical review surfaced
concrete gaps against the PDF's own "consider error handling, input validation, meaningful HTTP status
codes" requirement:

- **`GET /health`** added — trivial to implement, commonly probed by graders/CI, costs nothing.
- **Async-safe graph invocation**: `/query` calls `graph.ainvoke(...)`, not the sync `.invoke()`, inside
  the `async def` route handler — otherwise a long-running graph traversal blocks the FastAPI event loop
  and starves concurrent requests. If any node uses a sync-only client, it's wrapped via
  `run_in_threadpool` rather than called directly.
- **Ingestion dedup**: `/ingest` hashes incoming document content (e.g., SHA-256) before chunking;
  re-ingesting an unchanged file/URL is a no-op (or explicit re-index if content changed), preventing
  silent duplicate chunks from accumulating in the vector store across repeated calls.
- **Large ingestion jobs**: for uploads likely to exceed a reasonable request timeout, `/ingest` returns
  `202 Accepted` with a job/document id immediately and processes ingestion in a background task;
  `GET /documents` reflects `status: "processing" | "ready" | "failed"` per document. For a small 3–5
  doc corpus this may be unnecessary in practice, but the design should state the tradeoff explicitly
  rather than silently assuming synchronous ingestion is always fine.
- **Feedback validation**: `/feedback` returns `404` if `query_id` doesn't match a known prior query
  (tracked via a small in-memory LRU or the same metadata store used for `/documents`); a second
  submission for the same `query_id` updates the existing record rather than silently duplicating it.
- **Consistent error envelope**: all endpoints return errors as `{error: {code, message, detail?}}` with
  correct status codes (`400` bad input, `404` unknown resource, `422` validation, `500` unexpected) —
  stated once as a cross-cutting convention (e.g., a FastAPI exception handler) rather than left
  ad hoc per route.
- **Embedding model/version pinning**: the embedding model name/version used at ingestion time is
  recorded in the document registry; `/ingest` (or a startup check) warns/errors if a query-time
  embedding model differs from what was used to build the index, since silent drift here degrades
  retrieval quality without any visible error.

---

## 9. Implementation Phasing (MVP → Bonus) **[v2, new]**

A critical objection to the v1 draft: 7 nodes, 2 independent retry loops, a document registry, a
feedback store, and a hallucination checker is a lot of surface area for a 2-day assignment whose own
instructions say twice to prioritize the core loop over extras. Making the cut line explicit pre-empts
that objection and demonstrates the prioritization judgment the rubric is actually looking for.

**Day 1 — must ship (this alone is a complete, scoreable submission):**
- Ingestion pipeline (load → chunk → embed → store) + `run_ingestion.py` script
- Core 4-node graph: `query_analysis → retrieve → grade_documents → generate`, with the single
  required conditional edge (relevant → generate; irrelevant + under retry cap → `transform_query` →
  retrieve; irrelevant + at cap → `fallback`)
- State schema exactly as specified in Section 6, minus the bonus-only fields (`is_grounded`,
  `regen_count`, `web_search_docs`)
- `POST /query`, `POST /ingest`, `GET /documents`, `GET /health`
- A 3–5 document corpus + README with chunking rationale, setup, and example requests

**Day 2 morning — should ship if Day 1 finished on schedule:**
- `POST /feedback` + validation
- Error envelope + input validation across all routes
- `eval/golden_qa.json` + `run_eval.py` (cheap, high-signal rigor demonstration)
- Ingestion dedup by content hash

**Day 2 afternoon — bonus only if time remains, and cut without regret if not:**
- Hallucination-check node + its own conditional edge/regen loop
- Web search fallback (`web_search_docs`, Tavily/Serper)
- Conversation memory / session history
- Minimal Streamlit/Gradio UI

This ordering is the honest, defensible answer if a reviewer asks "why doesn't the hallucination check
have more polish" — the README should state this phasing explicitly rather than leaving the reviewer to
guess whether incomplete bonus work reflects poor planning or correct prioritization.

---

## 10. Why This Design Scores Highly

1. **Directly matches the reference architecture the PDF points to** (Adaptive/Corrective RAG) while adding a differentiator (hallucination loop) — shows the candidate read the recommended material and extended it thoughtfully rather than copying a tutorial.
2. **State schema is explicit, typed, and answers the two questions the PDF asks verbatim** ("what data flows between nodes", "how do you track retries") — this is called out twice as a core evaluation criterion, and the design gives a direct, defensible answer with a named field and a clear increment site.
3. **Conditional routing is a pure, testable predicate function separate from the side-effecting nodes** — demonstrates LangGraph fluency beyond just "it works," which distinguishes senior-quality submissions.
4. **Retry logic is provably bounded** — no infinite-loop risk, explicitly guards against the most common self-corrective RAG failure mode, and is easy to unit-test (`test_graph_routing.py` exercising `retry_count` at boundary values).
5. **Clean separation of API / orchestration / services / ingestion** — reviewers skimming the repo can immediately locate the graph logic they care about most, and the architecture is swappable (LLM provider, vector store) without touching the graph, which reads as production judgment rather than notebook code.
6. **Citations are structurally guaranteed**, not just prompted for — because `graded_docs` metadata flows through the whole pipeline, `generate` can map answer segments to `source_id` deterministically rather than hoping the LLM remembers to cite.
7. **Core-first prioritization matches the PDF's own stated grading philosophy** ("prioritize the core RAG pipeline + document grading logic over additional features," "we value clear thinking... over feature completeness") — bonus features (hallucination check, web fallback) are designed as clean optional extensions of the same pattern rather than bolted-on afterthoughts, so they can be cut under time pressure without destabilizing the core deliverable.
8. **Traceability fields** (`grading_results`, `retry_count`, `fallback_used`, `query_id`) turn every response into a self-documenting trace of the self-corrective process — this makes it trivial to write a compelling README walkthrough with a real example showing the retry loop firing, which is exactly the kind of evidence a reviewer wants without having to dig through logs.
9. **API surface maps 1:1 to the mandatory endpoint table** with no scope creep, and the metadata/feedback stores are decoupled from the vector store so `/documents` and `/feedback` are fast and don't require a full vector-store round trip — shows attention to the "consider error handling, input validation, meaningful status codes" note, now backed by a concrete error-envelope convention, feedback validation, and async-safe invocation (Section 8).
10. **Folder structure signals engineering maturity** (tests directory specifically targeting graph routing/state, config via env vars, ingestion as both a standalone script and API endpoint per the PDF's explicit "can be a standalone script or run at startup" allowance) — low-risk, high-signal choices for a 2-day take-home graded partly on documentation and clarity of thought.
11. **[v2]** **The state schema is now internally consistent** — every field referenced in the data-flow narrative and API response is actually declared in the schema, with explicit reducer semantics per field. This closes the single most likely "gotcha" a LangGraph-literate reviewer would probe (how do your list fields behave across loop iterations?), and turns a potential deduction into a demonstrated strength.
12. **[v2]** **Retry loops are diversity-guaranteed, not just count-bounded** — `seen_chunk_ids` ensures a retry actually changes the retrieved set, so the self-corrective mechanism does real work rather than looping through cosmetically different but substantively identical attempts.
13. **[v2]** **Explicit MVP-vs-bonus phasing (Section 9) directly answers the take-home's own stated grading philosophy** with a concrete cut line, which is strong evidence of the "clear thinking over feature completeness" the PDF explicitly says it values — and gives an honest, prepared answer if a reviewer asks why any bonus feature is incomplete.
14. **[v2]** **A minimal golden Q&A eval set is cheap insurance against the most damaging failure mode of any take-home RAG demo** — the reviewer asking a question live and getting a wrong/ungrounded answer. Even 8–10 hand-checked examples materially reduce that risk and give the README a concrete "it works" artifact beyond a screenshot.
