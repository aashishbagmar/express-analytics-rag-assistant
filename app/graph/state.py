"""
LangGraph state schema for the RAG Technical Documentation Assistant.

This module defines the single source of truth for all data that flows between
graph nodes: query_analysis -> retrieve -> grade_documents -> (transform_query loop)
-> generate -> (hallucination_check loop, bonus) -> fallback -> END.

Design rules encoded here (see ARCHITECTURE.md, Section 6):
1. Every field is consumed by at least one node and produced by exactly one node
   (or a small, explicitly listed set of nodes) - no unused/decorative fields.
2. Fields are split into two reducer categories:
   - "replace" fields: a node overwrites the previous value outright. This is the
     TypedDict default behavior in LangGraph (no Annotated reducer needed).
   - "append" fields: history must accumulate across retry/regen loop iterations,
     so they are declared with `Annotated[list, operator.add]` to opt into
     additive merging instead of the default overwrite.
3. Retry/regen limits (max_retries, max_regen) are intentionally NOT part of this
   state. They are run-time configuration passed via `graph.ainvoke(state, config=...)`
   so that static limits never get mixed into the object LangGraph checkpoints on
   every superstep. Only the counters that track progress against those limits
   (retry_count, regen_count) live in state.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict


# --------------------------------------------------------------------------- #
# Supporting structures
#
# These are not top-level state fields themselves; they describe the shape of
# items stored inside the list-typed state fields below (retrieved_docs,
# graded_docs, web_search_docs, grading_results, citations).
# --------------------------------------------------------------------------- #


class DocChunk(TypedDict):
    """
    A single retrieved chunk plus its source metadata.

    Produced by: retrieve (from vector search) and, for the bonus fallback path,
    by the web search node (fetch_web_results), which populates web_search_docs
    using this same shape so downstream nodes (grade_documents, generate) can
    treat both sources uniformly.
    Consumed by: grade_documents (reads `text` to judge relevance), generate
    (reads `text` + `source_id` to build the grounded prompt and citations).
    """

    chunk_id: str          # stable id, e.g. f"{source_id}:{chunk_index}" - used for seen_chunk_ids dedup
    text: str               # the chunk's raw text content
    source_id: str          # id of the parent document, links back to the document registry
    source_title: str       # human-readable document title, shown in citations
    section_heading: Optional[str]   # nearest markdown/HTML heading above this chunk, if any
    chunk_index: int        # position of this chunk within its source document
    similarity_score: float  # vector similarity score from the retriever (0-1 or distance, retriever-defined)


class GradeResult(TypedDict):
    """
    The LLM grader's verdict for a single chunk on a single attempt.

    Produced by: grade_documents (one GradeResult per chunk per grading call).
    Consumed by: no node re-reads this to make decisions (routing uses graded_docs
    directly) - it exists purely for traceability, exposed in the final API
    response so a reviewer/tester can see *why* each chunk was kept or dropped,
    and so retries are auditable rather than opaque.
    """

    chunk_id: str                        # matches DocChunk.chunk_id
    verdict: Literal["relevant", "irrelevant"]
    rationale: str                       # short LLM-provided justification for the verdict
    attempt_number: int                  # which retrieval attempt (0 = initial, 1..N = after a rewrite) this came from


class Citation(TypedDict):
    """
    A single source reference attached to the final generated answer.

    Produced by: generate (derived from whichever graded_docs / web_search_docs
    chunks were actually used to ground the answer).
    Consumed by: nothing inside the graph - this is terminal output returned to
    the FastAPI layer and serialized in the /query response.
    """

    source_id: str
    source_title: str
    chunk_index: int


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #


class GraphState(TypedDict):
    """
    The complete state object threaded through every node of the LangGraph
    StateGraph. Each field's docstring-style comment below states:
      - WHY the field exists (what decision or output it enables)
      - UPDATED BY: the node(s) that write to it
      - READ BY: the node(s) that read it

    Reducer semantics:
      - Fields with no Annotated wrapper use LangGraph's default TypedDict
        behavior: the node's return value REPLACES the previous value.
      - Fields wrapped in Annotated[list, operator.add] APPEND the node's
        returned list to the existing one, which is required for any field
        that must retain history across the transform_query/generate loops.
    """

    # ------------------------------------------------------------------ #
    # Identity & original request (immutable for the life of the graph)
    # ------------------------------------------------------------------ #

    query_id: str
    """
    WHY: correlates this graph run with the /feedback endpoint and with logs/
    traces, so a user's thumbs-up/down can be tied back to the exact retrieval
    and generation path that produced the answer.
    UPDATED BY: query_analysis (set once, at graph entry, from the id generated
    by the FastAPI route before invocation).
    READ BY: generate (embeds it in the response payload); not otherwise used
    for control flow.
    """

    question: str
    """
    WHY: the user's original, unmodified question. Kept separate from
    rewritten_query so citations/answers are always grounded against what the
    user actually asked, even after several query rewrites have occurred.
    UPDATED BY: query_analysis (set once, at graph entry; never modified again).
    READ BY: generate (final answer must address the *original* question, not
    just the latest rewritten form); hallucination_check (bonus, grounds its
    check against the original intent).
    """

    # ------------------------------------------------------------------ #
    # Query analysis & rewriting
    # ------------------------------------------------------------------ #

    rewritten_query: str
    """
    WHY: retrieval quality improves when the raw question is expanded/clarified
    (synonyms, disambiguation) before it hits the vector store. This is the
    query actually used for similarity search.
    UPDATED BY: query_analysis (initial rewrite/expansion); transform_query
    (subsequent reformulations after a failed grading pass). Each write
    REPLACES the previous value - only the latest attempt's query matters for
    the next retrieval call.
    READ BY: retrieve (this is the search query, not `question`).
    """

    query_type: Literal["conceptual", "how_to", "troubleshooting", "api_reference"]
    """
    WHY: classifying the question lets retrieval/grading adapt (e.g. wider k
    for troubleshooting queries, stricter relevance threshold for api_reference
    queries) instead of using one-size-fits-all parameters.
    UPDATED BY: query_analysis (classified once from the original question).
    READ BY: retrieve (adjusts top-k / search parameters); grade_documents
    (adjusts relevance strictness). If not wired up in a given implementation
    pass, this field must still be populated but documented as informational
    only - never left silently unused without a note.
    """

    previous_queries: Annotated[list[str], operator.add]
    """
    WHY: prevents transform_query from generating a reformulation it already
    tried (and which already failed), forcing genuine diversity across retry
    attempts rather than looping on near-identical queries.
    UPDATED BY: transform_query (appends the query it is about to replace, i.e.
    the value of `rewritten_query` at the time of rewriting).
    READ BY: transform_query (checked before proposing the next reformulation).
    Reducer: APPEND - full history must survive across every retry iteration.
    """

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    retrieved_docs: list[DocChunk]
    """
    WHY: holds the raw top-k similarity-search results for the CURRENT attempt
    only. Kept separate from graded_docs so grading always operates on one
    clean, attempt-scoped batch rather than an ever-growing mixed history.
    UPDATED BY: retrieve (REPLACES the previous value on every call, including
    after a transform_query loop-back).
    READ BY: grade_documents (the input it grades); the router
    decide_generation_path (an empty list here, distinct from "graded but
    irrelevant", routes straight to fallback instead of burning a retry).
    Reducer: REPLACE.
    """

    seen_chunk_ids: Annotated[list[str], operator.add]
    """
    WHY: without this, a retry with a semantically similar rewritten query can
    re-surface the exact same irrelevant chunks, making the retry loop do no
    real self-correction. Tracking every chunk_id ever retrieved lets `retrieve`
    exclude or down-rank them on subsequent attempts, guaranteeing retries
    actually diversify the candidate set.
    UPDATED BY: retrieve (appends the chunk_ids of everything it just fetched,
    after using the existing list to filter/exclude candidates).
    READ BY: retrieve (reads the accumulated list before searching, on every
    attempt after the first).
    Reducer: APPEND - must accumulate across all retrieval attempts in this run.
    """

    # ------------------------------------------------------------------ #
    # Document grading (self-corrective core)
    # ------------------------------------------------------------------ #

    graded_docs: list[DocChunk]
    """
    WHY: the filtered subset of retrieved_docs the grader judged relevant -
    this, not retrieved_docs, is what generation is grounded on. Keeping it as
    a distinct field (rather than mutating retrieved_docs in place) preserves
    an audit trail of what was fetched vs. what was actually used.
    UPDATED BY: grade_documents (REPLACES the previous value each grading pass
    with only the chunks verdicted "relevant" this attempt).
    READ BY: the router decide_generation_path (len(graded_docs) > 0 is the
    core relevant/irrelevant branch condition - relevant_count is a derived
    value, `len(graded_docs)`, never stored separately to avoid drift); generate
    (its grounding context); hallucination_check (bonus, what the answer is
    checked against).
    Reducer: REPLACE.
    """

    grading_results: Annotated[list[GradeResult], operator.add]
    """
    WHY: pure traceability - lets the final API response and any test/debug
    tooling show exactly which chunks were graded relevant/irrelevant and why,
    across every attempt, proving the self-corrective behavior actually ran
    rather than just trusting the graph completed.
    UPDATED BY: grade_documents (appends one GradeResult per chunk it grades,
    every attempt).
    READ BY: nothing inside the graph reads this for control flow; it is
    surfaced in the terminal API response for transparency only.
    Reducer: APPEND - full history across all attempts is the point of this field.
    """

    retry_count: int
    """
    WHY: the field the assignment explicitly asks the candidate to design -
    tracks how many times the retrieval->grading cycle has been retried after
    finding no relevant documents. Bounds the self-correction loop.
    UPDATED BY: transform_query (increments by 1 immediately before looping
    back to retrieve; never incremented inside the router itself, which must
    stay a pure/side-effect-free predicate).
    READ BY: the router decide_generation_path (compared against the run-config
    `max_retries` to choose transform_query vs. fallback); generate (echoes
    retries_used into the response); the (optional) LangGraph `recursion_limit`
    is an independent, execution-level backstop for this same loop.
    Reducer: REPLACE (a single incrementing int, written by exactly one node).
    Starts at 0.
    """

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    answer: Optional[str]
    """
    WHY: holds the generated answer text; nullable because it does not exist
    until the generate node runs (or may remain None if fallback short-circuits
    without generating anything).
    UPDATED BY: generate (REPLACES on every generation/regeneration attempt).
    READ BY: hallucination_check (bonus, the text it verifies against
    graded_docs); the API layer (terminal output).
    Reducer: REPLACE.
    """

    citations: list[Citation]
    """
    WHY: the assignment requires citations/references in the response; deriving
    this structurally from graded_docs/web_search_docs (rather than trusting
    the LLM to remember to cite in free text) guarantees every answer has
    verifiable source attribution.
    UPDATED BY: generate (REPLACES with the citations for the current answer;
    recomputed if hallucination_check triggers a regeneration).
    READ BY: the API layer (terminal output); not read by any other node.
    Reducer: REPLACE.
    """

    # ------------------------------------------------------------------ #
    # Hallucination check (bonus loop)
    # ------------------------------------------------------------------ #

    is_grounded: Optional[bool]
    """
    WHY: bonus Self-RAG-style groundedness verdict - confirms the generated
    answer is actually supported by graded_docs/web_search_docs rather than
    hallucinated, before returning it to the user.
    UPDATED BY: hallucination_check (REPLACES on every check; None until that
    node runs, and permanently None in implementations that skip the bonus).
    READ BY: the router decide_hallucination_path (grounded -> END; not
    grounded and under cap -> loop back to generate; not grounded and
    exhausted -> fallback); used to help derive `confidence`.
    Reducer: REPLACE.
    """

    regen_count: int
    """
    WHY: bounds the hallucination-repair loop the same way retry_count bounds
    the retrieval loop, guaranteeing termination even if generation keeps
    failing groundedness checks.
    UPDATED BY: generate's regeneration branch (increments by 1 immediately
    before regenerating with a stricter grounding instruction; never
    incremented inside the router).
    READ BY: the router decide_hallucination_path (compared against run-config
    `max_regen`).
    Reducer: REPLACE. Starts at 0.
    """

    # ------------------------------------------------------------------ #
    # Fallback / web search (bonus)
    # ------------------------------------------------------------------ #

    fallback_used: bool
    """
    WHY: surfaces to the client whether the response came from the normal
    grounded-generation path or from the "insufficient context"/web-search
    fallback, so callers can distinguish a confident answer from a
    best-effort one.
    UPDATED BY: fallback (set True when entered); generate (left False on the
    normal happy path - initialized False at graph entry by query_analysis).
    READ BY: the API layer (terminal output); used to help derive `confidence`.
    Reducer: REPLACE.
    """

    web_search_docs: Optional[list[DocChunk]]
    """
    WHY: bonus web-search fallback path. Kept as a field distinct from
    graded_docs so it is always clear, in traces and in the response, whether
    generation was grounded on the local corpus or on a web-search fallback.
    UPDATED BY: fallback's web-search branch (REPLACES with search results,
    shaped as DocChunk for uniform handling; remains None if the bonus is not
    implemented or if fallback resolves to a plain "I don't know" instead).
    READ BY: generate (uses these instead of graded_docs when this path was
    taken).
    Reducer: REPLACE.
    """

    error: Optional[str]
    """
    WHY: gives every node a defined place to report a caught exception
    (embedding call failure, LLM timeout, vector store error) instead of
    letting it propagate as an unhandled 500, satisfying the assignment's
    "consider error handling" requirement at the graph level, not just the API
    level.
    UPDATED BY: any node, on catching an exception it cannot recover from
    (REPLACES; last error wins, which is fine since an error routes straight
    to fallback and the graph does not continue past it).
    READ BY: the router decide_generation_path / decide_hallucination_path
    (a non-None error routes straight to fallback); the API layer (surfaced in
    the error envelope).
    Reducer: REPLACE.
    """

    confidence: Optional[Literal["high", "medium", "low"]]
    """
    WHY: gives the client a single, auditable signal for how much to trust the
    answer, derived deterministically rather than left to the LLM to self-report.
    "high" = generated on the first attempt with is_grounded True (or
    hallucination_check not implemented); "medium" = required retries and/or
    regeneration but still grounded; "low" = served via fallback_used.
    UPDATED BY: generate (computed as the final step before returning, using
    retry_count, regen_count, is_grounded, and fallback_used).
    READ BY: the API layer (terminal output only).
    Reducer: REPLACE.
    """
