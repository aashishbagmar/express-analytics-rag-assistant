# PROJECT 1 — SELF-CORRECTIVE RAG

Scope note: This document is based only on the code in the current workspace. It does not rely on internet facts or external assumptions. Where the code is silent, I say so plainly.

---

## Architecture overview

```mermaid
flowchart TD
    U[User question] --> A[POST /query\napp/api/query.py]
    A --> B[get_compiled_graph\napp/api/query.py]
    B --> C[build_graph\napp/graph/builder.py]
    C --> D[query_analysis\napp/graph/nodes/query_analysis.py]
    D --> E[retrieval\napp/graph/nodes/retrieval.py]
    E --> F[grading\napp/graph/nodes/grading.py]
    F --> G{len(graded_docs) > 0?\ndecide_generation_path}
    G -- yes --> H[generation\napp/graph/nodes/generation.py]
    G -- no --> I{retry_count < max_retries?}
    I -- yes --> J[transform_query\napp/graph/nodes/transform_query.py]
    J --> E
    I -- no --> K[fallback\napp/graph/nodes/fallback.py]
    H --> L[answer, citations, confidence]
    K --> M[low-confidence fallback answer]

    U2[Document upload or corpus ingest] --> X[POST /upload or /ingest\napp/api/upload.py / app/api/ingest.py]
    X --> Y[IngestionPipeline\napp/ingestion/pipeline.py]
    Y --> Z[DocumentLoader\napp/ingestion/loaders.py]
    Z --> W[DocumentChunker\napp/ingestion/chunking.py]
    W --> V[VectorStoreService.add_documents\napp/services/vector_store.py]
    V --> CDB[ChromaDB]
```

---

## A. Project Understanding

### 1) Give me a 30-second explanation of the project.
- Explain: This project is a local technical-document RAG assistant. It loads supported files, splits them into chunks, embeds them with a sentence-transformer model, stores them in ChromaDB, retrieves relevant passages for a user question, grades them for relevance, retries the query when needed, and returns a grounded answer with citations. The graph is built as a LangGraph state machine in [app/graph/builder.py](app/graph/builder.py#L85-L122).
- Reference: [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/services/vector_store.py](app/services/vector_store.py#L44-L163), [app/api/query.py](app/api/query.py#L33-L72), [app/ingestion/pipeline.py](app/ingestion/pipeline.py#L35-L85).
- Interview Answer: It is a self-corrective retrieval-augmented generation system for local technical docs, built around FastAPI, LangGraph, ChromaDB, and Ollama.

### 2) Give me a 2-minute explanation of the project.
- Explain: The system starts with a local corpus or uploaded file. The ingestion pipeline loads supported document formats, normalizes text, splits long files into chunks, preserves source metadata, and upserts them into a persistent Chroma collection. The query path creates an initial graph state, classifies and rewrites the question, embeds the query, and retrieves top-k chunks. Those chunks are sent to a grading node where an LLM decides whether each chunk is relevant; if the LLM fails or is disabled, the system falls back to a deterministic heuristic based on similarity and keyword overlap. If no relevant chunks are found, the graph selects the query-rewrite node, updates the retry count, reformulates the query, and loops back to retrieval. Once relevant context exists, the generation node builds a grounded answer from that context and derives structural citations from metadata rather than trusting the LLM to cite correctly. If the pipeline cannot produce a safe answer, the fallback node returns a low-confidence, honest “not enough relevant information” response.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L110-L191), [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L121), [app/services/vector_store.py](app/services/vector_store.py#L44-L210), [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L101-L315), [app/graph/nodes/fallback.py](app/graph/nodes/fallback.py#L64-L115).
- Interview Answer: This project turns technical documentation into an evidence-backed answer engine. The core feature is not just retrieval; it is self-correction: after a bad retrieval pass, it rewrites the query and retries, and it only generates from documented context that passed a relevance check.

### 3) What exact problem were you trying to solve?
- Explain: The project tries to answer questions from a corpus of technical documentation without hallucinating or relying on the model’s generic memory. The design enforces groundedness by retrieving document chunks, grading them for relevance, and generating only from the accepted context. This is visible in the graph design and generator instructions in [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453), and [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L101-L315).
- Reference: [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L101-L315), [app/services/llm.py](app/services/llm.py#L43-L109).
- Interview Answer: The problem was building a local, source-grounded documentation assistant that can answer technical questions with context and citations, while still recovering when retrieval is weak or question phrasing is poor.

### 4) Who is the intended user?
- Explain: The project is designed for users who ask questions about local technical documentation, such as developers working with FastAPI, Pydantic, LangChain, or similar docs. The system is exposed through FastAPI endpoints and a Streamlit front end, as described in [app/main.py](app/main.py#L25-L53), and the README says it is a technical documentation assistant for the Express Analytics assignment.
- Reference: [app/main.py](app/main.py#L25-L53), [README.md](README.md#L1-L220), [app/api/query.py](app/api/query.py#L33-L72).
- Interview Answer: The intended user is a technical user or developer querying a local documentation corpus through a chat or API interface.

### 5) What makes this different from a normal document chatbot?
- Explain: It is not a plain chat model over documents. It explicitly retrieves evidence from a vector store, filters and grades chunks for relevance, retries the query on failure, and constrains generation to the accepted context. This is a core difference from a normal chatbot because the document evidence is externalized into Chroma and the graph route is stateful.
- Reference: [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L101-L315).
- Interview Answer: It is a self-corrective RAG pipeline, not a generic chat wrapper: it retrieves evidence, checks relevance, retries failed search paths, and grounds answers in source chunks.

### 6) Why did you decide to build a RAG system?
- Explain: The code uses retrieval and generation separately. The vector-store service stores indexed chunks, and the generation node is specifically instructed to use only the relevant context. This design is a direct response to the need to answer from the local corpus instead of relying on an LLM’s general knowledge.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L107-L210), [app/services/llm.py](app/services/llm.py#L41-L109), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L101-L315).
- Interview Answer: Because the model alone is not the correct source of truth; the answer should come from indexed documentation that is actually available and relevant.

### 7) Why did you make it self-corrective?
- Explain: The graph is designed so that when no relevant documents are found, the system does not stop; it rewrites the question, updates the retry count, and re-enters retrieval. This retry behavior is implemented in [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/graph/router.py](app/graph/router.py#L56-L102), and [app/graph/nodes/transform_query.py](app/graph/nodes/transform_query.py#L72-L266).
- Reference: [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/nodes/transform_query.py](app/graph/nodes/transform_query.py#L72-L266), [app/graph/state.py](app/graph/state.py#L154-L247).
- Interview Answer: Because retrieval quality is not guaranteed on the first pass, and the project explicitly adds a retry loop that diversifies the query instead of looping on the same result set.

### 8) What are the major components of the system?
- Explain: The major pieces are the API layer, the ingestion pipeline, the chunker and loaders, the embedding service, the Chroma vector store, the graph builder and nodes, the LLM service, and the storage layer. The exact modules are in [app/main.py](app/main.py#L25-L53), [app/api](app/api), [app/ingestion](app/ingestion), [app/services](app/services), [app/graph](app/graph), and [app/storage](app/storage).
- Reference: [app/main.py](app/main.py#L25-L53), [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/ingestion/pipeline.py](app/ingestion/pipeline.py#L35-L85), [app/services/vector_store.py](app/services/vector_store.py#L44-L210), [app/services/llm.py](app/services/llm.py#L77-L209).
- Interview Answer: The project has six core layers: ingestion, embedding, vector search, LangGraph self-correction, LLM answer generation, and FastAPI API exposure.

### 9) Draw the complete architecture.
- Explain: The architecture combines ingestion and query pathways. Ingestion loads docs, chunks them, embeds them, and writes to Chroma. The query path takes a user question, rewrites it, retrieves candidate chunks, grades them, retries or falls back, and generates a grounded answer.
- Reference: [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/ingestion/pipeline.py](app/ingestion/pipeline.py#L35-L85), [app/services/vector_store.py](app/services/vector_store.py#L44-L210), [app/api/query.py](app/api/query.py#L33-L72).
- Interview Answer: The architecture is a two-pipeline system: a document ingestion pipeline for indexing and a query pipeline for answer generation and self-correction.

### 10) Explain the complete data flow from document upload to final answer.
- Explain: A document is uploaded to the /upload endpoint in [app/api/upload.py](app/api/upload.py#L68-L152). The file is validated, saved under data/uploads, and ingested through the pipeline. The loader reads text from the file format, the chunker splits it into chunks with metadata, and the vector-store service embeds the chunks and upserts them into the Chroma collection. When a user later asks a question, the graph retrieves similar chunks, grades them, and generates an answer that is returned alongside citations and confidence.
- Reference: [app/api/upload.py](app/api/upload.py#L68-L152), [app/ingestion/pipeline.py](app/ingestion/pipeline.py#L35-L85), [app/ingestion/loaders.py](app/ingestion/loaders.py#L110-L191), [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L121), [app/services/vector_store.py](app/services/vector_store.py#L107-L210), [app/api/query.py](app/api/query.py#L33-L72).
- Interview Answer: Upload → validate → save → parse → chunk → embed → upsert → query → retrieval → grading → generation → response.

### 11) Explain the complete data flow from user question to final answer.
- Explain: The API receives a question in [app/api/query.py](app/api/query.py#L33-L72), builds an initial GraphState, and invokes the compiled graph. The graph runs `query_analysis` → `retrieval` → `grading` → conditional router → `transform_query` loop or `generation`. The generation node builds context from `graded_docs`, calls the LLM if available, and falls back to an extractive answer when the LLM is unavailable. The API serializes the result into `QueryResponse` with `answer`, `citations`, `confidence`, `retry_count`, and `fallback_used`.
- Reference: [app/api/query.py](app/api/query.py#L33-L72), [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L101-L315), [app/schemas/query.py](app/schemas/query.py#L9-L31).
- Interview Answer: The question enters the graph, is rewritten and embedded, relevant chunks are retrieved and graded, the router chooses generation or retry, and the final result is shaped into the API response.

### 12) What happens if the user asks something that is not present in the documents?
- Explain: The router checks whether any `graded_docs` remain. If none, it retries while `retry_count < max_retries`; once the retry budget is exhausted, it routes to the fallback node. The fallback message is a constant, honest response saying there is not enough relevant information. This is implemented in [app/graph/router.py](app/graph/router.py#L56-L102) and [app/graph/nodes/fallback.py](app/graph/nodes/fallback.py#L64-L115).
- Reference: [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/nodes/fallback.py](app/graph/nodes/fallback.py#L64-L115), [app/graph/state.py](app/graph/state.py#L154-L247).
- Interview Answer: The system does not fabricate an answer; it either rewrites the query and retries or returns a low-confidence fallback saying the indexed documentation is insufficient.

---

## B. Document Ingestion

### 13) Which document formats does your system support?
- Explain: `DocumentLoader.SUPPORTED_EXTENSIONS` supports `.md`, `.txt`, `.html`, `.htm`, `.pdf`, and `.docx`.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L110-L191).
- Interview Answer: The current implementation supports Markdown, plain text, HTML, PDF, and DOCX files.

### 14) Which loader is used for each document type?
- Explain: In `load_file`, `.pdf` calls `_load_pdf`, `.docx` calls `_load_docx`, and all other supported text-like files call `_load_text`. HTML is normalized through `_HTMLTextExtractor` inside `_load_text`.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L120-L191).
- Interview Answer: PDF uses PyMuPDF, DOCX uses python-docx, and all others are read as text with HTML normalization when needed.

### 15) How is text extracted from PDF?
- Explain: The loader opens the PDF with `fitz.open(file_path)`, iterates each page, calls `page.get_text()`, strips it, and appends the page text into a list. It then joins the non-empty pages with `\n\n`.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L145-L170).
- Interview Answer: PDF text is extracted page by page using PyMuPDF, then concatenated in document order.

### 16) How is text extracted from DOCX?
- Explain: It loads the Word document with `DocxDocument(str(file_path))`, iterates through `docx_doc.paragraphs`, keeps only non-empty paragraphs, and joins them with `\n`.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L172-L183).
- Interview Answer: DOCX extraction is paragraph-based, preserving document order as plain text.

### 17) How is text extracted from TXT?
- Explain: `_load_text` reads the file with UTF-8 or UTF-8-SIG and strips whitespace. It does not perform any extra parsing beyond normalization.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L124-L143).
- Interview Answer: TXT is read as plain UTF-8 text and normalized for downstream chunking.

### 18) How is text extracted from Markdown?
- Explain: Markdown is treated as a text file; `_load_text` reads it, applies `.strip()`, and then the chunker later splits it while preserving headings. There is no Markdown-specific parser in the code.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L124-L143), [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L121).
- Interview Answer: Markdown is read as raw text and then chunked with heading-aware splitting logic.

### 19) What happens if a document cannot be parsed?
- Explain: The loader catches `OSError` during file reads and logs a warning. If text extraction fails or the file is unreadable, it returns an empty list and skips the file so one bad file does not kill ingestion. This is visible in `load_file` and `load_directory`.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L120-L191).
- Interview Answer: It warns and drops the bad file rather than crashing the entire ingestion batch.

### 20) How do you handle empty documents?
- Explain: After extraction, `if not content.strip(): logger.warning(...); return []` in `load_file`. The chunker also skips documents whose `page_content.strip()` is empty.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L120-L191), [app/ingestion/chunking.py](app/ingestion/chunking.py#L62-L85).
- Interview Answer: Empty files are dropped with warnings and never indexed.

### 21) How do you handle corrupted documents?
- Explain: For PDFs it catches `OSError`, `RuntimeError`, and `ValueError`; for DOCX it catches `OSError` and `ValueError`; in general it logs the issue and returns empty list. There is no retry or quarantine mechanism in this code.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L145-L183).
- Interview Answer: Corrupt files are treated as skip-on-error entries and logged without failing the whole ingestion process.

### 22) What metadata do you attach to each document?
- Explain: `_build_metadata` attaches `source_id`, `source_title`, `source_path`, `filename`, and then `file_type` is added if missing. The PDF loader also adds `page_count`.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L145-L191).
- Interview Answer: Every source document keeps a deterministic source_id, title, filesystem path, filename, and file type; PDFs also include page_count.

### 23) What metadata do you attach to each chunk?
- Explain: The chunker copies the source metadata and adds `chunk_index` and `section_heading`. `section_heading` is the nearest preceding Markdown heading in the original document. It also preserves information like `source_id`, `source_title`, `source_path`, and `filename` through the chunk metadata.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L121).
- Interview Answer: Each chunk keeps source provenance plus chunk position and its current section heading so citations can be mapped back precisely.

### 24) How do you generate a document ID?
- Explain: The document metadata uses the resolved file path, hashes it with `hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:16]`, and stores the result as `source_id`.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L184-L191).
- Interview Answer: A source_id is a 16-character SHA-1 hash of the resolved file path.

### 25) How do you generate a chunk ID?
- Explain: In the vector store, chunks are assigned a deterministic ID with `source_id:chunk_index:content_hash` using `VectorStoreService._document_id`. For grading, a fallback `chunk_id` may be `source_id:chunk_index` when metadata is present, or a hash string when not.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L359-L379), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L468-L507).
- Interview Answer: Chunk IDs are deterministic and derived from source identity, chunk position, and content hash, which keeps upserts idempotent.

### 26) How do you detect duplicate documents?
- Explain: There is no full content-hash duplicate detector in the `DocumentRegistry` skeleton; the actual system uses deterministic IDs and Chroma upserts. The upload route also blocks duplicate filenames by refusing to overwrite an existing file in `data/uploads`.
- Reference: [app/api/upload.py](app/api/upload.py#L95-L152), [app/services/document_registry.py](app/services/document_registry.py#L1-L38), [app/services/vector_store.py](app/services/vector_store.py#L359-L379).
- Interview Answer: Duplicate detection is currently implemented as a deterministic-ID/upsert pattern plus a duplicate-filename check in upload, not a full content-hash registry.

### 27) How does incremental indexing work?
- Explain: `replace_documents_for_sources` deletes all existing chunks whose `source_id` belongs to the incoming documents, then adds the new chunk set. This is “incremental” because it refreshes only the affected document sources instead of clearing the entire collection.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L107-L163).
- Interview Answer: It performs a source-scoped refresh: delete old chunks for the same source IDs, then reinsert the fresh chunks.

### 28) What happens if ingestion fails halfway through?
- Explain: The code does not implement a transactional rollback for the whole corpus. The ingestion pipeline is best-effort and skips bad files, and the endpoint will error if no chunks are created or if the vector update fails. There is no multi-file transaction with rollback.
- Reference: [app/ingestion/pipeline.py](app/ingestion/pipeline.py#L35-L85), [app/api/ingest.py](app/api/ingest.py#L36-L69), [app/api/upload.py](app/api/upload.py#L68-L152).
- Interview Answer: The system fails fast at the route level for a broken overall ingest, but individual bad files are skipped rather than crashing the pipeline.

### 29) Can the same document be indexed twice?
- Explain: Yes, it can be re-ingested, but the code is designed to be idempotent at the chunk level because each chunk ID is deterministic. Chroma’s `upsert` is used in [app/services/vector_store.py](app/services/vector_store.py#L107-L140), so repeated inserts replace the same IDs instead of creating duplicates.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L107-L140), [app/services/vector_store.py](app/services/vector_store.py#L359-L379).
- Interview Answer: Reindexing the same source is allowed; the deterministic chunk IDs make it effectively idempotent.

### 30) How would you prevent duplicate chunks?
- Explain: The safest approach in this codebase would be to keep using deterministic IDs (`source_id`, `chunk_index`, and content hash) and/or compare against a content-hash registry before insert. This is already how `_document_id` is constructed.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L359-L379).
- Interview Answer: Use a deterministic ID per chunk and set it as the Chroma ID so any duplicate insert becomes an upsert rather than a second copy.

---

## C. Chunking

### 31) Why do you need chunking?
- Explain: Long documents must be split into retrieval-friendly units so that each candidate chunk stays focused, carries useful metadata, and fits into the embedding and retrieval pipeline. The project does this in [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L121).
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L121), [app/services/vector_store.py](app/services/vector_store.py#L107-L140).
- Interview Answer: Chunking keeps each vector item small, specific, and easier to grade for relevance than a whole document.

### 32) What chunking strategy did you implement?
- Explain: It uses `RecursiveCharacterTextSplitter` with `chunk_size=800`, `chunk_overlap=100`, `add_start_index=True`, and a custom separator order that prefers Markdown headings and then paragraph breaks.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L63).
- Interview Answer: The strategy is recursive character splitting with a heading-aware boundary preference, not a semantic embedding-based segmenter.

### 33) What exactly is semantic chunking in your implementation?
- Explain: In this code, “semantic” is closer to “section-aware chunking” than an LLM-driven semantic splitter. It extracts Markdown headings and stores the nearest previous heading as `section_heading`. This preserves topical grouping even though the actual split is still character-based.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L121).
- Interview Answer: It is section-aware recursive chunking, not full semantic segmentation by model; the chunks are still created by character boundaries and then annotated with headings.

### 34) What chunk size did you use?
- Explain: `DocumentChunker.__init__` sets `chunk_size=800` and `chunk_overlap=100`.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L63).
- Interview Answer: The implementation currently uses 800-character chunks with 100-character overlap.

### 35) What chunk overlap did you use?
- Explain: The overlap is 100 characters.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L63).
- Interview Answer: The overlap is 100 characters, intended to reduce information loss across adjacent chunks.

### 36) Why did you choose those values?
- Explain: The code hard-codes them in the chunker constructor and does not include adaptive tuning logic. The intent is to keep chunks long enough to carry context but short enough to stay focused.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L63).
- Interview Answer: They are fixed defaults chosen to balance retrieval recall and chunk specificity, even though the project does not implement auto-tuning.

### 37) How would you know whether your chunk size is too small?
- Explain: If retrieval produces fragments that do not contain enough context to answer a question, the system would show lots of partial or low-relevance chunks. In practice, the signal is poor answer quality and repeated retries due to missing topic context. The code’s retry loop captures this pattern indirectly.
- Reference: [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/nodes/transform_query.py](app/graph/nodes/transform_query.py#L72-L266).
- Interview Answer: If chunks are too small, the system starts seeing fragmented context and loses the necessary surrounding explanation; retrieval quality drops and retries increase.

### 38) How would you know whether your chunk size is too large?
- Explain: If chunks contain too many unrelated concepts, the grader will reject them as irrelevant or the system will retrieve a broad chunk that mixes topics; the result is lower precision and more noisy context. This is the tradeoff the code is designed to mitigate by using a moderate size and overlap.
- Reference: [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453), [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L63).
- Interview Answer: If chunks are too large, they become noisy and blend multiple topics, which hurts relevance grading and retrieval precision.

### 39) What happens when important context crosses two chunks?
- Explain: The overlap prevents that from being a silent break. With `chunk_overlap=100`, adjacent chunks keep a little of the previous text, and `section_heading` helps maintain topic continuity.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L121).
- Interview Answer: The overlap reduces the chance that a critical sentence gets split away from its surrounding explanation.

### 40) How do you preserve document/page/source information after chunking?
- Explain: The chunker copies the original document metadata into each chunk and adds `chunk_index` and `section_heading`. Metadata preservation is explicit in `metadata = dict(chunk.metadata)` and then an update with new keys.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L68-L104).
- Interview Answer: All original source metadata is copied down to each chunk, and the chunk adds its own position and section context.

### 41) How would you evaluate different chunking strategies?
- Explain: You would compare retrieval quality using the same corpus and query set under different chunk_size / chunk_overlap combinations, then measure recall and precision. The code has no built-in benchmark, so this would be an evaluation layer added outside the current implementation.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L121), [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105).
- Interview Answer: I would run a fixed evaluation set, vary chunk size and overlap, and compare retrieval precision / recall across those runs.

### 42) What would you change if retrieval quality was poor because of chunking?
- Explain: I would tune `chunk_size` and `chunk_overlap`, increase heading-aware splitting, and consider reducing topic mixing by using smaller chunks with heavier overlap. This is the natural change because the chunker is centralized in one place.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L63), [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105).
- Interview Answer: I would tighten the splitter, increase overlap, and re-evaluate using a fixed query set before changing the retrieval model itself.

---

## D. Embeddings

### 43) Which embedding model did you use?
- Explain: The code defaults to `all-MiniLM-L6-v2` in `EmbeddingService.DEFAULT_MODEL_NAME` and uses `SentenceTransformer(self._model_name)`.
- Reference: [app/services/embeddings.py](app/services/embeddings.py#L36-L127).
- Interview Answer: The application uses the sentence-transformers model `all-MiniLM-L6-v2` by default.

### 44) Why did you choose all-MiniLM-L6-v2?
- Explain: The code chooses it explicitly as the default model and loads it lazily. There is no extra rationale in the code beyond the implementation default.
- Reference: [app/services/embeddings.py](app/services/embeddings.py#L36-L127).
- Interview Answer: The project chooses it because it is the configured default model for the local embedding service; the code does not provide a broader comparative rationale.

### 45) What is an embedding?
- Explain: In this implementation, it is the numeric vector created by `SentenceTransformer.encode` for a document or query. The vector is then stored in Chroma and used to compare semantic similarity.
- Reference: [app/services/embeddings.py](app/services/embeddings.py#L45-L127), [app/services/vector_store.py](app/services/vector_store.py#L107-L140), [app/services/vector_store.py](app/services/vector_store.py#L208-L256).
- Interview Answer: An embedding is a dense numeric representation of a text chunk that lets the system compare meaning via vector distance.

### 46) What is the embedding dimension of your model?
- Explain: The code does not hardcode or expose the embedding dimension. It simply calls `encode` and converts the result to a list of floats. The dimension is therefore an implementation detail of the sentence-transformer model, not a project-defined constant.
- Reference: [app/services/embeddings.py](app/services/embeddings.py#L45-L127).
- Interview Answer: The project does not declare an explicit dimension constant; the dimension is whatever the selected SentenceTransformer model produces.

### 47) How is a document converted into embeddings?
- Explain: `VectorStoreService.add_documents` collects the text list, calls `self.embedding_service.embed_documents(texts)`, and then calls `collection.upsert(ids=..., documents=..., embeddings=embeddings, metadatas=...)`.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L107-L140), [app/services/embeddings.py](app/services/embeddings.py#L45-L127).
- Interview Answer: Each chunk text is embedded by the embedding service and uploaded to Chroma with its metadata.

### 48) How is a query converted into an embedding?
- Explain: `similarity_search` validates the query, calls `self.embedding_service.embed_query(query)`, and uses that single embedding in `collection.query(query_embeddings=[query_embedding], ...)`.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L193-L256).
- Interview Answer: The user’s query is embedded as a single vector and used to search the indexed chunk embeddings.

### 49) Are document and query embeddings generated using the same model?
- Explain: Yes. The same `EmbeddingService` instance is used for both document and query embeddings, and the collection metadata stores the selected model ID. The vector store also seeds the collection with `metadata={"embedding_model": self.embedding_service.model_id()}`.
- Reference: [app/services/embeddings.py](app/services/embeddings.py#L36-L127), [app/services/vector_store.py](app/services/vector_store.py#L57-L97), [app/services/vector_store.py](app/services/vector_store.py#L107-L140).
- Interview Answer: Yes, both are generated by the same model because the same embedding service handles both paths.

### 50) What similarity metric do you use?
- Explain: The actual vector store query returns Chroma distances; the project then converts distance to a similarity-like scalar using `1 /(1 + distance)` in `_distance_to_similarity`. So the project uses Chroma’s distance semantics internally and converts them to a normalized score for the app.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L208-L256), [app/services/vector_store.py](app/services/vector_store.py#L420-L447).
- Interview Answer: The code uses Chroma distance values and converts them to a bounded similarity score in Python.

### 51) Why did you choose that similarity metric?
- Explain: The project comment says the transformation is monotonic and keeps the score bounded in `(0,1]`, which preserves ordering while giving a stable similarity-like measure for the retrieval node. That is exactly what `_distance_to_similarity` does.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L420-L447).
- Interview Answer: The conversion is chosen because it keeps ranking order intact while producing a normalized score that is easier to reason about downstream.

### 52) What happens if you change the embedding model after documents are indexed?
- Explain: The code has no migration logic. The collection is created with metadata containing the embedding model id and the search code expects the collection to match the current model. If a different embedding model is used without rebuilding, old and new vectors are not aligned in the same collection.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L57-L97), [app/services/vector_store.py](app/services/vector_store.py#L107-L140), [app/services/embeddings.py](app/services/embeddings.py#L36-L127).
- Interview Answer: A model change is a reindexing event. The safe behavior is to rebuild or replace the collection, not to mix embeddings from two models in the same Chroma collection.

### 53) Would old and new embeddings be compatible?
- Explain: Not automatically. The project stores the embedding model metadata on the collection, but it does not implement cross-model compatibility. Compatibility is effectively “same model, same collection”.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L57-L97), [app/services/embeddings.py](app/services/embeddings.py#L36-L127).
- Interview Answer: No, not without reindexing. The project does not implement a compatibility layer across embedding models.

### 54) How would you re-index the collection safely?
- Explain: The safe reindex path is to rebuild the collection with the new embedding model and then insert all documents again. `build_index` deletes the existing collection and recreates it before adding documents. That is the explicit rebuild path.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L57-L97).
- Interview Answer: Delete and recreate the Chroma collection, then re-embed and insert all documents under the new model.

---

## E. ChromaDB / Vector Search

### 55) Why did you choose ChromaDB?
- Explain: The code uses a persistent local Chroma client and a named collection. This is the concrete vector system used by the project. The service wraps persistence and similarity search behind `VectorStoreService`.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L97), [app/services/vector_store.py](app/services/vector_store.py#L208-L256).
- Interview Answer: ChromaDB is the actual vector database in this implementation because it is the object the project directly instantiates and persists on disk.

### 56) Why ChromaDB instead of FAISS?
- Explain: There is no FAISS implementation in the codebase; the project directly uses ChromaDB and persistent storage. This means the answer is effectively: the code is built around ChromaDB and not around FAISS.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L97).
- Interview Answer: No FAISS path exists in the code; the implementation is built around ChromaDB.

### 57) What exactly is stored in ChromaDB?
- Explain: The collection stores document text, embedding vectors, metadata, and IDs. In `add_documents`, it does `collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)`.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L107-L140).
- Interview Answer: Each indexed chunk is stored with its content, vector embedding, metadata, and a deterministic ID.

### 58) How are embeddings stored?
- Explain: Embeddings are passed as the `embeddings` argument to `collection.upsert`, and the embedding service creates them from the chunk text before insertion.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L107-L140), [app/services/embeddings.py](app/services/embeddings.py#L45-L127).
- Interview Answer: Embeddings are generated by the embedding service and stored in Chroma as part of the upsert payload.

### 59) How is metadata stored?
- Explain: Metadata is sanitized in `_sanitize_metadata` so it is Chroma-safe and then restored in `_restore_metadata`. This supports strings, ints, floats, and bools, with `None` and JSON values stored behind internal marker keys.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L316-L419).
- Interview Answer: Metadata is serialized into Chroma-compatible scalar values and restored when retrieved.

### 60) What is the collection structure?
- Explain: The project uses a persistent Chroma client and a default collection name of `technical_docs`. The collection is created lazily with `get_or_create_collection(name=self.collection_name, metadata={"embedding_model": ...})`.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L97).
- Interview Answer: A single persistent collection named `technical_docs` holds the chunk documents and their metadata.

### 61) What distance/similarity metric is configured?
- Explain: Not explicitly configured in code. Chroma distance is returned by `collection.query`, then converted to similarity in `_distance_to_similarity`. There is no custom metric setup in this implementation.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L208-L256), [app/services/vector_store.py](app/services/vector_store.py#L420-L447).
- Interview Answer: The code relies on Chroma’s default distance metric and then converts it to a normalized similarity score.

### 62) What is top-k?
- Explain: Top-k is the number of nearest chunks returned by the similarity query. In the graph node, the retrieval call defaults to `k=5`.
- Reference: [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L18-L28), [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L87), [app/core/config.py](app/core/config.py#L56-L69).
- Interview Answer: Top-k is how many candidate chunks are returned for a query; in this code, the default is 5.

### 63) What top-k value did you use?
- Explain: The implementation default is 5 (`DEFAULT_TOP_K = 5`). `Settings.default_top_k` also exists and defaults to 5, but the retrieval function itself uses the explicit default argument.
- Reference: [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L18-L28), [app/core/config.py](app/core/config.py#L56-L69).
- Interview Answer: The default is 5 results per query.

### 64) Why did you choose that top-k?
- Explain: The code hard-codes 5 and does not justify it statistically in the project itself. It is simply the default retrieval size used by the graph.
- Reference: [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L18-L28).
- Interview Answer: The default of 5 is a balance between recall and context size in the current implementation, but the code does not provide a deeper tuning rationale.

### 65) What happens if top-k is too low?
- Explain: The system can miss relevant chunks and route to fallback sooner. This is a direct consequence of fewer candidate passages being available for grading.
- Reference: [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105).
- Interview Answer: If k is too low, relevant passages may never reach grading, and the system may fail with a fallback response even when the answer exists in the corpus.

### 66) What happens if top-k is too high?
- Explain: The graph receives more candidate chunks and more work to grade. That can create noisy context and reduce precision, but also helps recall. The code’s grading node is the filter that resolves this tradeoff.
- Reference: [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453), [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105).
- Interview Answer: A higher k increases recall but also raises the chance of irrelevant chunks reaching the grader, which increases noise and grading cost.

### 67) How do you filter results by document metadata?
- Explain: There is no metadata filtering in the similarity-search query itself. The code does source-aware deletion in `replace_documents_for_sources`, but query-time metadata filtering is not implemented in this retrieval function.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L107-L163), [app/services/vector_store.py](app/services/vector_store.py#L208-L256), [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105).
- Interview Answer: The current implementation does not perform metadata-based filtering during retrieval; it relies on semantic search plus grading.

### 68) How do you handle an empty retrieval result?
- Explain: `VectorStoreService.similarity_search` returns an empty list when the collection is empty, and the graph route then sees empty `retrieved_docs` or empty `graded_docs` and chooses `transform_query` or `fallback`.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L208-L256), [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/nodes/fallback.py](app/graph/nodes/fallback.py#L64-L115).
- Interview Answer: The system treats empty retrieval as a normal retry/fallback condition; it does not hallucinate a result.

### 69) What happens if ChromaDB is unavailable?
- Explain: `VectorStoreService` wraps Chroma errors in `VectorStoreServiceError`, and the API routes catch that and raise a 500. The query endpoint also catches general exceptions and returns a 500. There is no graceful degradation to another vector store in the current code.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L97), [app/api/query.py](app/api/query.py#L33-L72), [app/api/ingest.py](app/api/ingest.py#L36-L69).
- Interview Answer: The system fails with an HTTP 500 and logs the exception instead of silently answering from stale or alternate data.

### 70) How would you scale vector retrieval?
- Explain: This implementation is a single persistent local Chroma collection. To scale, the code would need a service-level change: a managed or sharded vector store, additional replicas, or a background indexing service. The current code is not designed for multi-node horizontal scaling.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L97), [app/services/vector_store.py](app/services/vector_store.py#L208-L256).
- Interview Answer: I would move from a single local persistent collection to a larger managed vector service or sharded deployment with queuing and load balancing.

---

## F. Retrieval

### 71) Explain your retrieval function line by line.
- Explain: `retrieve_documents` first validates that `rewritten_query` exists and is non-empty, then validates that `k > 0`; it creates a `VectorStoreService` if one is not injected; it calls `store.similarity_search(query=query, k=k)`; then it normalizes each result into a `RetrievedDocument` with `content`, `metadata`, and `score`; finally it returns `{"retrieved_docs": retrieved_docs}`. This is the complete retrieval contract.
- Reference: [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105).
- Interview Answer: Retrieval is a thin wrapper around the vector store: validate query, fetch nearest chunks, normalize them, and return the state payload for grading.

### 72) What exactly happens when a query reaches retrieval?
- Explain: The retrieval node reads the current `rewritten_query` from state, validates it, and asks the vector store for nearest neighbors using the current search term. It does not do any query rewriting there; it simply executes the vector search.
- Reference: [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105), [app/services/vector_store.py](app/services/vector_store.py#L193-L256).
- Interview Answer: A retrieval call turns the current rewritten_query into an embedding, searches the Chroma collection, and returns candidate chunks for grading.

### 73) How is the query embedded?
- Explain: `similarity_search` calls `self.embedding_service.embed_query(query)` and then uses the resulting embedding in `collection.query(query_embeddings=[query_embedding], ...)`.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L193-L256), [app/services/embeddings.py](app/services/embeddings.py#L45-L127).
- Interview Answer: The query is embedded through the same sentence-transformer service used for indexed chunks.

### 74) How are candidate chunks selected?
- Explain: Chroma returns `documents`, `metadatas`, and `distances` for the nearest results. The vector-store service turns them into a list of dictionaries and passes those back to the retrieval node.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L193-L256).
- Interview Answer: Candidate chunks are selected by Chroma’s nearest-neighbor query and ranked by embedding distance.

### 75) How is similarity calculated?
- Explain: Chroma produces a distance, and `_distance_to_similarity` converts it to `1/(1+distance)`. This is a monotonic transformation that preserves ordering while yielding a bounded score.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L420-L447).
- Interview Answer: The project uses Chroma distance internally and converts it to a normalized similarity score in Python.

### 76) Do you use metadata filtering?
- Explain: Not during the retrieval query. The retrieval function does not pass a metadata filter to Chroma. Source-scoped delete logic exists, but query-time metadata filtering is not implemented.
- Reference: [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105), [app/services/vector_store.py](app/services/vector_store.py#L107-L163).
- Interview Answer: No metadata filter is used in the active retrieval path.

### 77) Do you use dense retrieval, sparse retrieval or hybrid retrieval?
- Explain: This system uses dense retrieval only. It embeds query and chunk text and searches with vector similarity in Chroma. There is no BM25 or sparse index in the code.
- Reference: [app/services/embeddings.py](app/services/embeddings.py#L36-L127), [app/services/vector_store.py](app/services/vector_store.py#L193-L256).
- Interview Answer: The active implementation is dense retrieval only.

### 78) Why did you choose that retrieval strategy?
- Explain: Because the project is built around embeddings and Chroma; the code is entirely centered on dense nearest-neighbor lookup with a vector store. The LLM grading step then filters the returned candidates semantically.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L193-L256), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453).
- Interview Answer: The implementation chooses dense retrieval because it aligns with the project’s embedding-based document store and relevance-grading workflow.

### 79) Do you rerank retrieved documents?
- Explain: No explicit reranking step exists. The retrieval node returns top-k results, and the grading node filters them based on LLM relevance or a heuristic fallback; there is no ranking stage after retrieval.
- Reference: [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453).
- Interview Answer: No, the project relies on the vector store’s ranking followed by relevance grading rather than a separate reranker.

### 80) If not, why not?
- Explain: The implementation intentionally keeps the retrieval layer simple and uses the grading step as the filter. That is consistent with the project’s graph design and avoids unnecessary complexity in the current phase.
- Reference: [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453).
- Interview Answer: The project keeps retrieval simple and lets grading decide which retrieved chunks are actually useful.

### 81) How would you improve retrieval quality?
- Explain: I would tune chunk size and overlap, improve query rewriting, add metadata filters, and consider a reranker or hybrid search. The code does not implement those features yet.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L121), [app/graph/nodes/transform_query.py](app/graph/nodes/transform_query.py#L72-L266), [app/services/vector_store.py](app/services/vector_store.py#L208-L256).
- Interview Answer: I would improve chunk quality and query diversity first, then add reranking or metadata filters if recall and precision remain weak.

### 82) How would you measure retrieval quality?
- Explain: I would evaluate a set of known questions and expected supporting chunks to compute recall@k and MRR. Those are standard hit-based retrieval metrics and fit the code structure.
- Reference: [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453).
- Interview Answer: I would create a small gold set of queries and expected documents, then compute retrieval metrics on that set.

### 83) How would you calculate Recall@k?
- Explain: Recall@k = number of relevant documents retrieved in the top-k / total number of relevant documents for that query. This is a standard retrieval formula and is not implemented in the current code.
- Reference: None in code; this is a standard metric explanation, not a code implementation.
- Interview Answer: For a query, I would count how many relevant source chunks appear in the top-k and divide by the total relevant chunks for that query.

### 84) How would you calculate MRR?
- Explain: MRR is the reciprocal of the rank of the first relevant result: $\frac{1}{r}$ where $r$ is the position of the first relevant item in the ranked list. The code does not compute it as part of the system today.
- Reference: None in code; this is a standard metric explanation, not a code implementation.
- Interview Answer: I would rank retrieved chunks for a query and take the reciprocal of the position of the first truly relevant document.

---

## G. LangGraph / Self-Correction

### 85) Why did you use LangGraph?
- Explain: The project creates a `StateGraph` and wires it with conditional edges in `build_graph` so the graph can route from retrieval to grading to query reformulation or generation. This is exactly the pattern used for self-correction.
- Reference: [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/graph/router.py](app/graph/router.py#L56-L102).
- Interview Answer: LangGraph gives a structured state machine for retrieval, grading, retry, generation, and fallback without embedding that logic inside the API layer.

### 86) What are all the nodes in your graph?
- Explain: The active graph includes `query_analysis`, `retrieval`, `grading`, `transform_query`, `generation`, and `fallback`.
- Reference: [app/graph/builder.py](app/graph/builder.py#L25-L82), [app/graph/builder.py](app/graph/builder.py#L85-L122).
- Interview Answer: The current graph is six nodes: query analysis, retrieval, grading, query transformation, generation, and fallback.

### 87) What does each node do?
- Explain: `query_analysis` rewrites and classifies the question; `retrieval` fetches chunks; `grading` filters relevant chunks; `transform_query` revises the search after failed grading; `generation` creates the final answer; `fallback` returns a safe low-confidence reply when the graph cannot proceed.
- Reference: [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/graph/nodes/query_analysis.py](app/graph/nodes/query_analysis.py#L97-L202), [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453), [app/graph/nodes/transform_query.py](app/graph/nodes/transform_query.py#L72-L266), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L101-L315), [app/graph/nodes/fallback.py](app/graph/nodes/fallback.py#L64-L115).
- Interview Answer: Each node has a distinct responsibility: interpret the question, fetch evidence, judge evidence, retry search, write the answer, or fail safely.

### 88) What is the graph state?
- Explain: `GraphState` is a `TypedDict` that holds the values passed between nodes, including question metadata, retrieval results, graded chunks, retry counts, answer text, citations, and fallback status. It is defined in [app/graph/state.py](app/graph/state.py#L97-L334).
- Reference: [app/graph/state.py](app/graph/state.py#L97-L334).
- Interview Answer: Graph state is the shared memory for the entire LangGraph run: it stores both raw inputs and intermediate evidence/results.

### 89) List every state field.
- Explain: The active fields are: `query_id`, `question`, `rewritten_query`, `query_type`, `previous_queries`, `retrieved_docs`, `seen_chunk_ids`, `graded_docs`, `grading_results`, `retry_count`, `answer`, `citations`, `is_grounded`, `regen_count`, `fallback_used`, `web_search_docs`, `error`, `confidence`.
- Reference: [app/graph/state.py](app/graph/state.py#L97-L334).
- Interview Answer: The graph stores identity, original request, rewritten query, retrieval artifacts, grading artifacts, retry bookkeeping, answer/citations, and fallback/error state.

### 90) Which node modifies each state field?
- Explain: `query_analysis` sets `rewritten_query`, `query_type`, and `previous_queries`; `retrieval` sets `retrieved_docs`; `grading` sets `graded_docs` and `grading_results`; `transform_query` updates `rewritten_query`, `previous_queries`, and `retry_count`; `generation` sets `answer`, `citations`, and `confidence`; `fallback` sets `answer`, `fallback_used`, and `confidence`. `query_id` and `question` are initialized by the API before graph invocation.
- Reference: [app/api/query.py](app/api/query.py#L33-L72), [app/graph/state.py](app/graph/state.py#L97-L334), [app/graph/builder.py](app/graph/builder.py#L85-L122).
- Interview Answer: The API initializes identity and the original question, then the graph-specific nodes own the rest of the fields.

### 91) What are the conditional edges?
- Explain: There is a conditional edge after `grading` that calls `decide_generation_path`. The route options are `generate`, `transform_query`, and `fallback`, defined in the graph builder.
- Reference: [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/graph/router.py](app/graph/router.py#L56-L102).
- Interview Answer: The route after grading determines whether the graph should answer, retry the prompt, or stop with a fallback.

### 92) What determines which edge is selected?
- Explain: `decide_generation_path` checks for a recorded `error`, then if there are any `graded_docs`, then if `retry_count < max_retries`, else fallback. The exact logic is in the router.
- Reference: [app/graph/router.py](app/graph/router.py#L56-L102).
- Interview Answer: If relevant evidence exists, the graph generates; if not and retry budget remains, it rewrites; otherwise it falls back.

### 93) What happens after retrieval?
- Explain: After retrieval, the graph always moves to `grading`. The grading node validates the chunks and either accepts them as relevant or rejects them. Then the router decides whether to generate or retry.
- Reference: [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453).
- Interview Answer: Retrieval is followed immediately by relevance grading, and only then does the graph decide whether to answer or fix the query.

### 94) Explain the document-grading node.
- Explain: `grade_documents` reads `question` and `retrieved_docs`, tries batched LLM grading when the LLM is enabled and multiple docs exist, then falls back to per-chunk grading with a deterministic heuristic. It returns `graded_docs` and `grading_results`.
- Reference: [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453).
- Interview Answer: The grading node is the self-corrective gate: it decides which retrieved chunks are actually usable evidence.

### 95) What exactly does the grader evaluate?
- Explain: It evaluates whether a retrieved chunk helps answer the user’s question, based on question-to-chunk semantic relevance and lexical overlap. The LLM prompt is explicit: “check whether the chunk contains information that helps answer the question”.
- Reference: [app/services/llm.py](app/services/llm.py#L43-L69), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453).
- Interview Answer: It judges whether a chunk is relevant to the user’s question, not whether it is merely similar by embedding score alone.

### 96) What does the grader return?
- Explain: The grader returns a structured verdict: `verdict`, `confidence`, and `reason`, and the graph stores this as `grading_results` alongside the relevant filtered chunks as `graded_docs`.
- Reference: [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453), [app/services/llm.py](app/services/llm.py#L77-L209).
- Interview Answer: It returns a relevance verdict with a short rationale and confidence value for each chunk.

### 97) How do you validate the grader output?
- Explain: `LLMService._parse_grade_response` verifies it is a JSON object, contains a verdict of `relevant` or `irrelevant`, contains numeric confidence, and contains non-empty reason text. If invalid, it raises `LLMServiceError` and the node falls back to the heuristic.
- Reference: [app/services/llm.py](app/services/llm.py#L77-L209).
- Interview Answer: The output is validated structurally before the graph trusts it.

### 98) What happens if the grader says the retrieved documents are irrelevant?
- Explain: If the grader rejects them and no `graded_docs` remain, the router returns `transform_query` if `retry_count < max_retries`; otherwise it routes to fallback. The system never generates from empty or rejected evidence.
- Reference: [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/nodes/fallback.py](app/graph/nodes/fallback.py#L64-L115).
- Interview Answer: It triggers a retry loop or a safe fallback instead of forcing a bad answer.

### 99) Explain the query-rewriting node.
- Explain: `transform_query` reads the original question and current query, tracks previously tried queries, and chooses the next reformulation from a fixed strategy list. It increments `retry_count` and sets `rewritten_query` to a distinct candidate.
- Reference: [app/graph/nodes/transform_query.py](app/graph/nodes/transform_query.py#L72-L266).
- Interview Answer: The rewrite node deliberately changes the search formulation to avoid repeating failed queries and to broaden or clarify the search.

### 100) When exactly is query rewriting triggered?
- Explain: It is triggered when `decide_generation_path` sees `len(graded_docs) == 0` and `retry_count < max_retries`.
- Reference: [app/graph/router.py](app/graph/router.py#L56-L102).
- Interview Answer: Query rewriting happens only after retrieval and grading fail to yield useful evidence while retry budget remains.

### 101) How is the rewritten query generated?
- Explain: It cycles through a fixed set of strategies: broaden scope, extract key entities, switch to keyword phrasing, add documentation-search framing, and strip framework qualifiers. If all are exhausted, it falls back to a guaranteed unique “attempt N” remake.
- Reference: [app/graph/nodes/transform_query.py](app/graph/nodes/transform_query.py#L130-L266).
- Interview Answer: It uses a deterministic strategy cycle intentionally designed to diversify the next retrieval attempt rather than just rephrasing the same search.

### 102) How do you prevent query rewriting from making the query worse?
- Explain: The node keeps a history in `previous_queries` and normalizes the query before comparing it. It also uses the original question when constructing new reformulations, ensuring it does not compound the query gradually. It will not repeat prior attempts.
- Reference: [app/graph/nodes/transform_query.py](app/graph/nodes/transform_query.py#L72-L266), [app/graph/state.py](app/graph/state.py#L174-L211).
- Interview Answer: It compares normalized queries and blocks duplicates, while cycling through different reformulation styles so retry attempts stay diverse and bounded.

### 103) How many times can the system retry?
- Explain: The project default is `max_retries = 2` in settings; the router stops retrying when retry_count reaches that cap. In the graph, `retry_count` is incremented inside `transform_query` and compared against `max_retries` in the router.
- Reference: [app/core/config.py](app/core/config.py#L53-L69), [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/nodes/transform_query.py](app/graph/nodes/transform_query.py#L72-L116).
- Interview Answer: The current default retry budget is two retries after the initial pass.

### 104) Where is the retry count stored?
- Explain: It is stored in `GraphState.retry_count`.
- Reference: [app/graph/state.py](app/graph/state.py#L220-L247).
- Interview Answer: The retry count lives in the graph state itself.

### 105) How do you prevent infinite loops?
- Explain: The loop is bounded by `retry_count < max_retries`, it tracks previous queries to prevent repeats, and LangGraph also uses `recursion_limit=25` when the API invokes the graph. That combination prevents runaway retries.
- Reference: [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/nodes/transform_query.py](app/graph/nodes/transform_query.py#L72-L116), [app/api/query.py](app/api/query.py#L33-L72).
- Interview Answer: Infinite loops are prevented by a hard retry cap, duplicate-query blocking, and a recursion limit in the graph invocation config.

### 106) What happens after the retry limit is reached?
- Explain: The router routes to fallback. The fallback node produces a low-confidence answer saying the system cannot find enough relevant documentation.
- Reference: [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/nodes/fallback.py](app/graph/nodes/fallback.py#L64-L115).
- Interview Answer: After the retry budget is exhausted, the system stops and returns a safe fallback response instead of inventing an answer.

### 107) What is your fallback behavior?
- Explain: `handle_fallback` returns a constant message: “I could not find enough relevant information in the indexed documentation to answer this question.” It also sets `fallback_used=True` and `confidence="low"`.
- Reference: [app/graph/nodes/fallback.py](app/graph/nodes/fallback.py#L64-L115).
- Interview Answer: Fallback is a safe, low-confidence abstention with no fabricated answer.

### 108) Why is this workflow better than a single LangChain chain?
- Explain: The graph makes the decision logic explicit. The routing is stateful and inspectable, and each node owns a single responsibility. This is better than a monolithic chain because it allows debugability, retries, and fallback behavior without embedding all logic into a single chain object.
- Reference: [app/graph/builder.py](app/graph/builder.py#L85-L122), [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/state.py](app/graph/state.py#L97-L334).
- Interview Answer: The graph provides an explicit, auditable control flow for retrieval, grading, retries, and fallback, which is easier to reason about and maintain than a single chain.

---

## H. LLM / Ollama

### 109) Which LLM are you using?
- Explain: The code’s active default is `llama3:8b` in `Settings.ollama_model`. The README mentions `qwen2.5:14b`, but the implementation default in code is `llama3:8b`.
- Reference: [app/core/config.py](app/core/config.py#L35-L53), [README.md](README.md#L1-L220).
- Interview Answer: The code defaults to `llama3:8b` for Ollama, while the README references a different local model name as a note.

### 110) Why did you choose that model?
- Explain: The project does not include a model-selection rationale beyond the default settings object. It is configured as a local Ollama model and is easy to swap.
- Reference: [app/core/config.py](app/core/config.py#L35-L53).
- Interview Answer: The code chooses it as the default runtime configuration; it does not include a deeper model-justification section in the implementation itself.

### 111) Why Ollama?
- Explain: The service posts to `http://localhost:11434/api/chat` and uses `ollama_base_url` and `ollama_model` from settings. The system is intentionally local and keyless.
- Reference: [app/core/config.py](app/core/config.py#L35-L53), [app/services/llm.py](app/services/llm.py#L116-L169).
- Interview Answer: Ollama is used because the application is designed to run a local, no-API-key LLM service next to the app.

### 112) Why not use a hosted API?
- Explain: The code does not implement a hosted provider path, and the configuration is explicitly local. The current design is intentionally built around a self-hosted local model.
- Reference: [app/core/config.py](app/core/config.py#L35-L53), [app/services/llm.py](app/services/llm.py#L116-L169).
- Interview Answer: The implementation is not designed for a hosted provider at the moment; it is built around a local Ollama endpoint.

### 113) Where does the model run?
- Explain: It runs on the local Ollama server at the configured base URL, defaulting to `http://localhost:11434`.
- Reference: [app/core/config.py](app/core/config.py#L35-L53), [app/services/llm.py](app/services/llm.py#L116-L169).
- Interview Answer: The model runs on a local Ollama server rather than in a remote cloud API.

### 114) How is the LLM called from your application?
- Explain: The application uses `httpx.post` to `.../api/chat` with a structured JSON payload: `model`, `messages`, `format: "json"`, `stream: False`, and `temperature: 0.0`.
- Reference: [app/services/llm.py](app/services/llm.py#L116-L169).
- Interview Answer: The app calls Ollama through HTTP with JSON-mode structured output and zero temperature for reproducibility.

### 115) What prompt does the final generation step receive?
- Explain: The `generate_answer` method builds a user prompt with the question and context, then asks the model to return JSON of the form `{"answer": "..."}`. The system message says, “Use ONLY the supplied context. Do not invent facts. If the answer is not supported by the context, explicitly say that the context does not contain enough information.”
- Reference: [app/services/llm.py](app/services/llm.py#L41-L69), [app/services/llm.py](app/services/llm.py#L167-L209).
- Interview Answer: The final generation prompt includes the user question and only the accepted chunks, with explicit instructions to stay grounded and not fabricate facts.

### 116) What information is included in the generation context?
- Explain: The generation node builds the context strictly from `graded_docs` content, in order, with `[n]` markers for each chunk. It is an explicit “context from graded documents only” approach.
- Reference: [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L117-L206).
- Interview Answer: The answer context is limited to the relevant graded chunks and nothing else from the graph state.

### 117) How do you prevent the model from using information outside the retrieved evidence?
- Explain: The generation prompt explicitly says “Use ONLY the supplied context,” and `_build_context` only includes `graded_docs`. The answer path also falls back to an extractive answer if the LLM call fails or returns unusable output.
- Reference: [app/services/llm.py](app/services/llm.py#L41-L69), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L117-L206).
- Interview Answer: The system deliberately constructs the context from accepted evidence only and instructs the model not to use anything outside it.

### 118) How do you handle hallucinations?
- Explain: The active graph does not wire the bonus `hallucination_check` node; the builder comments explicitly say the bonus loop is intentionally not wired in this phase. The actual implementation therefore relies on retrieval quality and the generation prompt, not a live groundedness check. The code does include `hallucination_check.py`, but it is not connected to the default graph.
- Reference: [app/graph/builder.py](app/graph/builder.py#L1-L72), [app/graph/nodes/hallucination_check.py](app/graph/nodes/hallucination_check.py), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L117-L206).
- Interview Answer: The shipped implementation uses a grounded context prompt and safe fallback, but it does not currently run an active hallucination-check loop in the main LangGraph path.

### 119) How do you handle malformed LLM output?
- Explain: The LLM service validates the JSON responses and raises `LLMServiceError` if the structure is wrong. The grading and generation nodes catch these exceptions and fall back to a deterministic heuristic or extractive answer.
- Reference: [app/services/llm.py](app/services/llm.py#L77-L209), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L117-L206).
- Interview Answer: Bad LLM output is treated as an LLM failure and replaced by a deterministic fallback instead of crashing the graph.

### 120) What happens if Ollama is unavailable?
- Explain: `httpx.HTTPError` is caught and converted to `LLMServiceError`. The grade/generation nodes catch `LLMServiceError` and use their fallback path, so the graph does not stop or send an invalid answer.
- Reference: [app/services/llm.py](app/services/llm.py#L116-L169), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L117-L206).
- Interview Answer: The graph fails over to a deterministic heuristic or extractive answer rather than crashing.

### 121) What happens if the model times out?
- Explain: `httpx.post(..., timeout=self._settings.llm_request_timeout_seconds)` sets the timeout; if it fires, `httpx.HTTPError` is raised and treated as an LLM failure. The system then uses fallback logic.
- Reference: [app/services/llm.py](app/services/llm.py#L116-L169), [app/core/config.py](app/core/config.py#L35-L53).
- Interview Answer: A timeout is handled as a provider failure and immediately degraded to a safe fallback path.

### 122) How do you control LLM latency?
- Explain: The application sets a single request timeout in settings and uses `stream=False` with a small prompt context built only from the graded chunks. There is no more advanced async batching or streaming in the current code.
- Reference: [app/core/config.py](app/core/config.py#L35-L53), [app/services/llm.py](app/services/llm.py#L116-L169), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L117-L206).
- Interview Answer: Latency is controlled mainly by small retrieval context, a fixed timeout, and the fact that the model is asked only for a single structured answer or relevance verdict.

### 123) How do you control LLM token usage?
- Explain: The app controls token usage by limiting the generation prompt to the accepted `graded_docs` context only, with no broad corpus context. The fallback generator also limits the number of context chunks to five.
- Reference: [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L117-L206), [app/services/llm.py](app/services/llm.py#L167-L209).
- Interview Answer: Token usage is kept low by building the prompt from only the relevant chunks, with a hard cap on the extractive fallback context.

### 124) How would you reduce inference cost?
- Explain: Lower the number of retrieved chunks, disable or reduce LLM grading, use a smaller local model, or cache frequent queries. The current code does not implement any of these optimizations, but the architecture is designed so they can be inserted at the service layer.
- Reference: [app/core/config.py](app/core/config.py#L35-L53), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453), [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105).
- Interview Answer: I would reduce the value of `k`, shrink the context, and lower the need for LLM grading when a cheaper deterministic fallback is sufficient.

### 125) How would you replace the LLM with another model?
- Explain: The code isolates this behind `LLMService` so the graph and nodes do not call an external provider directly. To swap models, change the settings value or replace the implementation in `_call_chat` while preserving the same public methods.
- Reference: [app/core/config.py](app/core/config.py#L35-L53), [app/services/llm.py](app/services/llm.py#L77-L209).
- Interview Answer: I would change the configuration or the provider implementation behind `LLMService`, without touching the graph nodes themselves.

---

## I. Citations / Confidence / Hallucination

### 126) How are citations generated?
- Explain: `generate_answer` calls `_build_citations` on `graded_docs`, which reads metadata values like `source_id`, `source_title`, and `chunk_index` and deduplicates them by `(source_id, chunk_index)`.
- Reference: [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L211-L315).
- Interview Answer: Citations are generated structurally from the metadata in the accepted chunks, not from free-form model citations.

### 127) Where does citation source information come from?
- Explain: It comes from each chunk’s metadata, especially `source_id`, `source_title`, and `chunk_index`, which are carried forward by the chunker and preserved in Chroma associations.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L68-L104), [app/services/vector_store.py](app/services/vector_store.py#L316-L419), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L211-L315).
- Interview Answer: Citation source information is taken directly from each chunk’s stored metadata, not guessed by the model.

### 128) How do you map an answer back to a source chunk?
- Explain: In the generation flow, each chunk’s metadata is turned into a `Citation` with `source_id`, `source_title`, and `chunk_index`. The values are stored in the final API response; the mapping is therefore source-and-chunk-based, not text-based.
- Reference: [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L211-L315), [app/schemas/query.py](app/schemas/query.py#L9-L31).
- Interview Answer: The mapping is source metadata-based: each returned citation points to a concrete source document and chunk index.

### 129) What metadata is required for citations?
- Explain: `source_id`, `source_title`, and `chunk_index` are enough to create a citation object. `source_id` and `chunk_index` are the deduplication key; `source_title` becomes the visible label.
- Reference: [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L211-L315).
- Interview Answer: A citation requires the source ID, title, and chunk index.

### 130) What exactly does your confidence score represent?
- Explain: The project defines `confidence` as a `Literal["high", "medium", "low"]` and computes it deterministically from the number of graded docs, retry count, and fallback status. It is not a probability score.
- Reference: [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L211-L315), [app/graph/state.py](app/graph/state.py#L319-L334).
- Interview Answer: Confidence here is a qualitative label describing how well grounded the answer is, not a statistical probability.

### 131) How is the confidence score calculated?
- Explain: It returns `low` if fallback was used; otherwise `high` when `retry_count == 0` and there are at least two graded docs; otherwise `medium`.
- Reference: [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L211-L315).
- Interview Answer: High confidence means first-pass grounded answer from multiple supporting chunks; medium means it was retried or thinly supported; low means fallback.

### 132) Is the confidence score calibrated?
- Explain: No. It is a deterministic heuristic, not a calibrated probability model. The code says it is computed deterministically from retry and evidence counts.
- Reference: [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L211-L315).
- Interview Answer: No, it is not calibrated; it is a heuristic signal based on the size and freshness of the evidence set.

### 133) Is it a probability or a heuristic?
- Explain: It is a heuristic label, not a probability.
- Reference: [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L211-L315).
- Interview Answer: It is a heuristic assessment, not a probability estimate.

### 134) Can a high-confidence answer still be wrong?
- Explain: Yes. The project’s active graph does not include a live groundedness-check loop in the main path; it only notes a bonus hallucination-check design, and the builder explicitly says that bonus loop is intentionally not wired. So a high-confidence answer can still be wrong in the current implementation because there is no active strict abstention check.
- Reference: [app/graph/builder.py](app/graph/builder.py#L1-L72), [app/graph/nodes/hallucination_check.py](app/graph/nodes/hallucination_check.py), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L211-L315).
- Interview Answer: Yes, because the graph does not currently enforce a full groundedness validation on the main generation path.

### 135) How do you detect unsupported answers?
- Explain: There is a dedicated hallucination-check node in [app/graph/nodes/hallucination_check.py](app/graph/nodes/hallucination_check.py), but it is not wired into the compiled graph in [app/graph/builder.py](app/graph/builder.py#L85-L122). In the shipped code, the active detection mechanism is basically “don’t generate unless relevant docs exist and use only the provided context.”
- Reference: [app/graph/builder.py](app/graph/builder.py#L1-L72), [app/graph/nodes/hallucination_check.py](app/graph/nodes/hallucination_check.py).
- Interview Answer: The code includes a groundedness-check design but the active graph does not currently run it.

### 136) What happens when evidence is insufficient?
- Explain: The graph routes to fallback. The fallback node returns a safe low-confidence answer and marks `fallback_used=True`.
- Reference: [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/nodes/fallback.py](app/graph/nodes/fallback.py#L64-L115).
- Interview Answer: The system abstains and tells the user it could not find enough relevant information rather than answering from weak evidence.

### 137) How would you implement a stricter abstention mechanism?
- Explain: I would wire the `hallucination_check` node into the graph, require that answer terms appear in the accepted evidence, and refuse generation when evidence coverage is too weak. That is the natural enhancement path visible in the node and builder comments.
- Reference: [app/graph/builder.py](app/graph/builder.py#L1-L72), [app/graph/nodes/hallucination_check.py](app/graph/nodes/hallucination_check.py).
- Interview Answer: I would add a groundedness check that verifies the answer is supported by the accepted context and route back to fallback when not.

---

## J. FastAPI / Backend

### 138) Why FastAPI?
- Explain: The codebase uses FastAPI to expose the API endpoints and manage request/response schemas while also using threadpool wrappers for blocking operations. This is visible in [app/main.py](app/main.py#L25-L53) and route modules.
- Reference: [app/main.py](app/main.py#L25-L53), [app/api/query.py](app/api/query.py#L33-L72), [app/api/ingest.py](app/api/ingest.py#L36-L69).
- Interview Answer: FastAPI is used because it is a clean ASGI framework for a local API that serves the RAG workflow and ingestion endpoints.

### 139) What are your API endpoints?
- Explain: The app exposes `/health`, `/query`, `/upload`, `/ingest`, `/documents`, and `/feedback`.
- Reference: [app/main.py](app/main.py#L25-L53), [app/api/health.py](app/api/health.py#L24-L27), [app/api/query.py](app/api/query.py#L33-L72), [app/api/upload.py](app/api/upload.py#L68-L152), [app/api/ingest.py](app/api/ingest.py#L36-L69), [app/api/documents.py](app/api/documents.py#L32-L57), [app/api/feedback.py](app/api/feedback.py#L28-L49).
- Interview Answer: The current backend exposes health, query, upload, ingest, document listing, and feedback endpoints.

### 140) What does each endpoint do?
- Explain: `/health` returns a ready signal; `/query` runs the graph and returns grounded answer; `/upload` stores and ingests a file; `/ingest` ingests a corpus path and reindexes source docs; `/documents` lists indexed documents; `/feedback` stores a feedback record.
- Reference: [app/api/health.py](app/api/health.py#L24-L27), [app/api/query.py](app/api/query.py#L33-L72), [app/api/upload.py](app/api/upload.py#L68-L152), [app/api/ingest.py](app/api/ingest.py#L36-L69), [app/api/documents.py](app/api/documents.py#L32-L57), [app/api/feedback.py](app/api/feedback.py#L28-L49).
- Interview Answer: The API is split into a health check, a question-answer endpoint, indexing endpoints, a document listing endpoint, and a feedback store.

### 141) What are the request schemas?
- Explain: `QueryRequest` has a `question` string; `IngestRequest` has a `path` string; `FeedbackRequest` is used for feedback persistence. Upload is handled through multipart file upload rather than a JSON request schema.
- Reference: [app/schemas/query.py](app/schemas/query.py#L9-L31), [app/schemas/ingest.py](app/schemas/ingest.py#L9-L26), [app/api/upload.py](app/api/upload.py#L68-L152), [app/api/feedback.py](app/api/feedback.py#L28-L49).
- Interview Answer: Request schemas are small Pydantic models for query, ingest, and feedback, while upload uses multipart file input.

### 142) What are the response schemas?
- Explain: `QueryResponse` includes `query_id`, `answer`, `citations`, `confidence`, `retry_count`, and `fallback_used`. `IngestResponse` includes `documents_processed`, `chunks_created`, and `status`. `FeedbackResponse` includes `query_id` and `status`.
- Reference: [app/schemas/query.py](app/schemas/query.py#L9-L31), [app/schemas/ingest.py](app/schemas/ingest.py#L9-L26), [app/api/feedback.py](app/api/feedback.py#L28-L49).
- Interview Answer: The API response is serialized to a narrow Pydantic contract that exposes structured answer metadata and ingestion status.

### 143) Where is validation performed?
- Explain: Validation occurs in FastAPI request models and in custom server-side checks inside the route functions, such as upload filename checking and empty-content checks. The internal graph nodes also validate state fields before using them.
- Reference: [app/schemas/query.py](app/schemas/query.py#L9-L31), [app/api/upload.py](app/api/upload.py#L95-L152), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453).
- Interview Answer: Validation sits both in the API layer and inside the graph logic, especially before node execution.

### 144) How are exceptions handled?
- Explain: The app registers a global `@app.exception_handler(Exception)` in [app/main.py](app/main.py#L25-L53) to return a generic 500. Route functions also catch exceptions and raise `HTTPException` with appropriate status codes when the failure is known.
- Reference: [app/main.py](app/main.py#L25-L53), [app/api/query.py](app/api/query.py#L33-L72), [app/api/ingest.py](app/api/ingest.py#L36-L69), [app/api/upload.py](app/api/upload.py#L68-L152).
- Interview Answer: The backend catches operational errors as HTTP exceptions and falls back to a consistent 500 envelope for unexpected exceptions.

### 145) How are timeouts handled?
- Explain: LLM requests have a timeout configured in `Settings.llm_request_timeout_seconds` and passed to `httpx.post`. There is no explicit timeout for Chroma operations in this implementation.
- Reference: [app/core/config.py](app/core/config.py#L35-L53), [app/services/llm.py](app/services/llm.py#L116-L169).
- Interview Answer: LLM timeouts are explicit and the system falls back; vector-store timeouts are not implemented in the code today.

### 146) What happens if the LLM is slow?
- Explain: It raises `LLMServiceError` or times out, and the grade or generation node falls back to heuristic or extractive paths. The graph does not stall indefinitely because the LLM call has a timeout and the node is written to fail over.
- Reference: [app/services/llm.py](app/services/llm.py#L116-L169), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L117-L206).
- Interview Answer: Slow LLM calls degrade gracefully into deterministic fallback logic instead of breaking the query path.

### 147) What happens if the vector database is slow?
- Explain: There is no explicit vector-store timeout in the current service, so the call will simply depend on Chroma’s behavior. If Chroma errors out, it raises `VectorStoreServiceError`, which the route turns into a 500.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L97), [app/services/vector_store.py](app/services/vector_store.py#L193-L256), [app/api/query.py](app/api/query.py#L33-L72).
- Interview Answer: The current backend treats vector-store failures as hard errors and responds with an HTTP 500.

### 148) How would you add authentication?
- Explain: The current code has no auth layer. A standard FastAPI pattern would be to add a dependency using a token or session-based verifier, then protect the query, upload, ingest, and feedback routers with it.
- Reference: [app/main.py](app/main.py#L25-L53), [app/api/query.py](app/api/query.py#L33-L72).
- Interview Answer: I would add a FastAPI dependency that validates tokens or sessions before allowing the API route to proceed.

### 149) How would you add authorization?
- Explain: The project currently has no role-based permissions. Authorization would usually be implemented by attaching a user identity to the request and then checking the user’s access against a per-user document registry or a token claim.
- Reference: [app/services/document_registry.py](app/services/document_registry.py#L1-L38), [app/api/query.py](app/api/query.py#L33-L72).
- Interview Answer: I would add a user-context dependency and enforce document or route permissions using the authenticated identity.

### 150) How would you make the API multi-user?
- Explain: The code currently does not implement user separation. A multi-user API would require user scopes, per-user document collections, and user-aware metadata within the vector store and ingestion paths.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L97), [app/services/document_registry.py](app/services/document_registry.py#L1-L38).
- Interview Answer: I would isolate document stores and request context by user and prevent cross-user reads through namespacing or collection-level isolation.

### 151) How would you isolate documents between users?
- Explain: The most direct method would be user-specific collection names or a user_id prefix in source IDs and document metadata. The current code is not implementing this yet.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L97), [app/services/vector_store.py](app/services/vector_store.py#L359-L379).
- Interview Answer: I would namespace documents by user in the collection or metadata so queries never search another user’s corpus.

### 152) How would you rate-limit the API?
- Explain: The project does not include rate limiting. In FastAPI, the usual approach would be middleware or an external gateway that enforces request quotas per user or IP.
- Reference: [app/main.py](app/main.py#L25-L53).
- Interview Answer: I would add a request quota middleware or gateway policy before the app receives the request.

### 153) How would you monitor the API?
- Explain: The code currently uses Python logging, but there is no metrics exporter or tracing setup in the main app. Monitoring would mean logs for requests, graph steps, and errors with query IDs and latency metrics.
- Reference: [app/core/logging.py](app/core/logging.py), [app/api/query.py](app/api/query.py#L33-L72), [app/main.py](app/main.py#L25-L53).
- Interview Answer: I would keep log events for request latency, query id, model failures, vector-store errors, and ingestion counts, then aggregate them via a monitoring tool.

---

## K. Testing / Evaluation

### 154) What tests did you write?
- Explain: The workspace includes a suite of tests under the `tests` directory, covering API, chunking, embeddings, fallback behavior, feedback store, generation, grading, graph routing, hallucination checking, ingestion, incremental ingest, query analysis, retrieval, state schema, transform query, and vector store.
- Reference: [tests](tests), [tests/test_api_endpoints.py](tests/test_api_endpoints.py), [tests/test_chunking.py](tests/test_chunking.py), [tests/test_embeddings.py](tests/test_embeddings.py), [tests/test_fallback.py](tests/test_fallback.py), [tests/test_feedback_store.py](tests/test_feedback_store.py), [tests/test_generation.py](tests/test_generation.py), [tests/test_grading.py](tests/test_grading.py), [tests/test_graph_builder.py](tests/test_graph_builder.py), [tests/test_graph_routing.py](tests/test_graph_routing.py), [tests/test_hallucination_check.py](tests/test_hallucination_check.py), [tests/test_ingest_incremental.py](tests/test_ingest_incremental.py), [tests/test_ingestion.py](tests/test_ingestion.py), [tests/test_query_analysis.py](tests/test_query_analysis.py), [tests/test_retrieval.py](tests/test_retrieval.py), [tests/test_state_schema.py](tests/test_state_schema.py), [tests/test_transform_query.py](tests/test_transform_query.py), [tests/test_vector_store.py](tests/test_vector_store.py).
- Interview Answer: The codebase includes a broad unit and integration-style test suite covering ingestion, retrieval, grading, routing, and API behavior.

### 155) How did you test ingestion?
- Explain: There are dedicated tests for ingestion and incremental ingest, which validate the file handling pipeline and the reindex behavior. The code clearly contains ingestion-specific test files.
- Reference: [tests/test_ingestion.py](tests/test_ingestion.py), [tests/test_ingest_incremental.py](tests/test_ingest_incremental.py).
- Interview Answer: Ingestion was tested for both a normal ingest path and an incremental reindex path.

### 156) How did you test retrieval?
- Explain: There is a retrieval-focused test file alongside the graph and vector-store tests. It exercises the retrieval path and result normalization.
- Reference: [tests/test_retrieval.py](tests/test_retrieval.py), [tests/test_vector_store.py](tests/test_vector_store.py).
- Interview Answer: Retrieval was tested by asserting the vector-store and retrieval node produce expected candidate chunks.

### 157) How did you test query rewriting?
- Explain: There are dedicated `query_analysis` and `transform_query` tests. The project contains explicit test files for both.
- Reference: [tests/test_query_analysis.py](tests/test_query_analysis.py), [tests/test_transform_query.py](tests/test_transform_query.py).
- Interview Answer: Query rewriting was tested by verifying the transformed query is different, valid, and non-repeating across attempts.

### 158) How did you test the retry loop?
- Explain: There is a graph routing test suite and a transform-query test suite. Those are the natural places for retry-loop validation. The graph route itself is in [app/graph/router.py](app/graph/router.py#L56-L102).
- Reference: [tests/test_graph_routing.py](tests/test_graph_routing.py), [tests/test_transform_query.py](tests/test_transform_query.py), [app/graph/router.py](app/graph/router.py#L56-L102).
- Interview Answer: The retry loop is tested at the routing level by checking when the graph chooses `transform_query` versus `fallback`.

### 159) How did you test hallucination handling?
- Explain: There is a dedicated hallucination-check test file. The project also includes a bonus hallucination-check node, even though it is not wired into the main graph builder at this phase.
- Reference: [tests/test_hallucination_check.py](tests/test_hallucination_check.py), [app/graph/nodes/hallucination_check.py](app/graph/nodes/hallucination_check.py), [app/graph/builder.py](app/graph/builder.py#L1-L72).
- Interview Answer: Hallucination handling is covered by tests and by a dedicated node, though the main graph builder intentionally leaves that bonus loop out of the active path.

### 160) How did you test the API?
- Explain: There is an API endpoint test file for the backend. This is the direct test coverage for the route layer.
- Reference: [tests/test_api_endpoints.py](tests/test_api_endpoints.py).
- Interview Answer: The API endpoints are tested directly, including the route contracts and behaviors.

### 161) What happens if a test document is empty?
- Explain: The loader and chunker skip empty documents with warnings and do not index them. This is the same rule used in production ingestion.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L120-L191), [app/ingestion/chunking.py](app/ingestion/chunking.py#L62-L85).
- Interview Answer: Empty documents are ignored rather than indexed.

### 162) What happens if no relevant chunk exists?
- Explain: The graph router chooses `transform_query` if there is still retry budget, else `fallback`. That means no relevant chunk is treated as a normal self-corrective failure mode, not a successful answer condition.
- Reference: [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/nodes/fallback.py](app/graph/nodes/fallback.py#L64-L115).
- Interview Answer: The system does not answer from empty evidence; it retries or falls back.

### 163) How would you create a RAG golden dataset?
- Explain: I would collect representative user questions with their expected evidence chunks and expected answers, then run them through the full retrieval and generation pipeline to compare actual output to the gold record. This is a design for evaluation, not a code implementation in the current repo.
- Reference: [tests](tests), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L117-L206).
- Interview Answer: A golden dataset would contain queries, relevant source chunks, and the expected answer or key evidence markers so retrieval and answer quality can be measured consistently.

### 164) How would you perform regression testing?
- Explain: I would keep the existing unit tests and rerun them after changes, while also maintaining a fixed evaluation set for retrieval and answer quality. This is the natural strategy for a graph system with multiple moving parts.
- Reference: [tests](tests), [app/graph/router.py](app/graph/router.py#L56-L102).
- Interview Answer: Regression testing would combine targeted unit tests with a small golden dataset so routing, retrieval, and answer outputs remain stable across updates.

### 165) What metric would you optimize first?
- Explain: I would optimize retrieval recall@k first, because if the right evidence never reaches grading, downstream answer quality cannot recover. This is the first metric I would focus on in a retrieval-first system.
- Reference: [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453).
- Interview Answer: Retrieval recall@k would be my first optimization target because all downstream generation quality depends on finding the right evidence first.

---

## L. Production / Scaling / Security

### 166) How would you deploy this system?
- Explain: The code is a Python FastAPI service with a local Chroma persistent directory and a local Ollama model. A production deployment would run the app via a process manager or container, keep Chroma and Ollama as services, and expose FastAPI behind a reverse proxy or load balancer.
- Reference: [app/main.py](app/main.py#L25-L53), [app/services/vector_store.py](app/services/vector_store.py#L44-L97), [app/core/config.py](app/core/config.py#L35-L53).
- Interview Answer: I would deploy the API as a Python service, keep Chroma and Ollama as local or managed services, and front the API with a standard reverse proxy or gateway.

### 167) How would you handle 100 users?
- Explain: The current code is single-process and local-storage based. At 100 users, I would add user-aware isolation, queue background tasks, and constrain concurrency around the shared Chroma store and local LLM server.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L97), [app/api/query.py](app/api/query.py#L33-L72).
- Interview Answer: The main challenges are request concurrency, document isolation, and making sure the local vector store and LLM service can handle parallel loads.

### 168) How would you handle 10,000 users?
- Explain: You would move from one local persistent instance to a multi-instance architecture with a managed vector store, worker processes for ingestion, and a shared deployment for the API. The current code does not provide that architecture yet.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L97), [app/main.py](app/main.py#L25-L53).
- Interview Answer: At that scale, the bottlenecks move to the LLM and vector store, so I would scale both horizontally and isolate the active corpus per user or tenant.

### 169) How would you handle 100,000 users?
- Explain: This requires a managed vector database, background ingestion jobs, user-level isolation, and a scalable API layer. The current code is not designed for that scale.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L97), [app/api/query.py](app/api/query.py#L33-L72), [app/core/config.py](app/core/config.py#L35-L53).
- Interview Answer: A 100k-user design would need strong infrastructure separation, queue-based indexing, and a horizontally scalable API plus managed vector service.

### 170) Which component becomes the bottleneck first?
- Explain: In this code, the likely first bottleneck is the LLM service because each relevance decision and answer generation call hits the local Ollama server. The vector store is also a strong candidate, especially if queries or ingests get heavy. Both are visible in the architecture.
- Reference: [app/services/llm.py](app/services/llm.py#L116-L169), [app/services/vector_store.py](app/services/vector_store.py#L208-L256), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453).
- Interview Answer: The most likely first bottleneck is the LLM call path, followed by the vector store under heavier ingest or query load.

### 171) How would you scale ingestion?
- Explain: The code currently performs local, synchronous ingestion. To scale, I would queue files, process them in workers, and batch inserts into the vector store to reduce I/O and lock contention.
- Reference: [app/api/upload.py](app/api/upload.py#L68-L152), [app/ingestion/pipeline.py](app/ingestion/pipeline.py#L35-L85), [app/services/vector_store.py](app/services/vector_store.py#L107-L140).
- Interview Answer: Ingestion would be moved to a worker queue and batched to avoid blocking the API while indexing large corpora.

### 172) How would you scale retrieval?
- Explain: I would maintain a managed vector database and tune k, document sharding, and caching. The current code is a single local collection and not designed for large distributed retrieval.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L97), [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105).
- Interview Answer: Retrieval scales by moving to a managed or distributed vector store and by controlling the candidate set size.

### 173) How would you scale the API?
- Explain: I would deploy multiple API workers behind a load balancer and keep state minimal and externalized. The current app is a standard FastAPI service with stateless graph execution and no complex server-side session state.
- Reference: [app/main.py](app/main.py#L25-L53), [app/api/query.py](app/api/query.py#L33-L72).
- Interview Answer: A stateless FastAPI deployment with multiple workers behind a load balancer is the natural path for scaling.

### 174) How would you queue document ingestion?
- Explain: The current code does not implement a queue. In a production system, I would add a task queue such as Celery or RQ and have the API dispatch ingestion jobs to workers, then store job status and result metadata.
- Reference: [app/api/upload.py](app/api/upload.py#L68-L152), [app/api/ingest.py](app/api/ingest.py#L36-L69).
- Interview Answer: I would use a background job queue so upload and ingest do not block the API request thread.

### 175) How would you handle concurrent indexing?
- Explain: The current code uses deterministic IDs and Chroma upserts, which helps idempotence, but it does not implement locking or a job scheduler. A production version would need source-level locks or job-level deduplication.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L107-L140), [app/services/vector_store.py](app/services/vector_store.py#L359-L379).
- Interview Answer: I would enforce a single-writer-per-source rule and rely on deterministic IDs so concurrent writes become safe upserts rather than duplicates.

### 176) How would you secure uploaded documents?
- Explain: The app validates file extension and size before accepting uploads. It also stores the files under a controlled directory and rejects unsupported or empty files. This is a good start, but it is not full malware or content scanning.
- Reference: [app/api/upload.py](app/api/upload.py#L68-L152).
- Interview Answer: I would preserve the current extension and size validation and add quarantine, scanning, and strict file-handling policies for untrusted uploads.

### 177) How would you prevent one user from accessing another user’s documents?
- Explain: There is no user isolation in the current implementation. A robust design would namespace documents by user and enforce user-specific access during query and ingestion.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L97), [app/services/document_registry.py](app/services/document_registry.py#L1-L38).
- Interview Answer: I would isolate by user-specific collection or source namespace and enforce that namespace in every query and document access path.

### 178) How would you protect against prompt injection?
- Explain: The current code aims to protect by building the final generation prompt from the accepted chunks and instructing the model to use only that supplied context. That limits prompt injection from the user question but does not sanitize documents before indexing.
- Reference: [app/services/llm.py](app/services/llm.py#L41-L69), [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L117-L206).
- Interview Answer: I would keep the model constrained to retrieved evidence and also treat uploaded documentation as untrusted content that must be sanitized or filtered before indexing.

### 179) How would you protect against malicious documents?
- Explain: The code currently validates file type and size, but there is no malware scanning or sandboxing. The best next step would be to isolate extraction in a safer environment and reject unsupported content before it hits the index.
- Reference: [app/api/upload.py](app/api/upload.py#L68-L152), [app/ingestion/loaders.py](app/ingestion/loaders.py#L120-L191).
- Interview Answer: I would keep the file type/size checks and add stricter sandboxing and content validation around parsing and extraction.

### 180) How would you protect API secrets?
- Explain: The project uses `BaseSettings` with environment-driven config and an optional `.env` file, which is the correct pattern for local config. The code does not hardcode secrets in the source tree.
- Reference: [app/core/config.py](app/core/config.py#L26-L69).
- Interview Answer: I would keep all secrets in environment variables or a secret store instead of embedding them in code.

### 181) What logs would you store?
- Explain: The code uses Python logging across the API, graph, ingestion, and storage layers. The relevant log patterns include warnings on bad files, retrieval failures, and model calls. For production, I would store request IDs, query IDs, ingestion results, and failure reasons.
- Reference: [app/core/logging.py](app/core/logging.py), [app/api/query.py](app/api/query.py#L33-L72), [app/services/vector_store.py](app/services/vector_store.py#L44-L97), [app/services/llm.py](app/services/llm.py#L116-L169).
- Interview Answer: I would log request metadata, query IDs, ingestion counts, retrieval errors, model exceptions, and fallback triggers while keeping the logs keyed to the user and document context.

### 182) What information should never be logged?
- Explain: The code does not define a secret redaction policy, but the safe rule is not to log secret values, tokens, or sensitive document content beyond what is necessary for debugging. This is a general security principle rather than an explicit code requirement.
- Reference: None in code; this is a safe design rule, not an explicit implementation.
- Interview Answer: I would never log API keys, model credentials, passwords, or raw sensitive document contents beyond what is indispensable for system debugging.

---

## M. Code-Level Cross-Examination

### 183) Open the main entry point and explain it line by line.
- Explain: The main entry point is `create_app` in [app/main.py](app/main.py#L25-L53). It constructs the FastAPI app, sets title/description/version, includes the routers for health, query, ingest, upload, documents, and feedback, and installs a global exception handler that converts unexpected errors to HTTP 500. The file ends by creating the module-level `app = create_app()` object.
- Reference: [app/main.py](app/main.py#L25-L53).
- Interview Answer: The entry point wires the app together: routers, app metadata, and one global exception handler.

### 184) Open the ingestion service and explain it line by line.
- Explain: `IngestionPipeline` validates input paths and delegates to `DocumentLoader.load_file` / `load_directory`, then to `DocumentChunker.split`. It returns chunked documents without embeddings or vector-store writes. This keeps the load-and-chunk stage separate from the index stage.
- Reference: [app/ingestion/pipeline.py](app/ingestion/pipeline.py#L35-L85), [app/ingestion/loaders.py](app/ingestion/loaders.py#L110-L191), [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L121).
- Interview Answer: The ingestion service is intentionally a pure load-and-chunk boundary. It does not write to Chroma or invoke the LLM.

### 185) Open the chunking implementation and explain it line by line.
- Explain: `DocumentChunker` initializes a `RecursiveCharacterTextSplitter` with a chunk size and overlap, splits the document, extracts heading offsets from markdown, and attaches `chunk_index` plus `section_heading` to each chunk. It is a centralized chunking layer for all supported ingestion formats.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L121).
- Interview Answer: The chunker defines the exact unit of retrieval and preserves source context using metadata and headings.

### 186) Open the embedding service and explain it line by line.
- Explain: `EmbeddingService` lazily loads a shared `SentenceTransformer` model by name, exposes `embed_documents` and `embed_query`, and converts outputs into plain Python float lists. It is a provider boundary that hides the model implementation.
- Reference: [app/services/embeddings.py](app/services/embeddings.py#L36-L127).
- Interview Answer: The embedding service is the single place responsible for converting text into vectors and maintaining model reuse.

### 187) Open the vector-store service and explain it line by line.
- Explain: `VectorStoreService` initializes a persistent Chroma client, creates or reuses a collection named `technical_docs`, sanitizes metadata for Chroma compatibility, stores embeddings, and runs similarity queries. It also defines source-scoped deletion and conversion from Chroma distance to normalized score.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L210), [app/services/vector_store.py](app/services/vector_store.py#L316-L447).
- Interview Answer: The vector-store service is the concrete storage layer: it owns Chroma, metadata handling, upsert logic, and similarity search.

### 188) Open the retrieval node and explain it line by line.
- Explain: `retrieve_documents` validates the query, calls the vector store, normalizes returned results into `RetrievedDocument`, and returns `retrieved_docs`. It is intentionally thin; the actual search is in `VectorStoreService.similarity_search`.
- Reference: [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105), [app/services/vector_store.py](app/services/vector_store.py#L193-L256).
- Interview Answer: Retrieval is a thin adapter between the graph state and the underlying vector query.

### 189) Open the grading node and explain it line by line.
- Explain: `grade_documents` reads `retrieved_docs`, optionally uses batched LLM relevance grading, validates the results, and falls back to a deterministic heuristic when the LLM fails or is disabled. It returns `graded_docs` and `grading_results`.
- Reference: [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453).
- Interview Answer: The grading node makes the key relevance decision and acts as the self-corrective gate for evidence quality.

### 190) Open the query-rewrite node and explain it line by line.
- Explain: `transform_query` gets the current question, tracks previous queries, chooses the next unique reformulation strategy, increments retry_count, and returns the new query. This closes the retry loop in the graph.
- Reference: [app/graph/nodes/transform_query.py](app/graph/nodes/transform_query.py#L72-L266).
- Interview Answer: Query rewrite is a deterministic retry mechanism designed to diversify the search without repeating earlier failed questions.

### 191) Open the final-generation node and explain it line by line.
- Explain: `generate_answer` validates `graded_docs`, builds a context from those chunks, asks the LLM for a grounded answer, falls back to an extractive answer if needed, derives citations from metadata, and computes confidence. This is the final stage of the graph.
- Reference: [app/graph/nodes/generation.py](app/graph/nodes/generation.py#L101-L315).
- Interview Answer: Generation is the final step that converts accepted evidence into a grounded answer with source citations and a confidence label.

### 192) Open the FastAPI query endpoint and explain it line by line.
- Explain: `/query` in [app/api/query.py](app/api/query.py#L33-L72) builds the initial graph state, runs `graph.invoke`, validates the returned answer and confidence, converts citations to the response model, and returns `QueryResponse`. It also sets `query_id` and tracks `fallback_used` and `retry_count`.
- Reference: [app/api/query.py](app/api/query.py#L33-L72), [app/schemas/query.py](app/schemas/query.py#L9-L31).
- Interview Answer: The route is the API boundary between HTTP and graph execution; it turns the user’s question into state and serializes the graph’s output back to JSON.

### 193) Introduce an error into retrieval and debug it.
- Explain: If retrieval fails, the `VectorStoreServiceError` is caught in `retrieve_documents` and logged. The route layer catches it and converts it to an HTTP 500. There is no special debug harness in the code, but the logging and exception handling are explicit.
- Reference: [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105), [app/services/vector_store.py](app/services/vector_store.py#L193-L256), [app/api/query.py](app/api/query.py#L33-L72).
- Interview Answer: I would make the retrieval call fail and inspect the stack trace / logs; the service is intentionally designed to surface the failure cleanly as a VectorStoreServiceError.

### 194) Change top-k and explain the expected effect.
- Explain: Increasing `k` fetches more neighbors, which can improve recall but increase noise and grading work. Decreasing `k` reduces noise but can miss relevant evidence; the router may then go to retry or fallback.
- Reference: [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105), [app/graph/router.py](app/graph/router.py#L56-L102).
- Interview Answer: Larger `k` gives more recall but more noise; smaller `k` reduces noise but can miss the evidence needed to answer.

### 195) Change chunk size and explain the expected effect.
- Explain: Smaller chunks increase retrieval specificity but may break context; larger chunks may preserve more context but reduce relevance and increase noise. This is the direct tradeoff in the chunker and retrieval path.
- Reference: [app/ingestion/chunking.py](app/ingestion/chunking.py#L43-L121), [app/graph/nodes/grading.py](app/graph/nodes/grading.py#L148-L453).
- Interview Answer: Smaller chunks improve precision but fragment context; larger chunks improve context continuity but reduce specificity.

### 196) Add a retry limit.
- Explain: This is already present as `max_retries` in settings and enforced in the router. The transform node incrementally increases `retry_count` and the graph exits to fallback when the cap is reached.
- Reference: [app/core/config.py](app/core/config.py#L56-L69), [app/graph/router.py](app/graph/router.py#L56-L102), [app/graph/nodes/transform_query.py](app/graph/nodes/transform_query.py#L72-L116).
- Interview Answer: The retry limit already exists as a configuration-bound loop guard in the graph.

### 197) Add a new document type.
- Explain: I would extend `SUPPORTED_EXTENSIONS`, add a new branch in `load_file`, and implement a `load_<type>` extraction routine. This is the clear extension point in the loader.
- Reference: [app/ingestion/loaders.py](app/ingestion/loaders.py#L110-L191).
- Interview Answer: Add the extension to the supported set and a format-specific extraction path in the loader before chunking.

### 198) Add a new metadata filter.
- Explain: The current code has no query-time metadata filter, but the right extension point is the vector-search call or the `VectorStoreService` query wrapper, where the collection could accept metadata filters before or after embedding search.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L193-L256), [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105).
- Interview Answer: I would add an optional metadata filter parameter to `similarity_search` and pass it through the vector-store query.

### 199) Replace the embedding model.
- Explain: This is done by changing the value in `Settings.embedding_model_name` or by passing a different model to `EmbeddingService`. A safe switch requires reindexing the collection to match the new embedding space.
- Reference: [app/core/config.py](app/core/config.py#L47-L69), [app/services/embeddings.py](app/services/embeddings.py#L36-L127), [app/services/vector_store.py](app/services/vector_store.py#L57-L97).
- Interview Answer: Swap the model in the service and rebuild the vector collection; otherwise the old vectors and new embeddings will not match.

### 200) Replace ChromaDB with another vector store.
- Explain: The vector-store abstraction is isolated in `VectorStoreService`, so a new backend would require implementing the same methods: `build_index`, `add_documents`, `replace_documents_for_sources`, `similarity_search`, and the metadata restoration helpers. The graph would not need to change much.
- Reference: [app/services/vector_store.py](app/services/vector_store.py#L44-L210), [app/graph/nodes/retrieval.py](app/graph/nodes/retrieval.py#L49-L105).
- Interview Answer: The graph interacts with the vector store through a stable service interface, so a backend swap would mainly be a service-layer refactor.

### 201) Replace Ollama with another LLM provider.
- Explain: The code isolates provider logic in `LLMService._call_chat`, which is the key replace point. All graph nodes call the service’s high-level methods, not a provider-specific client, so the swap can be localized.
- Reference: [app/services/llm.py](app/services/llm.py#L77-L209).
- Interview Answer: I would keep the same service interface and swap only the provider-specific HTTP implementation behind `_call_chat`.

---

## Interview Answer (short form)

This project is a local, self-corrective RAG assistant for technical documents. The system ingests supported document types, chunks them, embeds them into ChromaDB, retrieves candidate chunks for a user question, grades them for relevance, retries the query when needed, and returns a grounded answer with citations and a low-confidence fallback when the evidence is insufficient. The core code is organized around a LangGraph state machine in [app/graph/builder.py](app/graph/builder.py#L85-L122), a vector store in [app/services/vector_store.py](app/services/vector_store.py#L44-L210), and an Ollama-backed LLM boundary in [app/services/llm.py](app/services/llm.py#L77-L209). The design goal is not merely to chat with documents, but to answer from retrieval evidence, preserve source traceability, and fail safely when the corpus does not contain enough support.

---

## Final note

This write-up is intentionally evidence-based and conservative. Where the code does not implement a feature or where the implementation is a bonus not connected to the active graph, I call that out directly instead of guessing.
