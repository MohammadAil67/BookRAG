# BookRAG — Architecture & Design

## High-Level Overview

BookRAG is an AI-powered tutoring assistant aimed at **middle and high school students**. It ingests a textbook PDF and lets students ask questions in natural language, receive grounded answers, and take auto-generated quizzes — all through a multi-page Streamlit web app.

The system supports both **English and Bengali** throughout the entire pipeline (query understanding, retrieval, answer generation, and topic tracking).

### User-Facing Features

| View | Description |
|---|---|
| **Chat** | Conversational Q&A backed by the book content |
| **Practice** | Auto-generated multiple-choice quizzes by topic and difficulty |
| **Study Plan** | Structured study scheduling |
| **Progress Tracker** | Tracks quiz scores and conversation history |
| **System Logs** | Real-time pipeline debug output |

### System Entry Points

- **`ui_main.py`** — Streamlit app; routes between the five views above
- **`main.py`** — Minimal CLI loop for testing without the UI

---

## Component Map

```
PDFprocessing.py      ← OCR ingestion (Tesseract + Poppler, multithreaded)
       │
       ▼
  chunks + embeddings (cached as .pkl files)
       │
       ▼
rag_system.py (RAGSystem)
  ├── models.py         ← GroqLLM, Embedder (BGE-M3), Reranker (CrossEncoder)
  ├── retrieval.py      ← HybridRetriever (BM25 + vector), MultiQueryRetriever
  ├── processing.py     ← PromptBuilder, AdvancedQueryRefiner, MultiQueryGenerator
  ├── query_decomposition.py ← QueryDecomposer, SmartContextApplier
  └── topics.py         ← EnhancedTopicTracker, CitationValidator
       │
       ▼
config.py             ← Config, SystemUtils, HistoryObject
```

---

## Technical Deep Dive

### 1. PDF Ingestion (`PDFprocessing.py`)

- Converts each PDF page to an image using **pdf2image** (Poppler backend).
- Runs **Tesseract OCR** with `eng+ben` language pack concurrently across pages using `ThreadPoolExecutor` (defaults to 80 % of CPU threads).
- Applies `clean_ocr_text()` to normalise whitespace, fix common OCR artefacts, and handle Bengali punctuation (e.g. `|` → `।`).
- Chunks the concatenated text with **LangChain `RecursiveCharacterTextSplitter`** (`chunk_size=1000`, `overlap=200`).
- Results are **pickled** to `<pdf_name>_chunks.pkl` / `<pdf_name>_bge_embeddings.pkl` so re-runs skip OCR entirely.

### 2. Embedding & Indexing (`rag_system.py`, `models.py`)

- **Model**: `BAAI/bge-m3` via `SentenceTransformer`, **INT8 quantized** with `torch.quantization.quantize_dynamic` for faster CPU inference.
- All chunk embeddings are L2-normalised and persisted in the `.pkl` cache.
- A **BM25** index (`rank-bm25`) is built in-memory on every startup from the cached chunks.

### 3. Retrieval Stack (`retrieval.py`)

#### HybridRetriever
1. **Vector search** — cosine similarity between the query embedding and all chunk embeddings; top-`INITIAL_RETRIEVAL_K` (15) results.
2. **BM25 search** — top-`INITIAL_RETRIEVAL_K` results by BM25 score.
3. **Candidate union** — deduplicated set from both searches.
4. **Reranking** — `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` scores each candidate; top-`FINAL_TOP_K` (11) are kept.

#### MultiQueryRetriever
- Wraps `HybridRetriever` with **parallel multi-query expansion**: the LLM generates 2 rephrasings of the original query, all three are retrieved concurrently via `ThreadPoolExecutor`, candidates are merged, and a final reranking pass is run against the **original** query.

### 4. Query Intelligence (`processing.py`, `query_decomposition.py`)

| Class | Role |
|---|---|
| `AdvancedQueryRefiner` | Detects pronouns / vague follow-ups and uses the LLM to rewrite the query with full context before retrieval. |
| `MultiQueryGenerator` | Produces `n` alternative phrasings to improve recall. |
| `QueryDecomposer` | Classifies queries as *comparison*, *analytical*, *multi-part*, or *simple*. Complex types are broken into 2–4 focused sub-queries via rule-based patterns with an LLM fallback. |
| `SmartContextApplier` | Appends the current tracked topic to vague short queries. |

Language detection uses a simple Unicode range check (Bengali: `\u0980–\u09FF`). All classes contain bilingual prompt templates and keyword sets.

### 5. LLM (`models.py`)

- **Provider**: [Groq](https://groq.com) — `llama-3.1-8b-instant`
- **Defaults**: `temperature=0.5`, `max_tokens=1500`
- Query variant generation uses `temperature=0.7`; query rewriting uses `temperature=0.3`.

### 6. Answer Generation & Grounding (`rag_system.py`)

Simple query path (`_ask_simple`):
1. Get topic context → refine query → retrieve chunks → build prompt → generate answer.
2. **Citation validation**: embed the answer and each retrieved chunk, compute cosine similarities. If `max_similarity < 0.6`, the answer is regenerated with a stricter "use only the context" instruction.

Decomposed query path (`_ask_with_decomposition`):
1. Decompose into sub-queries → parallel retrieval for each → merge unique chunks.
2. If more than `FINAL_TOP_K` candidates exist, rerank with the original query.
3. Use a specialised prompt template per decomposition type (comparison vs. analytical vs. general).
4. Apply the same citation grounding check.

### 7. Topic Tracking (`topics.py`)

`EnhancedTopicTracker` maintains a list of `Topic` objects with:
- **Semantic merging**: if a new topic embedding is cosine-similar (> 0.75) to an existing one, keywords and entity mentions are merged rather than creating a duplicate.
- **Confidence decay**: topics not mentioned in recent turns lose confidence (`−0.1 × turns_elapsed`).
- **Hard reset**: explicit topic-change phrases trigger a 70 % confidence cut across all topics.
- **Dynamic K**: retrieval `K` scales with the confidence of the current topic (5 / 7 / 10 for high / medium / low confidence).

### 8. Configuration (`config.py`)

| Parameter | Default | Purpose |
|---|---|---|
| `INITIAL_RETRIEVAL_K` | 15 | Candidates pulled per search type |
| `FINAL_TOP_K` | 11 | Chunks passed to the LLM |
| `RERANK_THRESHOLD` | −2.0 | Minimum reranker score to keep a chunk |
| `MAX_CONVERSATION_HISTORY` | 10 | Turns kept in the rolling chat context |

Environment variables `PDF_PATH`, `GROQ_API_KEY`, `POPPLER_PATH`, and `TESSERACT_PATH` override defaults.

---

## Data Flow (End-to-End)

```
User question
     │
     ▼
QueryDecomposer.should_decompose()
     ├── Simple ──► AdvancedQueryRefiner.refine()
     │                    │
     │                    ▼
     │             MultiQueryRetriever.retrieve()
     │             (parallel HybridRetriever × 3 queries)
     │                    │
     │                    ▼
     │             PromptBuilder.build() ──► GroqLLM.generate()
     │                    │
     │                    ▼
     │             CitationValidator.validate_answer()
     │                    │ low confidence?
     │                    ▼
     │             GroqLLM.generate() [stricter prompt]
     │
     └── Complex ► QueryDecomposer.decompose()
                   (2-4 sub-queries)
                         │
                         ▼
                   MultiQueryRetriever per sub-query (parallel)
                         │
                         ▼
                   Reranker (if > FINAL_TOP_K chunks)
                         │
                         ▼
                   _build_decomposed_prompt() ──► GroqLLM.generate()
                         │
                         ▼
                   CitationValidator.validate_answer()

Both paths:
     │
     ▼
EnhancedTopicTracker.update()   ← semantic topic merge / decay
_update_history()               ← rolling conversation window
     │
     ▼
Answer returned to UI / CLI
```

---

## Key Dependencies

| Package | Role |
|---|---|
| `groq` | LLM API client (LLaMA 3.1) |
| `sentence-transformers` | BGE-M3 embedder + CrossEncoder reranker |
| `rank-bm25` | BM25 sparse retrieval |
| `scikit-learn` | Cosine similarity |
| `torch` | INT8 model quantization |
| `pytesseract` / `pdf2image` | OCR pipeline |
| `langchain-text-splitters` | Recursive chunking |
| `streamlit` | Web UI |
| `plotly` | Progress charts |
