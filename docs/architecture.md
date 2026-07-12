# Architecture

## Flow

1. **Ingest** (`api/ingest.py`) — a corpus adapter yields documents → text is chunked
   (window + overlap) → each chunk is embedded via Ollama → rows land in Postgres/pgvector.
2. **Retrieve** (`api/retrieve.py`) — the question is embedded; pgvector returns the nearest
   chunks by cosine similarity. (V2: fuse with keyword/BM25, then cross-encoder rerank.)
3. **Gate** — if the top similarity is below `MIN_SIMILARITY`, refuse instead of hallucinate.
4. **Generate** (`api/llm.py`) — Gemma is given the numbered passages and a strict
   grounding prompt; it answers with `[n]` citations, streamed token-by-token over SSE.
5. **Evaluate** (`eval/run_eval.py`) — gold questions measure retrieval hit-rate, answer
   faithfulness, and refusal precision.

## Why these choices

- **pgvector over a toy vector DB** — one datastore, real SQL, production-credible, and it
  lets V2 hybrid retrieval reuse Postgres full-text search.
- **Local Ollama for embeddings + generation** — zero API cost, runs on a single GPU, and
  matches an AI-native workflow.
- **Corpus adapter interface** — the engine is generic; swapping the knowledge base is a
  ~40-line file (`api/corpora/`). Ships with a generic markdown loader and a crystal-battle
  competitive-Pokemon analyst.

## Component map

| Path | Responsibility |
|------|----------------|
| `api/corpora/` | Pluggable knowledge sources → documents |
| `api/ingest.py` | Chunk → embed → store |
| `api/db.py` | pgvector schema + writes |
| `api/retrieve.py` | Vector search (+ V2 hybrid/rerank) |
| `api/llm.py` | Grounded, cited, streamed generation |
| `api/main.py` | FastAPI `/ask` (SSE) |
| `web/` | React + TS streaming chat UI |
| `eval/` | Gold questions + metrics |
