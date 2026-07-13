# Architecture

## Flow

1. **Ingest** (`api/ingest.py`) — a corpus adapter yields documents → text is chunked
   (window + overlap) → each chunk is embedded via Ollama → rows land in Postgres/pgvector.
2. **Condense** (`api/llm.py`) — when the request carries chat history, the follow-up is
   rewritten into a standalone question ("what about its speed?" → "what is Kingambit's
   speed?"). This is the whole conversational-memory mechanism: the server keeps no session
   state, and everything downstream still sees a single self-contained question.
3. **Retrieve** (`api/retrieve.py`) — the question is embedded; pgvector returns the nearest
   chunks by cosine similarity. (V2: fuse with keyword/BM25, then cross-encoder rerank.)
4. **Gate** — if the top similarity is below `MIN_SIMILARITY`, refuse instead of hallucinate.
5. **Generate** (`api/llm.py`) — Gemma is given the numbered passages and a strict
   grounding prompt; it answers with `[n]` citations, streamed token-by-token over SSE.
   History never enters this prompt: answers stay grounded in the fresh passages only.
6. **Evaluate** (`eval/run_eval.py`) — gold questions measure retrieval hit-rate, answer
   faithfulness, refusal precision, follow-up (condense → retrieve) hit-rate, paraphrase
   hit-rate (frozen rewordings, see `eval/gen_paraphrases.py`), and ungrounded entity
   mentions; `eval/band_report.py` reports gate-signal bands and margins.

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
| `api/main.py` | FastAPI `/ask` (SSE) + `/retrieve` (JSON, no generation) |
| `web/` | React + TS streaming chat UI |
| `eval/` | Gold questions + metrics |
