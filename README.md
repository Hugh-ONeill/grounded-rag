# grounded-rag

> Local, citation-grounded question answering over any document corpus: pgvector retrieval,
> a tuned "I don't know" gate, source attribution, and an evaluation harness that measures
> whether the answers are actually right. No API keys, no per-token cost.

![demo](docs/demo.gif)

## What it is

Ask a question, get an answer that **quotes its sources** and says *"I don't know"* when the
retrieved context doesn't support an answer. Runs entirely locally on a single GPU, using
Ollama for both embeddings and generation.

## Why it exists

Most RAG demos wire up libraries and never check whether retrieval works. This one is built
around an evaluation harness with gold questions, and the harness earned its keep on day one:
the first eval run revealed that the anti-hallucination gate had never fired (details below).
Every answer is grounded in numbered passages, so you can see exactly where it came from.

## Architecture

```
Documents → Chunk (overlap + metadata) → Embed (Ollama) → Postgres + pgvector
                                                               │
Query → Embed → Vector retrieve (cosine top-k) → "I don't know" gate
                                                               │
                       Gemma (Ollama) + retrieved context → answer + [citations]
                                                               │
                                  eval/ harness → hit-rate@k, faithfulness, refusal
```

Hybrid retrieval (vector + BM25) and cross-encoder reranking are roadmap items; their
signatures are already stubbed in [`api/retrieve.py`](api/retrieve.py). See
[docs/architecture.md](docs/architecture.md) for the detailed flow.

## Stack & rationale

| Layer | Choice | Why |
|------|--------|-----|
| Vector store | **Postgres + pgvector** | Production-grade, SQL-backed; not a throwaway in-memory store |
| Embeddings | **`nomic-embed-text` (Ollama)** | Local, free, runs on one GPU |
| LLM | **Gemma (Ollama)** | Local generation, no API cost |
| Backend | **FastAPI** | Async, typed, streams responses (SSE) |
| Frontend | **React + TypeScript (Vite)** | Streaming chat UI with cited sources |
| Infra | **Docker Compose** (→ k8s manifests) | One command to run the whole system |

## Key engineering decisions

**Documents are citable units.** Corpus adapters yield small, self-contained documents (one
Pokemon's usage profile, one team paste), so a citation like `gen9ou_chaos#Kingambit` points
at something a reader can actually check, rather than "page 37 of a blob".

**Chunking is a 900-character window with 150 overlap.** Most per-species documents fit in a
single chunk, so retrieved passages carry complete fact blocks. Simple character windows stay
until the eval shows they are the weak link.

**The "I don't know" threshold is tuned, not guessed.** The gate refuses when top-1 cosine
similarity falls below `MIN_SIMILARITY`. It was first set to 0.30, and the eval immediately
reported 0% refusal precision: cosine similarities from `nomic-embed-text` run hot, and even
"What is the capital of France?" retrieves at 0.40 against this corpus. Measured over the gold
set, on-topic questions score 0.64 to 0.78 and off-topic ones 0.40 to 0.54, so the threshold
now sits at 0.60, inside the clean gap between the two bands. This is the project's thesis in
miniature: without an eval, the gate would have silently never fired.

**Aggregation questions get synthetic rankings documents.** "What is the most used Pokemon?"
is a corpus-wide comparison, and no single species chunk contains the answer, so top-k
retrieval returns famous-sounding species and the model answers with the max over whatever it
happened to see. An early build confidently said Heracross at 6.1% when the true answer is
Snorlax on 96.6% of teams. The fix has two halves: the corpus adapter emits one usage-rankings
document per format, making the ranking itself a citable chunk, and the prompt forbids
computing superlatives by comparing retrieved passages. Notably, hybrid retrieval and
reranking would not have helped here; this failure class needs a corpus-shape fix, and it is
now covered by its own gold questions.

**The grounding prompt is strict.** The generator sees only the numbered passages, must cite
as `[n]`, and is instructed to refuse rather than invent. Thinking mode is disabled
(`think: false`): factual extraction does not benefit from it and it is roughly 10x slower
for this shape of answer.

**Retrieval is deliberately simple, for now.** Pure vector search already scores 100%
hit-rate@5 on the current gold set, so hybrid retrieval and reranking wait until a corpus or
question set exposes an actual vector-only failure, rather than being added for the buzzwords.

## Evaluation

Run it yourself: `python -m eval.run_eval`. Over 30 gold questions grounded in real Smogon
usage data (27 answerable, including corpus-wide aggregations, and 3 deliberately
unanswerable), the current build scores:

| Metric | Score |
|--------|-------|
| Retrieval hit-rate@k | 100% (27/27) |
| Answer faithfulness | 100% (24/24) |
| Refusal precision (no-answer) | 100% (3/3) |

Method: hit-rate@k checks that the expected source appears among the retrieved top-k;
faithfulness checks that the generated answer contains expected key terms; refusal precision
checks that the gate fires on unanswerable questions. Retrieval and refusal are deterministic;
generation is temperature-sampled, so faithfulness moves between 95% and 100% across runs.
Gold questions live in [eval/questions.yaml](eval/questions.yaml).

## Corpora

The engine is corpus-agnostic via a small [`CorpusLoader`](api/corpora/base.py) interface.
Shipped adapters:

- **`markdown_dir`**: point it at any folder of `.md` / `.txt` files.
- **`crystal_battle`**: a competitive-Pokemon meta analyst over real Smogon usage statistics
  (per-species moves, items, abilities, teammates, and checks for Gen 9 OU and Gen 2 OU) plus
  example team builds, extracted from the [crystal-battle](https://github.com/Hugh-ONeill/crystal-battle)
  project. Ask *"What item does Kingambit most commonly run?"* and get a cited answer from
  actual ladder data.

## Run it

```bash
cp .env.example .env            # set CORPUS, OLLAMA_HOST, model names
docker compose up -d                       # postgres+pgvector, api, web
docker compose exec api python -m ingest   # build the index from the configured corpus
# open http://localhost:5173
```

No Docker? Any Postgres with the pgvector extension works:

```bash
python -m venv .venv && .venv/bin/pip install -e ./api
# point DATABASE_URL in .env at your Postgres, then:
.venv/bin/python api/ingest.py
.venv/bin/uvicorn --app-dir api main:app --port 8000
.venv/bin/python -m eval.run_eval
```

## Roadmap

- [x] MVP: ingest → vector search → cited answer → chat UI
- [x] Streaming responses (SSE)
- [x] "I don't know" threshold, tuned via the eval
- [x] Eval harness with published numbers
- [ ] Hybrid retrieval (vector + BM25) and cross-encoder reranking
- [ ] Monotype moveset tables and replay ingestion for the crystal-battle corpus
- [ ] Validate the k8s manifests end to end
- [ ] Query rewriting, conversation memory
