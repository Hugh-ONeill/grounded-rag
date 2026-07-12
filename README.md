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
Query → Embed → Hybrid retrieve (vector + keyword, RRF) → cross-encoder rerank → gate
                                                               │
                       Gemma (Ollama) + retrieved context → answer + [citations]
                                                               │
                                  eval/ harness → hit-rate@k, faithfulness, refusal
```

See [docs/architecture.md](docs/architecture.md) for the detailed flow.

## Stack & rationale

| Layer | Choice | Why |
|------|--------|-----|
| Vector store | **Postgres + pgvector** | Production-grade, SQL-backed; not a throwaway in-memory store |
| Embeddings | **`nomic-embed-text` (Ollama)** | Local, free, runs on one GPU |
| Reranker | **ms-marco MiniLM cross-encoder** | Reads (question, passage) together; also scores the refusal gate |
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

**The "I don't know" threshold is tuned, not guessed, and re-tuned as the corpus grows.**
The gate refuses when top-1 cosine similarity falls below `MIN_SIMILARITY`. It was first set
to 0.30, and the eval immediately reported 0% refusal precision: cosine similarities from
`nomic-embed-text` run hot, and even "What is the capital of France?" retrieves at 0.40.
Measured bands on the first corpus (on-topic 0.64 to 0.78, off-topic 0.40 to 0.54) put the
threshold at 0.60. Then the corpus grew 10x with the PokeAPI adapter and the eval caught the
gate breaking again: "What is the boiling point of water?" started retrieving Hydro Steam, a
Water-type move about boiling water, at 0.649. The threshold moved to 0.66 and the margin
between bands narrowed from ~0.10 to ~0.02. Then it closed entirely: "Is a Sitrus Berry better
than an Oran Berry?" tops out at 0.639 while the boiling-point question reaches 0.649, so no
similarity threshold can separate them. The gate is now scored by the cross-encoder reranker,
whose bands do separate (answerable questions bottom out at 0.099, off-topic tops at 0.059)
and, unlike cosine bands, do not depend on how crowded the embedding space is. Model choice
was measured too, with a surprise: tiny ms-marco MiniLM beats bge-reranker-v2-m3 here, because
bge's off-topic floor (~0.50) sits inside its own answerable band while MiniLM pins off-topic
pairs to 0.000. Keyword-rank and similarity thresholds survive as the fallback gate when the
reranker is disabled. Without an eval, none of this drift would have been visible.

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

**Hybrid retrieval was added when the measurements demanded it, not before.** Pure vector
search carried a 100% hit-rate until the corpus grew and comparison questions arrived:
"Is a Sitrus Berry better than an Oran Berry?" names two exact entities, but the chatty
phrasing dilutes the embedding below the refusal threshold, while an off-topic question with
adjacent vocabulary scores higher. The keyword leg (Postgres full-text, titles weighted
heaviest, OR semantics so multi-entity comparisons match) catches exactly what the vector leg
dilutes; reciprocal rank fusion merges the two, and the cross-encoder reorders the fused pool.
Comparisons between things in context are also explicitly permitted by the prompt, which
previously blocked them along with corpus-wide superlatives.

**Postgres full-text search has no IDF, so term selection supplies it.** First contact with
the keyword leg was a flood: for "What is Garganacl's most used move?", OR-querying every
question word made 900 move documents title-match the word "move", and ts_rank scores a title
hit on generic "move" identically to one on rare "garganacl". The fix computes lexeme document
frequencies once (ts_stat) and searches only the question's rarest terms, which is exactly the
IDF weighting the built-in ranking lacks. The eval caught this as a real retrieval regression
the moment the corpus grew past what the vector leg could carry alone.

## Evaluation

Run it yourself: `python -m eval.run_eval`. Over 48 gold questions spanning both corpora
(45 answerable, covering usage stats, corpus-wide aggregations, stat superlatives, species
data, moves, abilities, items, learnsets, in-context comparisons, and usage-versus-movepool
intent, plus 3 deliberately unanswerable), the current build scores:

| Metric | Score |
|--------|-------|
| Retrieval hit-rate@k | 100% (45/45) |
| Answer faithfulness | 100% (41/41) |
| Refusal precision (no-answer) | 100% (3/3) |

Method: hit-rate@k checks that the expected source appears among the retrieved top-k;
faithfulness checks that the generated answer contains expected key terms; refusal precision
checks that the gate fires on unanswerable questions. Retrieval and refusal are deterministic;
generation is temperature-sampled, so faithfulness moves between 95% and 100% across runs.
Gold questions live in [eval/questions.yaml](eval/questions.yaml).

## Corpora

The engine is corpus-agnostic via a small [`CorpusLoader`](api/corpora/base.py) interface,
and the index is multi-corpus: every chunk carries a corpus tag, ingest rebuilds one corpus
at a time, and retrieval (and the `/ask` API) can optionally scope to a single corpus.
Shipped adapters:

- **`markdown_dir`**: point it at any folder of `.md` / `.txt` files.
- **`crystal_battle`**: a competitive-Pokemon meta analyst over real Smogon usage statistics
  (per-species moves, items, abilities, teammates, and checks for Gen 9 OU and Gen 2 OU) plus
  example team builds, extracted from the [crystal-battle](https://github.com/Hugh-ONeill/crystal-battle)
  project. Ask *"What item does Kingambit most commonly run?"* and get a cited answer from
  actual ladder data.
- **`pokeapi`**: general Pokemon knowledge from the [PokeAPI](https://github.com/PokeAPI/pokeapi)
  CSV dataset: one citable document per species (types, base stats, abilities, Pokedex
  entries), per move, per ability, per item, and per learnset. ~4,500 documents; a sparse
  checkout of just `data/v2/csv` is enough (see `.env.example`). Two dataset gaps are handled
  in the adapter: items missing effect prose fall back to their flavor text, and gen 9 items
  (Booster Energy, Covert Cloak, ...) have no text in the dataset at all, so the top few are
  hand-transcribed and marked as such in their metadata.

## Run it

```bash
cp .env.example .env            # set OLLAMA_HOST, model names, corpus paths
docker compose up -d                       # postgres+pgvector, api, web
docker compose exec api python -m ingest crystal_battle pokeapi   # build the index
# open http://localhost:5173
```

No Docker? Any Postgres with the pgvector extension works:

```bash
python -m venv .venv && .venv/bin/pip install -e ./api
# point DATABASE_URL in .env at your Postgres, then:
.venv/bin/python api/ingest.py crystal_battle pokeapi
.venv/bin/uvicorn --app-dir api main:app --port 8000
.venv/bin/python -m eval.run_eval
```

## Roadmap

- [x] MVP: ingest → vector search → cited answer → chat UI
- [x] Streaming responses (SSE)
- [x] "I don't know" threshold, tuned via the eval
- [x] Eval harness with published numbers
- [x] Hybrid retrieval (vector + weighted full-text, reciprocal rank fusion)
- [x] Cross-encoder reranking, now also scoring the refusal gate
- [ ] Monotype moveset tables and replay ingestion for the crystal-battle corpus
- [ ] Validate the k8s manifests end to end
- [ ] Query rewriting, conversation memory
