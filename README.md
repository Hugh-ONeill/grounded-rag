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

Query (follow-up? + history → standalone rewrite)
  → Router ── computed (matchup / speed / stat query)? → tools → pseudo-passages ──────┐
           └── otherwise → Embed → Hybrid retrieve (RRF) → cross-encoder rerank → gate ┤
                                                                                       │
                              Gemma (Ollama) + passages → answer + [citations]
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
| Tools | **Type chart + stats, computed** | Deterministic answers rendered as citable pseudo-passages |
| Infra | **Docker Compose** (→ k8s manifests) | One command to run the whole system |

## Key engineering decisions

**Documents are citable units.** Corpus adapters yield small, self-contained documents (one
Pokemon's usage profile, one team paste), so a citation like `gen9ou_chaos#Kingambit` points
at something a reader can actually check, rather than "page 37 of a blob".

**Chunking is a 900-character window with 150 overlap, and continuation chunks keep their
provenance.** Most per-species documents fit in a single chunk, so retrieved passages carry
complete fact blocks. The promised "until the eval shows chunking is the weak link" moment
arrived with the long Smogon analyses: a mid-document window carried no hint of what document
it came from, which starved the reranker and broke format attribution in answers. Every
continuation chunk is now stamped with its document title at ingest.

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

**Conversational memory is a query rewrite, not server state.** Every stage of the pipeline
(router rules, entity matching, embedding, the reranker gate) is keyed on the literal question
text, so a follow-up like "what about its speed?" used to retrieve garbage or refuse. The fix
is one step at the front: when a request carries chat history, Gemma condenses the follow-up
into a standalone question ("Is it weak to Fighting?" → "Is Kingambit weak to Fighting?"),
and everything downstream, including tool routing and the eval, still sees a single
self-contained question, unchanged in meaning and tuning. Two deliberate constraints: the
server holds no sessions (the transcript rides along in the request, so any frontend that
owns its own conversation state, like an embedded companion or agent, can use `/ask` as a
pure knowledge tool), and history never enters the answer prompt, so a prior answer can never
become uncited evidence. Clients that narrate for themselves get `/retrieve`: the same routed
passages, tool outputs, gate verdict, and follow-up condensation as `/ask`, returned as plain
JSON with no generation step, so computed answers (calcs, matchups, speed checks) come back
in milliseconds and the external voice does its own narrating instead of paraphrasing Gemma's. The UI surfaces the rewrite as "searched as: ...", which makes
condensation failures visible instead of silent, and the eval scores follow-ups as their own
metric so single-turn numbers stay comparable. The follow-up gold questions immediately
caught something real: "what about its stock price?" condenses to "what about Kingambit's
stock price?", and a question that names a corpus entity scores that entity's own page right
at the reranker's answerable floor. No retrieval signal can call it unanswerable, because
the entity genuinely is in the corpus. Refusal turns out to be layered: the gate catches
off-topic questions, and the strict grounding prompt catches on-entity questions the sources
cannot answer. The refusal metric now accepts either layer and reports which one fired.

**Hybrid retrieval was added when the measurements demanded it, not before.** Pure vector
search carried a 100% hit-rate until the corpus grew and comparison questions arrived:
"Is a Sitrus Berry better than an Oran Berry?" names two exact entities, but the chatty
phrasing dilutes the embedding below the refusal threshold, while an off-topic question with
adjacent vocabulary scores higher. The keyword leg (Postgres full-text, titles weighted
heaviest, OR semantics so multi-entity comparisons match) catches exactly what the vector leg
dilutes; reciprocal rank fusion merges the two, and the cross-encoder reorders the fused pool.
Comparisons between things in context are also explicitly permitted by the prompt, which
previously blocked them along with corpus-wide superlatives.

**Computed questions route to deterministic tools, and the LLM stays the narrator.**
"Is Earthquake effective against Skarmory?" is not a retrieval question: the answer is type
math. A rule-based router recognizes matchup, speed-comparison, speed-tier ("what outspeeds
Garchomp?": faster mons from the usage top 30, plus the Choice Scarf ×1.5 math), and
stat-superlative questions
(only when it positively identifies real Pokemon, move, or type names; anything uncertain
falls through to retrieval) and calls tools that compute the answer from the PokeAPI data:
the full type chart, base stats, per-Pokemon abilities. Tool output is rendered as a citable
pseudo-passage, so the citation prompt, the refusal gate, and the eval treat it exactly like
retrieved text. Damage questions ("Does Garchomp's Earthquake OHKO Heatran?") go one level
deeper: they run through the [poke-engine](https://github.com/pmariglia/poke-engine) battle
engine (the same Rust engine behind the crystal-battle agent), with real base stats and
stated assumptions (level 100, neutral spreads, no items or abilities), returning true damage
rolls and n-HKO verdicts. Battle state is parsed from the question (boosts, setup moves,
items, natures, EV spreads, burn, weather, terrain, Tera, and doubles spread reduction),
with modifiers bound to the Pokemon they describe ("a bold max HP Milotic" invests the
defender; "Choice Band Scizor" equips the attacker), and "what would it take to OHKO?" questions
run a tiered escalation search: levers are applied cumulatively, largest structural choices
first (nature, then EVs with IVs always assumed perfect, then item, then weather, Tera, and
boosts), recomputing engine rolls at each rung until the KO is guaranteed, with immunities
correctly reported as unfixable by investment. The same search runs in reverse for defenders:
"what would Garganacl need to survive Choice Band Garchomp's Earthquake?" escalates defensive
nature, HP and defense EVs, Assault Vest, a screen, weather, and the best defensive Tera
(picked from the type chart, including outright immunities) until the hit is survived.
Sources follow the answer, not just the question: when the generated answer names an entity
(the priority move turns out to be Sucker Punch), that entity's own documents — its data entry
and its Bulbapedia article — are appended to the cited sources in a second SSE event. The
damage tools also cross-reference their own answers against the rest of the corpus: a
calc into Bronzong reports the raw numbers and then warns that Bronzong runs Levitate on 95%
of its observed gen9ou sets, which makes it immune — the engine assumes no ability, and the
tool says so instead of letting a technically-correct number mislead. When the immunity is
certain (Levitate is Rotom-Wash's only possible ability, or observed usage is 100%), the
answer flips to definitive: "no, it does nothing — unless the immunity is removed (Gravity,
Smack Down, an Iron Ball, or a Mold Breaker attacker), in which case: guaranteed OHKO."  The tools also carry the conditional mechanics a raw type chart misses: a
Ground-vs-Flying immunity is reported together with its Gravity/Smack Down/Iron Ball and
Roost variants (with recomputed multipliers), and ability-based immunities like Levitate are
flagged from data. Deliberately not LLM function-calling: the router is a dozen lines of
rules that cannot hallucinate a tool call, and tool questions get gold answers with
deterministic ground truth.

**Every corpus growth breaks something measurable, and the eval finds it.** Adding the
Bulbapedia corpus immediately produced two regressions. Generic mechanics pages seduced the
reranker (a page defining "priority" outranked Kingambit's own stats for "what priority move
does Kingambit carry"), fixed by an ordering-only boost for passages whose title carries one
of the question's rare terms. And popular Pokemon names turned out to exceed the keyword leg's
document-frequency ceiling, because every teammate list mentions them; the ceiling was also
computed against the wrong denominator (the max lexeme frequency rather than the document
count). The boost deliberately does not feed the refusal gate: a first attempt that did let
"Point Card" open the gate for a question about boiling points, and refusal precision fell to
33% until ordering and gating were separated.

**Phrasing robustness is measured, and the first measurement scored 78%.** The gold set only
ever tested one phrasing per fact, and past retrieval bugs had all hidden behind exactly that:
each was found by ad-hoc poking, not by the eval. So every gold question now has frozen Gemma
rewordings (entity names guarded verbatim) scored as their own row, and the first sweep's 51
misses were all real. The router knew "OHKO" but not "one-shot", "OHKO'd", or "knocked out in
one hit"; it knew "faster" but not "compare the speed"; "fastest" but not "quickest";
"effective against" but not "susceptible to". The fix widens the verb vocabularies, which is
safe here because every router branch still requires positively identified entities before it
fires, and moves the calc branches ahead of matchup so both can be broad without hijacking
each other. The sweep also caught a retrieval class the unigram IDF cannot serve: every word
of "Which Pokemon is known as the Sleeping Pokemon?" is too common to survive the document
frequency ceiling, but the phrase is unique, so the keyword leg now falls back to
adjacent-bigram phrase queries (with accent variants, because the corpus spells "Pokémon" and
questions type "pokemon"). Three sweeps later the row reads 100%, and each fix is pinned by
the paraphrase that found it.

**Postgres full-text search has no IDF, so term selection supplies it.** First contact with
the keyword leg was a flood: for "What is Garganacl's most used move?", OR-querying every
question word made 900 move documents title-match the word "move", and ts_rank scores a title
hit on generic "move" identically to one on rare "garganacl". The fix computes lexeme document
frequencies once (ts_stat) and searches only the question's rarest terms, which is exactly the
IDF weighting the built-in ranking lacks. The eval caught this as a real retrieval regression
the moment the corpus grew past what the vector leg could carry alone.

## Evaluation

Run it yourself: `python -m eval.run_eval`. Over 87 gold questions (77 single-turn answerable,
covering usage stats, corpus-wide aggregations, stat superlatives, species data, moves,
abilities, items, learnsets, encyclopedic prose, competitive strategy, in-context comparisons,
usage-versus-movepool intent,
and computed answers:
type matchups with conditional immunities, speed checks, typed stat queries, battle-state-aware
engine damage calculations, and tiered OHKO and survival escalation searches; 6 conversational
follow-ups whose referent lives in a prior turn, including one that must route into a tool
after condensation; 4 deliberately unanswerable, one of them a follow-up; plus 237 frozen
paraphrases of the whole set), the current build scores:

| Metric | Score |
|--------|-------|
| Retrieval hit-rate@k | 100% (77/77) |
| Follow-up hit-rate (condense → retrieve) | 100% (6/6) |
| Paraphrase hit-rate | 100% (237/237) |
| Answer faithfulness | 100% (72/72) |
| Refusal precision (no-answer) | 100% (4/4: 3 gate, 1 generator) |
| Ungrounded entity mentions | 0 (over 72 generated answers) |

Method: hit-rate@k checks that the expected source appears among the retrieved top-k;
faithfulness checks that the generated answer contains expected key terms; refusal precision
checks that the gate fires on unanswerable questions; follow-up hit-rate runs the
conversational path (condense the follow-up against its transcript, then route) and is scored
as its own row so the single-turn numbers stay comparable. Ungrounded entity mentions counts
known Pokemon and move names a generated answer uses that appear in no retrieved passage:
knowledge leakage that keyword checks cannot see, because an answer can contain every
expected term and still be decorated with world knowledge. Paraphrase hit-rate scores frozen
Gemma rewordings of each gold question ([eval/paraphrases.yaml](eval/paraphrases.yaml),
entity names guarded verbatim) against the same expected source, because retrieval bugs have
repeatedly hidden behind the one phrasing the gold set happened to use; paraphrases of
unanswerable questions must still be refused. Retrieval and refusal on
single-turn questions are deterministic; generation and condensation are temperature-sampled,
so faithfulness moves between 95% and 100% across runs.
Gold questions live in [eval/questions.yaml](eval/questions.yaml). The harness exits nonzero
on any miss, and a local pre-push hook runs it before every publish. A companion command,
`python -m eval.band_report`, prints each gate signal's answerable and no-answer bands with
margins against the configured thresholds; the bands moved with every corpus growth, and this
makes the drift visible before the eval breaks.

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
- **`bulbapedia`**: encyclopedic prose from [Bulbapedia](https://bulbapedia.bulbagarden.net/),
  read from a local cache written by a polite crawler (`python api/corpora/bulbapedia_crawl.py`,
  batched MediaWiki API queries, maxlag cooperation, descriptive User-Agent, resumable cache).
  Section-level filtering keeps the knowledge (lead, Biology, Effect, mechanics pages) and
  drops anime episodes, TCG, galleries, and sprites. Bulbapedia content is CC BY-NC-SA 2.5:
  the cache never ships, every document carries its source URL, and this stays a local
  personal-use corpus.
- **`smogon`**: expert competitive strategy from Smogon's Strategy Dex via the
  [@pkmn data mirror](https://data.pkmn.cc) (Smogon discourages scraping their site; the
  mirror exists for the ecosystem): peer-reviewed per-Pokemon analyses and named recommended
  sets with exact spreads, for Gen 9 OU and Gen 2 OU. The sets also power the calc tools'
  standard-set mode: "does standard Kingambit's Sucker Punch OHKO standard Dragapult?" runs
  the engine with both Pokemon's real Smogon builds (nature, EVs, item, ability) instead of
  neutral assumptions. Local personal use, attributed, never redistributed.
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

The damage-calc tool needs [poke-engine](https://github.com/pmariglia/poke-engine) built
from source with gen 9 features (`maturin build --release --no-default-features --features
terastallization` in `poke-engine-py`, then pip install the wheel); without it, damage
questions fall through to retrieval.

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
- [x] Question router + deterministic tools (type matchups with conditional immunities, speed checks, stat queries)
- [x] poke-engine damage calculator tool (neutral spreads, true engine rolls, n-HKO verdicts)
- [x] Battle-state-aware calc inputs: boosts, setup moves, items, natures, EVs, burn, weather, Tera
- [x] Tiered OHKO escalation search ("what would it take to OHKO?")
- [x] Defensive escalation search ("what would it take to survive?")
- [x] Cross-referenced ability immunities in damage answers, with observed usage rates
- [x] Terrain, doubles spread reduction, and per-Pokemon modifier binding in calc questions
- [ ] Monotype moveset tables and replay ingestion for the crystal-battle corpus
- [ ] Validate the k8s manifests end to end
- [x] Conversational memory: stateless history-in-request, follow-up condensation, its own eval metric
- [x] `/retrieve`: routed passages + tool outputs without generation, for external narrators
- [x] Ungrounded-entity eval metric (knowledge leakage the keyword proxy can't see)
- [x] Band-drift report: gate-signal bands and threshold margins as one command
- [x] Pre-push eval hook (harness exits nonzero on any miss)
- [x] Paraphrase-robustness gold: frozen rewordings of every gold question, scored as their own row
- [x] Speed-tiers tool: "what outspeeds X?" from the usage top 30, with the Choice Scarf math
