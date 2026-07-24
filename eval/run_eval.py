"""Evaluation harness — the section recruiters remember.

Measures, over a set of gold questions:
  - retrieval hit-rate@k : did the right source make it into the top-k?
  - faithfulness proxy   : does the answer contain the expected keywords?
  - refusal precision    : on no-answer questions, did we correctly say "I don't know"?
  - follow-up hit-rate   : items with `history` are condensed to a standalone question
                           first (the conversational-memory path), then scored on
                           retrieval separately so single-turn numbers stay comparable
  - ungrounded entities  : known Pokemon/move names the answer mentions that appear in
                           NO retrieved passage — knowledge leakage the keyword proxy
                           can't see (the model decorating answers from world knowledge)
  - paraphrase hit-rate  : frozen rewordings of each gold question (eval/paraphrases.yaml,
                           see gen_paraphrases.py) must land on the same expect_source, or
                           still be refused for no-answer originals — phrasing robustness,
                           scored separately so the canonical-phrasing rows stay comparable

Run:  python -m eval.run_eval
Then paste the printed table into the README's Evaluation section.
Exits nonzero on any miss in the strict signals: retrieval, follow-up, and paraphrase
(deterministic routing) plus refusal and ungrounded-entity leaks (generation-touched, but
each miss is a hard grounding failure worth stopping for). Faithfulness is a
temperature-sampled keyword proxy that wobbles run to run, so it gates on a band
(FAITHFULNESS_FLOOR, 95-100%) rather than a flat 100%. Reread the printed miss lines
before treating any dip as a regression.
"""
import asyncio
import re
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
from retrieve import passes_threshold
from router import route, find_entity_names   # noqa: E402
from llm import answer_stream, condense_question    # noqa: E402

# Faithfulness (does the answer contain the expected keywords) is a temperature-sampled
# proxy: a lone wording-driven miss is sampling noise, not a regression, so the gate accepts
# this documented band instead of demanding a flat 100%. Everything else stays strict.
FAITHFULNESS_FLOOR = 0.95


def _src_hit(item, passages) -> bool:
    """expect_source may be one substring or an any-of list: "how do I deal with
    Kingambit?" is answered equally well by the chaos checks data or the Smogon
    analysis prose, and pinning one of them makes paraphrase misses meaningless."""
    want = item.get("expect_source", "∅")
    wants = want if isinstance(want, list) else [want]
    return any(w in (p["source"] or "") for p in passages for w in wants)


def _squash(s: str) -> str:
    """Lowercase alphanumerics only: 'Shadow Ball' matches the raw 'shadowball' in
    chaos docs. Loses word boundaries, so short common-word move names ('rest')
    almost always match something — acceptable for a leak check, not a hit check."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


async def _full_answer(q, passages):
    return "".join([tok async for tok in answer_stream(q, passages)])


async def main():
    qfile = Path(__file__).parent / "questions.yaml"
    if not qfile.exists():
        qfile = Path(__file__).parent / "questions.example.yaml"
    questions = yaml.safe_load(qfile.read_text())

    pfile = Path(__file__).parent / "paraphrases.yaml"
    paraphrases = {}
    if pfile.exists():
        paraphrases = {e["question"]: e["paraphrases"] for e in yaml.safe_load(pfile.read_text())}

    hits = faith = faith_total = refuse_ok = refuse_total = refuse_gate = 0
    answerable = fu_hits = fu_total = leaks = para_hits = para_total = 0

    for item in questions:
        q = item["question"]
        if item.get("history"):
            q = await condense_question(q, item["history"])
        passages = await route(q, corpus=item.get("corpus"))

        # phrasing robustness: reworded variants must land on the same source, or,
        # for no-answer originals, still be refused by one of the two layers
        for p in paraphrases.get(item["question"], []):
            pp = await route(p, corpus=item.get("corpus"))
            para_total += 1
            if item.get("no_answer"):
                ok = (not passes_threshold(pp)
                      or "don't know" in (await _full_answer(p, pp)).lower())
            else:
                ok = _src_hit(item, pp)
            para_hits += ok
            if not ok:
                print(f"  paraphrase miss: {p!r} (of: {item['question']!r})")

        if item.get("no_answer"):
            # refusal is layered: the retrieval gate fires on off-topic questions, but a
            # question that names a real corpus entity ("what about Kingambit's stock
            # price?") scores that entity's own page at the answerable floor, and only
            # the grounded generator can refuse it. Count either layer, report which fired.
            refuse_total += 1
            if not passes_threshold(passages):
                refuse_ok += 1
                refuse_gate += 1
            elif "don't know" in (await _full_answer(q, passages)).lower():
                refuse_ok += 1
            else:
                print(f"  refusal miss: {item['question']!r} -> asked as: {q!r}")
            continue

        hit = _src_hit(item, passages)
        if item.get("history"):
            fu_total += 1
            fu_hits += hit
            if not hit:
                print(f"  follow-up miss: {item['question']!r} -> condensed: {q!r}")
        else:
            answerable += 1
            hits += hit
        # faithfulness proxy (only if keywords given)
        kws = item.get("expect_keywords") or []
        if kws:
            faith_total += 1
            raw_ans = await _full_answer(q, passages)
            if all(k.lower() in raw_ans.lower() for k in kws):
                faith += 1
            else:
                missing = [k for k in kws if k.lower() not in raw_ans.lower()]
                print(f"  faithfulness miss: {item['question']!r} missing {missing!r}")
            # knowledge-leakage check: entities the answer names must exist in some
            # retrieved passage OR in the question itself (echoing the question's own
            # terms is not decoration: "after a Swords Dance" gets stripped to +2
            # before the calc tool runs, so the passages never name the move).
            # Lowercase occurrences are skipped: a move named "rest" or "protect"
            # matching ordinary prose is not an entity mention.
            ctx = _squash(" ".join(p["content"] for p in passages) + " " + q)
            ents = find_entity_names(raw_ans)
            for name in ents["pokemon"] + ents["moves"]:
                if _squash(name) in ctx:
                    continue
                m = re.search(rf"\b{re.escape(name)}(?:['’]s|s)?\b", raw_ans, re.I)
                if m and not m.group(0)[0].isupper():
                    continue
                leaks += 1
                print(f"  ungrounded entity: {name!r} in answer to {item['question']!r}")

    def pct(a, b):
        return f"{100*a/b:.0f}%" if b else "n/a"

    print("\n| Metric | Score |")
    print("|--------|-------|")
    print(f"| Retrieval hit-rate@k | {pct(hits, answerable)} ({hits}/{answerable}) |")
    print(f"| Follow-up hit-rate (condense → retrieve) | {pct(fu_hits, fu_total)} ({fu_hits}/{fu_total}) |")
    print(f"| Paraphrase hit-rate | {pct(para_hits, para_total)} ({para_hits}/{para_total}) |")
    print(f"| Answer faithfulness | {pct(faith, faith_total)} ({faith}/{faith_total}) |")
    print(f"| Refusal precision (no-answer) | {pct(refuse_ok, refuse_total)} ({refuse_ok}/{refuse_total}: "
          f"{refuse_gate} gate, {refuse_ok - refuse_gate} generator) |")
    print(f"| Ungrounded entity mentions | {leaks} (over {faith_total} generated answers) |")

    faith_ratio = faith / faith_total if faith_total else 1.0
    if faith_total and faith < faith_total and faith_ratio >= FAITHFULNESS_FLOOR:
        print(f"\nfaithfulness {faith}/{faith_total} ({100*faith_ratio:.0f}%) is within the "
              f"accepted {int(FAITHFULNESS_FLOOR*100)}-100% band (temperature-sampled wobble); "
              f"gate passes. Check the miss lines above if this is unexpected.")

    if (hits < answerable or fu_hits < fu_total or refuse_ok < refuse_total
            or leaks or para_hits < para_total or faith_ratio < FAITHFULNESS_FLOOR):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
