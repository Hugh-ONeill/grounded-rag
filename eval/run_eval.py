"""Evaluation harness — the section recruiters remember.

Measures, over a set of gold questions:
  - retrieval hit-rate@k : did the right source make it into the top-k?
  - faithfulness proxy   : does the answer contain the expected keywords?
  - refusal precision    : on no-answer questions, did we correctly say "I don't know"?
  - follow-up hit-rate   : items with `history` are condensed to a standalone question
                           first (the conversational-memory path), then scored on
                           retrieval separately so single-turn numbers stay comparable

Run:  python -m eval.run_eval
Then paste the printed table into the README's Evaluation section.
"""
import asyncio
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
from retrieve import passes_threshold
from router import route   # noqa: E402
from llm import answer_stream, condense_question    # noqa: E402


async def _full_answer(q, passages):
    return "".join([tok async for tok in answer_stream(q, passages)])


async def main():
    qfile = Path(__file__).parent / "questions.yaml"
    if not qfile.exists():
        qfile = Path(__file__).parent / "questions.example.yaml"
    questions = yaml.safe_load(qfile.read_text())

    hits = faith = faith_total = refuse_ok = refuse_total = refuse_gate = 0
    answerable = fu_hits = fu_total = 0

    for item in questions:
        q = item["question"]
        if item.get("history"):
            q = await condense_question(q, item["history"])
        passages = await route(q, corpus=item.get("corpus"))

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

        hit = any(item.get("expect_source", "∅") in (p["source"] or "") for p in passages)
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
            ans = (await _full_answer(q, passages)).lower()
            if all(k.lower() in ans for k in kws):
                faith += 1

    def pct(a, b):
        return f"{100*a/b:.0f}%" if b else "n/a"

    print("\n| Metric | Score |")
    print("|--------|-------|")
    print(f"| Retrieval hit-rate@k | {pct(hits, answerable)} ({hits}/{answerable}) |")
    print(f"| Follow-up hit-rate (condense → retrieve) | {pct(fu_hits, fu_total)} ({fu_hits}/{fu_total}) |")
    print(f"| Answer faithfulness | {pct(faith, faith_total)} ({faith}/{faith_total}) |")
    print(f"| Refusal precision (no-answer) | {pct(refuse_ok, refuse_total)} ({refuse_ok}/{refuse_total}: "
          f"{refuse_gate} gate, {refuse_ok - refuse_gate} generator) |")


if __name__ == "__main__":
    asyncio.run(main())
