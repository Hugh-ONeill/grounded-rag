"""Evaluation harness — the section recruiters remember.

Measures, over a set of gold questions:
  - retrieval hit-rate@k : did the right source make it into the top-k?
  - faithfulness proxy   : does the answer contain the expected keywords?
  - refusal precision    : on no-answer questions, did we correctly say "I don't know"?

Run:  python -m eval.run_eval
Then paste the printed table into the README's Evaluation section.
"""
import asyncio
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
from retrieve import retrieve, passes_threshold   # noqa: E402
from llm import answer_stream                       # noqa: E402


async def _full_answer(q, passages):
    return "".join([tok async for tok in answer_stream(q, passages)])


async def main():
    qfile = Path(__file__).parent / "questions.yaml"
    if not qfile.exists():
        qfile = Path(__file__).parent / "questions.example.yaml"
    questions = yaml.safe_load(qfile.read_text())

    hits = faith = faith_total = refuse_ok = refuse_total = 0
    answerable = 0

    for item in questions:
        q = item["question"]
        passages = await retrieve(q, corpus=item.get("corpus"))

        if item.get("no_answer"):
            refuse_total += 1
            if not passes_threshold(passages):
                refuse_ok += 1
            continue

        answerable += 1
        # retrieval hit
        if any(item.get("expect_source", "∅") in (p["source"] or "") for p in passages):
            hits += 1
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
    print(f"| Answer faithfulness | {pct(faith, faith_total)} ({faith}/{faith_total}) |")
    print(f"| Refusal precision (no-answer) | {pct(refuse_ok, refuse_total)} ({refuse_ok}/{refuse_total}) |")


if __name__ == "__main__":
    asyncio.run(main())
