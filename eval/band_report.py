"""Band-drift report: the gate's safety margins, measured on demand.

The refusal/retrieval bands (top-1 similarity, keyword rank, reranker score) were
re-measured ad hoc every time a corpus grew, and every growth moved them. This makes
it one command: per-gold-question top-1 signals, answerable vs no-answer bands, and
each signal's margin against its configured threshold. Run it after any ingest;
drift becomes visible before the eval breaks.

Run:  python -m eval.band_report
"""
import asyncio
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
from config import settings          # noqa: E402
from router import route             # noqa: E402
from llm import condense_question    # noqa: E402


async def main():
    qfile = Path(__file__).parent / "questions.yaml"
    if not qfile.exists():
        qfile = Path(__file__).parent / "questions.example.yaml"
    questions = yaml.safe_load(qfile.read_text())

    rows = []
    for item in questions:
        q = item["question"]
        if item.get("history"):
            q = await condense_question(q, item["history"])
        passages = await route(q, corpus=item.get("corpus"))
        top = passages[0] if passages else {}
        rows.append({
            "q": item["question"],
            "no_answer": bool(item.get("no_answer")),
            "tool": (top.get("source") or "").startswith("tool#"),
            "sim": top.get("similarity") or 0.0,
            "kw": top.get("kw_rank") or 0.0,
            "rr": top.get("rerank_score"),
        })

    tool_rows = [r for r in rows if r["tool"]]
    ans = [r for r in rows if not r["tool"] and not r["no_answer"]]
    noans = [r for r in rows if not r["tool"] and r["no_answer"]]

    def band(rs, key):
        vals = [r[key] for r in rs if r[key] is not None]
        return (min(vals), max(vals)) if vals else (float("nan"), float("nan"))

    print(f"\n{len(rows)} gold questions: {len(ans)} answerable (retrieval-routed), "
          f"{len(noans)} no-answer, {len(tool_rows)} tool-routed (synthetic scores, excluded from bands)")

    print("\n| Signal | Answerable band | No-answer band | Threshold | Margins (floor/ceiling) |")
    print("|--------|-----------------|----------------|-----------|-------------------------|")
    for key, label, thr in (("rr", "rerank", settings.min_rerank_score),
                            ("sim", "similarity", settings.min_similarity),
                            ("kw", "keyword rank", settings.min_keyword_rank)):
        alo, ahi = band(ans, key)
        nlo, nhi = band(noans, key)
        print(f"| {label} | {alo:.3f} to {ahi:.3f} | {nlo:.3f} to {nhi:.3f} | {thr} "
              f"| {alo - thr:+.3f} / {thr - nhi:+.3f} |")

    eps = 0.02
    for r in ans:
        if r["rr"] is not None and r["rr"] < settings.min_rerank_score + eps:
            print(f"\n  at the gate floor (rerank {r['rr']:.3f}): {r['q']!r}")
    for r in noans:
        if r["rr"] is not None and r["rr"] >= settings.min_rerank_score:
            print(f"  gate-passing no-answer (rerank {r['rr']:.3f}, generator must refuse): {r['q']!r}")

    print("\nPer-question top-1 signals (sorted by rerank):")
    for r in sorted(rows, key=lambda r: (r["rr"] is None, r["rr"] or 0)):
        tag = "tool " if r["tool"] else ("NOANS" if r["no_answer"] else "     ")
        rr = f"{r['rr']:.3f}" if r["rr"] is not None else "  -  "
        print(f"  {tag} rr={rr} sim={r['sim']:.3f} kw={r['kw']:.3f}  {r['q'][:70]}")


if __name__ == "__main__":
    asyncio.run(main())
