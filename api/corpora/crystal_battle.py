"""crystal-battle adapter — a competitive-Pokemon meta analyst.

Turns the Smogon usage data already sitting in your crystal-battle checkout into citable,
natural-language documents. Each Pokemon becomes one document describing its real-world
movesets, items, abilities, common teammates, and what checks/counters it — so the RAG
system answers things like "what beats Great Tusk?" from actual ladder data, with a citation.

Set CRYSTAL_BATTLE_PATH in .env. Robust to missing files (only yields what it finds).
"""
import json
from pathlib import Path
from config import settings


def _top(d: dict, denom: float, n: int = 8, min_pct: float = 3.0):
    """Sort a {name: weight} dict to a 'name (xx%)' list, conventional Smogon normalization."""
    if not d or denom <= 0:
        return []
    ranked = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for name, w in ranked:
        pct = 100.0 * w / denom
        if pct < min_pct:
            break
        out.append(f"{name} ({pct:.0f}%)")
        if len(out) >= n:
            break
    return out


def _species_doc(name: str, s: dict, fmt: str, types: dict):
    # chaos dict values are WEIGHTED counts while "Raw count" is unweighted, so
    # value/raw deflates every share. The weighted set count is the sum of any
    # single-slot attribute (each set has exactly one ability), and value/that
    # matches Smogon's published percentages.
    raw = (sum(s.get("Abilities", {}).values()) or sum(s.get("Items", {}).values())
           or s.get("Raw count") or 1)
    typ = " / ".join(types.get(name, [])) if types else ""
    lines = [f"{name}" + (f" ({typ}-type)" if typ else "") + f" — {fmt} usage data."]
    if s.get("usage") is not None:
        lines.append(f"Usage: {100*s['usage']:.1f}% of teams.")
    for label, key, denom in [
        ("Commonly used moves", "Moves", raw),
        ("Commonly held items", "Items", raw),
        ("Abilities", "Abilities", raw),
        ("Tera Types", "Tera Types", raw),
        ("Common teammates", "Teammates", raw),
    ]:
        vals = _top(s.get(key, {}), denom)
        if vals:
            lines.append(f"{label}: {', '.join(vals)}.")
    # Checks and Counters has a different shape: {name: [n, score, stddev]}
    cc = s.get("Checks and Counters", {})
    if cc:
        ranked = sorted(cc.items(), key=lambda kv: kv[1][1] if isinstance(kv[1], list) else 0, reverse=True)
        names = [n for n, _ in ranked[:6]]
        if names:
            lines.append(f"Checked / countered by: {', '.join(names)}.")
    return {
        "source": f"{fmt}_chaos#{name}",
        "title": f"{name} ({fmt})",
        "content": "\n".join(lines),
        "metadata": {"kind": "usage_stats", "format": fmt, "species": name},
    }


def _rankings_doc(fmt: str, data: dict):
    """Synthetic aggregation document. Top-k similarity retrieval cannot answer
    corpus-wide superlatives ("what is the most used Pokemon?"): no single species
    chunk contains the comparison. So the ranking is emitted as its own citable
    document, which the phrasing of such questions retrieves naturally."""
    ranked = sorted(data.items(), key=lambda kv: kv[1].get("usage") or 0, reverse=True)[:30]
    if not ranked:
        return None
    top_name, top_stats = ranked[0]
    lines = [
        f"{fmt} usage rankings: the most used Pokemon in {fmt}, ranked by share of teams.",
        f"The most used Pokemon in {fmt} is {top_name}, on {100*(top_stats.get('usage') or 0):.1f}% of teams.",
    ]
    lines += [f"{i}. {n} ({100*(s.get('usage') or 0):.1f}%)" for i, (n, s) in enumerate(ranked, 1)]
    return {
        "source": f"{fmt}_chaos#usage_rankings",
        "title": f"{fmt} usage rankings",
        "content": "\n".join(lines),
        "metadata": {"kind": "usage_rankings", "format": fmt},
    }


def load():
    root = Path(settings.crystal_battle_path)
    if not root.exists():
        raise FileNotFoundError(f"CRYSTAL_BATTLE_PATH not found: {root}")

    # type lookup (optional)
    types = {}
    tp = root / "showdown" / "species_types.json"
    if tp.exists():
        types = json.loads(tp.read_text())

    # 1) Smogon usage stats — the richest source (gen9ou_chaos.json, gen2ou_chaos.json, ...)
    for chaos in (root / "showdown").glob("*_chaos.json"):
        fmt = chaos.stem.replace("_chaos", "")
        data = json.loads(chaos.read_text()).get("data", {})
        for name, s in data.items():
            yield _species_doc(name, s, fmt, types)
        rankings = _rankings_doc(fmt, data)
        if rankings:
            yield rankings

    # 2) Per-type monotype moveset tables (raw text is already human-readable)
    for txt in (root / "monotype" / "smogon_stats").glob("*.txt"):
        yield {
            "source": f"monotype_stats/{txt.name}",
            "title": txt.stem,
            "content": txt.read_text(encoding="utf-8", errors="ignore"),
            "metadata": {"kind": "monotype_stats"},
        }

    # 3) Example team builds
    for team in (root / "monotype" / "teams").glob("teams_v*.txt"):
        yield {
            "source": f"teams/{team.name}",
            "title": team.stem,
            "content": team.read_text(encoding="utf-8", errors="ignore"),
            "metadata": {"kind": "team_paste"},
        }

    # TODO (stretch): ingest showdown/replays/*.json as per-turn narrative documents
    #   ("Turn 7: P1 switched Heatran into Earthquake ...") for play-pattern questions.
