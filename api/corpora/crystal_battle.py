"""crystal-battle adapter — a competitive-Pokemon meta analyst.

Turns the Smogon usage data already sitting in your crystal-battle checkout into citable,
natural-language documents. Each Pokemon becomes one document describing its real-world
movesets, items, abilities, common teammates, and what checks/counters it — so the RAG
system answers things like "what beats Great Tusk?" from actual ladder data, with a citation.

Set CRYSTAL_BATTLE_PATH in .env. Robust to missing files (only yields what it finds).
"""
import json
import re
from pathlib import Path
from config import settings
from . import monotype_data as md


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
    lines = [f"{name}" + (f" ({typ}-type)" if typ else "") + f" — {_pretty_fmt(fmt)} usage data."]
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


def _pretty_fmt(fmt: str) -> str:
    """gen2ou -> "Gen 2 OU (gen2ou)": questions say the spaced form, exact-token
    search needs the tag, so documents carry both."""
    m = re.match(r"gen(\d+)(\w+)", fmt)
    return f"Gen {m.group(1)} {m.group(2).upper()} ({fmt})" if m else fmt


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
        f"{_pretty_fmt(fmt)} usage rankings: the most used Pokemon in {_pretty_fmt(fmt)}, "
        f"ranked by the share of teams they appear on.",
        f"The most used Pokemon in {_pretty_fmt(fmt)} is {top_name}, appearing on "
        f"{100*(top_stats.get('usage') or 0):.1f}% of teams.",
        # cover the registers real users ask in ("a common sight", "on nearly every
        # team"), not just the "most used" phrasing the first gold question used
        f"The top entries are a common sight in {_pretty_fmt(fmt)}, seen on nearly every team.",
    ]
    lines += [f"{i}. {n} ({100*(s.get('usage') or 0):.1f}%)" for i, (n, s) in enumerate(ranked, 1)]
    return {
        "source": f"{fmt}_chaos#usage_rankings",
        "title": f"{fmt} usage rankings",
        "content": "\n".join(lines),
        "metadata": {"kind": "usage_rankings", "format": fmt},
    }


_MONO_FMT = "gen9monotype"
_MONO_LABEL = "Gen 9 Monotype (gen9monotype)"


def _pct_list(pairs, n):
    return ", ".join(f"{nm} ({p:.0f}%)" for nm, p in pairs[:n])


def _monotype_docs(cb_path):
    """gen9monotype per-type documents from crystal-battle's Smogon dumps: one
    usage-skeleton doc per type, per-(type, mon) set docs for the meta core, a
    type-matchup doc per type, and an overall type-strength ranking. Replaces the
    old raw-file dump, which globbed the wrong path (files live in month subdirs)
    and emitted one unsearchable giant chunk per table."""
    sdir = md.stats_dir(cb_path)
    month = md.latest_month(sdir)
    if not month:
        return
    matchup = md.matchup_chart(cb_path, month=month)

    for typ in md.TYPES:
        Typ = typ.capitalize()

        usage = md.type_usage(cb_path, typ, month=month)
        if usage:
            top = usage[:20]
            lines = [
                f"{_MONO_LABEL} mono-{Typ} usage rankings: the most used and most "
                f"common Pokémon on {Typ}-type teams in monotype, ranked by the share "
                f"of {Typ} teams they appear on.",
                f"The core {Typ}-type Pokémon in Gen 9 Monotype, seen on nearly every "
                f"{Typ} team: " + ", ".join(n for n, _ in top[:6]) + ".",
            ]
            lines += [f"{i}. {n} ({p:.0f}%)" for i, (n, p) in enumerate(top, 1)]
            yield {
                "source": f"gen9monotype_usage#{Typ}",
                "title": f"{Typ} monotype team — most used Pokémon (Gen 9 Monotype)",
                "content": "\n".join(lines),
                "metadata": {"kind": "monotype_usage", "format": _MONO_FMT, "type": typ},
            }

        movesets = md.type_moveset(cb_path, typ, month=month)
        for mon, s in sorted(movesets.items(), key=lambda kv: kv[1]["raw"], reverse=True)[:15]:
            parts = [f"{mon} on {Typ}-type teams in {_MONO_LABEL} — how it is built and "
                     f"played on {Typ} monotype teams."]
            if s["moves"]:
                parts.append("Commonly used moves: " + _pct_list(s["moves"], 8) + ".")
            if s["items"]:
                parts.append("Common items: " + _pct_list(s["items"], 4) + ".")
            if s["abilities"]:
                parts.append("Ability: " + _pct_list(s["abilities"], 2) + ".")
            if s["teammates"]:
                parts.append(f"Common {Typ} teammates: " + _pct_list(s["teammates"], 6) + ".")
            if s["checks"]:
                parts.append("Checked / countered by: " + ", ".join(s["checks"][:6]) + ".")
            yield {
                "source": f"gen9monotype_set#{Typ}-{mon}",
                "title": f"{mon} on {Typ} (Gen 9 Monotype)",
                "content": "\n".join(parts),
                "metadata": {"kind": "monotype_set", "format": _MONO_FMT,
                             "type": typ, "species": mon},
            }

        row = matchup.get(typ)
        if row:
            others = [(o, w) for o, w in row.items() if o != typ and o in md.TYPES]
            fav = sorted((x for x in others if x[1] >= 55), key=lambda x: -x[1])
            unf = sorted((x for x in others if x[1] <= 45), key=lambda x: x[1])
            pv = lambda items: ", ".join(f"{o.capitalize()} ({w:.0f}%)" for o, w in items)
            lines = [f"In {_MONO_LABEL}, {Typ}-type teams — type matchup spread (win rate "
                     f"vs each opposing monotype at 1500 ELO)."]
            if fav:
                lines.append(f"{Typ} teams are favored against: " + pv(fav) + ".")
            if unf:
                lines.append(f"{Typ} teams struggle against: " + pv(unf) + ".")
            yield {
                "source": f"gen9monotype_matchup#{Typ}",
                "title": f"{Typ} monotype — type matchups (Gen 9 Monotype)",
                "content": "\n".join(lines),
                "metadata": {"kind": "monotype_matchup", "format": _MONO_FMT, "type": typ},
            }

    if matchup:
        strength = []
        for typ in md.TYPES:
            vals = [w for o, w in matchup.get(typ, {}).items() if o != typ and o in md.TYPES]
            if vals:
                strength.append((typ, sum(vals) / len(vals)))
        strength.sort(key=lambda x: -x[1])
        if strength:
            lines = [f"{_MONO_LABEL} type strength ranking: the strongest, best, and most "
                     f"dominant types in monotype, by average win rate across all type "
                     f"matchups (1500 ELO). The weakest types are at the bottom."]
            lines += [f"{i}. {t.capitalize()} (avg {w:.0f}% win rate)"
                      for i, (t, w) in enumerate(strength, 1)]
            yield {
                "source": "gen9monotype_matchup#type_rankings",
                "title": "Gen 9 Monotype — strongest types",
                "content": "\n".join(lines),
                "metadata": {"kind": "monotype_type_ranking", "format": _MONO_FMT},
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

    # 2) gen9monotype per-type stats: usage skeletons, per-mon sets, and type
    #    matchups as structured, citable docs (see _monotype_docs).
    yield from _monotype_docs(root)

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
