"""Smogon Strategy Dex adapter — expert competitive prose and recommended sets,
read from the local @pkmn mirror cache (see smogon_fetch.py).

Where the usage-stats corpus says what players DO and the dex corpora say what
things ARE, these analyses say what you SHOULD run and why: peer-reviewed
overviews, named sets with exact spreads, and written checks-and-counters
reasoning. One document per analysis section, one per recommended set.
"""
import json
import re
from pathlib import Path
from config import settings

FORMATS = {"gen9": ["ou"], "gen2": ["ou"]}
FMT_LABEL = {"gen9ou": "Gen 9 OU (gen9ou)", "gen2ou": "Gen 2 OU (gen2ou)"}


def _strip_html(text: str) -> str:
    text = re.sub(r"</p>\s*<p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&#39;", "'").strip())


def _first(v):
    """Set fields hold options: lists mean pick-one, first is most standard."""
    return _first(v[0]) if isinstance(v, list) and v else v


def render_set(name: str, s: dict) -> str:
    """A recommended set as readable text, slashes preserving the options."""
    def opts(v):
        if isinstance(v, list):
            return " / ".join(opts(x) for x in v)
        return str(v)

    lines = [f"Recommended set: {name}."]
    if s.get("moves"):
        lines.append("Moves: " + "; ".join(opts(m) for m in s["moves"]) + ".")
    for key, label in (("item", "Item"), ("ability", "Ability"), ("nature", "Nature"),
                       ("teratypes", "Tera types")):
        if s.get(key):
            lines.append(f"{label}: {opts(s[key])}.")
    evs = s.get("evs")
    if evs:
        first = evs[0] if isinstance(evs, list) else evs
        lines.append("EVs: " + ", ".join(f"{v} {k.upper()}" for k, v in first.items()) + ".")
    return "\n".join(lines)


def load():
    d = Path(settings.smogon_cache).expanduser()
    if not d.exists():
        raise FileNotFoundError(f"no Smogon cache at {d}; run: python api/corpora/smogon_fetch.py")
    for gen, fmts in FORMATS.items():
        analyses = json.loads((d / f"analyses-{gen}.json").read_text())
        sets = json.loads((d / f"sets-{gen}.json").read_text())
        for fmt in fmts:
            tag = f"{gen}{fmt}"
            label = FMT_LABEL.get(tag, tag)
            for mon, by_fmt in analyses.items():
                a = by_fmt.get(fmt)
                if not a:
                    continue
                url = f"https://www.smogon.com/dex/{'sv' if gen == 'gen9' else 'gs'}/pokemon/{mon.lower().replace(' ', '-')}/"
                meta = {"kind": "smogon_analysis", "format": tag, "species": mon, "url": url}
                overview = _strip_html(a.get("overview", ""))
                if overview:
                    yield {
                        "source": f"smogon#{mon} ({tag})",
                        "title": f"{mon} — {label} analysis",
                        "content": f"{mon} — {label} competitive analysis (Smogon):\n{overview}",
                        "metadata": meta,
                    }
                mon_sets = (sets.get(mon) or {}).get(fmt, {})
                set_comments = a.get("sets") or {}
                for set_name, set_data in mon_sets.items():
                    comment = set_comments.get(set_name) or {}
                    desc = _strip_html(comment.get("description", "") if isinstance(comment, dict) else str(comment))
                    body = render_set(set_name, set_data)
                    if desc:
                        body += f"\nWhy this set works: {desc}"
                    yield {
                        "source": f"smogon#{mon} ({tag}) — {set_name}",
                        "title": f"{mon} {set_name} set ({label})",
                        "content": f"{mon} — {label} recommended set (Smogon):\n{body}",
                        "metadata": {**meta, "kind": "smogon_set", "set": set_name},
                    }


def standard_set(mon_name: str, gen: str = "gen9") -> tuple[str, dict] | None:
    """The most standard recommended set for a Pokemon (first in Smogon's own
    ordering, first option of every slashed choice), for the calc tools."""
    d = Path(settings.smogon_cache).expanduser()
    f = d / f"sets-{gen}.json"
    if not f.exists():
        return None
    sets = json.loads(f.read_text())
    for key, by_fmt in sets.items():
        if key.lower() == mon_name.lower():
            for fmt in FORMATS.get(gen, []):
                if by_fmt.get(fmt):
                    name, s = next(iter(by_fmt[fmt].items()))
                    return name, {k: _first(v) for k, v in s.items()}
    return None
