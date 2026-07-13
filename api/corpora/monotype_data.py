"""Parsers + accessors for crystal-battle's local gen9monotype Smogon stat dumps
(monotype/smogon_stats/<YYYY-MM>/{usage,moveset,matchup}). The percentages in
these text files are already computed, so no chaos-denominator normalization is
needed (unlike the JSON chaos files the OU adapter reads).

Shared by the crystal_battle corpus adapter (turns them into citable docs) and,
later, the monotype-aware tools. Everything is type-parameterized so the same
shapes generalize to other per-type formats.
"""
import re
from pathlib import Path

TYPES = ["bug", "dark", "dragon", "electric", "fairy", "fighting", "fire",
         "flying", "ghost", "grass", "ground", "ice", "normal", "poison",
         "psychic", "rock", "steel", "water"]

_SECTIONS = {"Abilities", "Items", "Spreads", "Moves", "Teammates", "Checks and Counters"}
_KEY = {"Abilities": "abilities", "Items": "items", "Spreads": "spreads",
        "Moves": "moves", "Teammates": "teammates"}


def stats_dir(crystal_battle_path) -> Path:
    return Path(crystal_battle_path) / "monotype" / "smogon_stats"


def latest_month(sdir: Path) -> str | None:
    months = sorted(p.name for p in sdir.glob("20*-*") if p.is_dir())
    return months[-1] if months else None


def _usage_file(sdir, month, typ, elo):
    return sdir / month / "usage" / f"gen9monotype-mono{typ}-{elo}.txt"


def _moveset_file(sdir, month, typ, elo):
    return sdir / month / "moveset" / f"gen9monotype-mono{typ}-{elo}.txt"


def _matchup_file(sdir, month, elo):
    return sdir / month / "matchup" / f"gen9monotype-matchup_chart-{elo}.txt"


def parse_usage(path: Path):
    """Usage table -> [(mon, usage_pct), ...] in rank order."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*([\d.]+)%", line)
        if m:
            out.append((m.group(1).strip(), float(m.group(2))))
    return out


def parse_matchup(path: Path):
    """Matchup chart -> {team_type: {opp_type: win_pct}}. Rows are pairs (a rating
    line then a percentage line); the percentage line holds the win rates."""
    result = {}
    if not path.exists():
        return result
    cols = None
    cur = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        head, rest = cells[0], cells[1:]
        if cols is None:
            if head == "" and any(c in TYPES for c in rest):
                cols = rest
            continue
        if head in TYPES:
            cur = head          # rating line; percentages follow on the next line
        elif head == "" and cur:
            pcts = {}
            for t, v in zip(cols, rest):
                v = v.replace("%", "").strip()
                try:
                    pcts[t] = float(v)
                except ValueError:
                    pass
            result[cur] = pcts
            cur = None
    return result


def parse_moveset(path: Path):
    """Moveset dump -> {mon: {abilities, items, spreads, moves, teammates, checks, raw}}.
    Each list is [(name, pct), ...] (the 'Other' bucket dropped); checks is [name, ...];
    raw is the unweighted count. Handles files that omit Checks-and-Counters."""
    mons = {}
    if not path.exists():
        return mons
    cur = None
    section = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("|"):        # borders (+---+) are skipped
            continue
        body = line.strip("|").strip()
        if not body:
            continue
        if body in _SECTIONS:
            section = body
            continue
        if body.startswith("Raw count:"):
            if cur:
                cur["raw"] = int(re.search(r"\d+", body).group())
            continue
        if body.startswith(("Avg. weight", "Viability Ceiling")):
            continue
        m = re.match(r"(.+?)\s+([\d.]+)%$", body)
        if m and section in _KEY:
            name = m.group(1).strip()
            if name != "Other":
                cur[_KEY[section]].append((name, float(m.group(2))))
            continue
        if section == "Checks and Counters" and cur:
            mc = re.match(r"(.+?)\s+[\d.]+\s*\(", body)
            if mc:
                cur["checks"].append(mc.group(1).strip())
                continue
        # a bare name between borders that matched nothing above -> a new mon block
        cur = {"abilities": [], "items": [], "spreads": [], "moves": [],
               "teammates": [], "checks": [], "raw": 0}
        mons[body] = cur
        section = None
    return mons


# ---- convenience accessors (resolve the latest month, one type at a time) ----

def type_usage(crystal_battle_path, typ, elo=1500, month=None):
    sdir = stats_dir(crystal_battle_path)
    month = month or latest_month(sdir)
    return parse_usage(_usage_file(sdir, month, typ, elo)) if month else []


def type_moveset(crystal_battle_path, typ, elo=1500, month=None):
    sdir = stats_dir(crystal_battle_path)
    month = month or latest_month(sdir)
    return parse_moveset(_moveset_file(sdir, month, typ, elo)) if month else {}


def matchup_chart(crystal_battle_path, elo=1500, month=None):
    sdir = stats_dir(crystal_battle_path)
    month = month or latest_month(sdir)
    return parse_matchup(_matchup_file(sdir, month, elo)) if month else {}
