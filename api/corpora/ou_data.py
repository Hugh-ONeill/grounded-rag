"""Accessor for single-format Smogon chaos JSON (gen9ou, gen9ubers, ...), exposing
the SAME shape as monotype_data so the team-builder tools stay format-agnostic:
`usage(fmt)` -> [(mon, usage_pct)] and `movesets(fmt)` -> {mon: {abilities, items,
spreads, moves, tera, teammates, checks, raw}} with each list [(name, pct), ...].

Chaos values are WEIGHTED counts; the correct share denominator is the weighted set
count = sum(Abilities.values()) (a set has exactly one ability), which matches
Smogon's published percentages. Reads crystal-battle's showdown/<fmt>_chaos.json.
"""
import json
from functools import lru_cache
from pathlib import Path


def _path(cb, fmt):
    return Path(cb) / "showdown" / f"{fmt}_chaos.json"


@lru_cache(maxsize=4)
def _load(cb, fmt):
    p = _path(cb, fmt)
    return json.loads(p.read_text()).get("data", {}) if p.exists() else {}


def _top(d, denom, n, min_pct=1.0):
    out = []
    if not d or denom <= 0:
        return out
    for name, w in sorted(d.items(), key=lambda kv: kv[1], reverse=True):
        if not name or name in ("Other", "Nothing"):
            continue
        pct = 100.0 * w / denom
        if pct < min_pct:
            break
        out.append((name, pct))
        if len(out) >= n:
            break
    return out


def usage(cb, fmt="gen9ou"):
    data = _load(cb, fmt)
    return sorted(((m, 100.0 * s.get("usage", 0)) for m, s in data.items()),
                  key=lambda x: -x[1])


def _cc_score(v):
    """The counter-reliability score from a Checks & Counters value, handling both the
    dict shape ({n, p, d}, p = KO/force-out fraction) and the older list shape."""
    if isinstance(v, dict):
        return v.get("p", 0.0)
    if isinstance(v, list) and len(v) > 1:
        return v[1] / 100.0 if v[1] > 1 else v[1]
    return 0.0


def _cc_encounters(v):
    return (v.get("n", 0) if isinstance(v, dict) else (v[0] if isinstance(v, list) and v else 0))


def counters_of(cb, fmt, mon):
    """Scored Checks & Counters for one mon: [(name, score_pct)] high-first, where the
    score is the weighted fraction of matchups where that Pokemon KOed the mon or
    forced it out. Low-sample entries (< 250 encounters) are dropped as noise."""
    s = _load(cb, fmt).get(mon) or {}
    cc = {n: v for n, v in s.get("Checks and Counters", {}).items()
          if n not in ("Other", "Nothing") and _cc_encounters(v) >= 250}
    ranked = sorted(cc.items(), key=lambda kv: _cc_score(kv[1]), reverse=True)
    return [(n, _cc_score(v) * 100) for n, v in ranked]


def movesets(cb, fmt="gen9ou"):
    data = _load(cb, fmt)
    out = {}
    for name, s in data.items():
        denom = (sum(s.get("Abilities", {}).values()) or sum(s.get("Items", {}).values())
                 or s.get("Raw count") or 1)
        cc = s.get("Checks and Counters", {})
        checks = [n for n, _ in sorted(cc.items(), key=lambda kv: _cc_score(kv[1]),
                  reverse=True) if n not in ("Other", "Nothing")][:6]
        out[name] = {
            "abilities": _top(s.get("Abilities", {}), denom, 3),
            "items": _top(s.get("Items", {}), denom, 5),
            "moves": _top(s.get("Moves", {}), denom, 12),
            "spreads": _top(s.get("Spreads", {}), denom, 6, min_pct=0.0),
            "tera": _top(s.get("Tera Types", {}), denom, 3),
            "teammates": _top(s.get("Teammates", {}), denom, 12),
            "checks": checks,
            "raw": int(s.get("Raw count") or 0),
        }
    return out
