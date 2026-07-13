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


def movesets(cb, fmt="gen9ou"):
    data = _load(cb, fmt)
    out = {}
    for name, s in data.items():
        denom = (sum(s.get("Abilities", {}).values()) or sum(s.get("Items", {}).values())
                 or s.get("Raw count") or 1)
        cc = s.get("Checks and Counters", {})
        checks = [n for n, _ in sorted(cc.items(),
                  key=lambda kv: kv[1][1] if isinstance(kv[1], list) else 0, reverse=True)][:6]
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
