"""Question router: computed questions go to tools, everything else to RAG.

Deliberately rule-based rather than LLM function-calling: the rules only fire
when they positively identify the entities involved (real Pokemon names, real
type names), so a miss falls through to retrieval instead of a wrong tool call.
The LLM stays the narrator, never the dispatcher.
"""
import re
import tools
from retrieve import retrieve_hybrid

STATS = {
    "speed": "Speed", "fastest": "Speed", "slowest": "Speed",
    "hp": "HP", "attack": "Attack", "defense": "Defense", "defence": "Defense",
    "special attack": "Special Attack", "special defense": "Special Defense",
    "bulkiest": "HP",
}

_MATCHUP = re.compile(
    r"super effective|not very effective|effective against|weak (?:to|against)|resists?|immune|\bhits?\b", re.I)
_SPEED_VS = re.compile(r"\bfaster\b|\boutspeeds?\b|\bslower\b", re.I)
_SUPERLATIVE = re.compile(r"\b(fastest|slowest|highest|lowest|best|most|bulkiest)\b", re.I)
_DAMAGE = re.compile(r"\bohko\b|\b\dhko\b|how much damage|damage does|\bkill\b", re.I)


def _find_names(question: str, names, limit: int = 3) -> list[str]:
    """Word-boundary matches of known names in the question, longest first
    (prefers "marowak-alola" over "marowak"; "unknown" must not match Unown)."""
    q = question.lower()
    # tolerate possessives with or without the apostrophe ("garchomps earthquake")
    hits = [n for n in names if re.search(rf"\b{re.escape(n)}(?:['\u2019]s|s)?\b", q)]
    hits.sort(key=len, reverse=True)
    out = []
    for h in hits:
        if not any(h in kept for kept in out):
            out.append(h)
    return out[:limit]


def _find_mons(question: str) -> list[str]:
    return _find_names(question, tools.known_pokemon())


def _find_moves(question: str) -> list[str]:
    return _find_names(question, tools.known_moves())


def _find_types(question: str) -> list[str]:
    """Type names in question order."""
    q = question.lower()
    found = [(m.start(), t) for t in tools.known_types()
             for m in [re.search(rf"\b{t.lower()}\b", q)] if m]
    return [t for _, t in sorted(found)]


async def route(question: str, corpus: str | None = None) -> list[dict]:
    """Return passages for the question: from a tool when one clearly applies,
    otherwise from hybrid retrieval."""
    mons = _find_mons(question)
    types = _find_types(question)

    # "who is faster, X or Y" / "does X outspeed Y"
    if _SPEED_VS.search(question) and len(mons) == 2:
        p = tools.speed_check(mons[0], mons[1])
        if p:
            return p

    # "is Earthquake effective against Skarmory" / "is Fire effective against Grass"
    # / "what is Garchomp weak to"
    if _MATCHUP.search(question):
        moves = _find_moves(question)
        if mons and moves:
            name, typ = tools.known_moves()[moves[0]]
            return tools.type_matchup(typ, mons[0], via_move=name)
        if mons and types:
            return tools.type_matchup(types[0], mons[0])
        if mons:
            return tools.defensive_profile(mons[0])
        if len(types) >= 2:
            return tools.type_matchup(types[0], types[1])
        if len(types) == 1:
            return tools.defensive_profile(types[0])

    # "does Garchomp's Earthquake OHKO Heatran" / "how much damage does X's Y do to Z"
    if _DAMAGE.search(question) and len(mons) >= 2:
        moves = _find_moves(question)
        if moves:
            q = question.lower()
            move_pos = q.find(moves[0])
            to_m = re.search(r"\bto\s+([a-z0-9' -]+)", q)
            defender = next((m for m in mons if to_m and m in to_m.group(1)), None)
            if defender:
                attacker = next(m for m in mons if m != defender)
            else:
                attacker = min(mons, key=lambda m: abs(q.find(m) - move_pos))
                defender = next(m for m in mons if m != attacker)
            name, _ = tools.known_moves()[moves[0]]
            p = tools.damage_calc(attacker, name, defender)
            if p:
                return p

    # "fastest Ghost-type Pokemon", "highest Attack among Water types"
    if _SUPERLATIVE.search(question) and types and not mons:
        stat = next((STATS[k] for k in STATS if re.search(rf"\b{k}\b", question, re.I)), None)
        if stat:
            lowest = bool(re.search(r"\b(slowest|lowest)\b", question, re.I))
            p = tools.stat_query(stat, type_filter=types[0], lowest=lowest)
            if p:
                return p

    return await retrieve_hybrid(question, corpus=corpus)
