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
_DAMAGE = re.compile(r"\bohko\b|\b\dhko\b|how much damage|damage does|\bkill\b|\bsurviv", re.I)
_SURVIVE_HOW = re.compile(
    r"(?:what would|what does|what do|how (?:can|could|do)|needs?|needed|required?|take[s]? (?:for|to)|make).{0,60}\b(?:survive|tank|withstand)\b", re.I)
_OHKO_HOW = re.compile(
    r"(?:what would|what does|what do|how (?:can|could|do)|needs?|needed|required?|take[s]? (?:for|to)|make).{0,60}\bohko\b", re.I)

# battle-state modifiers, parsed out of the question BEFORE move detection so
# "after a Swords Dance" is a boost, not the attacking move
_SETUP = {"swords dance": {"Attack": 2}, "dragon dance": {"Attack": 1, "Speed": 1},
          "nasty plot": {"Special Attack": 2}, "calm mind": {"Special Attack": 1, "Special Defense": 1},
          "shell smash": {"Attack": 2, "Special Attack": 2, "Speed": 2},
          "bulk up": {"Attack": 1, "Defense": 1}}
_OFFENSE_ITEMS = {"choice band": "Choice Band", "banded": "Choice Band", "choice specs": "Choice Specs",
                  "specs": "Choice Specs", "life orb": "Life Orb", "expert belt": "Expert Belt"}
_DEFENSE_ITEMS = {"assault vest": "Assault Vest", "eviolite": "Eviolite"}
_NATURES = {"adamant": ("Attack", "Special Attack"), "jolly": ("Speed", "Special Attack"),
            "modest": ("Special Attack", "Attack"), "timid": ("Speed", "Attack"),
            "impish": ("Defense", "Special Attack"), "bold": ("Defense", "Attack"),
            "careful": ("Special Defense", "Special Attack"), "calm": ("Special Defense", "Attack")}
_OFFENSIVE_NATURES = {"adamant", "jolly", "modest", "timid"}
_EV_STATS = {"attack": "Attack", "atk": "Attack", "spa": "Special Attack",
             "special attack": "Special Attack", "hp": "HP", "def": "Defense",
             "defense": "Defense", "spd": "Special Defense", "special defense": "Special Defense",
             "speed": "Speed", "spe": "Speed"}


def _parse_mods(question: str):
    """Extract battle-state modifiers; return (stripped_question, atk_mods, def_mods, weather).
    Offensive kit binds to the attacker, defensive kit to the defender."""
    q = question.lower()
    atk, dfn = {}, {}

    def strip(pattern):
        nonlocal q
        q = re.sub(pattern, " ", q)

    for phrase, boosts in _SETUP.items():
        if re.search(rf"(?:after|following)[a-z ]*\b{phrase}\b|\b{phrase}\b(?: boost(?:ed)?)?", q) \
                and re.search(rf"after|following|boost|\+", q):
            atk.setdefault("boosts", {}).update(boosts)
            strip(rf"(?:after (?:a |an |using )?)?\b{phrase}\b")
    m = re.search(r"\+([1-6])\b(?:\s*(attack|atk|spa|special attack|speed))?", q)
    if m:
        n = int(m.group(1))
        stat = _EV_STATS.get(m.group(2) or "", None)
        boosts = {stat: n} if stat else {"Attack": n, "Special Attack": n}
        atk.setdefault("boosts", {}).update(boosts)
        strip(r"\+[1-6]\b(?:\s*(?:attack|atk|spa|special attack|speed))?")
    for phrase, item in _OFFENSE_ITEMS.items():
        if re.search(rf"\b{phrase}\b", q):
            atk["item"] = item
            strip(rf"\b{phrase}\b")
            break
    for phrase, item in _DEFENSE_ITEMS.items():
        if re.search(rf"\b{phrase}\b", q):
            dfn["item"] = item
            strip(rf"\b{phrase}\b")
            break
    for nature, updown in _NATURES.items():
        if re.search(rf"\b{nature}\b", q):
            side = atk if nature in _OFFENSIVE_NATURES else dfn
            side["nature"] = updown
            side.setdefault("evs", {})[updown[0]] = 252
            strip(rf"\b{nature}\b")
    for m2 in re.finditer(r"(?:max|252)\s+(attack|atk|spa|special attack|hp|def|defense|spd|special defense|speed|spe)", q):
        stat = _EV_STATS[m2.group(1)]
        side = dfn if stat in ("HP", "Defense", "Special Defense") else atk
        side.setdefault("evs", {})[stat] = 252
    strip(r"(?:max|252)\s+(?:attack|atk|spa|special attack|hp|def|defense|spd|special defense|speed|spe)")
    if re.search(r"\bburn(?:ed|t)?\b", q):
        atk["status"] = "brn"
        strip(r"\bburn(?:ed|t)?\b")
    weather = "none"
    m3 = re.search(r"\bin (?:the )?(sun|rain|sand(?:storm)?|snow|hail)\b", q)
    if m3:
        weather = {"sandstorm": "sand"}.get(m3.group(1), m3.group(1))
        strip(r"\bin (?:the )?(?:sun|rain|sand(?:storm)?|snow|hail)\b")
    m4 = re.search(r"\btera[- ]?([a-z]+)\b", q)
    if m4 and m4.group(1).capitalize() in tools.known_types():
        atk["tera"] = m4.group(1)
        strip(r"\btera[- ]?[a-z]+\b")
    return q, atk, dfn, weather


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
            name, typ, _cls = tools.known_moves()[moves[0]]
            return tools.type_matchup(typ, mons[0], via_move=name)
        if mons and types:
            return tools.type_matchup(types[0], mons[0])
        if mons:
            return tools.defensive_profile(mons[0])
        if len(types) >= 2:
            return tools.type_matchup(types[0], types[1])
        if len(types) == 1:
            return tools.defensive_profile(types[0])

    # "what would X need to survive Y's Z" -> defensive escalation search
    if _SURVIVE_HOW.search(question) and len(mons) >= 2:
        q0, atk_mods, _dm, _wx = _parse_mods(question)
        moves = _find_names(q0, tools.known_moves())
        if moves:
            move_pos = q0.find(moves[0])
            attacker = min(mons, key=lambda m: abs(q0.find(m) - move_pos))
            defender = next(m for m in mons if m != attacker)
            name = tools.known_moves()[moves[0]][0]
            p = tools.survive_search(attacker, name, defender, atk_mods)
            if p:
                return p

    # "what would it take for X's Y to OHKO Z" -> tiered escalation search
    if _OHKO_HOW.search(question) and len(mons) >= 2:
        moves = _find_moves(question)
        if moves:
            q2 = question.lower()
            to_m = re.search(r"\b(?:to )?ohko\s+([a-z0-9' -]+)", q2)
            defender = next((m for m in mons if to_m and m in to_m.group(1)), None)
            if defender is None:
                move_pos = q2.find(moves[0])
                attacker = min(mons, key=lambda m: abs(q2.find(m) - move_pos))
                defender = next(m for m in mons if m != attacker)
            else:
                attacker = next(m for m in mons if m != defender)
            name = tools.known_moves()[moves[0]][0]
            p = tools.ohko_search(attacker, name, defender)
            if p:
                return p

    # "does Garchomp's Earthquake OHKO Heatran" / "how much damage does X's Y do to Z"
    # with optional battle state: boosts, items, natures, EVs, burn, weather, tera
    if _DAMAGE.search(question) and len(mons) >= 2:
        q, atk_mods, def_mods, weather = _parse_mods(question)
        moves = _find_names(q, tools.known_moves())
        if moves:
            move_pos = q.find(moves[0])
            to_m = re.search(r"\bto\s+([a-z0-9' -]+)", q)
            defender = next((m for m in mons if to_m and m in to_m.group(1)), None)
            if defender:
                attacker = next(m for m in mons if m != defender)
            else:
                attacker = min(mons, key=lambda m: abs(q.find(m) - move_pos))
                defender = next(m for m in mons if m != attacker)
            name = tools.known_moves()[moves[0]][0]
            p = tools.damage_calc(attacker, name, defender, atk_mods, def_mods, weather)
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
