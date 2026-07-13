"""Deterministic tools: computed answers rendered as citable pseudo-passages.

Retrieval answers "what does the corpus say"; these answer "what does the math
say". Each tool returns passages shaped like retrieval results (source, title,
content) with rerank_score 1.0, so the gate, the citation prompt, and the eval
all work unchanged. Data comes straight from the PokeAPI CSVs (same checkout
the corpus adapter reads).
"""
import csv
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from config import settings
from corpora import monotype_data as md
from corpora import ou_data as ou


def _toid(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

EN = "9"


def _passage(source: str, title: str, content: str) -> dict:
    return {"source": source, "title": title, "content": content,
            "similarity": 1.0, "kw_rank": 0.0, "rerank_score": 1.0}


@lru_cache(maxsize=1)
def _data():
    """Load pokemon (incl. forms), stats, types, and the type chart once."""
    csvdir = Path(settings.pokeapi_path) / "data" / "v2" / "csv"

    def rows(name):
        with open(csvdir / f"{name}.csv", newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)

    type_names = {r["type_id"]: r["name"] for r in rows("type_names")
                  if r["local_language_id"] == EN and int(r["type_id"]) < 10000}
    # damage_factor is percent: 0 / 50 / 100 / 200
    chart = {}
    for r in rows("type_efficacy"):
        att, dfn = type_names.get(r["damage_type_id"]), type_names.get(r["target_type_id"])
        if att and dfn:
            chart[(att, dfn)] = int(r["damage_factor"]) / 100

    species_name = {r["pokemon_species_id"]: r["name"] for r in rows("pokemon_species_names")
                    if r["local_language_id"] == EN}
    stat_names = {"1": "HP", "2": "Attack", "3": "Defense",
                  "4": "Special Attack", "5": "Special Defense", "6": "Speed"}
    p_stats, p_types = {}, {}
    for r in rows("pokemon_stats"):
        p_stats.setdefault(r["pokemon_id"], {})[stat_names.get(r["stat_id"], "?")] = int(r["base_stat"])
    for r in rows("pokemon_types"):
        t = type_names.get(r["type_id"])
        if t:
            p_types.setdefault(r["pokemon_id"], []).append((int(r["slot"]), t))

    ability_names = {r["ability_id"]: r["name"] for r in rows("ability_names")
                     if r["local_language_id"] == EN}
    p_abils = {}
    for r in rows("pokemon_abilities"):
        a = ability_names.get(r["ability_id"])
        if a:
            p_abils.setdefault(r["pokemon_id"], []).append(a)

    move_names = {r["move_id"]: r["name"] for r in rows("move_names")
                  if r["local_language_id"] == EN}
    dmg_class2 = {r["id"]: r["identifier"] for r in rows("move_damage_classes")}
    moves = {}  # lowercase move name -> (Name, type, class)
    spread_moves = set()  # hit multiple targets in doubles (0.75x): all-other-pokemon / all-opponents
    for r in rows("moves"):
        name, typ = move_names.get(r["id"]), type_names.get(r["type_id"])
        if name and typ:
            moves[name.lower()] = (name, typ, dmg_class2.get(r["damage_class_id"], "?"))
            if r["target_id"] in ("9", "11"):
                spread_moves.add(name.lower())

    def pretty(ident):
        return "-".join(p.capitalize() for p in ident.split("-"))

    mons = {}  # lowercase display name -> {name, types, stats}
    for r in rows("pokemon"):
        name = (species_name.get(r["species_id"], pretty(r["identifier"]))
                if r["is_default"] == "1" else pretty(r["identifier"]))
        mons[name.lower()] = {
            "name": name,
            "types": [t for _, t in sorted(p_types.get(r["id"], []))],
            "stats": p_stats.get(r["id"], {}),
            "abilities": p_abils.get(r["id"], []),
            "weight_kg": int(r["weight"] or 0) / 10,
        }
    # Showdown-id -> display-name maps, so the chaos-JSON tools (which carry ids like
    # "swordsdance"/"heavydutyboots") can render readable pastes and match roles.
    try:
        item_names = {r["item_id"]: r["name"] for r in rows("item_names")
                      if r["local_language_id"] == EN}
    except FileNotFoundError:
        item_names = {}
    moves_by_id = {_toid(n): n for n in move_names.values()}
    abilities_by_id = {_toid(n): n for n in ability_names.values()}
    items_by_id = {_toid(n): n for n in item_names.values()}

    return {"chart": chart, "mons": mons, "moves": moves, "spread": spread_moves,
            "types": sorted(set(type_names.values())),
            "moves_by_id": moves_by_id, "items_by_id": items_by_id,
            "abilities_by_id": abilities_by_id}


def known_pokemon() -> dict:
    return _data()["mons"]


def known_types() -> list[str]:
    return _data()["types"]


def known_moves() -> dict:
    return _data()["moves"]


# Abilities that negate or absorb an attacking type; the type chart alone
# misses these (Bronzong shrugs off Earthquake via Levitate).
ABILITY_IMMUNITIES = {
    "Levitate": "Ground", "Earth Eater": "Ground",
    "Flash Fire": "Fire", "Well-Baked Body": "Fire",
    "Water Absorb": "Water", "Storm Drain": "Water", "Dry Skin": "Water",
    "Volt Absorb": "Electric", "Lightning Rod": "Electric", "Motor Drive": "Electric",
    "Sap Sipper": "Grass", "Wind Rider": "Flying",
}


# ---- tools ------------------------------------------------------------------

def type_matchup(attacking_type: str, defender: str, via_move: str | None = None) -> list[dict]:
    """Effectiveness of an attacking type into a defender (Pokemon name or type)."""
    d = _data()
    att = attacking_type.capitalize()
    mon = d["mons"].get(defender.lower())
    def_types = mon["types"] if mon else [defender.capitalize()]
    mult = 1.0
    parts = []
    for t in def_types:
        f = d["chart"].get((att, t), 1.0)
        mult *= f
        parts.append(f"{f:g}x vs {t}")
    target = f"{mon['name']} ({' / '.join(def_types)})" if mon else def_types[0]
    verdict = ("has no effect on" if mult == 0 else
               "is super effective against" if mult > 1 else
               "is not very effective against" if mult < 1 else
               "deals neutral damage to")
    prefix = f"{via_move} is a {att}-type move. " if via_move else ""
    lines = [f"Type matchup, computed from the type chart: {prefix}{att} {verdict} {target}, "
             f"overall {mult:g}x ({', '.join(parts)})."]

    # a Ground-vs-Flying immunity is conditional, not absolute
    if att == "Ground" and "Flying" in def_types and mult == 0:
        grounded = 1.0
        for t2 in def_types:
            grounded *= 1.0 if t2 == "Flying" else d["chart"].get((att, t2), 1.0)
        roost_types = [t2 for t2 in def_types if t2 != "Flying"] or ["Normal"]
        roosted = 1.0
        for t2 in roost_types:
            roosted *= d["chart"].get((att, t2), 1.0)
        name = mon["name"] if mon else "the defender"
        lines.append(
            f"This immunity is conditional: if {name} is grounded (Gravity, Smack Down, "
            f"Thousand Arrows, or holding an Iron Ball), Ground moves hit for {grounded:g}x; "
            f"during a turn {name} uses Roost it loses its Flying type and takes {roosted:g}x.")
    # the chart cannot see ability-based immunities
    if mon:
        blockers = [a for a in mon.get("abilities", []) if ABILITY_IMMUNITIES.get(a) == att]
        if blockers and mult > 0:
            lines.append(f"Note: {mon['name']} can have the ability {', '.join(blockers)}, "
                         f"which negates or absorbs {att}-type moves entirely.")
    supports = ([f"pokedex#{mon['name']}"] if mon else []) + ([f"move#{via_move}"] if via_move else [])
    return [_passage(f"tool#type_matchup", f"{att} vs {target}", "\n".join(lines))] \
        + _corpus_docs(supports)


def defensive_profile(defender: str) -> list[dict]:
    """What is super effective / resisted / immune against a Pokemon or type."""
    d = _data()
    mon = d["mons"].get(defender.lower())
    def_types = mon["types"] if mon else [defender.capitalize()]
    target = f"{mon['name']} ({' / '.join(def_types)})" if mon else def_types[0]
    buckets = {}
    for att in d["types"]:
        mult = 1.0
        for t in def_types:
            mult *= d["chart"].get((att, t), 1.0)
        buckets.setdefault(mult, []).append(att)
    lines = [f"Defensive type profile of {target}, computed from the type chart:"]
    for mult in sorted(buckets, reverse=True):
        if mult != 1.0:
            label = {0.0: "immune to"}.get(mult, f"takes {mult:g}x from")
            lines.append(f"- {label}: {', '.join(buckets[mult])}")
    return [_passage(f"tool#type_matchup", f"defensive profile: {target}", "\n".join(lines))] \
        + _corpus_docs([f"pokedex#{mon['name']}"] if mon else [])


def speed_check(name_a: str, name_b: str) -> list[dict]:
    """Compare two Pokemon's base Speed."""
    d = _data()
    a, b = d["mons"].get(name_a.lower()), d["mons"].get(name_b.lower())
    if not a or not b:
        return []
    sa, sb = a["stats"].get("Speed", 0), b["stats"].get("Speed", 0)
    faster = a if sa > sb else b
    verdict = (f"{faster['name']} is faster at base stats" if sa != sb
               else "they are tied at base stats")
    content = (f"Base Speed comparison, computed from base stats: {a['name']} has base Speed "
               f"{sa}, {b['name']} has base Speed {sb}; {verdict}. Note: items (Choice Scarf), "
               f"natures, EVs, and boosts change effective speed in battle.")
    return [_passage("tool#speed_check", f"{a['name']} vs {b['name']} speed", content)] \
        + _corpus_docs([f"pokedex#{a['name']}", f"pokedex#{b['name']}"])


def _usage_rankings(fmt: str = "gen9ou") -> list[tuple[str, float]]:
    """(name, usage%) pairs parsed from the synthetic usage-rankings document —
    the top-30 meta filter for tools that would otherwise list 900 Pokemon."""
    try:
        from db import connect
        import re as _re
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT content FROM chunks WHERE source = %s ORDER BY id",
                        (f"{fmt}_chaos#usage_rankings",))
            text = "\n".join(r[0] for r in cur.fetchall())
        out, seen = [], set()
        # the doc spans chunks whose 150-char overlap repeats a ranking line
        for m in _re.finditer(r"^\d+\. (.+?) \(([\d.]+)%\)", text, _re.M):
            if m.group(1) not in seen:
                seen.add(m.group(1))
                out.append((m.group(1), float(m.group(2))))
        return out
    except Exception:
        return []


def speed_tiers(name: str, fmt: str = "gen9ou") -> list[dict]:
    """What outspeeds this Pokemon: faster mons from the usage top-30, speed ties,
    and the Choice Scarf math (x1.5) for slower mons that outspeed when scarfed."""
    d = _data()
    mon = d["mons"].get(name.lower())
    if not mon or not mon["stats"]:
        return []
    s = mon["stats"].get("Speed", 0)
    meta = _usage_rankings(fmt)
    if not meta:
        return []
    rows = []
    for n, u in meta:
        m = d["mons"].get(n.lower())
        if m and m["stats"] and n != mon["name"]:
            rows.append((n, m["stats"].get("Speed", 0), u))
    faster = sorted([r for r in rows if r[1] > s], key=lambda r: r[1], reverse=True)
    ties = [r for r in rows if r[1] == s]
    # level 100, 31 IV, 252 EV, neutral nature: stat = 2*base + 99. A scarfed mon
    # outspeeds when 1.5*(2b+99) > (2s+99), i.e. base above this floor:
    scarf_floor = int(((2 * s + 99) / 1.5 - 99) / 2) + 1
    scarfers = sorted([r for r in rows if scarf_floor <= r[1] <= s],
                      key=lambda r: r[1], reverse=True)

    def fmt_rows(rs):
        return ", ".join(f"{n} (base {sp}, {u:.1f}% usage)" for n, sp, u in rs)

    lines = [f"Speed tiers vs {mon['name']} (base Speed {s}), computed from base stats "
             f"against the {fmt} usage top 30."]
    lines.append(f"Faster than {mon['name']}: {fmt_rows(faster)}." if faster else
                 f"No Pokemon in the {fmt} usage top 30 has a higher base Speed than "
                 f"{mon['name']}.")
    if ties:
        lines.append(f"Speed ties with {mon['name']}: {fmt_rows(ties)}.")
    scarf = (f"Choice Scarf math: a Choice Scarf multiplies Speed by 1.5, so with max "
             f"investment on both sides, scarfed Pokemon with base Speed {scarf_floor} or "
             f"higher also outspeed {mon['name']}.")
    if scarfers:
        scarf += f" Among the top 30, that adds: {fmt_rows(scarfers)}."
    lines.append(scarf)
    lines.append("Note: natures, Speed EVs, boosts, paralysis, and Tailwind change "
                 "effective speed in battle; this assumes equal max investment.")
    return [_passage("tool#speed_tiers", f"what outspeeds {mon['name']}", "\n".join(lines))] \
        + _corpus_docs([f"pokedex#{mon['name']}", f"{fmt}_chaos#usage_rankings"])


def stat_query(stat: str, type_filter: str | None = None, n: int = 10,
               lowest: bool = False) -> list[dict]:
    """Top-N Pokemon by a base stat, optionally filtered to one type."""
    d = _data()
    rows = [(m["stats"].get(stat, 0), m["name"]) for m in d["mons"].values()
            if m["stats"] and (not type_filter or type_filter.capitalize() in m["types"])]
    rows.sort(reverse=not lowest)
    rows = rows[:n]
    if not rows:
        return []
    scope = f"{type_filter.capitalize()}-type Pokemon" if type_filter else "all Pokemon"
    order = "lowest" if lowest else "highest"
    lines = [f"{order.capitalize()} base {stat} among {scope} (computed from base stats, "
             f"alternate forms included): "
             + ", ".join(f"{i}. {name} ({v})" for i, (v, name) in enumerate(rows, 1))]
    return [_passage("tool#stat_query", f"{order} {stat}: {scope}", lines[0])]


def monotype_stat_query(type_name: str, stat: str, n: int = 10, lowest: bool = False) -> list[dict]:
    """Top-N by a base stat among the Pokemon actually used on a <type> monotype
    team (the meta skeleton from gen9monotype usage), not the whole species pool —
    "fastest on a Steel team" should mean the fastest Steel mon people run, not the
    fastest Steel-type in the game."""
    meta = md.type_usage(settings.crystal_battle_path, type_name.lower())
    if not meta:
        return []
    d = _data()
    rows = []
    for name, usage in meta[:20]:
        m = d["mons"].get(name.lower())
        if m and m["stats"]:
            rows.append((m["stats"].get(stat, 0), m["name"], usage))
    if not rows:
        return []
    rows.sort(reverse=not lowest)
    rows = rows[:n]
    Typ = type_name.capitalize()
    order = "lowest" if lowest else "highest"
    body = (f"{order.capitalize()} base {stat} among the Pokemon commonly used on "
            f"{Typ}-type teams in Gen 9 Monotype (gen9monotype usage; base stats computed): "
            + ", ".join(f"{i}. {name} (base {v}, {u:.0f}% usage)"
                        for i, (v, name, u) in enumerate(rows, 1)) + ".")
    return [_passage("tool#monotype_stat_query",
                     f"{order} {stat} on {Typ} monotype teams", body)] \
        + _corpus_docs([f"gen9monotype_usage#{Typ}"])


# competitive roles inferred from a mon's most-used moves/items (Stage 1 recommender)
_ROLE_MOVES = {
    "hazard setter": {"stealth rock", "spikes", "toxic spikes", "sticky web", "ceaseless edge", "stone axe"},
    "hazard control": {"rapid spin", "defog", "mortal spin", "tidy up"},
    "pivot": {"u-turn", "volt switch", "flip turn", "teleport", "parting shot", "chilly reception"},
    "setup sweeper": {"swords dance", "dragon dance", "nasty plot", "calm mind", "quiver dance",
                      "shell smash", "bulk up", "agility", "victory dance", "clangorous soul",
                      "no retreat", "belly drum"},
    "cleric": {"wish", "heal bell", "aromatherapy"},
    "screens": {"reflect", "light screen", "aurora veil"},
    "status spreader": {"will-o-wisp", "thunder wave", "toxic", "spore", "sleep powder", "glare", "nuzzle"},
}
_RECOVERY = {"recover", "roost", "slack off", "soft-boiled", "morning sun", "moonlight",
             "synthesis", "rest", "milk drink", "shore up", "strength sap", "wish", "jungle healing"}
_CHOICE_OFFENSE = {"choice band", "choice specs", "life orb"}


def _roles_from(moves: set, items: set) -> list[str]:
    roles = [role for role, kws in _ROLE_MOVES.items() if moves & kws]
    if moves & _RECOVERY:
        roles.append("wall / staller")
    if "choice scarf" in items:
        roles.append("speed control")
    if items & _CHOICE_OFFENSE and "setup sweeper" not in roles:
        roles.append("wallbreaker")
    return roles


def _team_roles(entry: dict) -> list[str]:
    """The roles a mon CAN fill, across its whole observed movepool/items — the
    recommender shows a mon's versatility."""
    return _roles_from({m.lower() for m, _ in entry.get("moves", [])},
                       {i.lower() for i, _ in entry.get("items", [])[:3]})


def _set_roles(entry: dict) -> list[str]:
    """The roles a mon actually fills in its generated set (top-4 moves + top item) —
    the generator must describe coverage honestly, by the sets it builds."""
    return _roles_from({m.lower() for m, _ in entry.get("moves", [])[:4]},
                       {entry["items"][0][0].lower()} if entry.get("items") else set())


def _rank_teammates(movesets, usage, have, want_role, n):
    """Rank candidates by teammate co-occurrence with the given core (or by usage when
    no core), each tagged with its roles. Returns (candidates, have_names)."""
    have_lc = {h.lower() for h in have}
    have_names = [m for m in movesets if m.lower() in have_lc]
    tm = {h: {x.lower(): p for x, p in movesets[h].get("teammates", [])} for h in have_names}
    cands = []
    for mon, entry in movesets.items():
        if mon.lower() in have_lc:
            continue
        roles = _team_roles(entry)
        if want_role and want_role not in roles:
            continue
        u = usage.get(mon, 0.0)
        coocc = (sum(tm[h].get(mon.lower(), 0.0) for h in have_names) / len(have_names)) if have_names else 0.0
        cands.append((mon, u, coocc, roles))
    cands.sort(key=lambda c: (c[2], c[1]) if have_names else (c[1], 0.0), reverse=True)
    return cands[:n], have_names


def _teammate_lines(cands, have_names, movesets):
    lines = []
    for mon, u, coocc, roles in cands:
        tag = f" — {', '.join(roles)}" if roles else ""
        extra = f", {coocc:.0f}% together" if have_names and coocc else ""
        lines.append(f"- {mon} ({u:.0f}% usage{extra}){tag}")
    if have_names:
        core_roles = set().union(*(set(_team_roles(movesets[h])) for h in have_names))
        missing = [r for r in _KEY_ROLES if r not in core_roles]
        if missing:
            lines.append("Roles your current core lacks: " + ", ".join(missing)
                         + " — prioritize teammates that provide them.")
    return lines


def recommend_teammates(type_name: str, have: list[str] | None = None,
                        want_role: str | None = None, n: int = 6) -> list[dict]:
    """Grounded monotype teammate suggestions for a type (usage, or co-occurrence with
    a given core), each tagged with its role, plus the roles the core is missing."""
    have = have or []
    cb = settings.crystal_battle_path
    ms = md.type_moveset(cb, type_name.lower())
    if not ms:
        return []
    usage = dict(md.type_usage(cb, type_name.lower()))
    cands, have_names = _rank_teammates(ms, usage, have, want_role, n)
    if not cands:
        return []
    Typ = type_name.capitalize()
    if have_names:
        head = (f"Recommended {Typ}-type teammates in Gen 9 Monotype to pair with "
                f"{', '.join(have_names)}, ranked by how often they are used together "
                f"(Smogon teammate stats):")
    else:
        head = (f"Recommended {Typ}-type teammates in Gen 9 Monotype — the staple "
                f"partners on {Typ} teams, ranked by usage:")
    lines = [head] + _teammate_lines(cands, have_names, ms)
    return [_passage("tool#recommend_teammates", f"{Typ} monotype teammates", "\n".join(lines))] \
        + _corpus_docs([f"gen9monotype_usage#{Typ}"])


def recommend_ou_teammates(have: list[str] | None = None,
                           want_role: str | None = None, n: int = 6) -> list[dict]:
    """Grounded Gen 9 OU teammate suggestions (usage, or co-occurrence with a given
    core), each tagged with its role, plus the roles the core is missing."""
    have = have or []
    ms = _ou_movesets("gen9ou")
    if not ms:
        return []
    usage = dict(_ou_usage("gen9ou"))
    cands, have_names = _rank_teammates(ms, usage, have, want_role, n)
    if not cands:
        return []
    if have_names:
        head = (f"Recommended Gen 9 OU teammates to pair with {', '.join(have_names)}, "
                f"ranked by how often they are used together (Smogon teammate stats):")
    else:
        head = "Recommended Gen 9 OU teammates — the staple partners, ranked by usage:"
    lines = [head] + _teammate_lines(cands, have_names, ms)
    return [_passage("tool#recommend_teammates", "Gen 9 OU teammates", "\n".join(lines))] \
        + _corpus_docs(["gen9ou_chaos#usage_rankings"])


# Stage 2 team generator: assemble a legal team, gated by the Showdown validator.
_NODE = shutil.which("node")
_VALIDATE_JS = os.path.expanduser("~/team-tools/validate_teams.js")
_STAT_ORDER = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
_KEY_ROLES = ["hazard setter", "hazard control", "speed control", "wall / staller", "pivot"]

# Archetype profiles: `build` = roles the generator fills for (its comp target),
# `expect` = roles the critic treats as required, `speed` = whether speed control
# is wanted. The same knob drives generation and criticism, so an HO team is not
# built with (or dinged for lacking) a wall, and stall is not dinged for no Scarf.
_ARCHETYPES = {
    # most offensive: fast breakers + setup sweepers behind hazard chip, no walls.
    "hyper offense": {"build": ["hazard setter", "speed control", "setup sweeper", "wallbreaker", "pivot"],
                      "expect": ["hazard setter", "speed control"], "speed": True, "wall": False},
    # offense with a defensive backbone: breakers + speed control + a wall or two.
    "bulky offense": {"build": ["hazard setter", "hazard control", "speed control", "setup sweeper", "wallbreaker", "wall / staller"],
                      "expect": ["hazard setter", "hazard control", "speed control"], "speed": True, "wall": True},
    # the all-rounder: full role coverage (hazards, removal, speed, a wall, a pivot).
    "balance": {"build": ["hazard setter", "hazard control", "speed control", "wall / staller", "pivot", "wallbreaker"],
                "expect": ["hazard setter", "hazard control", "speed control", "wall / staller", "pivot"], "speed": True, "wall": True},
    # fat balance: stacked walls + bulky pivots + hazards, but keeps one wallbreaker
    # as a wincon (more offense than stall, far more bulk than balance).
    "fat": {"build": ["wall / staller", "hazard setter", "hazard control", "pivot", "cleric", "wallbreaker"],
            "expect": ["wall / staller", "hazard setter", "hazard control", "pivot"], "speed": False, "wall": True},
    # most defensive: walls, recovery, hazards, status; wins by outlasting, no Scarf.
    "stall": {"build": ["wall / staller", "hazard setter", "hazard control", "cleric", "status spreader"],
              "expect": ["wall / staller", "hazard setter", "hazard control"], "speed": False, "wall": True},
}


def _archetype(name):
    return _ARCHETYPES.get((name or "balance").lower(), _ARCHETYPES["balance"])


@lru_cache(maxsize=4)
def _ou_movesets(fmt="gen9ou"):
    """gen9ou chaos movesets with Showdown ids rendered to display names, so role
    inference and set-building (written for the display-name monotype data) work
    unchanged. Teammate/check names and spreads are already display form."""
    d = _data()
    mbi, ibi, abi = d["moves_by_id"], d["items_by_id"], d["abilities_by_id"]

    def disp(name, table):
        return table.get(_toid(name), name.replace("-", " ").title())

    out = {}
    for mon, s in ou.movesets(settings.crystal_battle_path, fmt).items():
        out[mon] = {
            "abilities": [(disp(n, abi), p) for n, p in s["abilities"]],
            "items": [(disp(n, ibi), p) for n, p in s["items"]],
            "moves": [(disp(n, mbi), p) for n, p in s["moves"]],
            "spreads": s["spreads"],
            "tera": [(t.capitalize(), p) for t, p in s["tera"]],
            "teammates": s["teammates"],
            "checks": s["checks"],
            "raw": s["raw"],
        }
    return out


def _ou_usage(fmt="gen9ou"):
    return ou.usage(settings.crystal_battle_path, fmt)


def _mon_set_lines(mon: str, entry: dict, tera: bool = False) -> str:
    """A Showdown-paste set from the mon's most-used moves / item / ability / spread
    (plus Tera type, for formats that allow Terastallization)."""
    moves = [m for m, _ in entry.get("moves", [])[:4]]
    item = entry["items"][0][0] if entry.get("items") else None
    abil = entry["abilities"][0][0] if entry.get("abilities") else None
    spread = entry["spreads"][0][0] if entry.get("spreads") else None
    lines = [f"{mon} @ {item}" if item else mon]
    if abil:
        lines.append(f"Ability: {abil}")
    if spread and ":" in spread:
        nat, evs = spread.split(":")
        evs = [int(x) for x in evs.split("/")]
        es = " / ".join(f"{v} {_STAT_ORDER[i]}" for i, v in enumerate(evs) if v)
        if es:
            lines.append(f"EVs: {es}")
        lines.append(f"{nat} Nature")
    if tera and entry.get("tera"):
        lines.append(f"Tera Type: {entry['tera'][0][0]}")
    lines += [f"- {mv}" for mv in moves]
    return "\n".join(lines)


def _validate_team(paste: str, fmt: str = "gen9monotype", name: str = "Team") -> tuple[bool, bool, list[str]]:
    """Run the team through Showdown's TeamValidator for the given format.
    Returns (validated, clean, problems); validated=False when node/validator is
    unavailable (the tool then degrades to unvalidated sets rather than blocking)."""
    if not _NODE or not os.path.exists(_VALIDATE_JS):
        return False, True, []
    body = f"=== [{fmt}] {name} ===\n\n{paste}\n"
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(body)
            path = f.name
        r = subprocess.run([_NODE, _VALIDATE_JS, path, fmt],
                           capture_output=True, text=True, timeout=60)
    except Exception:
        return False, True, []
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
    clean = r.stdout.lstrip().startswith("OK ")
    problems = [l.strip() for l in r.stdout.splitlines() if l.startswith("  ")]
    return True, clean, problems


def _assemble_team(ms, usage, have, want, fmt, tera):
    """Shared role-aware assembly + set-building + validate + critique-refine loop
    (format-agnostic: monotype or gen9ou). Returns (team, paste, validated, clean,
    problems)."""
    pool = sorted(ms, key=lambda m: usage.get(m, 0), reverse=True)

    def roles_of(t):
        return set().union(*(set(_set_roles(ms[m])) for m in t)) if t else set()

    def coocc(c, t):
        if not t:
            return 0.0
        return sum({n.lower(): p for n, p in ms[m].get("teammates", [])}.get(c.lower(), 0.0)
                   for m in t) / len(t)

    def fill(team, banned):
        while len(team) < 6:
            miss = set(want) - roles_of(team)
            best = max((c for c in pool if c not in team and c not in banned),
                       key=lambda c: len(set(_set_roles(ms[c])) & miss) * 1000
                       + coocc(c, team) + usage.get(c, 0) * 0.5, default=None)
            if not best:
                break
            team.append(best)
        return team

    team = fill([m for m in ms if m.lower() in {h.lower() for h in have}], set())
    banned, paste = set(), ""
    for _ in range(4):
        paste = "\n\n".join(_mon_set_lines(m, ms[m], tera) for m in team)
        _, clean, problems = _validate_team(paste, fmt)
        if clean:
            break
        flagged = {m for m in team for p in problems if m.lower() in p.lower()}
        if not flagged:
            break                       # non-mon-specific problem: stop and report it
        banned |= flagged
        team = fill([m for m in team if m not in flagged], banned)

    # critique-driven refinement (the loop): if a key role is still missing AND the
    # meta has a mon that supplies it, swap out a redundant member and re-validate.
    for _ in range(2):
        missing = [r for r in want if r not in roles_of(team)]
        if not missing:
            break
        filler = next((c for c in pool if c not in team and c not in banned
                       and missing[0] in _set_roles(ms[c])), None)
        drop = _least_valuable(team, ms, usage)
        if not filler or not drop:
            break
        cand = [m for m in team if m != drop] + [filler]
        cpaste = "\n\n".join(_mon_set_lines(m, ms[m], tera) for m in cand)
        _, cclean, _ = _validate_team(cpaste, fmt)
        if cclean:
            team, paste = cand, cpaste
        else:
            banned.add(filler)

    validated, clean, problems = _validate_team(paste, fmt)
    return team, paste, validated, clean, problems


def _legality_line(validated, clean, problems, fmt):
    if validated and clean:
        return f"Legality: passes the {fmt} team validator."
    if validated:
        return "Note: the validator flagged " + "; ".join(problems[:3]) + "."
    return "Note: sets are from usage stats (team validator was unavailable)."


def generate_team(type_name: str, have: list[str] | None = None,
                  archetype: str = "balance") -> list[dict]:
    """Build a full 6-mon Gen 9 Monotype team of one type and archetype, gated by the
    Showdown validator."""
    arch = _archetype(archetype)
    cb = settings.crystal_battle_path
    ms = md.type_moveset(cb, type_name.lower())
    if not ms:
        return []
    usage = dict(md.type_usage(cb, type_name.lower()))
    team, paste, validated, clean, problems = _assemble_team(
        ms, usage, have or [], arch["build"], "gen9monotype", tera=False)
    Typ = type_name.capitalize()
    label = "" if archetype.lower() == "balance" else f" {archetype.lower()}"
    parts = [f"A Gen 9 Monotype{label} {Typ} team, built from usage and teammate stats:", "", paste, ""]
    parts += _analysis_lines(type_name, team, ms, arch)
    parts.append(_legality_line(validated, clean, problems, "gen9monotype"))
    return [_passage("tool#generate_team", f"{Typ} monotype team", "\n".join(parts))] \
        + _corpus_docs([f"gen9monotype_usage#{Typ}"])


def generate_ou_team(have: list[str] | None = None, archetype: str = "balance") -> list[dict]:
    """Build a full 6-mon Gen 9 OU team of an archetype from usage + teammate stats
    (Tera included), gated by the Showdown validator. Stage A output is the team plus a
    role-coverage note; the full OU critic (type coverage + threats) is analyze_ou_team."""
    arch = _archetype(archetype)
    ms = _ou_movesets("gen9ou")
    if not ms:
        return []
    usage = dict(_ou_usage("gen9ou"))
    team, paste, validated, clean, problems = _assemble_team(
        ms, usage, have or [], arch["build"], "gen9ou", tera=True)
    label = "" if archetype.lower() == "balance" else f" {archetype.lower()}"
    covered = set().union(*(set(_set_roles(ms[m])) for m in team)) if team else set()
    missing = [r for r in arch["expect"] if r not in covered]
    parts = [f"A Gen 9 OU{label} team, built from usage and teammate stats:", "", paste, "",
             "Roles covered: " + (", ".join(sorted(covered)) or "none") + "."]
    if missing:
        parts.append("Missing roles for this archetype: " + ", ".join(missing) + ".")
    parts.append(_legality_line(validated, clean, problems, "gen9ou"))
    return [_passage("tool#generate_team", "Gen 9 OU team", "\n".join(parts))] \
        + _corpus_docs(["gen9ou_chaos#usage_rankings"])


def _least_valuable(team, ms, usage):
    """The lowest-usage member whose key roles are all also covered by a teammate
    (safe to swap out without losing coverage); None if no member is redundant."""
    kr = {m: set(_set_roles(ms[m])) & set(_KEY_ROLES) for m in team if m in ms}
    redundant = [m for m in kr if len(kr) > 1
                 and kr[m] <= set().union(*(r for k, r in kr.items() if k != m))]
    return min(redundant, key=lambda m: usage.get(m, 0)) if redundant else None


def _team_analysis(type_name, team, ms, arch):
    role_by_mon = {m: set(_set_roles(ms[m])) for m in team if m in ms}
    covered = set().union(*role_by_mon.values()) if role_by_mon else set()
    missing = [r for r in arch["expect"] if r not in covered]
    rc = Counter(r for rs in role_by_mon.values() for r in rs)
    redundant = [r for r, c in rc.items() if c >= 3 and r in _KEY_ROLES]
    mc = md.matchup_chart(settings.crystal_battle_path).get(type_name.lower(), {})
    weak = sorted(((o, w) for o, w in mc.items() if o != type_name.lower() and w <= 45),
                  key=lambda x: x[1])[:5]
    return {"covered": covered, "missing": missing, "redundant": redundant, "weak": weak}


def _analysis_lines(type_name, team, ms, arch):
    a = _team_analysis(type_name, team, ms, arch)
    lines = ["Roles covered: " + (", ".join(sorted(a["covered"])) or "none") + "."]
    if a["missing"]:
        lines.append("Missing roles for this archetype: " + ", ".join(a["missing"]) + ".")
    if a["redundant"]:
        lines.append("Redundant (on 3+ members): " + ", ".join(a["redundant"]) + ".")
    if arch["speed"] and "speed control" not in a["covered"]:
        lines.append("No dedicated speed control (Choice Scarf / Tailwind) — watch for faster teams.")
    if a["weak"]:
        lines.append("Weakest against these monotypes: "
                     + ", ".join(f"{o.capitalize()} ({w:.0f}%)" for o, w in a["weak"]) + ".")
    return lines


def team_type(mons: list[str]) -> str | None:
    """The single type shared by every named mon (the monotype), or None."""
    d = _data()
    typesets = [set(d["mons"][m.lower()]["types"]) for m in mons if m.lower() in d["mons"]]
    if not typesets:
        return None
    common = set.intersection(*typesets)
    return next(iter(common)).lower() if common else None


def analyze_team(type_name: str, mons: list[str], archetype: str = "balance") -> list[dict]:
    """Critique a monotype team against its archetype: role coverage, gaps, redundancy,
    speed control, and the opposing monotypes it is weakest against."""
    cb = settings.crystal_battle_path
    ms = md.type_moveset(cb, type_name.lower())
    if not ms:
        return []
    keymap = {k.lower(): k for k in ms}
    team = [keymap[m.lower()] for m in mons if m.lower() in keymap]
    if not team:
        return []
    arch = _archetype(archetype)
    Typ = type_name.capitalize()
    lbl = "" if archetype.lower() == "balance" else f" {archetype.lower()}"
    lines = [f"Analysis of a Gen 9 Monotype{lbl} {Typ} team ({', '.join(team)}):"] \
        + _analysis_lines(type_name, team, ms, arch)
    return [_passage("tool#analyze_team", f"{Typ} monotype team analysis", "\n".join(lines))] \
        + _corpus_docs([f"gen9monotype_matchup#{Typ}"])


def _corpus_docs(sources: list[str]) -> list[dict]:
    """Fetch the corpus documents a tool computed FROM, as citable passages:
    a damage calc's provenance is the pokedex and move docs behind its inputs."""
    out = []
    try:
        from db import connect
        with connect() as conn, conn.cursor() as cur:
            for src in sources:
                cur.execute(
                    "SELECT corpus, source, title, content FROM chunks WHERE source = %s "
                    "ORDER BY id LIMIT 1", (src,))
                row = cur.fetchone()
                if row:
                    out.append({"corpus": row[0], "source": row[1], "title": row[2],
                                "content": row[3], "similarity": 0.0, "kw_rank": 0.0,
                                "rerank_score": 0.99})
    except Exception:
        pass
    return out


GROUNDING = "Gravity, Smack Down, an Iron Ball, or a Mold Breaker attacker"

# nature -> (raised stat, lowered stat); neutral natures omitted
NATURE_EFFECTS = {
    "Lonely": ("Attack", "Defense"), "Brave": ("Attack", "Speed"),
    "Adamant": ("Attack", "Special Attack"), "Naughty": ("Attack", "Special Defense"),
    "Bold": ("Defense", "Attack"), "Relaxed": ("Defense", "Speed"),
    "Impish": ("Defense", "Special Attack"), "Lax": ("Defense", "Special Defense"),
    "Timid": ("Speed", "Attack"), "Hasty": ("Speed", "Defense"),
    "Jolly": ("Speed", "Special Attack"), "Naive": ("Speed", "Special Defense"),
    "Modest": ("Special Attack", "Attack"), "Mild": ("Special Attack", "Defense"),
    "Quiet": ("Special Attack", "Speed"), "Rash": ("Special Attack", "Special Defense"),
    "Calm": ("Special Defense", "Attack"), "Gentle": ("Special Defense", "Defense"),
    "Sassy": ("Special Defense", "Speed"), "Careful": ("Special Defense", "Special Attack"),
}
_EV_KEYS = {"hp": "HP", "atk": "Attack", "def": "Defense",
            "spa": "Special Attack", "spd": "Special Defense", "spe": "Speed"}


def _apply_standard(mon: dict, mods: dict) -> dict:
    """Resolve {"standard": True} into the Pokemon's most standard Smogon set
    (nature, EVs, item, ability); explicit question modifiers stay on top."""
    if not mods.get("standard"):
        return mods
    from corpora.smogon import standard_set
    found = standard_set(mon["name"])
    if not found:
        return mods
    set_name, s = found
    base: dict = {"set_name": set_name}
    if s.get("nature") in NATURE_EFFECTS:
        base["nature"] = NATURE_EFFECTS[s["nature"]]
    if s.get("evs"):
        base["evs"] = {_EV_KEYS[k]: v for k, v in s["evs"].items() if k in _EV_KEYS}
    if s.get("item"):
        base["item"] = s["item"]
    if s.get("ability"):
        base["ability"] = s["ability"]
    out = dict(base)
    for k, v in mods.items():
        if k == "standard":
            continue
        if k in ("boosts", "evs") and k in out:
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _ability_immunity(mon: dict, move_type: str) -> dict | None:
    """Cross-reference: does this Pokemon have an ability that negates the incoming
    move type? Returns {ability, sole, observed, certain}: `sole` when it is the
    only possible ability in the dex data, `observed` from the usage-stats corpus,
    and `certain` when either makes the immunity effectively unconditional."""
    blockers = [ab for ab in mon.get("abilities", []) if ABILITY_IMMUNITIES.get(ab) == move_type]
    if not blockers:
        return None
    ability = blockers[0]
    sole = len(set(mon.get("abilities", []))) == 1
    observed = None
    try:
        from db import connect
        import re as _re
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT content FROM chunks WHERE corpus='crystal_battle' AND source LIKE %s LIMIT 1",
                (f"%chaos#{mon['name']}",))
            row = cur.fetchone()
        if row:
            m = _re.search(rf"{_engine_id(ability)} \((\d+)%\)", row[0])
            if m:
                observed = int(m.group(1))
    except Exception:
        pass
    return {"ability": ability, "sole": sole, "observed": observed,
            "certain": sole or (observed or 0) >= 99}


def _immunity_reason(mon: dict, imm: dict) -> str:
    if imm["sole"]:
        return f"{imm['ability']} is {mon['name']}'s only possible ability"
    if imm["observed"] is not None:
        return f"{mon['name']} runs {imm['ability']} on {imm['observed']}% of its observed gen9ou sets"
    return f"{mon['name']} can have {imm['ability']}"


def _ability_immunity_note(mon: dict, move_type: str, as_defender_option: bool = False) -> str | None:
    imm = _ability_immunity(mon, move_type)
    if not imm:
        return None
    reason = _immunity_reason(mon, imm)
    if as_defender_option:
        if imm["certain"]:
            return (f"In practice this is academic: {reason}, so it is outright immune to "
                    f"{move_type}-type moves with no investment at all.")
        return (f"Alternatively, {mon['name']} running {imm['ability']} is simply immune to "
                f"{move_type}-type moves, no investment needed ({reason}).")
    if imm["certain"]:
        return None  # certain immunities are handled as the headline, not a footnote
    return (f"Cross-reference: this calculation assumes no ability, but {reason}, "
            f"which would make it immune to {move_type}-type moves.")


# ---- damage calc (poke-engine) ------------------------------------------------

def _engine_id(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _level100_neutral(mon: dict, evs: dict | None = None, nature: tuple | None = None):
    """Final stats at level 100, 31 IVs, given EVs (default 0), given nature
    (default neutral). nature = (raised_stat, lowered_stat)."""
    s = mon["stats"]
    evs = evs or {}
    hp = 2 * s.get("HP", 1) + 141 + evs.get("HP", 0) // 4
    o = {}
    for k in ("Attack", "Defense", "Special Attack", "Special Defense", "Speed"):
        v = 2 * s.get(k, 1) + 36 + evs.get(k, 0) // 4
        if nature and nature[0] == k:
            v = int(v * 1.1)
        elif nature and nature[1] == k:
            v = int(v * 0.9)
        o[k] = v
    return hp, o


def _calc_rolls(a: dict, move_name: str, b: dict, am: dict, dm: dict, weather: str,
                terrain: str = "none"):
    """Engine damage core: returns (lo, hi, defender_max_hp), or (None, None, None)."""
    try:
        from poke_engine import (State, Side, SideConditions, Pokemon, Move, Weather,
                                 Terrain, calculate_damage)
    except ImportError:
        return None, None, None

    def build(m, mv, mods):
        hp, o = _level100_neutral(m, mods.get("evs"), mods.get("nature"))
        types = (m["types"] + ["typeless"])[:2]
        tera = mods.get("tera")
        return Pokemon(
            id=_engine_id(m["name"]), level=100,
            types=(types[0].lower(), types[1].lower()),
            base_types=(types[0].lower(), types[1].lower()),
            hp=hp, maxhp=hp,
            attack=o["Attack"], defense=o["Defense"],
            special_attack=o["Special Attack"], special_defense=o["Special Defense"],
            speed=o["Speed"], weight_kg=m["weight_kg"],
            item=_engine_id(mods["item"]) if mods.get("item") else "none",
            ability=_engine_id(mods["ability"]) if mods.get("ability") else "none",
            status=mods.get("status", "none"),
            terastallized=bool(tera), tera_type=(tera or "typeless").lower(),
            moves=[Move(id=mv, pp=16)])

    def side(m, mv, mods):
        boosts = mods.get("boosts", {})
        sc = SideConditions(reflect=5) if mods.get("screen") == "reflect" else (
             SideConditions(light_screen=5) if mods.get("screen") == "light_screen"
             else SideConditions())
        return Side(pokemon=[build(m, mv, mods)],
                    side_conditions=sc,
                    attack_boost=boosts.get("Attack", 0),
                    defense_boost=boosts.get("Defense", 0),
                    special_attack_boost=boosts.get("Special Attack", 0),
                    special_defense_boost=boosts.get("Special Defense", 0),
                    speed_boost=boosts.get("Speed", 0))

    state = State(side_one=side(a, _engine_id(move_name), am),
                  side_two=side(b, "tackle", dm),
                  weather=Weather(weather),
                  weather_turns_remaining=-1 if weather != "none" else 0,
                  terrain=Terrain(terrain),
                  terrain_turns_remaining=-1 if terrain != "none" else 0)
    rolls = calculate_damage(state, _engine_id(move_name), "tackle", True)[0]
    lo, hi = (min(rolls), max(rolls)) if rolls else (0, 0)
    def_hp, _ = _level100_neutral(b, dm.get("evs"), dm.get("nature"))
    return lo, hi, def_hp


def damage_calc(attacker: str, move: str, defender: str,
                attacker_mods: dict | None = None, defender_mods: dict | None = None,
                weather: str = "none", terrain: str = "none", spread: bool = False) -> list[dict]:
    """Damage rolls for attacker using move into defender, via poke-engine.
    Baseline: level 100, 31 IVs, 0 EVs, neutral natures, no items or abilities.
    mods: {"item": str, "boosts": {stat: n}, "nature": (up, down),
           "evs": {stat: n}, "status": "brn", "tera": type} per side."""
    am, dm = attacker_mods or {}, defender_mods or {}
    d = _data()
    a, b = d["mons"].get(attacker.lower()), d["mons"].get(defender.lower())
    move_entry = d["moves"].get(move.lower())
    if not a or not b or not move_entry:
        return []
    am, dm = _apply_standard(a, am), _apply_standard(b, dm)
    move_name = move_entry[0]
    lo, hi, def_hp = _calc_rolls(a, move_name, b, am, dm, weather, terrain)
    if lo is None:
        return []
    spread_note = ""
    if spread:
        if move.lower() in d["spread"]:
            lo, hi = int(lo * 0.75), int(hi * 0.75)
            spread_note = (f" The damage figures above already include the 0.75x spread "
                           f"reduction, because {move_name} hits multiple targets in doubles; "
                           f"do not reduce them again.")
        else:
            spread_note = f" Note: {move_name} is single-target, so no spread reduction applies in doubles."

    # rolls computed in _calc_rolls (shared with ohko_search)
    lo_pct, hi_pct = 100 * lo / def_hp, 100 * hi / def_hp
    if lo >= def_hp:
        verdict = "a guaranteed OHKO"
    elif hi >= def_hp:
        verdict = f"a possible OHKO ({hi_pct:.0f}% max roll)"
    elif lo > 0:
        import math
        n_hi, n_lo = math.ceil(def_hp / hi), math.ceil(def_hp / lo)
        verdict = (f"a guaranteed {n_lo}HKO" if n_lo == n_hi
                   else f"a {n_hi}-{n_lo}HKO depending on rolls")
    else:
        verdict = "no damage"
    def describe(name, mods):
        bits = []
        if mods.get("set_name"):
            bits.append(f"standard Smogon set: {mods['set_name']}")
        if mods.get("ability"):
            bits.append(mods["ability"])
        if mods.get("boosts"):
            bits += [f"{'+' if n > 0 else ''}{n} {s}" for s, n in mods["boosts"].items()]
        if mods.get("item"):
            bits.append(mods["item"])
        if mods.get("nature"):
            bits.append(f"+{mods['nature'][0]} nature")
        if mods.get("evs"):
            bits += [f"{n} {s} EVs" for s, n in mods["evs"].items()]
        if mods.get("status") == "brn":
            bits.append("burned")
        if mods.get("tera"):
            bits.append(f"Tera {mods['tera'].capitalize()}")
        return f"{name} ({', '.join(bits)})" if bits else name

    applied = (f" Weather: {weather}." if weather != "none" else "") + \
              (f" Terrain: {terrain}." if terrain != "none" else "") + spread_note
    imm = _ability_immunity(b, move_entry[1])
    if imm and imm["certain"] and lo > 0:
        content = (f"Damage calculation via the poke-engine battle engine: no. "
                   f"{_immunity_reason(b, imm)}, which makes it immune to "
                   f"{move_entry[1]}-type moves, so {describe(a['name'], am)}'s {move_name} "
                   f"normally does nothing. If the immunity is removed ({GROUNDING}), the "
                   f"calculation applies: {lo}-{hi} damage ({lo_pct:.0f}-{hi_pct:.0f}% of its "
                   f"{def_hp} max HP): {verdict}.{applied} Baseline assumptions unless stated: "
                   f"level 100, 31 IVs, 0 EVs, neutral natures, no items.")
    else:
        note = _ability_immunity_note(b, move_entry[1])
        content = (f"Damage calculation via the poke-engine battle engine: "
                   f"{describe(a['name'], am)}'s {move_name} does {lo}-{hi} damage to "
                   f"{describe(b['name'], dm)} ({lo_pct:.0f}-{hi_pct:.0f}% of its {def_hp} max HP): "
                   f"{verdict}.{applied} Baseline assumptions unless stated: level 100, 31 IVs, "
                   f"0 EVs, neutral natures, no items or abilities."
                   + (f" {note}" if note else ""))
    supports = [f"pokedex#{a['name']}", f"pokedex#{b['name']}", f"move#{move_name}"]
    if imm:
        supports.append(f"gen9ou_chaos#{b['name']}")
    return [_passage("tool#damage_calc", f"{a['name']} {move_name} vs {b['name']}", content)] \
        + _corpus_docs(supports)


def ohko_search(attacker: str, move: str, defender: str) -> list[dict]:
    """What would it take for attacker's move to OHKO defender? Applies levers
    cumulatively, largest structural choices first (nature, then EVs, then item,
    then battle state), recomputing true engine rolls at each rung. IVs are
    always assumed perfect (31)."""
    d = _data()
    a, b = d["mons"].get(attacker.lower()), d["mons"].get(defender.lower())
    entry = d["moves"].get(move.lower())
    if not a or not b or not entry:
        return []
    move_name, move_type, move_cls = entry
    if move_cls == "status":
        return [_passage("tool#ohko_search", f"{move_name} OHKO analysis",
                         f"{move_name} is a status move; it deals no damage and cannot OHKO.")]

    phys = move_cls == "physical"
    stat = "Attack" if phys else "Special Attack"
    nature_name = "Adamant" if phys else "Modest"
    item = "Choice Band" if phys else "Choice Specs"
    weather = {"Fire": "sun", "Water": "rain"}.get(move_type)

    am: dict = {}
    baseline = _calc_rolls(a, move_name, b, am, {}, "none")
    if baseline[0] is None:
        return []
    lo, hi, def_hp = baseline
    header = (f"OHKO analysis via the poke-engine battle engine: {a['name']}'s {move_name} "
              f"vs {b['name']} (both level 100, 31 IVs; defender uninvested).")
    if lo >= def_hp:
        return [_passage("tool#ohko_search", f"{a['name']} {move_name} vs {b['name']}",
                         f"{header} Already a guaranteed OHKO with no investment: "
                         f"{lo}-{hi} damage ({100*lo/def_hp:.0f}-{100*hi/def_hp:.0f}% of {def_hp} HP).")]
    if hi == 0:
        return [_passage("tool#ohko_search", f"{a['name']} {move_name} vs {b['name']}",
                         f"{header} {move_name} does no damage: {b['name']} is immune. No amount "
                         f"of offensive investment fixes an immunity; it must be removed by game "
                         f"state (for a Ground immunity: Gravity, Smack Down, or an Iron Ball).")]

    lines = [header,
             f"Baseline (neutral nature, no EVs, no item): {lo}-{hi} ({100*lo/def_hp:.0f}-{100*hi/def_hp:.0f}%): not an OHKO.",
             "Escalation, largest levers first:"]
    steps = [(f"{nature_name} nature", lambda: am.update(nature=(stat, "Speed"))),
             (f"252 {stat} EVs", lambda: am.update(evs={stat: 252})),
             (item, lambda: am.update(item=item))]
    if weather:
        steps.append((f"in {weather}", None))
    steps.append((f"Tera {move_type}", lambda: am.update(tera=move_type.lower())))
    for n in (1, 2, 3, 4, 6):
        steps.append((f"+{n} {stat}", lambda n=n: am.update(boosts={stat: n})))

    wx = "none"
    guaranteed_at = None
    possible_at = ["no investment (high roll)"] if hi >= def_hp else None
    applied = []
    for label, apply in steps:
        if apply:
            apply()
        else:
            wx = weather
        applied.append(label)
        lo, hi, def_hp = _calc_rolls(a, move_name, b, am, {}, wx)
        pct = f"{100*lo/def_hp:.0f}-{100*hi/def_hp:.0f}%"
        if possible_at is None and hi >= def_hp:
            possible_at = list(applied)
        if lo >= def_hp:
            guaranteed_at = list(applied)
            lines.append(f"- + {label}: {lo}-{hi} ({pct}): guaranteed OHKO.")
            break
        lines.append(f"- + {label}: {lo}-{hi} ({pct})")
    if guaranteed_at:
        lines.append(f"Verdict: guaranteed OHKO requires {', '.join(guaranteed_at)} (applied cumulatively).")
        if possible_at and possible_at != guaranteed_at:
            lines.append(f"A high roll can OHKO from {', '.join(possible_at)}.")
    elif possible_at:
        lines.append(f"Verdict: only a high-roll OHKO is possible, from {', '.join(possible_at)}; "
                     f"never guaranteed with these levers.")
    else:
        lines.append("Verdict: not an OHKO even with maximum investment, item, weather, Tera, "
                     "and +6; this move cannot OHKO this target one-on-one.")
    imm = _ability_immunity(b, move_type)
    if imm and imm["certain"]:
        lines.append(f"Important: {_immunity_reason(b, imm)}, so none of this lands unless the "
                     f"immunity is removed ({GROUNDING}); with it removed, the escalation above applies.")
    else:
        note = _ability_immunity_note(b, move_type)
        if note:
            lines.append(note + " In that case the whole escalation is moot.")
    supports = [f"pokedex#{a['name']}", f"pokedex#{b['name']}", f"move#{move_name}"]
    if imm:
        supports.append(f"gen9ou_chaos#{b['name']}")
    return [_passage("tool#ohko_search", f"{a['name']} {move_name} vs {b['name']}", "\n".join(lines))] \
        + _corpus_docs(supports)


def survive_search(attacker: str, move: str, defender: str,
                   attacker_mods: dict | None = None) -> list[dict]:
    """What would defender need to survive attacker's move? The mirror of
    ohko_search: defensive levers applied cumulatively, largest first (nature,
    then HP EVs, then Def/SpD EVs, then item, then screen/weather/Tera),
    recomputing engine rolls each rung. IVs always assumed perfect."""
    d = _data()
    a, b = d["mons"].get(attacker.lower()), d["mons"].get(defender.lower())
    entry = d["moves"].get(move.lower())
    if not a or not b or not entry:
        return []
    move_name, move_type, move_cls = entry
    am = attacker_mods or {}
    if move_cls == "status":
        return [_passage("tool#survive_search", f"{b['name']} vs {move_name}",
                         f"{move_name} is a status move; it deals no damage, so {b['name']} "
                         f"always survives it directly.")]
    phys = move_cls == "physical"
    def_stat = "Defense" if phys else "Special Defense"
    nature_label = "Bold (+Defense)" if phys else "Calm (+Sp. Def)"
    nature = ("Defense", "Attack") if phys else ("Special Defense", "Attack")
    screen = ("reflect", "Reflect") if phys else ("light_screen", "Light Screen")

    atk_desc = []
    if am.get("nature"): atk_desc.append(f"+{am['nature'][0]} nature")
    if am.get("evs"): atk_desc += [f"{n} {s} EVs" for s, n in am["evs"].items()]
    if am.get("item"): atk_desc.append(am["item"])
    if am.get("boosts"): atk_desc += [f"+{n} {s}" for s, n in am["boosts"].items()]
    atk_str = f"{a['name']} ({', '.join(atk_desc)})" if atk_desc else f"{a['name']} (uninvested)"

    dm: dict = {}
    lo, hi, def_hp = _calc_rolls(a, move_name, b, am, dm, "none")
    if lo is None:
        return []
    header = (f"Survival analysis via the poke-engine battle engine: can {b['name']} survive "
              f"{atk_str}'s {move_name}? Both level 100, 31 IVs.")
    if hi < def_hp:
        return [_passage("tool#survive_search", f"{b['name']} vs {move_name}",
                         f"{header} Yes, even uninvested: {lo}-{hi} damage into {def_hp} HP "
                         f"({100*lo/def_hp:.0f}-{100*hi/def_hp:.0f}%), a guaranteed survive.")]

    lines = [header,
             f"Uninvested: {lo}-{hi} into {def_hp} HP ({100*lo/def_hp:.0f}-{100*hi/def_hp:.0f}%): "
             + ("KO on a high roll." if lo < def_hp else "a guaranteed KO."),
             "Defensive escalation, largest levers first:"]
    steps = [(nature_label, lambda: dm.update(nature=nature)),
             ("252 HP EVs", lambda: dm.setdefault("evs", {}).update({"HP": 252})),
             (f"252 {def_stat} EVs", lambda: dm.setdefault("evs", {}).update({def_stat: 252}))]
    if not phys:
        steps.append(("Assault Vest", lambda: dm.update(item="Assault Vest")))
    steps.append((screen[1], lambda: dm.update(screen=screen[0])))
    wx_step = None
    if not phys and "Rock" in b["types"]:
        wx_step = "sand"
    elif phys and "Ice" in b["types"]:
        wx_step = "snow"
    if wx_step:
        steps.append((f"in {wx_step}", None))
    best_tera = min(d["types"], key=lambda tt: (d["chart"].get((move_type, tt), 1.0), tt))
    if d["chart"].get((move_type, best_tera), 1.0) < 1.0:
        steps.append((f"Tera {best_tera}", lambda: dm.update(tera=best_tera.lower())))

    wx = "none"
    survives_at = maybe_at = None
    applied = []
    for label, apply in steps:
        if apply:
            apply()
        else:
            wx = wx_step
        applied.append(label)
        lo, hi, def_hp = _calc_rolls(a, move_name, b, am, dm, wx)
        pct = f"{100*lo/def_hp:.0f}-{100*hi/def_hp:.0f}%"
        if maybe_at is None and lo < def_hp:
            maybe_at = list(applied)
        if hi < def_hp:
            survives_at = list(applied)
            lines.append(f"- + {label}: {lo}-{hi} ({pct}): guaranteed survive.")
            break
        lines.append(f"- + {label}: {lo}-{hi} ({pct})")
    note = _ability_immunity_note(b, move_type, as_defender_option=True)
    if survives_at:
        lines.append(f"Verdict: guaranteed survival requires {', '.join(survives_at)} "
                     f"(applied cumulatively).")
        if maybe_at and maybe_at != survives_at:
            lines.append(f"Low rolls are survivable from {', '.join(maybe_at)}.")
        if note:
            lines.append(note)
    elif maybe_at:
        lines.append(f"Verdict: survival is roll-dependent at best, from {', '.join(maybe_at)}; "
                     f"never guaranteed with these levers.")
    else:
        lines.append("Verdict: not survivable one-on-one even with full defensive investment, "
                     "a screen, and the best defensive Tera.")
        if note:
            lines.append(note)
    supports = [f"pokedex#{a['name']}", f"pokedex#{b['name']}", f"move#{move_name}"]
    if note:
        supports.append(f"gen9ou_chaos#{b['name']}")
    return [_passage("tool#survive_search", f"{b['name']} vs {move_name}", "\n".join(lines))] \
        + _corpus_docs(supports)


def entity_docs(pokemon: list[str], moves: list[str]) -> list[dict]:
    """Canonical documents for named entities, across corpora (pokedex/move
    entries plus their Bulbapedia articles), for answer-driven source expansion."""
    d = _data()
    sources = []
    for m in pokemon:
        name = d["mons"].get(m.lower(), {}).get("name", m)
        sources += [f"pokedex#{name}", f"bulbapedia#{name} (Pokémon)"]
    for mv in moves:
        entry = d["moves"].get(mv.lower())
        name = entry[0] if entry else mv
        sources += [f"move#{name}", f"bulbapedia#{name} (move)"]
    return _corpus_docs(sources)
