"""Deterministic tools: computed answers rendered as citable pseudo-passages.

Retrieval answers "what does the corpus say"; these answer "what does the math
say". Each tool returns passages shaped like retrieval results (source, title,
content) with rerank_score 1.0, so the gate, the citation prompt, and the eval
all work unchanged. Data comes straight from the PokeAPI CSVs (same checkout
the corpus adapter reads).
"""
import csv
from functools import lru_cache
from pathlib import Path
from config import settings

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
    for r in rows("moves"):
        name, typ = move_names.get(r["id"]), type_names.get(r["type_id"])
        if name and typ:
            moves[name.lower()] = (name, typ, dmg_class2.get(r["damage_class_id"], "?"))

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
    return {"chart": chart, "mons": mons, "moves": moves,
            "types": sorted(set(type_names.values()))}


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
    return [_passage(f"tool#type_matchup", f"{att} vs {target}", "\n".join(lines))]


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
    return [_passage(f"tool#type_matchup", f"defensive profile: {target}", "\n".join(lines))]


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
    return [_passage("tool#speed_check", f"{a['name']} vs {b['name']} speed", content)]


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


def _ability_immunity_note(mon: dict, move_type: str, as_defender_option: bool = False) -> str | None:
    """Cross-reference: does this Pokemon have a possible ability that negates the
    incoming move type? If the usage-stats corpus has observed ability data for it,
    cite the real split; otherwise report the possibility. The engine calc itself
    assumes no ability, so this is the correction the raw numbers need."""
    blockers = [ab for ab in mon.get("abilities", []) if ABILITY_IMMUNITIES.get(ab) == move_type]
    if not blockers:
        return None
    ability = blockers[0]
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
    if as_defender_option:
        base = (f"Alternatively, {mon['name']} running the ability {ability} is simply immune "
                f"to {move_type}-type moves, no investment needed")
    else:
        base = (f"Cross-reference: this calculation assumes no ability, but {mon['name']} can "
                f"have {ability}, which makes it immune to {move_type}-type moves")
    if observed is not None:
        base += f" (observed on {observed}% of its sets in gen9ou usage data)"
    return base + "."


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


def _calc_rolls(a: dict, move_name: str, b: dict, am: dict, dm: dict, weather: str):
    """Engine damage core: returns (lo, hi, defender_max_hp), or (None, None, None)."""
    try:
        from poke_engine import (State, Side, SideConditions, Pokemon, Move, Weather,
                                 calculate_damage)
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
                  weather_turns_remaining=-1 if weather != "none" else 0)
    rolls = calculate_damage(state, _engine_id(move_name), "tackle", True)[0]
    lo, hi = (min(rolls), max(rolls)) if rolls else (0, 0)
    def_hp, _ = _level100_neutral(b, dm.get("evs"), dm.get("nature"))
    return lo, hi, def_hp


def damage_calc(attacker: str, move: str, defender: str,
                attacker_mods: dict | None = None, defender_mods: dict | None = None,
                weather: str = "none") -> list[dict]:
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
    move_name = move_entry[0]
    lo, hi, def_hp = _calc_rolls(a, move_name, b, am, dm, weather)
    if lo is None:
        return []

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

    applied = f" Weather: {weather}." if weather != "none" else ""
    note = _ability_immunity_note(b, move_entry[1])
    content = (f"Damage calculation via the poke-engine battle engine: "
               f"{describe(a['name'], am)}'s {move_name} does {lo}-{hi} damage to "
               f"{describe(b['name'], dm)} ({lo_pct:.0f}-{hi_pct:.0f}% of its {def_hp} max HP): "
               f"{verdict}.{applied} Baseline assumptions unless stated: level 100, 31 IVs, "
               f"0 EVs, neutral natures, no items or abilities."
               + (f" {note}" if note else ""))
    return [_passage("tool#damage_calc", f"{a['name']} {move_name} vs {b['name']}", content)]


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
    note = _ability_immunity_note(b, move_type)
    if note:
        lines.append(note + " In that case the whole escalation is moot.")
    return [_passage("tool#ohko_search", f"{a['name']} {move_name} vs {b['name']}", "\n".join(lines))]


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
    return [_passage("tool#survive_search", f"{b['name']} vs {move_name}", "\n".join(lines))]
