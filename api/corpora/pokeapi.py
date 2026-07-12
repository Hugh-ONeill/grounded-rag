"""PokeAPI adapter — general Pokemon knowledge from the PokeAPI CSV dataset.

Point POKEAPI_PATH at a checkout of github.com/PokeAPI/pokeapi; only data/v2/csv
is read, so a sparse clone is enough (see .env.example). Emits small citable
documents: one per Pokemon (types, stats, abilities, Pokedex entries), one per
move, one per ability, one per item, and one learnset per Pokemon (latest
version group).
"""
import csv
import re
from collections import defaultdict
from pathlib import Path
from config import settings

EN = "9"  # local_language_id / language_id for English


def _rows(csvdir: Path, name: str):
    with open(csvdir / f"{name}.csv", newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def _clean_prose(text: str) -> str:
    """PokeAPI prose uses []{} wiki links: '[regular damage]{mechanic:regular-damage}'."""
    text = re.sub(r"\[([^\]]*)\]\{[^}:]*:([^}]*)\}",
                  lambda m: m.group(1) or m.group(2).replace("-", " "), text)
    return re.sub(r"\s+", " ", text).strip()


def _flavor(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _pretty(identifier: str) -> str:
    return "-".join(p.capitalize() for p in identifier.split("-"))


def load():
    root = Path(settings.pokeapi_path)
    csvdir = root / "data" / "v2" / "csv"
    if not csvdir.exists():
        raise FileNotFoundError(f"POKEAPI_PATH has no data/v2/csv: {root}")

    # ---- shared lookups (small tables) ----
    type_names = {r["type_id"]: r["name"] for r in _rows(csvdir, "type_names")
                  if r["local_language_id"] == EN}
    stat_names = {r["id"]: _pretty(r["identifier"]).replace("Special-", "Special ").replace("Hp", "HP")
                  for r in _rows(csvdir, "stats")}
    dmg_class = {r["id"]: r["identifier"] for r in _rows(csvdir, "move_damage_classes")}
    move_names = {r["move_id"]: r["name"] for r in _rows(csvdir, "move_names")
                  if r["local_language_id"] == EN}
    ability_names = {r["ability_id"]: r["name"] for r in _rows(csvdir, "ability_names")
                     if r["local_language_id"] == EN}
    version_names = {r["version_id"]: r["name"] for r in _rows(csvdir, "version_names")
                     if r["local_language_id"] == EN}
    vg_ident = {r["id"]: r["identifier"] for r in _rows(csvdir, "version_groups")}
    # version-group ids are NOT chronological (late-added groups like blue-japan get high
    # ids); the `order` column is the actual release order
    vg_order = {int(r["id"]): int(r["order"] or 0) for r in _rows(csvdir, "version_groups")}
    move_methods = {r["id"]: r["identifier"] for r in _rows(csvdir, "pokemon_move_methods")}

    species_name, species_genus = {}, {}
    for r in _rows(csvdir, "pokemon_species_names"):
        if r["local_language_id"] == EN:
            species_name[r["pokemon_species_id"]] = r["name"]
            species_genus[r["pokemon_species_id"]] = r["genus"]

    # ---- per-pokemon attribute tables ----
    p_stats = defaultdict(dict)
    for r in _rows(csvdir, "pokemon_stats"):
        p_stats[r["pokemon_id"]][r["stat_id"]] = r["base_stat"]

    p_types = defaultdict(list)
    for r in _rows(csvdir, "pokemon_types"):
        p_types[r["pokemon_id"]].append((int(r["slot"]), type_names.get(r["type_id"], "?")))

    p_abilities = defaultdict(list)
    for r in _rows(csvdir, "pokemon_abilities"):
        name = ability_names.get(r["ability_id"])
        if name:
            p_abilities[r["pokemon_id"]].append((int(r["slot"]), name, r["is_hidden"] == "1"))

    flavor_texts = defaultdict(list)  # species_id -> [(version, text)] unique
    seen_flavor = defaultdict(set)
    for r in _rows(csvdir, "pokemon_species_flavor_text"):
        if r["language_id"] != EN:
            continue
        text = _flavor(r["flavor_text"])
        key = text.lower()
        if key in seen_flavor[r["species_id"]]:
            continue
        seen_flavor[r["species_id"]].add(key)
        flavor_texts[r["species_id"]].append((version_names.get(r["version_id"], "?"), text))

    # ---- Pokemon documents (default and alternate forms) ----
    pokemon_display = {}  # pokemon_id -> display name (reused by learnsets)
    for r in _rows(csvdir, "pokemon"):
        pid, sid = r["id"], r["species_id"]
        is_default = r["is_default"] == "1"
        name = species_name.get(sid, _pretty(r["identifier"])) if is_default else _pretty(r["identifier"])
        pokemon_display[pid] = name

        types = " / ".join(t for _, t in sorted(p_types.get(pid, [])))
        genus = species_genus.get(sid, "")
        lines = [f"{name} ({types}-type Pokemon)." + (f" {genus}." if genus else "") + f" National dex number {sid}."]

        stats = p_stats.get(pid, {})
        if stats:
            parts = [f"{stat_names[s]} {v}" for s, v in sorted(stats.items(), key=lambda kv: int(kv[0]))]
            total = sum(int(v) for v in stats.values())
            lines.append(f"Base stats: {', '.join(parts)}. Base stat total {total}.")

        abils = sorted(p_abilities.get(pid, []))
        normal = [n for _, n, hidden in abils if not hidden]
        hidden = [n for _, n, hidden in abils if hidden]
        if normal:
            lines.append(f"Abilities: {', '.join(normal)}."
                         + (f" Hidden ability: {', '.join(hidden)}." if hidden else ""))

        if r["height"] and r["weight"]:
            lines.append(f"Height {int(r['height'])/10:g} m, weight {int(r['weight'])/10:g} kg.")

        if is_default and flavor_texts.get(sid):
            lines.append("Pokedex entries:")
            lines += [f"- {ver}: {text}" for ver, text in flavor_texts[sid][:12]]

        yield {
            "source": f"pokedex#{name}",
            "title": f"{name} (Pokedex)",
            "content": "\n".join(lines),
            "metadata": {"kind": "species", "dex": int(sid)},
        }

    # ---- Move documents ----
    effect_prose = {r["move_effect_id"]: r["short_effect"] for r in _rows(csvdir, "move_effect_prose")
                    if r["local_language_id"] == EN}
    move_flavor = {}
    for r in _rows(csvdir, "move_flavor_text"):  # keep the latest version group's text
        if r["language_id"] == EN and r["flavor_text"].strip():
            move_flavor[r["move_id"]] = _flavor(r["flavor_text"])

    for r in _rows(csvdir, "moves"):
        name = move_names.get(r["id"])
        typ = type_names.get(r["type_id"])
        if not name or not typ:
            continue  # shadow/unnamed moves
        cls = dmg_class.get(r["damage_class_id"], "?")
        specs = [f"Power {r['power'] or '-'}", f"accuracy {r['accuracy'] or '-'}", f"PP {r['pp'] or '-'}"]
        if r["priority"] not in ("", "0"):
            specs.append(f"priority {r['priority']}")
        lines = [f"{name} (move). {typ}-type, {cls}. {', '.join(specs)}."]
        effect = effect_prose.get(r["effect_id"], "")
        if effect:
            effect = _clean_prose(effect).replace("$effect_chance", r["effect_chance"] or "?")
            lines.append(f"Effect: {effect}")
        if r["id"] in move_flavor:
            lines.append(f"Description: {move_flavor[r['id']]}")
        yield {
            "source": f"move#{name}",
            "title": f"{name} (move)",
            "content": "\n".join(lines),
            "metadata": {"kind": "move", "type": typ, "class": cls},
        }

    # ---- Ability documents ----
    ability_prose = {r["ability_id"]: (r["short_effect"], r["effect"])
                     for r in _rows(csvdir, "ability_prose") if r["local_language_id"] == EN}
    for r in _rows(csvdir, "abilities"):
        if r["is_main_series"] != "1":
            continue
        name = ability_names.get(r["id"])
        if not name:
            continue
        short, long = ability_prose.get(r["id"], ("", ""))
        lines = [f"{name} (ability). {_clean_prose(short)}"]
        long = _clean_prose(long)
        if long and long.lower() != _clean_prose(short).lower():
            lines.append(f"Details: {long[:600]}")
        yield {
            "source": f"ability#{name}",
            "title": f"{name} (ability)",
            "content": "\n".join(lines),
            "metadata": {"kind": "ability"},
        }

    # ---- Item documents ----
    item_names = {r["item_id"]: r["name"] for r in _rows(csvdir, "item_names")
                  if r["local_language_id"] == EN}
    item_prose = {r["item_id"]: (r["short_effect"], r["effect"])
                  for r in _rows(csvdir, "item_prose") if r["local_language_id"] == EN}
    item_flavor = {}
    for r in _rows(csvdir, "item_flavor_text"):  # keep the latest version group's text
        if r["language_id"] == EN and r["flavor_text"].strip():
            item_flavor[r["item_id"]] = _flavor(r["flavor_text"])
    item_cats = {r["id"]: r["identifier"] for r in _rows(csvdir, "item_categories")}
    # bulk noise: hundreds of TMs, Dynamax crystals, sandwich ingredients, etc.
    SKIP_CATS = {"all-machines", "tm-materials", "dynamax-crystals", "unused",
                 "plot-advancement", "picnic", "sandwich-ingredients", "species-candies",
                 "curry-ingredients", "baking-only", "event-items", "data-cards",
                 "all-mail", "mulch", "dex-completion", "loot", "collectibles"}
    for r in _rows(csvdir, "items"):
        name = item_names.get(r["id"])
        short, long = item_prose.get(r["id"], ("", ""))
        flavor = item_flavor.get(r["id"], "")
        # newer items (Heavy-Duty Boots, Booster Energy, ...) have no effect prose
        # in the dataset; their flavor text describes the effect, so fall back to it
        if not name or not (short or flavor) or item_cats.get(r["category_id"], "") in SKIP_CATS:
            continue
        lines = [f"{name} (item). {_clean_prose(short) if short else flavor}"]
        long = _clean_prose(long)
        if long and long.lower() != _clean_prose(short).lower():
            lines.append(f"Details: {long[:600]}")
        if short and flavor:
            lines.append(f"Description: {flavor}")
        yield {
            "source": f"item#{name}",
            "title": f"{name} (item)",
            "content": "\n".join(lines),
            "metadata": {"kind": "item", "category": item_cats.get(r["category_id"], "")},
        }

    # ---- Stat rankings (one per base stat + total) ----
    # Corpus-wide superlatives ("which is the fastest Pokemon?") are unanswerable
    # by top-k retrieval: no species chunk contains the comparison. Same fix as
    # the usage rankings: make each ranking a citable document. Alternate forms
    # (megas, Deoxys-Speed) are included, since they hold many of the records.
    STAT_PHRASE = {
        "Speed": "the fastest Pokemon (highest base Speed)",
        "Attack": "the strongest physical attackers (highest base Attack)",
        "Defense": "the most physically defensive Pokemon (highest base Defense)",
        "Special Attack": "the strongest special attackers (highest base Special Attack)",
        "Special Defense": "the most specially defensive Pokemon (highest base Special Defense)",
        "HP": "the Pokemon with the most HP (highest base HP)",
    }
    stat_values = {}  # display stat name -> [(value, pokemon name)]
    totals = []
    for pid, stats in p_stats.items():
        name = pokemon_display.get(pid)
        if not name or not stats:
            continue
        for sid, v in stats.items():
            stat_values.setdefault(stat_names[sid], []).append((int(v), name))
        totals.append((sum(int(v) for v in stats.values()), name))
    for stat, phrase in STAT_PHRASE.items():
        ranked = sorted(stat_values.get(stat, []), reverse=True)[:25]
        if not ranked:
            continue
        top_v, top_n = ranked[0]
        lines = [f"{stat} rankings: {phrase}, across all Pokemon including alternate forms.",
                 f"{top_n} has the highest base {stat} of any Pokemon, at {top_v}."]
        lines += [f"{i}. {n} ({v})" for i, (v, n) in enumerate(ranked, 1)]
        yield {
            "source": f"stat_rankings#{stat}",
            "title": f"{stat} rankings",
            "content": "\n".join(lines),
            "metadata": {"kind": "stat_rankings", "stat": stat},
        }
    ranked = sorted(totals, reverse=True)[:25]
    lines = ["Base stat total rankings: the Pokemon with the strongest overall base stats.",
             f"{ranked[0][1]} has the highest base stat total of any Pokemon, at {ranked[0][0]}."]
    lines += [f"{i}. {n} ({v})" for i, (v, n) in enumerate(ranked, 1)]
    yield {
        "source": "stat_rankings#Base stat total",
        "title": "Base stat total rankings",
        "content": "\n".join(lines),
        "metadata": {"kind": "stat_rankings", "stat": "total"},
    }

    # PokeAPI's item text coverage ends before gen 9: Scarlet/Violet items exist in
    # item_names.csv but have no English prose or flavor text at all. These are the
    # most competitively relevant items, so their effects are transcribed by hand.
    GEN9_ITEM_SUPPLEMENT = {
        "Booster Energy": "Held: when the holder is a Paradox Pokemon with Protosynthesis or Quark Drive, this item is consumed to activate that ability, boosting the holder's highest stat.",
        "Covert Cloak": "Held: protects the holder from the additional effects of damaging moves, such as flinching or stat drops.",
        "Loaded Dice": "Held: the holder's multi-strike moves (like Bullet Seed or Icicle Spear) hit at least four times.",
        "Clear Amulet": "Held: prevents the holder's stats from being lowered by opposing moves or abilities like Intimidate.",
        "Mirror Herb": "Held: copies an opponent's stat increases once, boosting the holder the same way, then the item is consumed.",
        "Ability Shield": "Held: protects the holder's ability from being changed, suppressed, or replaced.",
        "Punching Glove": "Held: boosts the power of the holder's punching moves by 10% and removes their contact with the target.",
        "Fairy Feather": "Held: boosts the power of the holder's Fairy-type moves by 10%.",
        "Wellspring Mask": "Held: boosts the power of Ogerpon Wellspring Form's moves by 20%.",
        "Hearthflame Mask": "Held: boosts the power of Ogerpon Hearthflame Form's moves by 20%.",
        "Cornerstone Mask": "Held: boosts the power of Ogerpon Cornerstone Form's moves by 20%.",
    }
    for name, effect in GEN9_ITEM_SUPPLEMENT.items():
        yield {
            "source": f"item#{name}",
            "title": f"{name} (item)",
            "content": f"{name} (item). {effect}",
            "metadata": {"kind": "item", "category": "held-items", "text_source": "hand-transcribed"},
        }

    # ---- Learnset documents (latest version group per Pokemon) ----
    learn = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # pid -> vg -> method -> [(level, move)]
    for r in _rows(csvdir, "pokemon_moves"):
        mname = move_names.get(r["move_id"])
        if mname:
            learn[r["pokemon_id"]][int(r["version_group_id"])][r["pokemon_move_method_id"]].append(
                (int(r["level"] or 0), mname))

    METHOD_LABEL = {"1": "Level-up", "2": "Egg moves", "3": "Tutor", "4": "TM/machine"}
    for pid, by_vg in learn.items():
        name = pokemon_display.get(pid)
        if not name:
            continue
        # newest version group (by release order) that has level-up data; sparse
        # transfer-only groups would otherwise yield empty learnsets
        with_levelup = [vg for vg, methods in by_vg.items() if methods.get("1")]
        pool = with_levelup or list(by_vg)
        vg = max(pool, key=lambda v: vg_order.get(v, 0))
        lines = [f"{name} learnset ({vg_ident.get(str(vg), '?')}): the full movepool of "
                 f"moves {name} can learn (not what it commonly uses in competitive play)."]
        for method_id, label in METHOD_LABEL.items():
            entries = by_vg[vg].get(method_id)
            if not entries:
                continue
            if method_id == "1":
                entries.sort()
                lines.append(f"{label}: " + ", ".join(f"{m} (level {lv})" for lv, m in entries) + ".")
            else:
                lines.append(f"{label}: " + ", ".join(sorted(m for _, m in entries)) + ".")
        yield {
            "source": f"learnset#{name}",
            "title": f"{name} learnset",
            "content": "\n".join(lines),
            "metadata": {"kind": "learnset", "version_group": vg_ident.get(str(vg), "?")},
        }
