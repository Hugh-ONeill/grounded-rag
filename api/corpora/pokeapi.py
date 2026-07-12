"""PokeAPI adapter — general Pokemon knowledge from the PokeAPI CSV dataset.

Point POKEAPI_PATH at a checkout of github.com/PokeAPI/pokeapi; only data/v2/csv
is read, so a sparse clone is enough (see .env.example). Emits small citable
documents: one per Pokemon (types, stats, abilities, Pokedex entries), one per
move, one per ability, and one learnset per Pokemon (latest version group).
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
        lines = [f"{name} learnset ({vg_ident.get(str(vg), '?')}): moves {name} can learn."]
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
