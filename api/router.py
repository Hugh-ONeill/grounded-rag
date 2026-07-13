"""Question router: computed questions go to tools, everything else to RAG.

Deliberately rule-based rather than LLM function-calling: the rules only fire
when they positively identify the entities involved (real Pokemon names, real
type names), so a miss falls through to retrieval instead of a wrong tool call.
The LLM stays the narrator, never the dispatcher.
"""
import re
import tools
from retrieve import retrieve_hybrid, passes_threshold

STATS = {
    "speed": "Speed", "fastest": "Speed", "slowest": "Speed", "quickest": "Speed",
    "hp": "HP", "attack": "Attack", "defense": "Defense", "defence": "Defense",
    "special attack": "Special Attack", "special defense": "Special Defense",
    "bulkiest": "HP",
}

# The verb vocabularies below were each widened when the paraphrase gold set showed
# real-user rewordings dodging them ("one-shot", "knocked out in one hit", "quickest",
# "susceptible to", "compare the speed"). Safe to keep broad: every branch still
# requires positively-identified entities before it fires.
_MATCHUP = re.compile(
    r"super effective|not very effective|effective against|how effective|effectiveness"
    r"|weak (?:to|against)|weakness(?:es)?|resists?|susceptible|immune|affects?"
    r"|fare against|much damage|damage (?:to|against)|\bhit(?:s|ting)?\b", re.I)
_SPEED_VS = re.compile(
    r"\bfaster\b|\boutspeeds?\b|\boutpaces?\b|\boutruns?\b|\bslower\b|\bquicker\b"
    r"|higher speed|more speed|speed advantage|compare .{0,24}speed", re.I)
_SUPERLATIVE = re.compile(
    r"\b(fastest|slowest|highest|lowest|best|most|bulkiest|quickest|maximum|largest|greatest)\b", re.I)
# monotype intent: the word "monotype" or a "mono-<type>" phrasing. Superlative
# stat questions in this format must rank the monotype meta, not the whole dex.
_MONOTYPE = re.compile(
    r"\bmono[- ]?type\b|\bmono[- ]?(?:bug|dark|dragon|electric|fairy|fighting|fire|"
    r"flying|ghost|grass|ground|ice|normal|poison|psychic|rock|steel|water)\b", re.I)
# "a Steel team" / "fairy-type team" / "steel teammates" — an all-one-type team (and
# its same-type partners) only exists in monotype, so this signals monotype even
# without the word "monotype".
_TYPE_TEAM = re.compile(
    r"\b(?:bug|dark|dragon|electric|fairy|fighting|fire|flying|ghost|grass|ground|ice|"
    r"normal|poison|psychic|rock|steel|water)(?:[- ]?type)?\s+(?:teams?|teammates?|partners?)\b", re.I)
# teammate-recommendation intent (fires only with a type present, i.e. monotype)
_TEAMMATE = re.compile(
    r"\bteammates?\b|\bpartners?\b|pairs? (?:with|well|nicely|up)|pair well|goes? well"
    r"|round out|fill out|complement|what (?:else )?(?:to|should i|can i) (?:add|run|pair|bring)", re.I)
# build-a-team intent: "build me a Steel team", "make a monotype Fairy team"
_BUILD_TEAM = re.compile(
    r"\b(?:build|make|generate|create|come up with|put together|suggest|give me)\b.{0,40}\bteams?\b", re.I)
# critique-an-existing-team intent (needs the team's mons named)
_ANALYZE = re.compile(
    r"\b(?:analy[sz]e|analysis|critique|review|rate|assess|evaluate|feedback on|"
    r"how good is|wrong with|weakness(?:es)? (?:of|in)|problems? with)\b", re.I)


def _archetype_of(question: str) -> str:
    """Detect the requested team archetype, defaulting to balance."""
    q = question.lower()
    if re.search(r"hyper[- ]?offen[sc]e|\bho\b", q):
        return "hyper offense"
    if re.search(r"bulky[- ]?offen[sc]e", q):
        return "bulky offense"
    if "stall" in q:
        return "stall"
    if re.search(r"\bfat\b", q):
        return "fat"
    if re.search(r"\boffensive\b|\boffen[sc]e\b|\baggressive\b", q):
        return "hyper offense"
    return "balance"


def _variant_of(question: str) -> int:
    """Which team variant to return: 0 (base) unless the ask is for another/second/third."""
    q = question.lower()
    if re.search(r"\b(third|3rd)\b", q):
        return 2
    if re.search(r"\b(second|2nd)\b", q):
        return 1
    if re.search(r"\banother\b|\bdifferent\b|\balternat(?:e|ive)\b|\bother\b|\bvariant\b|\bone more\b", q):
        return 1
    return 0
_DAMAGE = re.compile(
    r"\bohko\w*|\b\dhko\b|one[- ]shot|how much damage|damage (?:does|output|calculation|against|to|on)\b"
    r"|\bcalculate\b|\boutput\b|knock(?:s|ed|ing)?(?: \w+)? out|taken out|\bsingle hit\b|\bone hit\b"
    r"|\bkill\b|\bsurviv", re.I)
_SURVIVE_HOW = re.compile(
    r"(?:what would|what does|what do|how (?:can|could|do)|is there (?:any|a) way|best way"
    r"|needs?|needed|required?|take[s]? (?:for|to)|make).{0,60}"
    r"\b(?:survive|tank|withstand|avoid being (?:knocked out|ohko\w*)|take (?:a|an|one)\b)", re.I)
_OHKO_HOW = re.compile(
    r"(?:what would|what does|what do|how (?:can|could|do)|how much|is there (?:any|a) way"
    r"|needs?|needed|must|required?|take[s]? (?:for|to)|make).{0,60}\b(?:ohko\w*|one[- ]shot)"
    # requirement verb can trail the KO word: "in order to OHKO X, how much must ..."
    r"|\b(?:ohko\w*|one[- ]shot)\b.{0,60}\b(?:must|needs?|required?)\b", re.I)

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


def _parse_mods(question: str, mons: list[str]):
    """Extract battle-state modifiers with proximity binding: a modifier phrase
    binds to the nearest Pokemon named shortly after it ("bold max hp skarmory"
    mods Skarmory), falling back to offensive-kit-to-attacker / defensive-kit-to-
    defender when no name follows. Returns (stripped_question, mods_by_mon,
    fallback_attacker_mods, fallback_defender_mods, field)."""
    q = question.lower()
    mon_pos = {}
    for mon in mons:
        m = re.search(rf"\b{re.escape(mon)}(?:['\u2019]s|s)?\b", q)
        if m:
            mon_pos[mon] = m.start()

    by_mon: dict[str, dict] = {m: {} for m in mons}
    fb_atk: dict = {}
    fb_dfn: dict = {}
    field = {"weather": "none", "terrain": "none", "spread": False}

    def bind(pos, fallback):
        after = [(p - pos, m) for m, p in mon_pos.items() if 0 < p - pos <= 40]
        if after:
            return by_mon[min(after)[1]]
        return fallback

    def strip(pattern):
        nonlocal q
        q = re.sub(pattern, " ", q)

    def merge_boost(target, boosts):
        target.setdefault("boosts", {}).update(boosts)

    for phrase, boosts in _SETUP.items():
        if re.search(rf"\b{phrase}\b", q) and re.search(r"after|following|boost|\+", q):
            merge_boost(fb_atk, boosts)  # setup describes the attacker's own action
            strip(rf"(?:after (?:a |an |using )?)?\b{phrase}\b")
    for m in re.finditer(r"\+([1-6])\b(?:\s*(attack|atk|spa|special attack|speed))?", q):
        stat = _EV_STATS.get(m.group(2) or "", None)
        boosts = {stat: int(m.group(1))} if stat else {"Attack": int(m.group(1)),
                                                       "Special Attack": int(m.group(1))}
        merge_boost(bind(m.start(), fb_atk), boosts)
    strip(r"\+[1-6]\b(?:\s*(?:attack|atk|spa|special attack|speed))?")
    for phrase, item in _OFFENSE_ITEMS.items():
        m = re.search(rf"\b{phrase}\b", q)
        if m:
            bind(m.start(), fb_atk)["item"] = item
            strip(rf"\b{phrase}\b")
            break
    for phrase, item in _DEFENSE_ITEMS.items():
        m = re.search(rf"\b{phrase}\b", q)
        if m:
            bind(m.start(), fb_dfn)["item"] = item
            strip(rf"\b{phrase}\b")
            break
    for nature, updown in _NATURES.items():
        m = re.search(rf"\b{nature}\b", q)
        if m:
            side = bind(m.start(), fb_atk if nature in _OFFENSIVE_NATURES else fb_dfn)
            side["nature"] = updown
            side.setdefault("evs", {})[updown[0]] = 252
            strip(rf"\b{nature}\b")
    for m in re.finditer(r"(?:max|252)\s+(attack|atk|spa|special attack|hp|def|defense|spd|special defense|speed|spe)", q):
        stat = _EV_STATS[m.group(1)]
        default = fb_dfn if stat in ("HP", "Defense", "Special Defense") else fb_atk
        bind(m.start(), default).setdefault("evs", {})[stat] = 252
    strip(r"(?:max|252)\s+(?:attack|atk|spa|special attack|hp|def|defense|spd|special defense|speed|spe)")
    # "typical"/"regular" are how paraphrased questions say "standard"; "normal" is
    # deliberately NOT a marker (it collides with the Normal type)
    for m in re.finditer(r"\bstandard\b|\btypical\b|\bregular\b|\bsmogon(?:['\u2019]s)? set\b|\bsmogon\b", q):
        bind(m.start(), fb_atk)["standard"] = True
    strip(r"\bstandard\b|\btypical\b|\bregular\b|\bsmogon(?:['\u2019]s)? set\b|\bsmogon\b")
    m = re.search(r"\bburn(?:ed|t)?\b", q)
    if m:
        bind(m.start(), fb_atk)["status"] = "brn"
        strip(r"\bburn(?:ed|t)?\b")
    m = re.search(r"\bin (?:the )?(sun|rain|sand(?:storm)?|snow|hail)\b", q)
    if m:
        field["weather"] = {"sandstorm": "sand"}.get(m.group(1), m.group(1))
        strip(r"\bin (?:the )?(?:sun|rain|sand(?:storm)?|snow|hail)\b")
    # "while grassy terrain is active" / "during grassy terrain" must parse as field
    # state, or "grassy terrain" gets picked up as the attacking move
    m = re.search(r"\b(?:in|during|while|under|with) (?:the )?(electric|grassy|psychic|misty) terrain\b", q)
    if m:
        field["terrain"] = m.group(1) + "terrain"
        strip(r"\b(?:in|during|while|under|with) (?:the )?(?:electric|grassy|psychic|misty) terrain\b(?: is active)?")
    if re.search(r"\bdoubles\b|\bvgc\b|\bspread\b|\bdouble battle\b", q):
        field["spread"] = True
        strip(r"\b(?:in |a |during )*doubles(?: format| play)?\b|\bvgc\b|\bspread\b|\bdouble battle\b")
    m = re.search(r"\btera[- ]?([a-z]+)\b", q)
    if m and m.group(1).capitalize() in tools.known_types():
        bind(m.start(), fb_atk)["tera"] = m.group(1)
        strip(r"\btera[- ]?[a-z]+\b")
    return q, by_mon, fb_atk, fb_dfn, field


def _merged(base: dict, extra: dict) -> dict:
    out = {k: v for k, v in base.items()}
    for k, v in extra.items():
        if k in ("boosts", "evs"):
            out.setdefault(k, {}).update(v)
        else:
            out[k] = v
    return out


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


def find_entity_names(text: str, limit: int = 4) -> dict:
    """Known Pokemon and move names appearing in free text. Used for
    answer-driven source expansion: when the generated answer names an entity
    (the priority move turns out to be Sucker Punch), that entity's own
    documents join the cited sources."""
    return {"pokemon": _find_names(text, tools.known_pokemon(), limit),
            "moves": _find_names(text, tools.known_moves(), limit)}


async def route(question: str, corpus: str | None = None) -> list[dict]:
    """Return passages for the question: from a tool when one clearly applies,
    otherwise from hybrid retrieval."""
    mons = _find_mons(question)
    types = _find_types(question)
    # monotype is opt-in: only surface its docs/tools when the question names
    # monotype or a single-type team, else default hard to OU (see retrieve_hybrid).
    mono_intent = bool(_MONOTYPE.search(question) or _TYPE_TEAM.search(question))
    # team tools need every named member (the calc tools cap _find_mons at 3)
    team_mons = _find_names(question, tools.known_pokemon(), limit=12)
    arch = _archetype_of(question)

    # a pokepast.es link: critique the real team by its actual sets, or (if the
    # question asks to build) use its members as the core to build around
    pp = re.search(r"https?://pokepast\.es/\w+", question, re.I)
    if pp:
        url = pp.group(0)
        if _BUILD_TEAM.search(question):
            p = tools.generate_ou_team(have=tools.paste_mons(url), archetype=arch,
                                       variant=_variant_of(question))
        else:
            p = tools.analyze_paste(url, archetype=arch)
        if p:
            return p

    # "who is faster, X or Y" / "does X outspeed Y"
    if _SPEED_VS.search(question) and len(mons) == 2:
        p = tools.speed_check(mons[0], mons[1])
        if p:
            return p

    # "what outspeeds Garchomp?" — one mon + a speed verb is the tiers question:
    # meta-relevant faster mons, ties, and the Choice Scarf math
    if _SPEED_VS.search(question) and len(mons) == 1:
        p = tools.speed_tiers(mons[0])
        if p:
            return p

    # "what would X need to survive Y's Z" -> defensive escalation search
    if _SURVIVE_HOW.search(question) and len(mons) >= 2:
        q0, by_mon, fb_atk, _fb_dfn, field = _parse_mods(question, mons)
        moves = _find_names(q0, tools.known_moves())
        if moves:
            move_pos = q0.find(moves[0])
            attacker = min(mons, key=lambda m: abs(q0.find(m) - move_pos))
            defender = next(m for m in mons if m != attacker)
            name = tools.known_moves()[moves[0]][0]
            p = tools.survive_search(attacker, name, defender, _merged(fb_atk, by_mon.get(attacker, {})))
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
        q, by_mon, fb_atk, fb_dfn, field = _parse_mods(question, mons)
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
            p = tools.damage_calc(attacker, name, defender,
                                  _merged(fb_atk, by_mon.get(attacker, {})),
                                  _merged(fb_dfn, by_mon.get(defender, {})),
                                  field["weather"], field["terrain"], field["spread"])
            if p:
                return p

    # "is Earthquake effective against Skarmory" / "is Fire effective against Grass"
    # / "what is Garchomp weak to" — checked AFTER the damage/OHKO/survive branches,
    # which are strictly more specific (two mons + a move), so matchup phrasings like
    # "does Earthquake do much damage to Skarmory" can't hijack real calc questions
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

    # monotype superlatives ("fastest on a Steel monotype team") rank the monotype
    # meta pool, not the whole dex; guards the generic stat_query branch below.
    if mono_intent and _SUPERLATIVE.search(question) and types:
        stat = next((STATS[k] for k in STATS if re.search(rf"\b{k}\b", question, re.I)), None)
        if stat:
            lowest = bool(re.search(r"\b(slowest|lowest)\b", question, re.I))
            p = tools.monotype_stat_query(types[0], stat, lowest=lowest)
            if p:
                return p

    # critique a team: monotype when the members share a type (or a monotype signal),
    # else gen9OU. Needs the members named; infers the shared type from them.
    if _ANALYZE.search(question) and len(team_mons) >= 2:
        typ = types[0] if (mono_intent and types) else tools.team_type(team_mons)
        p = (tools.analyze_team(typ, team_mons, archetype=arch) if typ
             else tools.analyze_ou_team(team_mons, archetype=arch))
        if p:
            return p

    # "build me a team": monotype when the question signals a single type, else default
    # to gen9OU (the format most people mean). Checked before the teammate recommender.
    if _BUILD_TEAM.search(question):
        var = _variant_of(question)
        p = (tools.generate_team(types[0], have=team_mons, archetype=arch, variant=var)
             if mono_intent and types
             else tools.generate_ou_team(have=team_mons, archetype=arch, variant=var))
        if p:
            return p

    # teammate recommender: monotype for a single-type signal ("good Steel teammates
    # for Gholdengo"), else gen9OU ("teammates for Kingambit").
    if _TEAMMATE.search(question):
        p = (tools.recommend_teammates(types[0], have=team_mons)
             if mono_intent and types
             else tools.recommend_ou_teammates(have=team_mons))
        if p:
            return p

    # "fastest Ghost-type Pokemon", "highest Attack among Water types"
    if _SUPERLATIVE.search(question) and types and not mons and not mono_intent:
        stat = next((STATS[k] for k in STATS if re.search(rf"\b{k}\b", question, re.I)), None)
        if stat:
            lowest = bool(re.search(r"\b(slowest|lowest)\b", question, re.I))
            p = tools.stat_query(stat, type_filter=types[0], lowest=lowest)
            if p:
                return p

    # "what counters/checks X" is a competitive-usage question: scope it to the
    # usage-stats corpus (the Counter move and X's own biology pages otherwise
    # crowd out the checks-and-counters data), falling through when it has nothing
    if re.search(r"\bcounters?\b|\bchecks?\b|most used|most common(?:ly)?|usually|\busage\b"
                 r"|almost every|nearly (?:all|every)|good answers to|deal with|\bbeats?\b"
                 r"|frequent(?:ly)?|\boften\b|relies on", question, re.I) and corpus is None:
        scoped = await retrieve_hybrid(question, corpus="crystal_battle", allow_monotype=mono_intent)
        if scoped and passes_threshold(scoped):
            return scoped

    return await retrieve_hybrid(question, corpus=corpus, allow_monotype=mono_intent)
