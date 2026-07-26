"""Canonical-preference metric: when a species' base page and a near-duplicate form
variant are both in the corpus, does retrieval put the BASE page first?

Why this exists: the PokeAPI adapter emits form variants (Gmax, Totem, in-battle
cosmetic forms) that carry the same typing, genus, dex number, stats and abilities as
the base species but none of the per-game flavor text. The stub is therefore far shorter,
which makes it *denser* on exactly the identity facts a genus question asks for, and the
cross-encoder ranks it above the real page. The retrieval hit-rate in run_eval.py cannot
see this: both documents ground the answer correctly, so the answer stays right and the
metric stays green while the canonical page silently falls out of the top-k.

The pairs are derived from the corpus, not hardcoded: a variant counts as a near-duplicate
only if every identity field matches its base. That keeps Mega/Alola/Hisui/Galar forms
(which genuinely differ) out of the metric, and it stays correct if the adapter changes.
Species whose names simply contain a hyphen (Chi-Yu, Ho-Oh, Porygon-Z, Hakamo-o) never
resolve to a base doc and are skipped.

Run:  python -m eval.canonical_preference
"""
import asyncio
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
from db import connect                    # noqa: E402
from retrieve import retrieve_hybrid, POOL  # noqa: E402

# identity fields parsed out of a pokedex document's opening block
_FIELDS = {
    "type":   re.compile(r"^\S.*?\(([\w /]+)-type Pokemon\)"),
    "genus":  re.compile(r"-type Pokemon\)\.\s*(.*?)\.\s*National dex"),
    "dex":    re.compile(r"National dex number (\d+)"),
    "stats":  re.compile(r"Base stats: (.*?)\. Base stat total"),
    "abil":   re.compile(r"Abilities: (.*?)\."),
    "hidden": re.compile(r"Hidden ability: (.*?)\."),
}


def _parse(text: str) -> dict:
    out = {}
    for name, rx in _FIELDS.items():
        m = rx.search(text)
        out[name] = m.group(1).strip() if m else None
    return out


def _base_of(name: str, by_name: dict[str, str]) -> str | None:
    """Resolve a form name to its base document by trying progressively shorter
    prefixes, longest first. Suffixes are not single segments: Pikachu-Original-Cap
    has to reach past 'Pikachu-Original' (not a document) to 'Pikachu'. Species whose
    name merely contains a hyphen never resolve, which is the point: 'Chi-Yu' would
    have to find 'Chi', and 'Hakamo-o' would have to find 'Hakamo'. Porygon-Z does
    resolve to Porygon, and is then correctly rejected on differing stats."""
    parts = name.split("-")
    for i in range(len(parts) - 1, 0, -1):
        base = by_name.get("-".join(parts[:i]))
        if base:
            return base
    return None


def _load_pairs() -> list[tuple[str, str, str]]:
    """(base_source, variant_source, genus) for every near-duplicate variant."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT source, replace(string_agg(content, ' ' ORDER BY id), E'\n', ' ')
            FROM chunks WHERE source LIKE 'pokedex#%%' GROUP BY source
        """)
        docs = dict(cur.fetchall())

    parsed = {s: _parse(c) for s, c in docs.items()}
    by_name = {s[len("pokedex#"):]: s for s in docs}

    pairs = []
    for name, src in sorted(by_name.items()):
        if "-" not in name:
            continue
        base_src = _base_of(name, by_name)
        if not base_src:
            continue  # hyphen is part of the species name (Chi-Yu, Hakamo-o)
        v, b = parsed[src], parsed[base_src]
        if all(v[f] is not None and v[f] == b[f] for f in _FIELDS):
            pairs.append((base_src, src, b["genus"]))
    return pairs


def _genus_question(genus: str) -> str:
    """'Sleeping Pokémon' -> 'Which Pokemon is known as the Sleeping Pokemon?'
    Accents are stripped because questions are typed unaccented while the corpus
    spells the genus with them."""
    plain = unicodedata.normalize("NFKD", genus).encode("ascii", "ignore").decode()
    return f"Which Pokemon is known as the {plain}?"


async def main():
    pairs = _load_pairs()
    if not pairs:
        print("no near-duplicate variants found; nothing to measure")
        return

    base_first = variant_first = neither = 0
    top8_base = top8_variant_only = 0
    losses = []
    by_family = defaultdict(lambda: [0, 0])

    for base_src, var_src, genus in pairs:
        q = _genus_question(genus)
        # ask for the whole fused pool so the metric measures ordering, not truncation
        passages = await retrieve_hybrid(q, k=POOL)
        srcs = [p["source"] for p in passages]
        bi = srcs.index(base_src) if base_src in srcs else None
        vi = srcs.index(var_src) if var_src in srcs else None
        # the whole suffix, not its last segment: "Original-Cap", not "Cap"
        family = var_src[len(base_src) + 1:]

        if bi is None and vi is None:
            neither += 1
            continue
        if vi is None or (bi is not None and bi < vi):
            base_first += 1
            by_family[family][0] += 1
        else:
            variant_first += 1
            by_family[family][1] += 1
            losses.append((q, base_src, bi, var_src, vi))

        top8 = srcs[:8]
        if base_src in top8:
            top8_base += 1
        elif var_src in top8:
            top8_variant_only += 1

    scored = base_first + variant_first
    print(f"near-duplicate variant pairs derived from the corpus: {len(pairs)}")
    print(f"  neither page retrieved at all (genus is ambiguous): {neither}\n")
    print(f"| Metric | Score |")
    print(f"|--------|-------|")
    pct = 100 * base_first / scored if scored else 0.0
    print(f"| Canonical preference (base ranks above its variant) | {pct:.0f}% ({base_first}/{scored}) |")
    print(f"| Base page present in top-8 | {top8_base}/{scored} |")
    print(f"| Only the variant stub in top-8 | {top8_variant_only}/{scored} |")

    if by_family:
        print("\nby form family (base-first / variant-first):")
        for fam, (bf, vf) in sorted(by_family.items(), key=lambda kv: -sum(kv[1])):
            print(f"  {fam:<10} {bf:>3} / {vf}")

    if losses:
        print(f"\nvariant outranks base ({len(losses)}):")
        for q, b, bi, v, vi in losses[:20]:
            br = "absent" if bi is None else f"rank {bi+1}"
            print(f"  {q}\n      base    {b} -> {br}\n      variant {v} -> rank {vi+1}")
        if len(losses) > 20:
            print(f"  ... and {len(losses)-20} more")


if __name__ == "__main__":
    asyncio.run(main())
