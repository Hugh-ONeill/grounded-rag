"""Bulbapedia adapter — encyclopedic prose from the local wikitext cache.

Reads the cache written by bulbapedia_crawl.py (never fetches the network) and
yields one citable document per kept section. Bulbapedia content is
CC BY-NC-SA 2.5: every document carries its source URL in metadata for
attribution, and the cache itself is never committed or redistributed.
"""
import json
import re
from pathlib import Path
from config import settings

# sections that are noise for a battle-knowledge expert
BLOCKED_SECTIONS = {
    "in the anime", "in the manga", "in the tcg", "in the tfg", "in other languages",
    "gallery", "sprites", "trivia", "references", "external links", "learnset",
    "by leveling up", "by tm", "by tm/hm", "by tr", "by breeding", "by tutoring",
    "by a prior evolution", "game locations", "in side games", "side game data",
    "in other games", "in spin-off games", "names", "related articles", "appearances",
    "in the pokémon adventures manga", "game data", "in animation", "errors",
    "in the spin-off games", "in spin-off games", "merchandise", "in pokémon go",
    "in the mystery dungeon series", "in other media", "cries", "forms",
    "other appearances", "major appearances", "minor appearances",
}


def _clean(text: str) -> str:
    """Wikitext -> plain prose. Imperfect by design; good enough to embed."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>/]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)          # tables
    text = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", text)
    # inline entity templates keep their display text: {{m|Earthquake}} -> Earthquake
    for _ in range(4):
        text = re.sub(r"\{\{(?:m|p|a|t|i|type|MSP|OBP)\|([^{}|]+)(?:\|[^{}]*)?\}\}",
                      r"\1", text)
        text = re.sub(r"\{\{tt\|([^{}|]+)\|[^{}]*\}\}", r"\1", text)
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)               # everything else
    text = re.sub(r"\[\[[^\]|]*\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    # drop lines that are only markup residue (----, ]], |cellpadding, lone *)
    text = "\n".join(l for l in text.split("\n")
                     if not re.fullmatch(r"[\s\[\]{}|*:;#=-]*", l))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sections(wikitext: str):
    """Yield (section_path, body) pairs, splitting on == headings ==."""
    parts = re.split(r"^(={2,4})\s*(.*?)\s*\1\s*$", wikitext, flags=re.M)
    # parts: [lead, marks, title, body, marks, title, body, ...]
    yield "Overview", parts[0]
    stack: list[str] = []
    for i in range(1, len(parts) - 2, 3):
        depth = len(parts[i]) - 2          # 0 for ==, 1 for ===, 2 for ====
        title = _clean(parts[i + 1])
        stack = stack[:depth] + [title]
        yield " / ".join(stack), parts[i + 2]


def load():
    d = Path(settings.bulbapedia_cache).expanduser()
    if not d.exists():
        raise FileNotFoundError(
            f"no Bulbapedia cache at {d}; run: python api/corpora/bulbapedia_crawl.py --pilot")
    for f in sorted(d.glob("*.json")):
        page = json.loads(f.read_text())
        if page.get("missing"):
            continue
        title = page["title"]
        url = "https://bulbapedia.bulbagarden.net/wiki/" + title.replace(" ", "_")
        for section, body in _sections(page["content"]):
            top = section.split(" / ")[0].lower()
            if top in BLOCKED_SECTIONS or section.lower() in BLOCKED_SECTIONS:
                continue
            prose = _clean(body)
            if len(prose) < 80:            # headers-only / empty sections
                continue
            label = title if section == "Overview" else f"{title} § {section}"
            yield {
                "source": f"bulbapedia#{label}",
                "title": label,
                "content": f"{label} (Bulbapedia):\n{prose}",
                "metadata": {"kind": "bulbapedia", "page": title, "section": section,
                             "url": url, "license": "CC BY-NC-SA 2.5"},
            }
