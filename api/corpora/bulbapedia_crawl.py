"""Polite Bulbapedia crawler: fetches page wikitext via the MediaWiki API into a
local cache. The cache is the corpus source; it is never committed (Bulbapedia
content is CC BY-NC-SA — ship the scraper, not the data).

Politeness: single-threaded, batched title queries, a delay between requests,
maxlag cooperation (backs off when the wiki is busy), a descriptive User-Agent,
and a resumable cache so nothing is fetched twice.

Usage (from the repo root):
  python api/corpora/bulbapedia_crawl.py --pilot   # ~45 high-value pages
  python api/corpora/bulbapedia_crawl.py --full    # species, moves, abilities, items
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

API = "https://bulbapedia.bulbagarden.net/w/api.php"
UA = "grounded-rag-crawler/0.1 (personal research project; https://github.com/Hugh-ONeill/grounded-rag)"
DELAY = 2.0        # seconds between requests
BATCH = 20         # titles per API request (50 is the hard limit; stay modest)

GENERAL_PAGES = [
    "Pokémon", "Pokémon world", "Pokémon Trainer", "Pokédex", "Pokémon battle",
    "Evolution", "Legendary Pokémon", "Mythical Pokémon", "Regional form",
    "Starter Pokémon", "Poké Ball", "Gym", "Pokémon League", "Move", "Egg",
    "Level", "Friendship", "Baby Pokémon", "Pseudo-legendary Pokémon",
]

MECHANICS_PAGES = [
    "Damage", "Stat", "Type", "Status condition", "Weather", "Terrain",
    "Entry hazard", "Priority", "Critical hit", "Same-type attack bonus",
    "Nature", "Effort values", "Individual values", "Accuracy", "Evasion",
    "Held item", "Terastal phenomenon", "Ability", "Egg Group", "Shiny Pokémon",
]

PILOT_EXTRA = [
    "Grassy Terrain (move)", "Earthquake (move)", "Stealth Rock (move)",
    "Roost (move)", "Gravity (move)", "Knock Off (move)",
    "Garchomp (Pokémon)", "Snorlax (Pokémon)", "Rotom (Pokémon)",
    "Great Tusk (Pokémon)", "Kingambit (Pokémon)",
    "Levitate (Ability)", "Regenerator (Ability)", "Supreme Overlord (Ability)",
    "Leftovers", "Choice Band", "Heavy-Duty Boots", "Booster Energy",
]


def cache_dir() -> Path:
    d = Path(settings.bulbapedia_cache).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fname(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", title) + ".json"


def _full_titles() -> list[str]:
    """Every species, move, ability, and named item from the PokeAPI CSVs,
    mapped to Bulbapedia's title conventions."""
    csvdir = Path(settings.pokeapi_path) / "data" / "v2" / "csv"

    def rows(name):
        with open(csvdir / f"{name}.csv", newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)

    EN = "9"
    titles = list(GENERAL_PAGES) + list(MECHANICS_PAGES)
    species = {r["pokemon_species_id"]: r["name"] for r in rows("pokemon_species_names")
               if r["local_language_id"] == EN}
    titles += [f"{n} (Pokémon)" for n in species.values()]
    moves = {r["move_id"]: r["name"] for r in rows("move_names") if r["local_language_id"] == EN}
    titles += [f"{n} (move)" for n in moves.values()]
    abilities = {r["ability_id"]: r["name"] for r in rows("ability_names")
                 if r["local_language_id"] == EN}
    titles += [f"{n} (Ability)" for n in abilities.values()]
    # items use plain titles; MediaWiki redirects resolve the ambiguous ones
    item_names = {r["item_id"]: r["name"] for r in rows("item_names")
                  if r["local_language_id"] == EN}
    prose = {r["item_id"] for r in rows("item_prose") if r["local_language_id"] == EN}
    titles += sorted({n for i, n in item_names.items() if i in prose})
    return titles


def crawl(titles: list[str]) -> None:
    d = cache_dir()
    todo = [t for t in titles if not (d / _fname(t)).exists()]
    print(f"[crawl] {len(titles)} titles, {len(todo)} not yet cached")
    client = httpx.Client(headers={"User-Agent": UA}, timeout=60)
    fetched = 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        params = {
            "action": "query", "prop": "revisions", "rvprop": "content",
            "rvslots": "main", "format": "json", "formatversion": "2",
            "redirects": "1", "maxlag": "5", "titles": "|".join(batch),
        }
        while True:
            r = client.get(API, params=params)
            if r.status_code == 200 and "maxlag" not in r.text[:200]:
                data = r.json()
                if "error" in data and data["error"].get("code") == "maxlag":
                    wait = int(r.headers.get("Retry-After", 5))
                    print(f"[crawl] wiki busy (maxlag), sleeping {wait}s")
                    time.sleep(wait)
                    continue
                break
            time.sleep(int(r.headers.get("Retry-After", 10)))
        redirected = {rd["to"]: rd["from"] for rd in data["query"].get("redirects", [])}
        for page in data["query"].get("pages", []):
            requested = redirected.get(page["title"], page["title"])
            if page.get("missing"):
                (d / _fname(requested)).write_text(json.dumps({"title": requested, "missing": True}))
                continue
            content = page["revisions"][0]["slots"]["main"]["content"]
            (d / _fname(requested)).write_text(json.dumps(
                {"title": page["title"], "requested": requested, "content": content}))
            fetched += 1
        print(f"[crawl] {min(i + BATCH, len(todo))}/{len(todo)} ({fetched} pages cached)")
        time.sleep(DELAY)
    print(f"[crawl] done: {fetched} new pages, cache at {d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.full:
        crawl(_full_titles())
    else:
        crawl(GENERAL_PAGES + MECHANICS_PAGES + PILOT_EXTRA)
