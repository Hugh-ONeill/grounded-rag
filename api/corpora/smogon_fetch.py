"""Fetch Smogon Strategy Dex analyses and sets from the @pkmn data mirror
(https://data.pkmn.cc, backed by pkmn.github.io/smogon) into a local cache.
Smogon discourages scraping smogon.com itself; this mirror exists for the
ecosystem. Same posture as the Bulbapedia cache: local personal use,
attributed, never committed or redistributed.

Usage (from the repo root):
  python api/corpora/smogon_fetch.py
"""
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

UA = "grounded-rag/0.1 (personal research project; https://github.com/Hugh-ONeill/grounded-rag)"
GENS = ["gen9", "gen2"]


def cache_dir() -> Path:
    d = Path(settings.smogon_cache).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch() -> None:
    d = cache_dir()
    with httpx.Client(headers={"User-Agent": UA}, timeout=120, follow_redirects=True) as client:
        for gen in GENS:
            for kind in ("analyses", "sets"):
                out = d / f"{kind}-{gen}.json"
                if out.exists():
                    print(f"[smogon] cached: {out.name}")
                    continue
                r = client.get(f"https://data.pkmn.cc/{kind}/{gen}.json")
                r.raise_for_status()
                out.write_bytes(r.content)
                print(f"[smogon] fetched {out.name} ({len(r.content)//1024} KB)")
    print(f"[smogon] cache at {d}")


if __name__ == "__main__":
    fetch()
