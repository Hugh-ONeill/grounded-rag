"""The corpus contract.

A corpus adapter is just a function that yields documents. Keep documents SMALL and
self-contained (one Pokémon, one move, one replay turn) so each becomes a citable unit
after chunking. A clean `source` string is what the UI shows as the citation.
"""
from typing import TypedDict

# Adapters may place this in `content` where a chunk must not straddle. Chunking is a
# blind character window, so a dense fact block followed by long prose gets sliced into
# one window carrying both, and the fact block's signal is diluted by whatever shares its
# chunk. That is measurable: a Gigantamax stub (280 chars, genus and stats only, no flavor
# text) out-ranked the real Snorlax page (900-char window, same genus buried among stats,
# abilities and dimensions) 0.99876 to 0.99046, because the cross-encoder rewards density.
# Splitting here lets an adapter say "this block stands alone" without duplicating it into
# its own document. Sections are windowed independently; continuation chunks are still
# stamped with the document title, so provenance is unchanged.
SECTION_BREAK = "\x1e"  # ASCII record separator: never appears in corpus prose


class Document(TypedDict):
    source: str        # citation label, e.g. "gen9ou_chaos.json#Heatran"
    title: str         # human-friendly heading
    content: str       # the text to embed + show
    metadata: dict     # anything extra (type, format, url, ...)
