"""The corpus contract.

A corpus adapter is just a function that yields documents. Keep documents SMALL and
self-contained (one Pokémon, one move, one replay turn) so each becomes a citable unit
after chunking. A clean `source` string is what the UI shows as the citation.
"""
from typing import TypedDict


class Document(TypedDict):
    source: str        # citation label, e.g. "gen9ou_chaos.json#Heatran"
    title: str         # human-friendly heading
    content: str       # the text to embed + show
    metadata: dict     # anything extra (type, format, url, ...)
