"""Corpus registry. Add an adapter, register it here, select it via CORPUS in .env."""
from config import settings
from . import markdown_dir, crystal_battle

_ADAPTERS = {
    "markdown_dir": markdown_dir.load,
    "crystal_battle": crystal_battle.load,
}


def load_corpus(name: str):
    """Yield documents: dicts with keys {source, title, content, metadata}."""
    if name not in _ADAPTERS:
        raise ValueError(f"unknown corpus {name!r}; options: {list(_ADAPTERS)}")
    return _ADAPTERS[name]()
