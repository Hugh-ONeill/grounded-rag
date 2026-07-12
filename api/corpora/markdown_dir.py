"""Generic adapter: ingest a folder of .md / .txt files. The safe, recruiter-legible demo."""
from pathlib import Path
from config import settings


def load():
    root = Path(settings.corpus_path)
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        yield {
            "source": str(path.relative_to(root)),
            "title": path.stem,
            "content": text,
            "metadata": {"format": path.suffix.lstrip(".")},
        }
