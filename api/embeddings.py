"""Embedding client — talks to local Ollama, batched via /api/embed."""
import httpx
from config import settings

BATCH = 64  # texts per request; Ollama handles the rest


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the local embedding model."""
    out: list[list[float]] = []
    async with httpx.AsyncClient(base_url=settings.ollama_host, timeout=300) as client:
        for i in range(0, len(texts), BATCH):
            r = await client.post(
                "/api/embed",
                json={"model": settings.embed_model, "input": texts[i:i + BATCH]},
            )
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
    return out


async def embed_one(text: str) -> list[float]:
    return (await embed([text]))[0]
