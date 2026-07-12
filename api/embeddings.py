"""Embedding client — talks to local Ollama. Boilerplate is done."""
import httpx
from config import settings


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the local embedding model."""
    out: list[list[float]] = []
    async with httpx.AsyncClient(base_url=settings.ollama_host, timeout=120) as client:
        for t in texts:
            r = await client.post("/api/embeddings", json={"model": settings.embed_model, "prompt": t})
            r.raise_for_status()
            out.append(r.json()["embedding"])
    return out


async def embed_one(text: str) -> list[float]:
    return (await embed([text]))[0]
