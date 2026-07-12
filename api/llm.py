"""Generation client — local Gemma via Ollama, with a grounding/citation prompt.

The prompt is deliberately strict about (a) only using the provided context and (b) citing
sources as [n]. This is your main lever against hallucination — tune it and write up what worked.
"""
import json
import httpx
from config import settings

SYSTEM = """You are a precise question-answering assistant. Answer ONLY from the numbered context
passages provided. Cite the passages you use inline as [1], [2], etc. If the context does not
contain the answer, say "I don't know based on the available sources." Do not invent facts.
The context is a small sample of a much larger corpus. For questions that ask for a
superlative over the whole corpus (like "which Pokemon is the most used?"), answer only if a
passage explicitly states that overall ranking; never derive one by comparing the few passages
you happen to see. But when the question names specific things (two items, two moves, two
Pokemon) and the context contains passages for them, do compare them: use the numbers in the
passages and explain the trade-off. Facts within a single passage are always fine to answer
directly."""


def _build_prompt(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i+1}] {p['content']}" for i, p in enumerate(passages))
    return f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer (with [n] citations):"


async def answer_stream(question: str, passages: list[dict]):
    """Yield answer tokens as they stream from Ollama (for SSE)."""
    payload = {
        "model": settings.llm_model,
        "prompt": _build_prompt(question, passages),
        "system": SYSTEM,
        "stream": True,
        "think": False,  # JSON-ish factual answers don't need thinking mode; keeps it fast
    }
    async with httpx.AsyncClient(base_url=settings.ollama_host, timeout=None) as client:
        async with client.stream("POST", "/api/generate", json=payload) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                tok = json.loads(line).get("response", "")
                if tok:
                    yield tok
