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
directly. For Pokemon, "what moves does X use/run" asks about competitive usage statistics
(prefer a usage-data passage when present); "what moves can X learn" asks about the
movepool/learnset. Competitive advice and usage statistics are format-specific: when answering
from a competitive analysis or usage data, name the format and generation (like "In Gen 9 OU")
in your first sentence; each passage states its format."""


CONDENSE_SYSTEM = """You rewrite follow-up questions to be self-contained. Given a conversation
and a follow-up question, resolve pronouns and references ("it", "that move", "what about X")
from the conversation and output ONE standalone question meaning the same thing. Keep the
user's wording where possible. Do not answer the question, add topics, or explain anything.
If the follow-up is already self-contained, output it unchanged. Output only the question."""


def _history_text(history: list[dict], max_turns: int = 6, max_chars: int = 400) -> str:
    lines = []
    for t in history[-max_turns:]:
        role = "User" if t.get("role") == "user" else "Assistant"
        content = (t.get("content") or "").strip()
        if len(content) > max_chars:
            # the referent is nearly always in the opening sentence of an answer
            content = content[:max_chars] + " …"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def condense_question(question: str, history: list[dict]) -> str:
    """Rewrite a follow-up into a standalone question using the conversation.

    Conversational memory lives entirely in this rewrite: the router, retrieval, the
    gate, and the answer prompt all still see one self-contained question, so none of
    their tuning changes meaning. Any failure falls back to the original question.
    """
    prompt = (f"Conversation:\n{_history_text(history)}\n\n"
              f"Follow-up question: {question}\n\nStandalone question:")
    payload = {
        "model": settings.llm_model,
        "prompt": prompt,
        "system": CONDENSE_SYSTEM,
        "stream": False,
        "think": False,  # short factual rewrite; thinking mode is 10x slower for the same output
    }
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_host, timeout=60) as client:
            r = await client.post("/api/generate", json=payload)
            r.raise_for_status()
            out = (r.json().get("response") or "").strip().strip('"').strip()
    except Exception:
        return question
    out = out.splitlines()[0].strip() if out else ""
    # a rewrite that vanishes or balloons is worse than no rewrite
    if not out or len(out) > 4 * max(len(question), 40):
        return question
    return out


def _build_prompt(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i+1}] {p['content']}" for i, p in enumerate(passages))
    directive = ""
    # format-specific advice must say its format; the instruction binds far better
    # right next to the question than buried in the system prompt. Keyed on the
    # passage's corpus, not its text: continuation chunks lack the header.
    if any(p.get("corpus") in ("smogon", "crystal_battle")
           or (p.get("source") or "").startswith(("smogon#", "gen9ou_chaos#", "gen2ou_chaos#"))
           for p in passages):
        directive = " Start your answer by naming the format this applies to (e.g. \"In Gen 9 OU, ...\")."
    return f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer (with [n] citations):{directive}"


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
