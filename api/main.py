"""FastAPI app: /health, /ask (streaming, cited). Wiring is done; tune the behavior."""
import json
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from retrieve import passes_threshold
from router import route, find_entity_names
import tools
from llm import answer_stream, condense_question

REFUSAL = "I don't know based on the available sources."

app = FastAPI(title="grounded-rag")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    corpus: str | None = None  # optional: scope retrieval to one corpus
    # Prior turns, oldest first. The server keeps no session state: memory is the
    # client's transcript riding along in the request, used only to condense a
    # follow-up into a standalone question before routing. This keeps /ask usable
    # as a pure knowledge tool by frontends that own their own conversation state.
    history: list[Turn] = []


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
async def ask(req: AskRequest):
    """Stream a cited answer as Server-Sent Events.

    First event (follow-ups only): the standalone rewrite of the question.
    Then: the retrieved sources (so the UI can render citation chips).
    Then: answer tokens. If retrieval is too weak, we refuse instead of hallucinating.
    """
    question = req.question
    if req.history:
        question = await condense_question(req.question, [t.model_dump() for t in req.history])
    passages = await route(question, corpus=req.corpus)

    async def gen():
        if question != req.question:
            yield ("event: question\n"
                   f"data: {json.dumps({'original': req.question, 'standalone': question})}\n\n")
        yield f"event: sources\ndata: {json.dumps(passages)}\n\n"
        if not passes_threshold(passages):
            yield f"event: token\ndata: {json.dumps(REFUSAL)}\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        parts = []
        # answer from the standalone question and fresh passages only; history never
        # enters the answer prompt, so prior turns can't become uncited evidence
        async for tok in answer_stream(question, passages):
            parts.append(tok)
            yield f"event: token\ndata: {json.dumps(tok)}\n\n"
        # answer-driven source expansion: entities the answer itself names get
        # their documents added to the cited sources (a second `sources` event;
        # the UI replaces its chips with the union)
        try:
            ents = find_entity_names("".join(parts))
            seen = {p["source"] for p in passages}
            extra = [doc for doc in tools.entity_docs(ents["pokemon"], ents["moves"])
                     if doc["source"] not in seen]
            if extra:
                yield f"event: sources\ndata: {json.dumps(passages + extra)}\n\n"
        except Exception:
            pass
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
