"""FastAPI app: /health, /ask (streaming, cited). Wiring is done; tune the behavior."""
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from retrieve import retrieve, passes_threshold
from llm import answer_stream

REFUSAL = "I don't know based on the available sources."

app = FastAPI(title="grounded-rag")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
async def ask(req: AskRequest):
    """Stream a cited answer as Server-Sent Events.

    First event: the retrieved sources (so the UI can render citation chips).
    Then: answer tokens. If retrieval is too weak, we refuse instead of hallucinating.
    """
    passages = await retrieve(req.question)

    async def gen():
        yield f"event: sources\ndata: {json.dumps(passages)}\n\n"
        if not passes_threshold(passages):
            yield f"event: token\ndata: {json.dumps(REFUSAL)}\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        async for tok in answer_stream(req.question, passages):
            yield f"event: token\ndata: {json.dumps(tok)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
