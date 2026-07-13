#!/usr/bin/env python3
"""MCP stdio server exposing grounded-rag's /retrieve as a `pokemon_expert` tool.

Built for AIRI's mcp.json (Claude-Desktop-style config), but any MCP client
works. Zero dependencies: stdlib json + urllib against the local API.

The tool returns raw reference notes and instructs the calling character to
rephrase in its own voice -- no citations. /retrieve is the right endpoint for
that: passages only, no generation, so the character's LLM is the only voice
(see the /retrieve docstring in api/main.py).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

RAG_URL = "http://127.0.0.1:8001/retrieve"
MAX_PASSAGE_CHARS = 1200
MAX_TOTAL_CHARS = 7000
LOG_PATH = os.path.expanduser(
    os.path.join(os.environ.get("XDG_STATE_HOME", "~/.local/state"),
                 "pokemon-expert-mcp.log"))


def log_lookup(question: str, outcome: str, titles: list[str]) -> None:
    """One JSON line per tool call, so 'did the character actually look this
    up?' is answerable with tail -f instead of guesswork."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "question": question,
                "outcome": outcome,
                "top_titles": titles[:3],
            }) + "\n")
    except OSError:
        pass

TOOL = {
    "name": "pokemon_expert",
    "description": (
        "Authoritative, up-to-date Pokemon reference: species, moves, abilities, items, "
        "stats, type matchups, damage calculations, competitive usage and sets, monotype "
        "team building/analysis/teammates, mechanics, and lore. ALWAYS call this FIRST for "
        "EVERY Pokemon question before you answer, no matter how simple it seems, because "
        "your own training data is often outdated or wrong on competitive facts. You MUST "
        "use the tool, never your own memory, for anything computed or competitive: usage, "
        "sets, spreads, items, damage, matchups, and building, analyzing, or rating teams. "
        "For general details the tool does not return, you MAY supplement from your own "
        "Pokemon knowledge, but never contradict the tool. It returns raw reference notes: "
        "weave them into your reply in your own voice and character; never cite, quote, or "
        "mention sources, documents, or the lookup itself."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "A single self-contained factual question, e.g. "
                "'does Choice Band Kingambit OHKO Corviknight with Sucker Punch?'",
            }
        },
        "required": ["question"],
    },
}


def lookup(question: str) -> str:
    req = urllib.request.Request(
        RAG_URL,
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        log_lookup(question, "api-offline", [])
        return (
            "The knowledge base is offline right now. Answer from your own "
            "general knowledge and be upfront that you couldn't double-check."
        )

    if not data.get("answerable") or not data.get("passages"):
        log_lookup(question, "no-answer", [])
        return (
            "No solid information found on this. Be honest that you're not sure "
            "rather than guessing specifics."
        )
    log_lookup(question, "answered",
               [p.get("title", "") for p in data["passages"]])

    notes, total = [], 0
    for p in data["passages"]:
        content = (p.get("content") or "").strip()[:MAX_PASSAGE_CHARS]
        title = (p.get("title") or "").strip()
        entry = f"* {title}: {content}" if title else f"* {content}"
        if total + len(entry) > MAX_TOTAL_CHARS:
            break
        notes.append(entry)
        total += len(entry)

    return (
        "Reference notes (rephrase naturally in your own voice; do not cite or "
        "mention sources):\n" + "\n".join(notes)
    )


def reply(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method, msg_id = msg.get("method"), msg.get("id")

        if method == "initialize":
            reply(msg_id, {
                "protocolVersion": msg.get("params", {}).get(
                    "protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "pokemon-expert", "version": "1.0.0"},
            })
        elif method == "ping":
            reply(msg_id, {})
        elif method == "tools/list":
            reply(msg_id, {"tools": [TOOL]})
        elif method == "tools/call":
            params = msg.get("params", {})
            if params.get("name") != "pokemon_expert":
                reply(msg_id, error={"code": -32602,
                                     "message": f"unknown tool {params.get('name')}"})
                continue
            question = (params.get("arguments") or {}).get("question", "")
            text = lookup(question) if question else "No question given."
            reply(msg_id, {"content": [{"type": "text", "text": text}],
                           "isError": False})
        elif msg_id is not None:
            # requests we don't implement; notifications fall through silently
            reply(msg_id, error={"code": -32601, "message": f"unknown method {method}"})


if __name__ == "__main__":
    main()
