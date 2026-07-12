import { useState } from "react";
import { ask, Source, Turn } from "./api";

// Minimal but functional chat UI: a running transcript with per-answer cited
// sources. Follow-ups are resolved server-side from the transcript we send along
// (the server keeps no session state); when the server rewrites a follow-up, the
// standalone question is shown under the user's message.
interface Msg {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  standalone?: string; // the rewrite of the PRECEDING user question, when one happened
}

export default function App() {
  const [q, setQ] = useState("");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);

  const patchLast = (patch: (m: Msg) => Msg) =>
    setMessages((ms) => [...ms.slice(0, -1), patch(ms[ms.length - 1])]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const question = q.trim();
    if (!question || busy) return;
    const history: Turn[] = messages.map(({ role, content }) => ({ role, content }));
    setBusy(true);
    setQ("");
    setMessages((ms) => [
      ...ms,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);
    try {
      await ask(
        question,
        history,
        (s) => patchLast((m) => ({ ...m, sources: s })),
        (t) => patchLast((m) => ({ ...m, content: m.content + t })),
        (standalone) => patchLast((m) => ({ ...m, standalone }))
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ maxWidth: 720, margin: "2rem auto", fontFamily: "system-ui" }}>
      <h1 style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        grounded-rag
        {messages.length > 0 && (
          <button
            type="button"
            onClick={() => setMessages([])}
            disabled={busy}
            style={{ fontSize: "0.5em", fontWeight: "normal" }}
          >
            new conversation
          </button>
        )}
      </h1>

      {messages.map((m, i) =>
        m.role === "user" ? (
          <section key={i} style={{ marginTop: 24 }}>
            <strong>{m.content}</strong>
            {messages[i + 1]?.standalone && (
              <div style={{ color: "#888", fontSize: "0.85em", marginTop: 4 }}>
                ↳ searched as: {messages[i + 1].standalone}
              </div>
            )}
          </section>
        ) : (
          <section key={i} style={{ marginTop: 12 }}>
            <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5 }}>
              {m.content || (busy && i === messages.length - 1 ? "…" : "")}
            </div>
            {(m.sources?.length ?? 0) > 0 && (
              <details style={{ marginTop: 8, fontSize: "0.9em" }}>
                <summary>Sources ({m.sources!.length})</summary>
                <ol>
                  {m.sources!.map((s, j) => (
                    <li key={j} title={s.content}>
                      <code>{s.source}</code> ({(s.similarity * 100).toFixed(0)}% match)
                    </li>
                  ))}
                </ol>
              </details>
            )}
          </section>
        )
      )}

      <form onSubmit={submit} style={{ display: "flex", gap: 8, marginTop: 24 }}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder={messages.length ? "Ask a follow-up…" : "Ask a question…"}
          style={{ flex: 1, padding: 8 }}
        />
        <button disabled={busy}>{busy ? "…" : "Ask"}</button>
      </form>
    </main>
  );
}
