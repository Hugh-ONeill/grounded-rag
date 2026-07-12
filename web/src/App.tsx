import { useState } from "react";
import { ask, Source } from "./api";

// Minimal but functional chat UI: ask a question, watch the answer stream in,
// see the cited sources as chips. Style it / polish it — this is your frontend showcase.
export default function App() {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim() || busy) return;
    setBusy(true);
    setAnswer("");
    setSources([]);
    try {
      await ask(q, setSources, (t) => setAnswer((a) => a + t));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ maxWidth: 720, margin: "2rem auto", fontFamily: "system-ui" }}>
      <h1>grounded-rag</h1>
      <form onSubmit={submit} style={{ display: "flex", gap: 8 }}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask a question…"
          style={{ flex: 1, padding: 8 }}
        />
        <button disabled={busy}>{busy ? "…" : "Ask"}</button>
      </form>

      {answer && (
        <section style={{ marginTop: 24, whiteSpace: "pre-wrap", lineHeight: 1.5 }}>
          {answer}
        </section>
      )}

      {sources.length > 0 && (
        <section style={{ marginTop: 24 }}>
          <h3>Sources</h3>
          <ol>
            {sources.map((s, i) => (
              <li key={i} title={s.content}>
                <code>{s.source}</code> ({(s.similarity * 100).toFixed(0)}% match)
              </li>
            ))}
          </ol>
        </section>
      )}
    </main>
  );
}
