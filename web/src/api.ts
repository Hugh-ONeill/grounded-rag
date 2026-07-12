// Thin client for the streaming /ask endpoint. Parses the SSE stream:
//   event: question -> follow-up rewritten as a standalone question (shown so the
//                      user can see what was actually retrieved)
//   event: sources  -> the retrieved citations (render as chips)
//   event: token    -> answer text, appended as it arrives
//   event: done     -> stream complete
export interface Source {
  source: string;
  title: string;
  content: string;
  similarity: number;
}

// A prior conversation turn. The server is stateless: the transcript rides along
// in each request and is only used to condense follow-ups before retrieval.
export interface Turn {
  role: "user" | "assistant";
  content: string;
}

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function ask(
  question: string,
  history: Turn[],
  onSources: (s: Source[]) => void,
  onToken: (t: string) => void,
  onQuestion?: (standalone: string) => void
): Promise<void> {
  const res = await fetch(`${API}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const events = buf.split("\n\n");
    buf = events.pop() ?? "";
    for (const ev of events) {
      const type = ev.match(/event: (\w+)/)?.[1];
      const data = ev.match(/data: ([\s\S]*)/)?.[1];
      if (!type || data === undefined) continue;
      if (type === "sources") onSources(JSON.parse(data));
      else if (type === "token") onToken(JSON.parse(data));
      else if (type === "question") onQuestion?.(JSON.parse(data).standalone);
    }
  }
}
