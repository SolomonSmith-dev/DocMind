import { FormEvent, useState } from "react";
import { AskResult, askQuestion } from "../api";
import CitationCard from "./CitationCard";

export default function AskPanel() {
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AskResult | null>(null);
  const [error, setError] = useState("");

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const res = await askQuestion(question, sessionId);
      setResult(res);
      setSessionId(res.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section style={styles.section}>
      <h2 style={styles.heading}>Ask a Question</h2>
      <form onSubmit={handleSubmit} style={styles.form}>
        <textarea
          style={styles.textarea}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask anything about your uploaded documents..."
          rows={3}
          required
        />
        <div style={styles.row}>
          <button style={styles.btn} type="submit" disabled={loading}>
            {loading ? "Thinking..." : "Ask"}
          </button>
          {sessionId && (
            <button
              style={styles.resetBtn}
              type="button"
              onClick={() => { setSessionId(null); setResult(null); setQuestion(""); }}
            >
              New session
            </button>
          )}
        </div>
      </form>

      {loading && <div style={styles.spinner} aria-label="Loading" />}

      {error && <p style={styles.error}>{error}</p>}

      {result && (
        <div style={styles.resultBox}>
          <div style={styles.metaRow}>
            <span style={styles.model}>{result.model}</span>
            <span style={{ ...styles.confidence, color: result.low_confidence ? "#ef4444" : "#15803d" }}>
              confidence: {Math.round(result.confidence * 100)}%
              {result.low_confidence && " (low)"}
            </span>
            <span style={styles.chunks}>{result.chunks_used} chunk(s) used</span>
          </div>
          <p style={styles.answer}>{result.answer}</p>
          {result.citations.length > 0 && (
            <div style={styles.citations}>
              <p style={styles.citationsLabel}>Sources</p>
              <div style={styles.citationList}>
                {result.citations.map((c, i) => (
                  <CitationCard key={i} citation={c} index={i} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

const styles: Record<string, React.CSSProperties> = {
  section: { marginBottom: "2rem" },
  heading: { fontSize: 18, fontWeight: 600, marginBottom: 12 },
  form: { display: "flex", flexDirection: "column", gap: 10 },
  textarea: {
    padding: "10px 12px",
    border: "1px solid #cbd5e1",
    borderRadius: 8,
    fontSize: 14,
    resize: "vertical",
    outline: "none",
    fontFamily: "inherit",
  },
  row: { display: "flex", gap: 8 },
  btn: {
    padding: "10px 24px",
    background: "#2563eb",
    color: "#fff",
    border: "none",
    borderRadius: 6,
    fontWeight: 600,
    fontSize: 14,
    cursor: "pointer",
  },
  resetBtn: {
    padding: "10px 16px",
    background: "none",
    color: "#64748b",
    border: "1px solid #e2e8f0",
    borderRadius: 6,
    fontSize: 14,
    cursor: "pointer",
  },
  spinner: {
    marginTop: 16,
    width: 32,
    height: 32,
    border: "3px solid #e2e8f0",
    borderTop: "3px solid #2563eb",
    borderRadius: "50%",
    animation: "spin 0.7s linear infinite",
  },
  error: { color: "#ef4444", fontSize: 14, marginTop: 10 },
  resultBox: {
    marginTop: 16,
    padding: "16px",
    background: "#fff",
    border: "1px solid #e2e8f0",
    borderRadius: 10,
  },
  metaRow: { display: "flex", gap: 12, marginBottom: 10, flexWrap: "wrap", fontSize: 12 },
  model: { color: "#64748b", fontFamily: "monospace" },
  confidence: { fontWeight: 600 },
  chunks: { color: "#94a3b8" },
  answer: { fontSize: 15, lineHeight: 1.7, color: "#0f172a", marginBottom: 12, whiteSpace: "pre-wrap" },
  citations: { marginTop: 12 },
  citationsLabel: { fontSize: 13, fontWeight: 600, color: "#475569", marginBottom: 8 },
  citationList: { display: "flex", flexDirection: "column", gap: 6 },
};
