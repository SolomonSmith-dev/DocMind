import { useState } from "react";
import { Citation } from "../api";

interface Props {
  citation: Citation;
  index: number;
}

export default function CitationCard({ citation, index }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div style={styles.card}>
      <button style={styles.toggle} onClick={() => setOpen((o) => !o)}>
        <span style={styles.badge}>[{index + 1}]</span>
        <span style={styles.filename}>{citation.source_filename}</span>
        <span style={styles.page}>p.{citation.page_number}</span>
        <span style={styles.chevron}>{open ? "▲" : "▼"}</span>
      </button>
      {open && <p style={styles.snippet}>{citation.text_snippet}</p>}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  card: {
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    overflow: "hidden",
  },
  toggle: {
    width: "100%",
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "10px 14px",
    background: "#f8fafc",
    border: "none",
    cursor: "pointer",
    textAlign: "left",
    fontSize: 13,
  },
  badge: {
    background: "#2563eb",
    color: "#fff",
    borderRadius: 4,
    padding: "1px 6px",
    fontSize: 11,
    fontWeight: 700,
  },
  filename: { flex: 1, fontWeight: 500 },
  page: { color: "#64748b" },
  chevron: { fontSize: 10, color: "#94a3b8" },
  snippet: {
    padding: "10px 14px",
    fontSize: 13,
    color: "#475569",
    lineHeight: 1.6,
    background: "#fff",
    borderTop: "1px solid #e2e8f0",
  },
};
