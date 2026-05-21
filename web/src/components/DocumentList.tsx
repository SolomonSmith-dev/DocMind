import { useEffect, useState } from "react";
import { DocumentInfo, listDocuments } from "../api";

interface Props {
  refreshKey: number;
}

export default function DocumentList({ refreshKey }: Props) {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    listDocuments()
      .then(setDocs)
      .catch(() => setError("Could not load documents."));
  }, [refreshKey]);

  if (error) return <p style={{ color: "#ef4444", fontSize: 14 }}>{error}</p>;
  if (docs.length === 0)
    return <p style={{ color: "#94a3b8", fontSize: 14 }}>No documents uploaded yet.</p>;

  return (
    <section style={styles.section}>
      <h2 style={styles.heading}>Documents ({docs.length})</h2>
      <ul style={styles.list}>
        {docs.map((d) => (
          <li key={d.document_id} style={styles.item}>
            <span style={styles.filename}>{d.filename}</span>
            <span style={styles.meta}>{d.pages} pages &middot; {d.chunks} chunks</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

const styles: Record<string, React.CSSProperties> = {
  section: { marginBottom: "2rem" },
  heading: { fontSize: 18, fontWeight: 600, marginBottom: 10 },
  list: { listStyle: "none", display: "flex", flexDirection: "column", gap: 6 },
  item: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "10px 14px",
    background: "#fff",
    border: "1px solid #e2e8f0",
    borderRadius: 8,
  },
  filename: { fontWeight: 500, fontSize: 14 },
  meta: { color: "#64748b", fontSize: 13 },
};
