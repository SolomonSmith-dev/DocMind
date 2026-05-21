import { DragEvent, useRef, useState } from "react";
import { uploadDocument, UploadResult } from "../api";

interface Props {
  onUploaded: () => void;
}

export default function Upload({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  async function handle(file: File) {
    setError("");
    setResult(null);
    setProgress(0);
    // Simulate progress while request is in-flight
    const interval = setInterval(() => {
      setProgress((p) => (p !== null && p < 85 ? p + 15 : p));
    }, 300);
    try {
      const res = await uploadDocument(file);
      clearInterval(interval);
      setProgress(100);
      setResult(res);
      onUploaded();
      setTimeout(() => setProgress(null), 800);
    } catch (e) {
      clearInterval(interval);
      setProgress(null);
      setError(e instanceof Error ? e.message : "Upload failed");
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handle(file);
  }

  return (
    <section style={styles.section}>
      <h2 style={styles.heading}>Upload Document</h2>
      <div
        style={{ ...styles.zone, ...(dragging ? styles.zoneDrag : {}) }}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        aria-label="Upload area"
      >
        <span style={styles.zoneIcon}>📄</span>
        <p style={styles.zoneText}>Drag &amp; drop a PDF, DOCX, or TXT here</p>
        <p style={styles.zoneHint}>or click to browse</p>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          style={{ display: "none" }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handle(f); }}
        />
      </div>

      {progress !== null && (
        <div style={styles.progressWrap} aria-label="Upload progress">
          <div style={{ ...styles.progressBar, width: `${progress}%` }} />
        </div>
      )}

      {error && <p style={styles.error}>{error}</p>}

      {result && (
        <div style={styles.success}>
          Uploaded <strong>{result.filename}</strong> — {result.pages} page(s), {result.chunks} chunk(s).
        </div>
      )}
    </section>
  );
}

const styles: Record<string, React.CSSProperties> = {
  section: { marginBottom: "2rem" },
  heading: { fontSize: 18, fontWeight: 600, marginBottom: 12 },
  zone: {
    border: "2px dashed #cbd5e1",
    borderRadius: 10,
    padding: "2rem",
    textAlign: "center",
    cursor: "pointer",
    transition: "border-color 0.2s, background 0.2s",
    background: "#f8fafc",
  },
  zoneDrag: { borderColor: "#2563eb", background: "#eff6ff" },
  zoneIcon: { fontSize: 36 },
  zoneText: { marginTop: 8, fontWeight: 500 },
  zoneHint: { color: "#94a3b8", fontSize: 13, marginTop: 4 },
  progressWrap: {
    marginTop: 10,
    height: 6,
    background: "#e2e8f0",
    borderRadius: 3,
    overflow: "hidden",
  },
  progressBar: {
    height: "100%",
    background: "#2563eb",
    transition: "width 0.3s ease",
  },
  error: { color: "#ef4444", marginTop: 8, fontSize: 14 },
  success: {
    marginTop: 10,
    padding: "10px 14px",
    background: "#f0fdf4",
    border: "1px solid #bbf7d0",
    borderRadius: 6,
    color: "#15803d",
    fontSize: 14,
  },
};
