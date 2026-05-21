import { useState } from "react";
import { logout } from "./api";
import AskPanel from "./components/AskPanel";
import DocumentList from "./components/DocumentList";
import Login from "./components/Login";
import Upload from "./components/Upload";

export default function App() {
  const [authed, setAuthed] = useState(!!sessionStorage.getItem("docmind_token"));
  const [refreshKey, setRefreshKey] = useState(0);

  if (!authed) {
    return <Login onLogin={() => setAuthed(true)} />;
  }

  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <span style={styles.logo}>DocMind</span>
        <button
          style={styles.logoutBtn}
          onClick={() => { logout(); setAuthed(false); }}
        >
          Sign out
        </button>
      </header>

      <main style={styles.main}>
        <div style={styles.left}>
          <Upload onUploaded={() => setRefreshKey((k) => k + 1)} />
          <DocumentList refreshKey={refreshKey} />
        </div>
        <div style={styles.right}>
          <AskPanel />
        </div>
      </main>

      {/* Keyframe for spinner */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: { minHeight: "100vh", display: "flex", flexDirection: "column" },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "14px 24px",
    background: "#fff",
    borderBottom: "1px solid #e2e8f0",
    position: "sticky",
    top: 0,
    zIndex: 10,
  },
  logo: { fontSize: 20, fontWeight: 700 },
  logoutBtn: {
    padding: "6px 14px",
    background: "none",
    border: "1px solid #e2e8f0",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: 13,
    color: "#64748b",
  },
  main: {
    display: "flex",
    gap: "2rem",
    padding: "2rem 2rem",
    maxWidth: 1100,
    margin: "0 auto",
    width: "100%",
    flexWrap: "wrap",
  },
  left: { flex: "0 0 340px", minWidth: 280 },
  right: { flex: 1, minWidth: 320 },
};
