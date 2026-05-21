const BASE = "";

function getToken(): string | null {
  return sessionStorage.getItem("docmind_token");
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function login(username: string, password: string): Promise<string> {
  const res = await fetch(`${BASE}/api/auth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error("Invalid credentials");
  const data = await res.json();
  sessionStorage.setItem("docmind_token", data.access_token);
  return data.access_token;
}

export function logout(): void {
  sessionStorage.removeItem("docmind_token");
}

export interface DocumentInfo {
  document_id: string;
  filename: string;
  pages: number;
  chunks: number;
  uploaded_at: string;
}

export interface UploadResult {
  document_id: string;
  filename: string;
  pages: number;
  chunks: number;
  status: string;
}

export async function uploadDocument(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/upload`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Upload failed (${res.status})`);
  }
  return res.json();
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch(`${BASE}/api/documents`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error("Failed to load documents");
  return res.json();
}

export interface Citation {
  page_number: number;
  source_filename: string;
  text_snippet: string;
}

export interface AskResult {
  answer: string;
  citations: Citation[];
  model: string;
  chunks_used: number;
  confidence: number;
  low_confidence: boolean;
  session_id: string | null;
}

export async function askQuestion(
  question: string,
  sessionId: string | null,
): Promise<AskResult> {
  const res = await fetch(`${BASE}/api/ask`, {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}
