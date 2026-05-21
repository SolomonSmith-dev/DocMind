import { useEffect, useRef, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

const buildUrl = (path) => `${API_BASE}${path}`

function App() {
  const [documents, setDocuments] = useState([])
  const [selectedDocument, setSelectedDocument] = useState('all')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [displayedAnswer, setDisplayedAnswer] = useState('')
  const [citations, setCitations] = useState([])
  const [model, setModel] = useState('')
  const [uploadProgress, setUploadProgress] = useState(null)
  const [uploadMessage, setUploadMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isAsking, setIsAsking] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const fileInputRef = useRef(null)

  const loadDocuments = async () => {
    try {
      const response = await fetch(buildUrl('/api/documents'))
      if (!response.ok) {
        throw new Error('Failed to load documents.')
      }
      const data = await response.json()
      setDocuments(data)
    } catch (error) {
      setErrorMessage(error.message)
    }
  }

  useEffect(() => {
    loadDocuments()
  }, [])

  useEffect(() => {
    if (!answer) {
      setDisplayedAnswer('')
      setIsStreaming(false)
      return
    }

    let index = 0
    setDisplayedAnswer('')
    setIsStreaming(true)
    const interval = setInterval(() => {
      index += 4
      setDisplayedAnswer(answer.slice(0, index))
      if (index >= answer.length) {
        clearInterval(interval)
        setIsStreaming(false)
      }
    }, 16)

    return () => clearInterval(interval)
  }, [answer])

  const startUpload = (file) => {
    if (!file) return
    if (file.type !== 'application/pdf') {
      setErrorMessage('Please select a PDF file to upload.')
      return
    }

    setErrorMessage('')
    setUploadMessage('')
    setUploadProgress(0)
    setIsUploading(true)

    const formData = new FormData()
    formData.append('file', file)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', buildUrl('/api/upload'))
    xhr.responseType = 'json'

    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) {
        setUploadProgress(Math.round((event.loaded / event.total) * 100))
      }
    })

    xhr.onload = () => {
      setIsUploading(false)
      if (xhr.status === 201) {
        setUploadMessage('Upload complete. Ready to query.')
        setUploadProgress(100)
        loadDocuments()
      } else {
        const detail =
          xhr.response?.detail || 'Upload failed. Please try again.'
        setErrorMessage(detail)
      }
    }

    xhr.onerror = () => {
      setIsUploading(false)
      setErrorMessage('Upload failed. Please try again.')
    }

    xhr.send(formData)
  }

  const handleFileChange = (event) => {
    const file = event.target.files?.[0]
    startUpload(file)
    event.target.value = ''
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    const file = event.dataTransfer.files?.[0]
    startUpload(file)
  }

  const handleAsk = async (event) => {
    event.preventDefault()
    const trimmed = question.trim()
    if (!trimmed) return

    setIsAsking(true)
    setErrorMessage('')
    setAnswer('')
    setDisplayedAnswer('')
    setCitations([])
    setModel('')

    try {
      const payload = { question: trimmed, top_k: 5 }
      if (selectedDocument !== 'all') {
        payload.document_id = selectedDocument
      }

      const response = await fetch(buildUrl('/api/ask'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error('Failed to get a response from DocMind.')
      }

      const data = await response.json()
      setAnswer(data.answer || '')
      setCitations(data.citations || [])
      setModel(data.model || '')
    } catch (error) {
      setErrorMessage(error.message)
    } finally {
      setIsAsking(false)
    }
  }

  const handleDelete = async (docId) => {
    try {
      setErrorMessage('')
      const response = await fetch(buildUrl(`/api/documents/${docId}`), {
        method: 'DELETE',
      })
      if (!response.ok) {
        throw new Error('Failed to delete document.')
      }
      await loadDocuments()
      if (selectedDocument === docId) {
        setSelectedDocument('all')
      }
    } catch (error) {
      setErrorMessage(error.message)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <p className="eyebrow">DocMind</p>
          <h1>Document Q&A for reviewers</h1>
          <p className="subtitle">
            Upload PDFs, ask questions, and review page-level citations in one
            place.
          </p>
        </div>
        <div className="status-card">
          <span className="status-label">API</span>
          <span className="status-value">http://localhost:8000</span>
        </div>
      </header>

      {(errorMessage || uploadMessage) && (
        <div className="banner" role="status">
          {errorMessage ? (
            <span className="error">{errorMessage}</span>
          ) : (
            <span className="success">{uploadMessage}</span>
          )}
        </div>
      )}

      <div className="grid">
        <section className="panel">
          <div className="panel-header">
            <h2>Upload PDF</h2>
            <button
              type="button"
              className="button button-secondary"
              onClick={() => fileInputRef.current?.click()}
            >
              Choose file
            </button>
          </div>
          <div
            className={`dropzone ${isDragging ? 'active' : ''}`}
            onDragOver={(event) => {
              event.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <p>Drag & drop a PDF here, or click to browse.</p>
            <p className="muted">PDF only • max 20MB</p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            hidden
          />
          {uploadProgress !== null && (
            <div className="progress">
              <progress value={uploadProgress} max="100" />
              <span>{isUploading ? 'Uploading…' : `${uploadProgress}%`}</span>
            </div>
          )}
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Documents</h2>
            <span className="muted">{documents.length} total</span>
          </div>
          {documents.length === 0 ? (
            <p className="muted">No documents uploaded yet.</p>
          ) : (
            <ul className="document-list">
              {documents.map((doc) => (
                <li key={doc.document_id} className="document-item">
                  <div>
                    <p className="document-title">{doc.filename}</p>
                    <p className="muted">
                      {doc.pages} pages • {doc.chunks} chunks
                    </p>
                  </div>
                  <button
                    type="button"
                    className="button button-danger"
                    onClick={() => handleDelete(doc.document_id)}
                  >
                    Delete
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <section className="panel ask-panel">
        <div className="panel-header">
          <h2>Ask a question</h2>
          <span className="muted">
            {documents.length ? 'Ready to query' : 'Upload a document first'}
          </span>
        </div>
        <form className="ask-form" onSubmit={handleAsk}>
          <label>
            Question
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="What are the key findings on page 4?"
              rows={3}
              required
            />
          </label>
          <label>
            Filter by document
            <select
              value={selectedDocument}
              onChange={(event) => setSelectedDocument(event.target.value)}
            >
              <option value="all">All documents</option>
              {documents.map((doc) => (
                <option key={doc.document_id} value={doc.document_id}>
                  {doc.filename}
                </option>
              ))}
            </select>
          </label>
          <button type="submit" className="button" disabled={isAsking}>
            {isAsking ? 'Asking…' : 'Ask DocMind'}
          </button>
        </form>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Answer</h2>
          <span className="muted">
            {model ? `Model: ${model}` : 'Responses stream in as they arrive'}
          </span>
        </div>
        <div className="answer">
          {displayedAnswer || (
            <span className="muted">
              Ask a question to see a cited response.
            </span>
          )}
          {isStreaming && <span className="cursor">▍</span>}
        </div>
        {citations.length > 0 && (
          <div className="citations">
            <h3>Citations</h3>
            {citations.map((citation, index) => (
              <details key={`${citation.page_number}-${index}`}>
                <summary>
                  Page {citation.page_number} • {citation.source_filename}
                </summary>
                <p>{citation.text_snippet}</p>
              </details>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

export default App
