"""ChromaDB vector storage and retrieval.

ChromaDB is our vector database. It stores document chunks alongside
their embedding vectors, then lets us find the most similar chunks
to a user's question using cosine similarity.

The flow:
1. Upload: chunks + embeddings → ChromaDB collection
2. Query: embed the question → find top-k similar chunks → return with scores

We use ChromaDB in client/server mode (separate Docker container) in
production, but fall back to in-process ephemeral mode for testing.
"""

from dataclasses import dataclass

import chromadb
import structlog

from docmind.core.embedding import generate_embeddings
from docmind.core.ingest import Chunk

logger = structlog.get_logger()

# Module-level client — initialized once
_client: chromadb.ClientAPI | None = None


def get_client(
    host: str = "localhost",
    port: int = 8000,
) -> chromadb.ClientAPI:
    """Get or create the ChromaDB client.

    In production (Docker), connects to the ChromaDB service.
    Falls back to ephemeral (in-memory) if connection fails,
    which is also what tests use.
    """
    global _client
    if _client is None:
        try:
            _client = chromadb.HttpClient(host=host, port=port)
            _client.heartbeat()  # verify connection
            logger.info("chromadb_connected", host=host, port=port)
        except Exception:
            logger.warning(
                "chromadb_http_failed_using_ephemeral",
                host=host,
                port=port,
            )
            _client = chromadb.EphemeralClient()
    return _client


def reset_client() -> None:
    """Reset the client. Used in tests to ensure isolation."""
    global _client
    _client = None


def get_collection(
    name: str = "docmind",
    host: str = "localhost",
    port: int = 8000,
) -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    client = get_client(host=host, port=port)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(
    chunks: list[Chunk],
    collection_name: str = "docmind",
    host: str = "localhost",
    port: int = 8000,
) -> int:
    """Generate embeddings for chunks and store them in ChromaDB.

    Each chunk is stored with:
    - id: unique identifier (doc_id + chunk_index)
    - embedding: dense vector from sentence-transformers
    - document: the chunk text
    - metadata: doc_id, page_number, source_filename, chunk_index
    """
    if not chunks:
        return 0

    collection = get_collection(collection_name, host, port)

    texts = [c.text for c in chunks]
    embeddings = generate_embeddings(texts)

    ids = [f"{c.doc_id}_{c.chunk_index}" for c in chunks]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "source_filename": c.source_filename,
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    logger.info(
        "chunks_stored",
        collection=collection_name,
        count=len(chunks),
        doc_id=chunks[0].doc_id,
    )

    return len(chunks)


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its similarity score."""

    text: str
    page_number: int
    score: float
    doc_id: str
    source_filename: str
    chunk_index: int


def retrieve_chunks(
    query: str,
    collection_name: str = "docmind",
    top_k: int = 5,
    doc_id: str | None = None,
    host: str = "localhost",
    port: int = 8000,
) -> list[RetrievalResult]:
    """Find the top-k most relevant chunks for a query.

    Embeds the query text, then searches ChromaDB using cosine similarity.
    Optionally filters by doc_id to search within a specific document.

    Returns results sorted by relevance (highest score first).
    """
    collection = get_collection(collection_name, host, port)

    query_embedding = generate_embeddings([query])

    where_filter = {"doc_id": doc_id} if doc_id else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    if results["documents"] and results["documents"][0]:
        for i, text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            # ChromaDB returns distances; for cosine, similarity = 1 - distance
            distance = results["distances"][0][i]
            score = round(1.0 - distance, 4)

            retrieved.append(
                RetrievalResult(
                    text=text,
                    page_number=meta["page_number"],
                    score=score,
                    doc_id=meta["doc_id"],
                    source_filename=meta["source_filename"],
                    chunk_index=meta["chunk_index"],
                )
            )

    logger.info(
        "chunks_retrieved",
        query_length=len(query),
        results=len(retrieved),
        top_score=retrieved[0].score if retrieved else None,
    )

    return retrieved
