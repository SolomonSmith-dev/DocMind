"""Embedding generation using sentence-transformers.

The embedding model converts text into dense vectors (lists of floats).
Two pieces of text that are semantically similar will have vectors that
are close together in this space — that's how semantic search works.

We use all-MiniLM-L6-v2: a small, fast model that produces 384-dimensional
vectors. It's good enough for document Q&A and runs on CPU without issues.

Key design decision: embeddings are generated at UPLOAD time, not query time.
This means uploads are slower (a few seconds per document), but queries are
instant because we only need to embed the short query string.
"""

import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()

# Module-level cache — model loads once, reused across requests
_model: SentenceTransformer | None = None


def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the embedding model, caching it for reuse.

    First call downloads the model (~80MB) and loads it into memory.
    Subsequent calls return the cached instance.
    """
    global _model
    if _model is None:
        logger.info("loading_embedding_model", model=model_name)
        _model = SentenceTransformer(model_name)
        logger.info("embedding_model_loaded", model=model_name)
    return _model


def generate_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> list[list[float]]:
    """Generate embedding vectors for a list of text chunks.

    Args:
        texts: List of text strings to embed.
        model_name: HuggingFace model identifier.

    Returns:
        List of embedding vectors (each is a list of floats).
        The length of each vector depends on the model (384 for MiniLM).
    """
    if not texts:
        return []

    model = get_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=False)

    logger.info(
        "embeddings_generated",
        num_texts=len(texts),
        vector_dim=len(embeddings[0]) if len(embeddings) > 0 else 0,
    )

    # Convert numpy arrays to plain lists for ChromaDB compatibility
    return [emb.tolist() for emb in embeddings]
