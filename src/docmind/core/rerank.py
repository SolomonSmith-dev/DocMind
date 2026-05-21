"""Cross-encoder re-ranking for improved retrieval precision.

The standard bi-encoder pipeline (sentence-transformers + ChromaDB cosine
similarity) retrieves fast but imprecisely -- it embeds query and document
independently and compares vectors. Cross-encoders see the query AND document
together in a single forward pass with cross-attention, which catches nuanced
relevance that bi-encoders miss.

The trade-off: cross-encoders cannot pre-compute document scores, so
re-ranking must happen at query time. We mitigate this by:
1. Retrieving a large initial candidate set (e.g., top-20) from ChromaDB
2. Re-scoring all candidates with the cross-encoder
3. Returning only the top-k final results

Model used: cross-encoder/ms-marco-MiniLM-L-6-v2 (small, fast, well-tested).
"""

import structlog

from docmind.core.retrieval import RetrievalResult

logger = structlog.get_logger()

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level model cache -- loaded once, reused across requests
_cross_encoder = None


def _get_cross_encoder():
    """Lazy-load the cross-encoder model (downloads on first call)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logger.info("cross_encoder_loading", model=_MODEL_NAME)
        _cross_encoder = CrossEncoder(_MODEL_NAME)
        logger.info("cross_encoder_loaded", model=_MODEL_NAME)
    return _cross_encoder


def rerank(
    query: str,
    chunks: list[RetrievalResult],
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Re-rank retrieved chunks using a cross-encoder model.

    Takes a list of candidate chunks (from bi-encoder retrieval), scores each
    pair of (query, chunk_text) with the cross-encoder, and returns the top-k
    results sorted by descending cross-encoder score.

    The returned RetrievalResult objects have their `score` field replaced with
    the cross-encoder score (logit, not normalized -- higher is more relevant).
    The relative ordering is what matters, not the absolute value.

    Returns at most top_k results (fewer if fewer candidates were provided).
    """
    if not chunks:
        return []

    model = _get_cross_encoder()

    pairs = [(query, chunk.text) for chunk in chunks]
    scores = model.predict(pairs)

    # Attach scores and sort descending
    scored = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, chunk in scored[:top_k]:
        reranked = RetrievalResult(
            text=chunk.text,
            page_number=chunk.page_number,
            score=float(score),
            doc_id=chunk.doc_id,
            source_filename=chunk.source_filename,
            chunk_index=chunk.chunk_index,
        )
        results.append(reranked)

    logger.info(
        "rerank_completed",
        query_length=len(query),
        candidates=len(chunks),
        returned=len(results),
        top_score=results[0].score if results else None,
    )

    return results
