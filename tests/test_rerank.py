"""Tests for cross-encoder re-ranking (Issue #3).

Covers:
- rerank() returns results sorted by descending score
- rerank() respects top_k
- rerank() handles empty input
- Reranked chunks have non-decreasing relevance score order (descending)
- DOCMIND_ENABLE_RERANK env var gate: disabled by default
- Integration: /api/ask uses rerank when enabled
"""

from unittest.mock import MagicMock, patch

import pytest

from docmind.core.rerank import rerank
from docmind.core.retrieval import RetrievalResult


def _make_chunk(text: str, score: float = 0.5, page: int = 1) -> RetrievalResult:
    return RetrievalResult(
        text=text,
        page_number=page,
        score=score,
        doc_id="doc1",
        source_filename="test.pdf",
        chunk_index=0,
    )


class TestRerankFunction:
    @patch("docmind.core.rerank._get_cross_encoder")
    def test_returns_sorted_by_descending_score(self, mock_get_model):
        mock_model = MagicMock()
        # cross-encoder scores in reverse order of input
        mock_model.predict.return_value = [0.3, 0.9, 0.1, 0.7]
        mock_get_model.return_value = mock_model

        chunks = [
            _make_chunk("low relevance text"),
            _make_chunk("high relevance text"),
            _make_chunk("very low text"),
            _make_chunk("medium relevance"),
        ]
        result = rerank("test query", chunks, top_k=4)

        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True), "Scores must be non-increasing"

    @patch("docmind.core.rerank._get_cross_encoder")
    def test_respects_top_k(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.8, 0.2, 0.6, 0.9]
        mock_get_model.return_value = mock_model

        chunks = [_make_chunk(f"chunk {i}") for i in range(5)]
        result = rerank("query", chunks, top_k=3)

        assert len(result) == 3

    @patch("docmind.core.rerank._get_cross_encoder")
    def test_score_order_non_decreasing_relevance(self, mock_get_model):
        """Reranked chunks must have non-decreasing relevance (i.e., sorted desc)."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.8, 0.5, 0.95, 0.1]
        mock_get_model.return_value = mock_model

        chunks = [_make_chunk(f"doc {i}") for i in range(5)]
        result = rerank("query", chunks, top_k=5)

        for i in range(len(result) - 1):
            assert result[i].score >= result[i + 1].score

    def test_empty_input_returns_empty(self):
        result = rerank("query", [], top_k=5)
        assert result == []

    @patch("docmind.core.rerank._get_cross_encoder")
    def test_top_k_capped_at_input_size(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.8]
        mock_get_model.return_value = mock_model

        chunks = [_make_chunk("a"), _make_chunk("b")]
        result = rerank("query", chunks, top_k=10)

        assert len(result) == 2

    @patch("docmind.core.rerank._get_cross_encoder")
    def test_score_field_updated_with_cross_encoder_score(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [3.14]
        mock_get_model.return_value = mock_model

        chunks = [_make_chunk("some text", score=0.99)]
        result = rerank("query", chunks, top_k=1)

        assert result[0].score == pytest.approx(3.14)


class TestRerankGating:
    def test_rerank_disabled_by_default(self, client, sample_pdf_bytes):
        """DOCMIND_ENABLE_RERANK defaults to False; /api/ask must not call reranker."""
        assert client.app.state.settings.enable_rerank is False

        with patch("docmind.api.ask.rerank") as mock_rerank:
            client.post(
                "/api/upload",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )
            client.post("/api/ask", json={"question": "What is this about?"})
            mock_rerank.assert_not_called()

    def test_rerank_enabled_calls_reranker(self, client, sample_pdf_bytes):
        """When enabled, /api/ask must call rerank()."""
        client.app.state.settings.enable_rerank = True

        with patch("docmind.api.ask.rerank") as mock_rerank:
            mock_rerank.return_value = []
            client.post(
                "/api/upload",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )
            client.post("/api/ask", json={"question": "What is this about?"})
            mock_rerank.assert_called_once()

    def test_query_rerank_enabled_calls_reranker(self, client, sample_pdf_bytes):
        """When enabled, /api/query must also call rerank()."""
        client.app.state.settings.enable_rerank = True

        with patch("docmind.api.query.rerank") as mock_rerank:
            mock_rerank.return_value = []
            client.post(
                "/api/upload",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )
            client.post("/api/query", json={"question": "What is this about?"})
            mock_rerank.assert_called_once()
