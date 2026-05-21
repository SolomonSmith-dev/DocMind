"""Tests for confidence scoring (Issue #7).

Covers:
- compute_confidence() unit behavior across edge cases
- _parse_llm_confidence() regex parsing
- AnswerResult carries confidence and low_confidence fields
- /api/ask response schema includes both fields
- low_confidence=True when no chunks are retrieved
- confidence is always in [0, 1]
"""

from unittest.mock import MagicMock, patch

import pytest

from docmind.core.chat import (
    _parse_llm_confidence,
    compute_confidence,
    generate_answer,
)
from docmind.core.retrieval import RetrievalResult


def _make_chunk(score: float) -> RetrievalResult:
    return RetrievalResult(
        text="sample text",
        page_number=1,
        score=score,
        doc_id="doc1",
        source_filename="test.pdf",
        chunk_index=0,
    )


class TestParseLlmConfidence:
    def test_standard_format(self):
        result = _parse_llm_confidence("The answer is X. Confidence: 8/10")
        assert result == pytest.approx(0.8)

    def test_lowercase(self):
        assert _parse_llm_confidence("confidence: 5/10") == pytest.approx(0.5)

    def test_out_of_10_format(self):
        assert _parse_llm_confidence("I'd say 7 out of 10.") == pytest.approx(0.7)

    def test_no_match_defaults_to_half(self):
        assert _parse_llm_confidence("I am very confident.") == pytest.approx(0.5)

    def test_empty_string_defaults_to_half(self):
        assert _parse_llm_confidence("") == pytest.approx(0.5)

    def test_ten_out_of_ten(self):
        assert _parse_llm_confidence("Confidence: 10/10") == pytest.approx(1.0)

    def test_zero_score(self):
        assert _parse_llm_confidence("Confidence: 0/10") == pytest.approx(0.0)


class TestComputeConfidence:
    def test_no_chunks_gives_low_score(self):
        # similarity=0, coverage=0, llm default=0.5
        # 0.5*0 + 0.3*0 + 0.2*0.5 = 0.1
        score = compute_confidence(chunks=[], top_k_requested=5, llm_response_text="")
        assert score == pytest.approx(0.1)

    def test_perfect_conditions(self):
        # similarity=1.0, coverage=5/5=1.0, llm=1.0
        # 0.5*1 + 0.3*1 + 0.2*1 = 1.0
        chunks = [_make_chunk(1.0)] * 5
        score = compute_confidence(
            chunks=chunks,
            top_k_requested=5,
            llm_response_text="Confidence: 10/10",
        )
        assert score == pytest.approx(1.0)

    def test_weights_sum_correctly(self):
        # similarity=0.8, coverage=3/5=0.6, llm=0.7
        # 0.5*0.8 + 0.3*0.6 + 0.2*0.7 = 0.4+0.18+0.14 = 0.72
        chunks = [_make_chunk(0.8)] * 3
        score = compute_confidence(
            chunks=chunks,
            top_k_requested=5,
            llm_response_text="Confidence: 7/10",
        )
        assert score == pytest.approx(0.72, abs=1e-4)

    def test_result_always_in_unit_interval(self):
        for sim in [0.0, 0.5, 1.0]:
            s = compute_confidence(
                chunks=[_make_chunk(sim)],
                top_k_requested=5,
                llm_response_text="Confidence: 5/10",
            )
            assert 0.0 <= s <= 1.0

    def test_top_similarity_uses_first_chunk_only(self):
        # chunks[0].score=0.9 is the top; chunks[1].score=0.1 is not used
        # 0.5*0.9 + 0.3*(2/5) + 0.2*0.5 = 0.45+0.12+0.1 = 0.67
        chunks = [_make_chunk(0.9), _make_chunk(0.1)]
        score = compute_confidence(
            chunks=chunks,
            top_k_requested=5,
            llm_response_text="",
        )
        assert score == pytest.approx(0.67, abs=1e-4)

    def test_below_default_threshold_when_no_chunks(self):
        score = compute_confidence(chunks=[], top_k_requested=5, llm_response_text="")
        assert score < 0.4


class TestGenerateAnswerConfidence:
    def test_no_chunks_result_has_low_confidence(self):
        result = generate_answer("test?", chunks=[], low_confidence_threshold=0.4)
        assert result.confidence == pytest.approx(0.1)
        assert result.low_confidence is True

    def test_confidence_fields_present(self):
        result = generate_answer("test?", chunks=[])
        assert hasattr(result, "confidence")
        assert hasattr(result, "low_confidence")

    def test_confidence_in_unit_interval_on_fallback(self):
        chunks = [_make_chunk(0.75)]
        result = generate_answer(
            "test question",
            chunks=chunks,
            ollama_base_url="http://localhost:99999",
            top_k_requested=5,
        )
        assert result.model == "fallback"
        assert 0.0 <= result.confidence <= 1.0

    @patch("docmind.core.chat.ollama_lib")
    def test_llm_confidence_parsed_from_response(self, mock_ollama):
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "content": "The revenue was $10M (page 5).\nConfidence: 8/10"
            }
        }
        mock_ollama.Client.return_value = mock_client

        chunks = [_make_chunk(0.9), _make_chunk(0.8)]
        result = generate_answer(
            "What was the revenue?",
            chunks=chunks,
            top_k_requested=5,
        )
        # 0.5*0.9 + 0.3*(2/5) + 0.2*0.8 = 0.45+0.12+0.16 = 0.73
        assert result.confidence == pytest.approx(0.73, abs=1e-4)
        assert result.low_confidence is False

    @patch("docmind.core.chat.ollama_lib")
    def test_confidence_annotation_stripped_from_answer(self, mock_ollama):
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "The answer is yes.\nConfidence: 7/10"}
        }
        mock_ollama.Client.return_value = mock_client

        result = generate_answer(
            "question?", chunks=[_make_chunk(0.8)], top_k_requested=5
        )
        assert "Confidence: 7/10" not in result.answer
        assert "The answer is yes." in result.answer

    @patch("docmind.core.chat.ollama_lib")
    def test_low_confidence_flag_set_when_below_threshold(self, mock_ollama):
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "I am not sure.\nConfidence: 1/10"}
        }
        mock_ollama.Client.return_value = mock_client

        chunks = [_make_chunk(0.1)]
        result = generate_answer(
            "obscure question",
            chunks=chunks,
            top_k_requested=10,
            low_confidence_threshold=0.4,
        )
        # 0.5*0.1 + 0.3*(1/10) + 0.2*0.1 = 0.05+0.03+0.02 = 0.10
        assert result.confidence < 0.4
        assert result.low_confidence is True


class TestAskEndpointConfidence:
    def test_response_has_confidence_fields(self, client, sample_pdf_bytes):
        client.post(
            "/api/upload",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
        )
        resp = client.post("/api/ask", json={"question": "What is this about?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "confidence" in data
        assert "low_confidence" in data

    def test_confidence_is_float_in_unit_interval(self, client, sample_pdf_bytes):
        client.post(
            "/api/upload",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
        )
        resp = client.post("/api/ask", json={"question": "What is this about?"})
        data = resp.json()
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_empty_collection_returns_low_confidence(self, client):
        client.app.state.settings.chroma_collection = "empty_confidence_test"
        resp = client.post("/api/ask", json={"question": "anything"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["low_confidence"] is True
        assert data["confidence"] < 0.4
