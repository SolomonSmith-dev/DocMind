from docmind.core.ingest import chunk_text


class TestChunking:
    def test_short_text_single_chunk(self):
        text = "This is a short sentence."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_splits_long_text(self):
        # Create text longer than chunk_size
        text = "word " * 200  # 1000 chars
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_overlap_exists(self):
        text = "A" * 50 + " " + "B" * 50 + " " + "C" * 50 + " " + "D" * 50
        chunks = chunk_text(text, chunk_size=60, chunk_overlap=20)
        # With overlap, some content should appear in adjacent chunks
        assert len(chunks) >= 2
        # Verify chunks cover the full text
        full = " ".join(chunks)
        assert "A" in full
        assert "D" in full

    def test_prefers_paragraph_boundaries(self):
        text = "First paragraph content here.\n\nSecond paragraph content here."
        chunks = chunk_text(text, chunk_size=40, chunk_overlap=10)
        # Should split at the paragraph boundary
        assert any("First paragraph" in c for c in chunks)
        assert any("Second paragraph" in c for c in chunks)

    def test_preserves_all_content(self):
        words = [f"word{i}" for i in range(50)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        # Every word should appear in at least one chunk
        for word in words:
            assert any(word in chunk for chunk in chunks), f"{word} missing from chunks"
