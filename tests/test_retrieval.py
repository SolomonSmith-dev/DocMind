from docmind.core.embedding import generate_embeddings
from docmind.core.ingest import Chunk
from docmind.core.retrieval import (
    reset_client,
    retrieve_chunks,
    store_chunks,
)


class TestEmbeddings:
    def test_generate_embeddings_returns_vectors(self):
        texts = ["hello world", "test document"]
        embeddings = generate_embeddings(texts)
        assert len(embeddings) == 2
        # all-MiniLM-L6-v2 produces 384-dim vectors
        assert len(embeddings[0]) == 384

    def test_empty_input(self):
        assert generate_embeddings([]) == []

    def test_similar_texts_have_close_embeddings(self):
        texts = [
            "the cat sat on the mat",
            "a cat was sitting on a mat",
            "quantum physics equations",
        ]
        embeddings = generate_embeddings(texts)

        # Compute cosine similarity manually
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x**2 for x in a) ** 0.5
            norm_b = sum(x**2 for x in b) ** 0.5
            return dot / (norm_a * norm_b)

        # Similar sentences should be more similar to each other
        # than to the unrelated sentence
        sim_similar = cosine_sim(embeddings[0], embeddings[1])
        sim_different = cosine_sim(embeddings[0], embeddings[2])
        assert sim_similar > sim_different


class TestChromaDBStorage:
    def setup_method(self):
        reset_client()

    def test_store_and_retrieve(self):
        chunks = [
            Chunk(
                text="Python is a programming language",
                doc_id="test-doc",
                page_number=1,
                chunk_index=0,
                source_filename="test.pdf",
            ),
            Chunk(
                text="Machine learning uses neural networks",
                doc_id="test-doc",
                page_number=2,
                chunk_index=1,
                source_filename="test.pdf",
            ),
        ]

        stored = store_chunks(chunks, collection_name="test_collection")
        assert stored == 2

        results = retrieve_chunks(
            query="what programming language",
            collection_name="test_collection",
            top_k=2,
        )
        assert len(results) > 0
        # The Python chunk should be the top result
        assert "Python" in results[0].text

    def test_retrieve_with_doc_filter(self):
        chunks_a = [
            Chunk(
                text="Document A talks about cats",
                doc_id="doc-a",
                page_number=1,
                chunk_index=0,
                source_filename="a.pdf",
            ),
        ]
        chunks_b = [
            Chunk(
                text="Document B talks about dogs",
                doc_id="doc-b",
                page_number=1,
                chunk_index=0,
                source_filename="b.pdf",
            ),
        ]

        store_chunks(chunks_a, collection_name="filter_test")
        store_chunks(chunks_b, collection_name="filter_test")

        results = retrieve_chunks(
            query="animals",
            collection_name="filter_test",
            top_k=5,
            doc_id="doc-a",
        )
        # Should only return chunks from doc-a
        assert all(r.doc_id == "doc-a" for r in results)

    def test_retrieve_returns_page_numbers(self):
        chunks = [
            Chunk(
                text="Content on page five about AI",
                doc_id="test-doc",
                page_number=5,
                chunk_index=0,
                source_filename="test.pdf",
            ),
        ]

        store_chunks(chunks, collection_name="page_test")

        results = retrieve_chunks(
            query="artificial intelligence",
            collection_name="page_test",
        )
        assert results[0].page_number == 5


class TestUploadThenQuery:
    """Integration test: upload a PDF, then query it."""

    def test_full_flow(self, client, sample_pdf_bytes):
        # Upload
        resp = client.post(
            "/api/upload",
            files={
                "file": ("test.pdf", sample_pdf_bytes, "application/pdf")
            },
        )
        assert resp.status_code == 201

        # Query
        resp = client.post(
            "/api/query",
            json={"question": "brown fox", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "brown fox"
        assert len(data["results"]) > 0
        # Should find the chunk containing "brown fox"
        assert any("fox" in r["text"] for r in data["results"])

    def test_query_empty_collection(self, client):
        # Use a unique collection name so previous test data doesn't leak
        client.app.state.settings.chroma_collection = "empty_test"
        resp = client.post(
            "/api/query",
            json={"question": "anything"},
        )
        assert resp.status_code == 200
        assert resp.json()["results"] == []
