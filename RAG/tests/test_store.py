"""Tests for the ChromaDB vector store."""

import pytest
from unittest.mock import MagicMock

from one_ai_rag.chunker import DocChunk
from one_ai_rag.store import VectorStore, RetrievedChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_chunk(
    chunk_id: str = "abc123_000",
    content: str = "Test documentation content about deploying apps.",
    section: str = "oneke",
) -> DocChunk:
    return DocChunk(
        chunk_id=chunk_id,
        content=content,
        token_count=10,
        source_url="https://docs.opennebula.io/test",
        source_title="Test Page",
        section=section,
        section_label="Test Section",
        heading_hierarchy=["Test Page", "Deployment"],
        has_code=False,
        breadcrumb=["Docs", "Test"],
    )


def make_embedder(dim: int = 384):
    """Create a mock embedder that returns deterministic vectors."""
    embedder = MagicMock()
    embedder.dimension = dim
    embedder.embed_texts.side_effect = lambda texts: [
        [float(hash(t) % 100) / 100.0] * dim for t in texts
    ]
    embedder.embed_query.side_effect = lambda t: [float(hash(t) % 100) / 100.0] * dim
    return embedder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_ingest_and_count(self, tmp_path):
        """Ingest chunks and verify they're stored."""
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        embedder = make_embedder()
        chunks = [make_chunk(f"chunk_{i}", f"Content {i}") for i in range(5)]

        count = store.ingest(chunks, embedder)
        assert count == 5
        assert store.collection.count() == 5

    def test_ingest_deduplication(self, tmp_path):
        """Re-ingesting same chunks should not create duplicates."""
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        embedder = make_embedder()
        chunks = [make_chunk("chunk_0", "Content")]

        store.ingest(chunks, embedder)
        count = store.ingest(chunks, embedder)

        assert count == 0  # Nothing new ingested
        assert store.collection.count() == 1

    def test_ingest_replace(self, tmp_path):
        """With replace=True, existing data should be cleared."""
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        embedder = make_embedder()

        store.ingest([make_chunk("old_chunk", "Old content")], embedder)
        assert store.collection.count() == 1

        store.ingest(
            [make_chunk("new_chunk", "New content")],
            embedder,
            replace=True,
        )
        assert store.collection.count() == 1

    def test_query_returns_results(self, tmp_path):
        """Query should return relevant chunks."""
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        embedder = make_embedder()

        chunks = [
            make_chunk("chunk_0", "How to deploy WordPress on OneKE"),
            make_chunk("chunk_1", "Managing virtual networks in OpenNebula"),
            make_chunk("chunk_2", "Helm chart configuration for Kubernetes"),
        ]
        store.ingest(chunks, embedder)

        results = store.query("deploy app", embedder, top_k=2, score_threshold=None)
        assert len(results) <= 2
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_query_with_section_filter(self, tmp_path):
        """Section filter should limit results."""
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        embedder = make_embedder()

        chunks = [
            make_chunk("chunk_0", "OneKE content", section="oneke"),
            make_chunk("chunk_1", "API content", section="api"),
        ]
        store.ingest(chunks, embedder)

        results = store.query(
            "content", embedder, section_filter="oneke", score_threshold=None,
        )
        assert all(r.section == "oneke" for r in results)

    def test_query_with_context(self, tmp_path):
        """query_with_context should return a formatted string."""
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        embedder = make_embedder()

        chunks = [make_chunk("chunk_0", "Documentation about deploying applications.")]
        store.ingest(chunks, embedder)

        context = store.query_with_context("deploy", embedder)
        assert isinstance(context, str)
        assert len(context) > 0

    def test_get_stats_empty(self, tmp_path):
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        stats = store.get_stats()
        assert stats["total_chunks"] == 0

    def test_get_stats_populated(self, tmp_path):
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        embedder = make_embedder()

        chunks = [
            make_chunk("c0", "Content", section="oneke"),
            make_chunk("c1", "Content", section="api"),
            make_chunk("c2", "Content", section="oneke"),
        ]
        store.ingest(chunks, embedder)

        stats = store.get_stats()
        assert stats["total_chunks"] == 3
        assert stats["sections"]["oneke"] == 2
        assert stats["sections"]["api"] == 1

    def test_clear(self, tmp_path):
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        embedder = make_embedder()

        store.ingest([make_chunk("c0", "x")], embedder)
        assert store.collection.count() == 1

        store.clear()
        # After clearing, a new collection should be empty
        assert store.collection.count() == 0


class TestRetrievedChunk:
    def test_format_for_context(self):
        chunk = RetrievedChunk(
            chunk_id="test_001",
            content="This is documentation about deploying apps.",
            score=0.85,
            metadata={},
            source_url="https://docs.opennebula.io/deploy",
            section="oneke",
            heading_hierarchy="OneKE > Deploying Apps",
        )
        formatted = chunk.format_for_context()
        assert "https://docs.opennebula.io/deploy" in formatted
        assert "OneKE > Deploying Apps" in formatted
        assert "documentation about deploying" in formatted
