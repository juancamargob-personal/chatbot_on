"""Tests for the retriever module."""

import pytest
from unittest.mock import MagicMock, patch

from one_ai_rag.retriever import OneAIRetriever, LangChainOpenNebulaRetriever
from one_ai_rag.store import RetrievedChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_retrieved_chunk(
    chunk_id: str = "test_001",
    content: str = "Documentation about deploying apps on OneKE.",
    score: float = 0.85,
    section: str = "oneke",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        score=score,
        metadata={"section": section},
        source_url="https://docs.opennebula.io/test",
        section=section,
        heading_hierarchy="OneKE > Deployment",
    )


# ---------------------------------------------------------------------------
# OneAIRetriever tests
# ---------------------------------------------------------------------------

class TestOneAIRetriever:
    def test_retrieve_delegates_to_store(self):
        """Retrieve should call store.query and return results."""
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        expected = [make_retrieved_chunk("c1"), make_retrieved_chunk("c2")]
        mock_store.query.return_value = expected

        retriever = OneAIRetriever(
            embedder=mock_embedder,
            store=mock_store,
            top_k=5,
            rerank=False,
        )
        results = retriever.retrieve("deploy wordpress")

        assert len(results) == 2
        mock_store.query.assert_called_once()

    def test_retrieve_respects_top_k(self):
        """Should limit results to top_k."""
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        # Return more than top_k
        mock_store.query.return_value = [
            make_retrieved_chunk(f"c{i}") for i in range(10)
        ]

        retriever = OneAIRetriever(
            embedder=mock_embedder,
            store=mock_store,
            top_k=3,
            rerank=False,
        )
        results = retriever.retrieve("query")
        assert len(results) == 3

    def test_retrieve_with_section_filter(self):
        """Section filter should be passed to store."""
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_store.query.return_value = []

        retriever = OneAIRetriever(
            embedder=mock_embedder,
            store=mock_store,
            rerank=False,
        )
        retriever.retrieve("query", section_filter="oneke")

        _, kwargs = mock_store.query.call_args
        assert kwargs.get("section_filter") == "oneke"

    def test_retrieve_with_reranking(self):
        """With rerank=True, should fetch more and rerank."""
        mock_store = MagicMock()
        mock_embedder = MagicMock()

        chunks = [make_retrieved_chunk(f"c{i}", score=0.5 + i * 0.1) for i in range(9)]
        mock_store.query.return_value = chunks

        mock_reranker = MagicMock()
        # Reverse the order with reranking scores
        mock_reranker.predict.return_value = list(range(9, 0, -1))

        retriever = OneAIRetriever(
            embedder=mock_embedder,
            store=mock_store,
            top_k=3,
            rerank=True,
        )
        retriever._reranker = mock_reranker

        results = retriever.retrieve("query")

        # Fetch k * 3 = 9 from store
        _, kwargs = mock_store.query.call_args
        assert kwargs.get("top_k") == 9  # 3 * 3

        # Reranker was called
        mock_reranker.predict.assert_called_once()

        # Return top 3 after reranking
        assert len(results) == 3

    def test_get_context_returns_string(self):
        """get_context should return a formatted context string."""
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_store.query_with_context.return_value = "[Source: url] Content here"

        retriever = OneAIRetriever(
            embedder=mock_embedder,
            store=mock_store,
            rerank=False,
        )
        context = retriever.get_context("deploy apps")

        assert isinstance(context, str)
        assert "Content here" in context
        mock_store.query_with_context.assert_called_once()


# ---------------------------------------------------------------------------
# LangChain retriever tests
# ---------------------------------------------------------------------------

class TestLangChainRetriever:
    def test_returns_langchain_documents(self):
        """LangChain retriever should return Document objects."""
        mock_inner = MagicMock(spec=OneAIRetriever)
        mock_inner.retrieve.return_value = [
            make_retrieved_chunk("c1", "Content about OneKE deployments"),
        ]

        retriever = LangChainOpenNebulaRetriever(
            inner=mock_inner,
            top_k=5,
        )
        docs = retriever._get_relevant_documents("deploy app")

        assert len(docs) == 1
        assert docs[0].page_content == "Content about OneKE deployments"
        assert docs[0].metadata["source"] == "https://docs.opennebula.io/test"
        assert docs[0].metadata["section"] == "oneke"
        assert docs[0].metadata["score"] == 0.85

    def test_passes_section_filter(self):
        """Section filter from the retriever should be forwarded."""
        mock_inner = MagicMock(spec=OneAIRetriever)
        mock_inner.retrieve.return_value = []

        retriever = LangChainOpenNebulaRetriever(
            inner=mock_inner,
            section_filter="api",
            top_k=3,
        )
        retriever._get_relevant_documents("query")

        mock_inner.retrieve.assert_called_once_with(
            query="query",
            section_filter="api",
            top_k=3,
        )

    def test_empty_results(self):
        mock_inner = MagicMock(spec=OneAIRetriever)
        mock_inner.retrieve.return_value = []

        retriever = LangChainOpenNebulaRetriever(inner=mock_inner)
        docs = retriever._get_relevant_documents("unknown topic")
        assert docs == []
