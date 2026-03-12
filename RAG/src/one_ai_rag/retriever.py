"""
LangChain-compatible retriever for OpenNebula documentation.

Wraps the VectorStore in a LangChain BaseRetriever so it plugs
directly into LangChain chains and agents.  Also provides a
standalone retriever class for use without LangChain.

Usage with LangChain:
    retriever = create_langchain_retriever()
    docs = retriever.invoke("How to deploy Helm charts on OneKE")

Usage standalone:
    retriever = OneAIRetriever()
    context = retriever.get_context("How to deploy Helm charts on OneKE")
"""

from __future__ import annotations

from typing import Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field as LCField

from one_ai_rag.config import settings
from one_ai_rag.embedder import BaseEmbedder, create_embedder
from one_ai_rag.store import RetrievedChunk, VectorStore


# ---------------------------------------------------------------------------
# Standalone retriever (no LangChain dependency for core usage)
# ---------------------------------------------------------------------------

class OneAIRetriever:
    """
    High-level retriever that combines embedding + vector store + optional reranking.

    This is the primary interface for retrieving documentation context.
    It can be used independently of LangChain.

    Usage:
        retriever = OneAIRetriever()
        context = retriever.get_context("Deploy WordPress on OneKE")
        chunks = retriever.retrieve("Deploy WordPress on OneKE")
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        store: Optional[VectorStore] = None,
        top_k: int = settings.retrieval_top_k,
        rerank: bool = settings.rerank,
    ):
        self.embedder = embedder or create_embedder("local")
        self.store = store or VectorStore()
        self.top_k = top_k
        self.rerank = rerank
        self._reranker = None

    def retrieve(
        self,
        query: str,
        section_filter: Optional[str] = None,
        code_only: bool = False,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant documentation chunks.

        Args:
            query: Natural language query
            section_filter: Limit to a specific doc section (e.g. "oneke")
            code_only: Only return chunks containing code examples
            top_k: Override default number of results

        Returns:
            List of RetrievedChunk sorted by relevance
        """
        k = top_k or self.top_k
        # Retrieve more if reranking (then trim after)
        fetch_k = k * 3 if self.rerank else k

        chunks = self.store.query(
            query_text=query,
            embedder=self.embedder,
            top_k=fetch_k,
            section_filter=section_filter,
            code_only=code_only,
        )

        if self.rerank and chunks:
            chunks = self._rerank_chunks(query, chunks, top_k=k)

        return chunks[:k]

    def get_context(
        self,
        query: str,
        section_filter: Optional[str] = None,
        max_tokens: int = 3000,
    ) -> str:
        """
        Retrieve and format documentation context for LLM injection.

        Returns a ready-to-use context string with source attributions.
        """
        return self.store.query_with_context(
            query_text=query,
            embedder=self.embedder,
            section_filter=section_filter,
            max_context_tokens=max_tokens,
        )

    def _rerank_chunks(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Rerank chunks using a cross-encoder model."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(settings.rerank_model)

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self._reranker.predict(pairs)

        # Attach rerank scores and sort
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_k]


# ---------------------------------------------------------------------------
# LangChain-compatible retriever
# ---------------------------------------------------------------------------

class LangChainOpenNebulaRetriever(BaseRetriever):
    """
    LangChain BaseRetriever wrapping the OneAI vector store.

    Plugs into LangChain chains:
        retriever = LangChainOpenNebulaRetriever(inner=OneAIRetriever())
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    """

    inner: OneAIRetriever
    section_filter: Optional[str] = None
    top_k: int = 6

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        """Retrieve documents (LangChain interface)."""
        chunks = self.inner.retrieve(
            query=query,
            section_filter=self.section_filter,
            top_k=self.top_k,
        )

        docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.content,
                metadata={
                    "source": chunk.source_url,
                    "section": chunk.section,
                    "heading_hierarchy": chunk.heading_hierarchy,
                    "score": chunk.score,
                    "chunk_id": chunk.chunk_id,
                },
            )
            docs.append(doc)

        return docs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_langchain_retriever(
    embedding_backend: str = "local",
    section_filter: Optional[str] = None,
    top_k: int = settings.retrieval_top_k,
    rerank: bool = settings.rerank,
    **embedder_kwargs,
) -> LangChainOpenNebulaRetriever:
    """
    Create a ready-to-use LangChain retriever.

    Args:
        embedding_backend: "local" or "openai"
        section_filter: Limit retrieval to a doc section
        top_k: Number of documents to retrieve
        rerank: Whether to apply cross-encoder reranking
        **embedder_kwargs: Passed to the embedder constructor

    Returns:
        LangChain-compatible retriever
    """
    embedder = create_embedder(embedding_backend, **embedder_kwargs)
    store = VectorStore()
    inner = OneAIRetriever(
        embedder=embedder,
        store=store,
        top_k=top_k,
        rerank=rerank,
    )
    return LangChainOpenNebulaRetriever(
        inner=inner,
        section_filter=section_filter,
        top_k=top_k,
    )
