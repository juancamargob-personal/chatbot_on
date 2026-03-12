"""
one-ai-rag: RAG pipeline for OpenNebula documentation.

Quickstart:
    from one_ai_rag import OneAIRetriever

    retriever = OneAIRetriever()
    context = retriever.get_context("How to deploy an app on OneKE")
    chunks  = retriever.retrieve("Helm chart deployment")

Full pipeline:
    from one_ai_rag import DocScraper, DocChunker, VectorStore, create_embedder

    # 1. Scrape docs
    scraper = DocScraper()
    pages = scraper.scrape_all()
    scraper.save(pages)

    # 2. Chunk
    chunker = DocChunker()
    chunks = chunker.chunk_pages(pages)
    chunker.save(chunks)

    # 3. Embed + ingest
    embedder = create_embedder("local")
    store = VectorStore()
    store.ingest(chunks, embedder)

    # 4. Query
    retriever = OneAIRetriever(embedder=embedder, store=store)
    results = retriever.retrieve("deploy WordPress")
"""

from one_ai_rag.chunker import DocChunk, DocChunker
from one_ai_rag.embedder import BaseEmbedder, LocalEmbedder, OpenAIEmbedder, create_embedder
from one_ai_rag.retriever import (
    LangChainOpenNebulaRetriever,
    OneAIRetriever,
    create_langchain_retriever,
)
from one_ai_rag.scraper import DocScraper, ScrapedPage
from one_ai_rag.store import RetrievedChunk, VectorStore

__all__ = [
    "DocScraper",
    "ScrapedPage",
    "DocChunker",
    "DocChunk",
    "BaseEmbedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "create_embedder",
    "VectorStore",
    "RetrievedChunk",
    "OneAIRetriever",
    "LangChainOpenNebulaRetriever",
    "create_langchain_retriever",
]
