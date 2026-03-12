"""
Embedding generator for documentation chunks.

Supports two backends:
* Local: sentence-transformers (free, private, no API calls)
* OpenAI: text-embedding-3-small (higher quality, costs money)

The embedder is used during ingestion (to populate ChromaDB) and can
also be used at query time if you need raw vectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from rich.console import Console
from rich.progress import track

from one_ai_rag.config import settings

console = Console()


class BaseEmbedder(ABC):
    """Abstract embedding interface."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...


class LocalEmbedder(BaseEmbedder):
    """
    Local embeddings using sentence-transformers.

    Default model: all-MiniLM-L6-v2 (384 dims, fast, decent quality).
    Runs entirely on CPU/GPU with no external API calls.
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        device: str = settings.embedding_device,
        batch_size: int = settings.embedding_batch_size,
    ):
        from sentence_transformers import SentenceTransformer

        console.print(f"Loading embedding model: [cyan]{model_name}[/cyan] on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # cosine similarity = dot product
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embeddings using text-embedding-3-small.

    Higher quality than local models but requires API key and costs money.
    Useful for comparison benchmarks or production deployments.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100,
    ):
        import os
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.batch_size = batch_size
        # text-embedding-3-small = 1536 dims by default
        self._dimension = 1536

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via OpenAI API."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query via OpenAI API."""
        response = self.client.embeddings.create(
            model=self.model,
            input=query,
        )
        return response.data[0].embedding

    @property
    def dimension(self) -> int:
        return self._dimension


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_embedder(backend: str = "local", **kwargs) -> BaseEmbedder:
    """
    Create an embedder instance.

    Args:
        backend: "local" for sentence-transformers, "openai" for OpenAI API
        **kwargs: Passed to the embedder constructor

    Returns:
        An embedder instance
    """
    if backend == "local":
        return LocalEmbedder(**kwargs)
    elif backend == "openai":
        return OpenAIEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}. Use 'local' or 'openai'.")
