"""
ChromaDB vector store for documentation chunks.

Handles collection management, ingestion, and querying with
metadata-based filtering.  ChromaDB is used because it's simple,
local, persistent, and requires no external services.

Usage:
    store = VectorStore()
    store.ingest(chunks, embedder)
    results = store.query("How to deploy apps on OneKE", embedder)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from rich.console import Console

from one_ai_rag.chunker import DocChunk
from one_ai_rag.config import settings
from one_ai_rag.embedder import BaseEmbedder

console = Console()


# ---------------------------------------------------------------------------
# Query result model
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store with its similarity score."""
    chunk_id: str
    content: str
    score: float            # Cosine similarity (higher = more similar)
    metadata: dict
    source_url: str
    section: str
    heading_hierarchy: str

    def format_for_context(self) -> str:
        """Format this chunk for injection into the LLM prompt context."""
        header = f"[Source: {self.source_url}]"
        if self.heading_hierarchy:
            header += f" [{self.heading_hierarchy}]"
        return f"{header}\n{self.content}"


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class VectorStore:
    """
    ChromaDB-backed vector store for document chunks.

    The store persists to disk so you only need to ingest once.
    Subsequent runs load the existing collection.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = None

    @property
    def collection(self) -> chromadb.Collection:
        """Lazy-load the collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ---------------------------------------------------------------------------
    # Ingestion
    # ---------------------------------------------------------------------------

    def ingest(
        self,
        chunks: list[DocChunk],
        embedder: BaseEmbedder,
        batch_size: int = 100,
        replace: bool = False,
    ) -> int:
        """
        Ingest document chunks into the vector store.

        Args:
            chunks: List of DocChunk objects to ingest
            embedder: Embedder to generate vectors
            batch_size: Number of chunks per batch
            replace: If True, delete existing collection first

        Returns:
            Number of chunks ingested
        """
        if replace:
            console.print("[yellow]Replacing existing collection...[/yellow]")
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = None

        # Deduplicate against existing chunks
        existing_ids = set()
        if self.collection.count() > 0:
            try:
                existing = self.collection.get()
                existing_ids = set(existing["ids"])
            except Exception:
                pass

        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            console.print("[yellow]All chunks already in store, nothing to ingest.[/yellow]")
            return 0

        console.print(
            f"Ingesting {len(new_chunks)} new chunks "
            f"({len(chunks) - len(new_chunks)} already exist)..."
        )

        ingested = 0
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]

            texts = [c.content for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [c.to_metadata() for c in batch]

            # Generate embeddings
            embeddings = embedder.embed_texts(texts)

            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            ingested += len(batch)
            if len(new_chunks) > batch_size:
                console.print(f"  Ingested {ingested}/{len(new_chunks)}...")

        console.print(f"[green]Ingestion complete: {ingested} chunks added[/green]")
        console.print(f"  Total chunks in store: {self.collection.count()}")
        return ingested

    # ---------------------------------------------------------------------------
    # Querying
    # ---------------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        embedder: BaseEmbedder,
        top_k: int = settings.retrieval_top_k,
        section_filter: Optional[str] = None,
        code_only: bool = False,
        score_threshold: Optional[float] = settings.retrieval_score_threshold,
    ) -> list[RetrievedChunk]:
        """
        Query the vector store for relevant chunks.

        Args:
            query_text: The natural language query
            embedder: Embedder for the query vector
            top_k: Number of results to return
            section_filter: Only return chunks from this section (e.g. "oneke")
            code_only: Only return chunks that contain code blocks
            score_threshold: Minimum cosine similarity score

        Returns:
            List of RetrievedChunk objects, sorted by relevance
        """
        # Build metadata filter
        where_filter = self._build_filter(section_filter, code_only)

        # Embed query
        query_embedding = embedder.embed_query(query_text)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Parse results
        retrieved: list[RetrievedChunk] = []
        if results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                # ChromaDB returns distances, convert to similarity
                # For cosine space: similarity = 1 - distance
                distance = results["distances"][0][idx]
                similarity = 1 - distance

                if score_threshold is not None and similarity < score_threshold:
                    continue

                metadata = results["metadatas"][0][idx]
                chunk = RetrievedChunk(
                    chunk_id=results["ids"][0][idx],
                    content=results["documents"][0][idx],
                    score=similarity,
                    metadata=metadata,
                    source_url=metadata.get("source_url", ""),
                    section=metadata.get("section", ""),
                    heading_hierarchy=metadata.get("heading_hierarchy", ""),
                )
                retrieved.append(chunk)

        return retrieved

    def query_with_context(
        self,
        query_text: str,
        embedder: BaseEmbedder,
        top_k: int = settings.retrieval_top_k,
        section_filter: Optional[str] = None,
        max_context_tokens: int = 3000,
    ) -> str:
        """
        Query and return formatted context string ready for LLM injection.

        Concatenates retrieved chunks into a single context block,
        respecting a token budget.

        Args:
            query_text: Natural language query
            embedder: Embedder instance
            top_k: Max chunks to retrieve
            section_filter: Optional section filter
            max_context_tokens: Token budget for the context

        Returns:
            Formatted context string
        """
        from one_ai_rag.chunker import estimate_tokens

        chunks = self.query(
            query_text=query_text,
            embedder=embedder,
            top_k=top_k,
            section_filter=section_filter,
        )

        if not chunks:
            return "(No relevant documentation found.)"

        context_parts: list[str] = []
        total_tokens = 0

        for chunk in chunks:
            formatted = chunk.format_for_context()
            chunk_tokens = estimate_tokens(formatted)

            if total_tokens + chunk_tokens > max_context_tokens:
                break

            context_parts.append(formatted)
            total_tokens += chunk_tokens

        separator = "\n\n---\n\n"
        return separator.join(context_parts)

    # ---------------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------------

    def _build_filter(
        self, section: Optional[str], code_only: bool
    ) -> Optional[dict]:
        """Build a ChromaDB where filter."""
        conditions = []
        if section:
            conditions.append({"section": {"$eq": section}})
        if code_only:
            conditions.append({"has_code": {"$eq": True}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def get_stats(self) -> dict:
        """Return statistics about the vector store."""
        count = self.collection.count()
        if count == 0:
            return {"total_chunks": 0, "sections": {}}

        # Sample metadata to get section breakdown
        all_data = self.collection.get(include=["metadatas"])
        sections: dict[str, int] = {}
        code_chunks = 0
        for meta in all_data["metadatas"]:
            sec = meta.get("section", "unknown")
            sections[sec] = sections.get(sec, 0) + 1
            if meta.get("has_code"):
                code_chunks += 1

        return {
            "total_chunks": count,
            "sections": sections,
            "code_chunks": code_chunks,
        }

    def clear(self) -> None:
        """Delete all data from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None
        console.print("[yellow]Vector store cleared[/yellow]")
