"""
Configuration for the RAG pipeline.

Centralises all tuneable parameters: doc sources, chunking strategy,
embedding model, ChromaDB paths, and retrieval settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Paths (relative to repository root)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]  # one-ai-rag/
DATA_DIR = _REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTORDB_DIR = DATA_DIR / "vectordb"


# ---------------------------------------------------------------------------
# Documentation sources
# ---------------------------------------------------------------------------

# OpenNebula documentation hierarchy to scrape.
# Updated for docs.opennebula.io v7.0 (Hugo/Docsy theme).
DOC_SOURCES: list[dict] = [
    {
        "base_url": "https://docs.opennebula.io/7.0/getting_started/try_opennebula/try_kubernetes_on_opennebula/",
        "section": "oneke_quickstart",
        "label": "OneKE Quick Start",
        "max_depth": 3,
    },
    {
        "base_url": "https://docs.opennebula.io/7.0/integrations/marketplace_appliances/oneke/",
        "section": "oneke",
        "label": "OneKE Service (Kubernetes)",
        "max_depth": 3,
    },
    {
        "base_url": "https://docs.opennebula.io/7.0/product/cluster_configuration/",
        "section": "cluster_config",
        "label": "Cluster Configuration (VMs, Networking, Storage)",
        "max_depth": 3,
    },
    {
        "base_url": "https://docs.opennebula.io/7.0/product/virtual_machines_operation/",
        "section": "vm_operations",
        "label": "VM & Workload Operations",
        "max_depth": 3,
    },
    {
        "base_url": "https://docs.opennebula.io/7.0/product/integration_references/system_interfaces/",
        "section": "api",
        "label": "API & System Interfaces (pyone, XML-RPC)",
        "max_depth": 3,
    },
    {
        "base_url": "https://docs.opennebula.io/7.0/product/control_plane_configuration/",
        "section": "control_plane",
        "label": "Control Plane Configuration",
        "max_depth": 2,
    },
    {
        "base_url": "https://docs.opennebula.io/7.0/software/",
        "section": "software",
        "label": "Software & Installation",
        "max_depth": 2,
    },
]

# Additional standalone pages to always include.
EXTRA_PAGES: list[str] = [
    # pyone Python bindings
    "https://docs.opennebula.io/7.0/product/integration_references/system_interfaces/python/",
    # XML-RPC API reference
    "https://docs.opennebula.io/7.0/product/integration_references/system_interfaces/api/",
    # Virtual network management
    "https://docs.opennebula.io/7.0/product/cluster_configuration/networking_system/manage_vnets/",
    # OneKE appliance page
    "https://docs.opennebula.io/7.0/integrations/marketplace_appliances/oneke/",
    # Marketplace appliances (how to download/manage)
    "https://docs.opennebula.io/7.0/product/apps-marketplace/managing_marketplaces/marketapps/",
]


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

class RAGSettings(BaseSettings):
    """All tuneable RAG parameters in one place."""

    # ----- Scraping -----
    scrape_delay_seconds: float = Field(
        default=1.0,
        description="Polite delay between HTTP requests when scraping",
    )
    scrape_timeout_seconds: int = Field(default=30)
    scrape_user_agent: str = Field(
        default="OneAI-RAG-Bot/0.1 (OpenNebula AI Assistant; educational project)",
    )
    max_pages_per_section: int = Field(
        default=200,
        description="Safety cap on pages scraped per section",
    )

    # ----- Chunking -----
    chunk_size: int = Field(
        default=1000,
        description="Target chunk size in tokens (approximate)",
    )
    chunk_overlap: int = Field(
        default=150,
        description="Overlap between consecutive chunks in tokens",
    )
    min_chunk_size: int = Field(
        default=100,
        description="Discard chunks smaller than this (tokens)",
    )
    preserve_code_blocks: bool = Field(
        default=True,
        description="Keep code blocks intact even if they exceed chunk_size",
    )

    # ----- Embeddings -----
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model for local embeddings",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for embedding model: 'cpu', 'cuda', 'mps'",
    )
    embedding_batch_size: int = Field(default=64)

    # ----- ChromaDB -----
    chroma_persist_dir: str = Field(
        default=str(VECTORDB_DIR),
        description="Directory where ChromaDB persists data",
    )
    chroma_collection_name: str = Field(default="opennebula_docs")

    # ----- Retrieval -----
    retrieval_top_k: int = Field(
        default=6,
        description="Number of chunks to retrieve per query",
    )
    retrieval_score_threshold: Optional[float] = Field(
        default=0.3,
        description="Minimum similarity score (cosine). None to disable.",
    )
    rerank: bool = Field(
        default=False,
        description="Whether to apply cross-encoder reranking (slower, more accurate)",
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    rerank_top_k: int = Field(
        default=4,
        description="Final number of chunks after reranking",
    )

    model_config = {"env_prefix": "ONEAI_RAG_"}


# Singleton for easy import
settings = RAGSettings()
