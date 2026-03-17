"""
one_ai_core.config
==================
All runtime settings for the orchestrator.

Every value can be overridden with an environment variable that has the
``ONEAI_CORE_`` prefix, e.g.::

    export ONEAI_CORE_OLLAMA_MODEL=mistral:7b-instruct-v0.3-q4_K_M
    export ONEAI_CORE_MAX_RETRIES=5
    export ONEAI_CORE_LLM_BACKEND=openai
    export OPENAI_API_KEY=sk-...

This mirrors the ``ONEAI_RAG_`` convention used in one-ai-rag/config.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Env-var helpers
# ---------------------------------------------------------------------------

def _env(key: str, default: str) -> str:
    return os.environ.get(f"ONEAI_CORE_{key}", default)

def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))

def _env_bool(key: str, default: bool) -> bool:
    return _env(key, str(default)).lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str   = _env("OLLAMA_BASE_URL",   "http://localhost:11434")
OLLAMA_MODEL: str      = _env("OLLAMA_MODEL",       "mistral:7b-instruct-v0.3-q4_K_M")
OLLAMA_TEMPERATURE: float = float(_env("OLLAMA_TEMPERATURE", "0.1"))
OLLAMA_TIMEOUT: int    = _env_int("OLLAMA_TIMEOUT", 120)

# ---------------------------------------------------------------------------
# LLM backend selector  ('ollama' | 'openai')
# ---------------------------------------------------------------------------

LLM_BACKEND: str = _env("LLM_BACKEND", "ollama")

# ---------------------------------------------------------------------------
# OpenAI  (used when LLM_BACKEND=openai)
# ---------------------------------------------------------------------------

# Reads the standard OPENAI_API_KEY first, then the prefixed variant.
OPENAI_API_KEY: str       = os.environ.get("OPENAI_API_KEY", _env("OPENAI_API_KEY", ""))
OPENAI_MODEL: str         = _env("OPENAI_MODEL",       "gpt-4o")
OPENAI_TEMPERATURE: float = float(_env("OPENAI_TEMPERATURE", "0.1"))
OPENAI_TIMEOUT: int       = _env_int("OPENAI_TIMEOUT", 60)

# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------

MAX_RETRIES: int = _env_int("MAX_RETRIES", 3)

# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

RAG_TOP_K: int   = _env_int("RAG_TOP_K",   5)
RAG_RERANK: bool = _env_bool("RAG_RERANK", True)
RAG_EMBEDDING_MODEL: str = _env(
    "RAG_EMBEDDING_MODEL",
    # Must match whatever model was used to build the ChromaDB collection.
    # The handoff doc confirms all-mpnet-base-v2 (768 dims) is the correct model.
    # Override with ONEAI_CORE_RAG_EMBEDDING_MODEL if you used a different one.
    "sentence-transformers/all-mpnet-base-v2",
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def _default_output_dir() -> Path:
    """
    Walk up from this file to find the repo root, then return <root>/output.
    Falls back to ~/one-ai-output if the repo root can't be determined.
    This is done in a function (not at module level) so a bad path never
    crashes the module before CoreConfig is defined.
    """
    try:
        # config.py lives at <repo>/core/src/one_ai_core/config.py
        # parents: [one_ai_core/, src/, core/, <repo-root>]
        repo_root = Path(__file__).resolve().parents[3]
        out = repo_root / "output"
        out.mkdir(parents=True, exist_ok=True)
        return out
    except Exception:
        fallback = Path.home() / "one-ai-output"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


OUTPUT_DIR: Path = Path(_env("OUTPUT_DIR", "")) or _default_output_dir()

# ---------------------------------------------------------------------------
# OpenNebula connection
# ---------------------------------------------------------------------------

ONE_ENDPOINT: str = _env("ONE_ENDPOINT", "http://localhost:2633/RPC2")
ONE_USER: str     = _env("ONE_USER",     "oneadmin")
ONE_PASSWORD: str = _env("ONE_PASSWORD", "")

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

DRY_RUN_DEFAULT: bool = _env_bool("DRY_RUN_DEFAULT", False)
VERBOSE: bool         = _env_bool("VERBOSE",          False)


# ---------------------------------------------------------------------------
# CoreConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class CoreConfig:
    # Ollama
    ollama_base_url: str   = field(default_factory=lambda: OLLAMA_BASE_URL)
    ollama_model: str      = field(default_factory=lambda: OLLAMA_MODEL)
    ollama_temperature: float = field(default_factory=lambda: OLLAMA_TEMPERATURE)
    ollama_timeout: int    = field(default_factory=lambda: OLLAMA_TIMEOUT)

    # LLM backend
    llm_backend: str = field(default_factory=lambda: LLM_BACKEND)

    # OpenAI
    openai_api_key: str       = field(default_factory=lambda: OPENAI_API_KEY)
    openai_model: str         = field(default_factory=lambda: OPENAI_MODEL)
    openai_temperature: float = field(default_factory=lambda: OPENAI_TEMPERATURE)
    openai_timeout: int       = field(default_factory=lambda: OPENAI_TIMEOUT)

    # Retry
    max_retries: int = field(default_factory=lambda: MAX_RETRIES)

    # RAG
    rag_top_k: int            = field(default_factory=lambda: RAG_TOP_K)
    rag_rerank: bool          = field(default_factory=lambda: RAG_RERANK)
    rag_embedding_model: str  = field(default_factory=lambda: RAG_EMBEDDING_MODEL)

    # Paths
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)

    # OpenNebula
    one_endpoint: str = field(default_factory=lambda: ONE_ENDPOINT)
    one_user: str     = field(default_factory=lambda: ONE_USER)
    one_password: str = field(default_factory=lambda: ONE_PASSWORD)

    # Flags
    dry_run_default: bool = field(default_factory=lambda: DRY_RUN_DEFAULT)
    verbose: bool         = field(default_factory=lambda: VERBOSE)

    # ------------------------------------------------------------------

    @property
    def active_model(self) -> str:
        """Human-readable label for the active LLM."""
        if self.llm_backend == "openai":
            return f"openai/{self.openai_model}"
        return f"ollama/{self.ollama_model}"

    def __repr__(self) -> str:
        # Never log API keys or passwords
        return (
            f"CoreConfig(backend={self.llm_backend!r}, "
            f"model={self.active_model!r}, "
            f"endpoint={self.one_endpoint!r}, "
            f"max_retries={self.max_retries}, "
            f"rag_top_k={self.rag_top_k})"
        )
