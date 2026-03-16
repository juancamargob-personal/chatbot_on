"""
one_ai_core.config
==================
All runtime settings for the orchestrator.

Every value can be overridden with an environment variable that has the
``ONEAI_CORE_`` prefix, e.g.::

    export ONEAI_CORE_OLLAMA_MODEL=mistral:7b-instruct-v0.3-q4_K_M
    export ONEAI_CORE_MAX_RETRIES=5

This mirrors the ``ONEAI_RAG_`` convention used in one-ai-rag/config.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    """Read an env var with the ONEAI_CORE_ prefix."""
    return os.environ.get(f"ONEAI_CORE_{key}", default)


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = _env(key, str(default)).lower()
    return val in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# LLM / Ollama
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
"""Base URL of the local Ollama server."""

OLLAMA_MODEL: str = _env("OLLAMA_MODEL", "mistral:7b-instruct-v0.3-q4_K_M")
"""Ollama model tag to use for config generation."""

OLLAMA_TEMPERATURE: float = float(_env("OLLAMA_TEMPERATURE", "0.1"))
"""Low temperature → more deterministic YAML output."""

OLLAMA_TIMEOUT: int = _env_int("OLLAMA_TIMEOUT", 120)
"""Seconds to wait for a single Ollama completion."""

# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------

MAX_RETRIES: int = _env_int("MAX_RETRIES", 3)
"""How many times to retry LLM generation after a validation failure."""

# ---------------------------------------------------------------------------
# RAG integration
# ---------------------------------------------------------------------------

RAG_TOP_K: int = _env_int("RAG_TOP_K", 5)
"""Number of RAG chunks to inject into the prompt context."""

RAG_RERANK: bool = _env_bool("RAG_RERANK", True)
"""Whether to apply cross-encoder reranking on RAG results."""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
"""Absolute path to the repo root (chatbot_on/)."""

OUTPUT_DIR: Path = Path(_env("OUTPUT_DIR", str(_PROJECT_ROOT / "output")))
"""Where generated YAML configs and Python scripts are written."""

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# OpenNebula connection (passed through to generated scripts)
# ---------------------------------------------------------------------------

ONE_ENDPOINT: str = _env("ONE_ENDPOINT", "http://localhost:2633/RPC2")
ONE_USER: str = _env("ONE_USER", "oneadmin")
ONE_PASSWORD: str = _env("ONE_PASSWORD", "")

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

DRY_RUN_DEFAULT: bool = _env_bool("DRY_RUN_DEFAULT", False)
"""If True, `one-ai apply` defaults to --dry-run unless overridden."""

VERBOSE: bool = _env_bool("VERBOSE", False)
"""Enable verbose debug logging."""


# ---------------------------------------------------------------------------
# Convenience dataclass (for passing config around as a single object)
# ---------------------------------------------------------------------------

@dataclass
class CoreConfig:
    ollama_base_url: str = field(default_factory=lambda: OLLAMA_BASE_URL)
    ollama_model: str = field(default_factory=lambda: OLLAMA_MODEL)
    ollama_temperature: float = field(default_factory=lambda: OLLAMA_TEMPERATURE)
    ollama_timeout: int = field(default_factory=lambda: OLLAMA_TIMEOUT)

    max_retries: int = field(default_factory=lambda: MAX_RETRIES)

    rag_top_k: int = field(default_factory=lambda: RAG_TOP_K)
    rag_rerank: bool = field(default_factory=lambda: RAG_RERANK)

    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)

    one_endpoint: str = field(default_factory=lambda: ONE_ENDPOINT)
    one_user: str = field(default_factory=lambda: ONE_USER)
    one_password: str = field(default_factory=lambda: ONE_PASSWORD)

    dry_run_default: bool = field(default_factory=lambda: DRY_RUN_DEFAULT)
    verbose: bool = field(default_factory=lambda: VERBOSE)

    def __repr__(self) -> str:
        # Hide password from repr / logs
        return (
            f"CoreConfig(model={self.ollama_model!r}, "
            f"endpoint={self.one_endpoint!r}, "
            f"max_retries={self.max_retries}, "
            f"rag_top_k={self.rag_top_k})"
        )
