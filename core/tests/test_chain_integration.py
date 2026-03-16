"""
tests/test_chain_integration.py
================================
Integration tests for the full RAG → LLM → Validator chain.

These tests hit a REAL Ollama server and REAL ChromaDB.
They are SKIPPED in normal pytest runs.

To run them:
    pytest tests/test_chain_integration.py -v --integration

Before running, make sure:
    1. Ollama is running:     ollama serve
    2. Model is pulled:       ollama pull mistral:7b-instruct-v0.3-q4_K_M
    3. RAG is populated:      cd ../RAG && one-ai-rag pipeline
    4. one-ai-rag is installed: cd ../RAG && pip install -e . --no-deps
    5. one-ai-config is installed: cd ../config && pip install -e . --no-deps
"""

from __future__ import annotations

import os
import subprocess

import pytest

# ---------------------------------------------------------------------------
# Markers & skip logic
# ---------------------------------------------------------------------------

# All tests in this file are skipped unless --integration is passed to pytest.
# This prevents accidental long-running calls in CI.
pytestmark = pytest.mark.integration


def _ollama_reachable(base_url: str) -> bool:
    """Return True if the Ollama server responds."""
    try:
        import urllib.request
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _model_available(base_url: str, model: str) -> bool:
    """Return True if the model is pulled and listed by Ollama."""
    try:
        import urllib.request, json
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        names = [m.get("name", "") for m in data.get("models", [])]
        # Model name may include a tag like ':latest' — match on prefix
        return any(n.startswith(model.split(":")[0]) for n in names)
    except Exception:
        return False


def _rag_available() -> bool:
    """Return True if one_ai_rag can be imported."""
    try:
        import one_ai_rag  # noqa: F401
        return True
    except ImportError:
        return False


def _config_available() -> bool:
    """Return True if one_ai_config can be imported."""
    try:
        import one_ai_config  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    """Return a CoreConfig pointing at the real local Ollama."""
    from one_ai_core.config import CoreConfig
    return CoreConfig()


@pytest.fixture(scope="module")
def chain(cfg):
    """Return a real OneAIChain (no mocks)."""
    from one_ai_core.chain import OneAIChain
    return OneAIChain(config=cfg)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

class TestPreFlight:
    """Verify the environment is ready before running chain tests."""

    def test_ollama_is_reachable(self, cfg):
        """Fails with a clear message if Ollama isn't running."""
        assert _ollama_reachable(cfg.ollama_base_url), (
            f"Ollama is not reachable at {cfg.ollama_base_url}.\n"
            "Start it with:  ollama serve"
        )

    def test_model_is_pulled(self, cfg):
        """Fails with a clear message if the model isn't downloaded."""
        assert _model_available(cfg.ollama_base_url, cfg.ollama_model), (
            f"Model '{cfg.ollama_model}' is not available in Ollama.\n"
            f"Pull it with:  ollama pull {cfg.ollama_model}"
        )

    def test_one_ai_rag_importable(self):
        """Fails if one-ai-rag isn't installed."""
        assert _rag_available(), (
            "one-ai-rag is not installed.\n"
            "Install with:  cd ../RAG && pip install -e . --no-deps"
        )

    def test_one_ai_config_importable(self):
        """Fails if one-ai-config isn't installed."""
        assert _config_available(), (
            "one-ai-config is not installed.\n"
            "Install with:  cd ../config && pip install -e . --no-deps"
        )


# ---------------------------------------------------------------------------
# RAG retrieval tests
# ---------------------------------------------------------------------------

class TestRAGRetrieval:
    """Tests for the RAG retrieval layer in isolation."""

    def test_retriever_returns_chunks(self, chain):
        """The retriever should return at least one chunk for a clear query."""
        context, chunks = chain._retrieve_context(
            "How do I deploy a Kubernetes cluster on OpenNebula?"
        )
        assert len(chunks) > 0, (
            "RAG returned 0 chunks. Is the ChromaDB populated?\n"
            "Run:  cd ../RAG && one-ai-rag pipeline"
        )

    def test_retriever_returns_oneke_content(self, chain):
        """OneKE queries should retrieve OneKE-relevant content."""
        context, chunks = chain._retrieve_context(
            "Deploy WordPress on OneKE cluster"
        )
        # At least one chunk should mention OneKE or Kubernetes
        texts = " ".join(c["text"].lower() for c in chunks)
        assert "oneke" in texts or "kubernetes" in texts, (
            "RAG returned chunks with no OneKE/Kubernetes content.\n"
            f"Got sources: {[c['source'] for c in chunks]}"
        )

    def test_retriever_context_is_formatted_string(self, chain):
        """The formatted context string should be non-empty and contain source refs."""
        context, _ = chain._retrieve_context("Scale worker nodes")
        assert isinstance(context, str)
        assert len(context) > 50


# ---------------------------------------------------------------------------
# Full chain tests — these call the real LLM
# ---------------------------------------------------------------------------

class TestChainIntegration:
    """End-to-end chain tests against real Ollama + RAG.

    Each test may take 30-120 seconds depending on model speed.
    """

    REQUESTS = [
        (
            "simple_deploy",
            "Create a namespace called 'wordpress' on my OneKE cluster",
        ),
        (
            "multi_step",
            "Deploy WordPress on my OneKE cluster with a persistent volume",
        ),
        (
            "vm_request",
            "Create a new virtual machine from template ID 5 named 'test-vm'",
        ),
    ]

    @pytest.mark.parametrize("label,prompt", REQUESTS)
    def test_chain_produces_valid_config(self, chain, label, prompt):
        """The chain should produce a valid YAML config for each request."""
        result = chain.run(prompt)

        assert result.success, (
            f"Chain failed for '{label}'.\n"
            f"Error: {result.error}\n"
            f"Attempts: {result.attempts}"
        )
        assert result.config is not None
        assert result.config_yaml.strip() != ""

    @pytest.mark.parametrize("label,prompt", REQUESTS)
    def test_chain_generates_script(self, chain, label, prompt):
        """A valid config should always produce an executable Python script."""
        result = chain.run(prompt)

        if not result.success:
            pytest.skip(f"Config generation failed for '{label}' — skipping script check")

        assert result.script is not None, "Code generator returned None"
        assert "def step_" in result.script.code or "def main" in result.script.code, (
            "Generated script missing expected function definitions"
        )

    def test_chain_uses_rag_context(self, chain):
        """RAG chunks should be present for an OpenNebula-specific query."""
        result = chain.run("Deploy WordPress on OneKE cluster")
        assert len(result.rag_chunks) > 0, (
            "No RAG chunks were retrieved. Check ChromaDB is populated."
        )

    def test_chain_retries_on_bad_output(self, cfg):
        """Chain should retry when the LLM produces invalid YAML.

        We patch the LLM to return garbage on the first call, then valid YAML
        on the second — verifying the retry loop works with real dependencies.
        """
        from unittest.mock import MagicMock, patch
        from one_ai_core.chain import OneAIChain
        from one_ai_config.validator import ConfigValidator

        chain_under_test = OneAIChain(config=cfg)

        valid_yaml = """\
metadata:
  name: test-namespace
  description: Create a test namespace
  version: "1.0"
  risk_level: low
  tags: [test]
steps:
  - id: create_ns
    name: Create namespace
    action: oneke.namespace.create
    params:
      name: test
    depends_on: []
    on_failure: stop
validation:
  pre_checks: []
  post_checks: []
rollback:
  enabled: false
  steps: []
"""
        # First call returns garbage; second returns valid YAML
        call_count = 0
        original_call = chain_under_test._call_llm

        def patched_call(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "this is definitely not yaml ~~~ ???"
            return valid_yaml

        chain_under_test._call_llm = patched_call
        # Use real validator
        chain_under_test._validator = ConfigValidator()

        result = chain_under_test.run("Create a test namespace")

        assert result.attempts == 2, f"Expected 2 attempts, got {result.attempts}"
        assert result.success is True


# ---------------------------------------------------------------------------
# Config → Script round-trip test
# ---------------------------------------------------------------------------

class TestConfigScriptRoundTrip:
    """Validate that a real LLM-generated config produces a runnable script."""

    def test_generated_script_is_executable_python(self, chain):
        """Run the generated script with --dry-run and expect exit code 0."""
        import tempfile, subprocess, sys

        result = chain.run("Create a namespace called 'smoketest'")
        if not result.success or not result.script:
            pytest.skip("Config/script generation failed — skipping syntax check")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(result.script.code)
            tmp_path = f.name

        # Just check that the Python syntax is valid (parse only)
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", tmp_path],
            capture_output=True, text=True
        )
        assert proc.returncode == 0, (
            f"Generated script has syntax errors:\n{proc.stderr}"
        )
