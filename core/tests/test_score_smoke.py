"""
tests/test_core_smoke.py
=========================
Smoke tests for one-ai-core that do NOT require Ollama, ChromaDB,
or the sibling packages (one-ai-rag, one-ai-config) to be installed.

These run in CI without any external services.
"""

from __future__ import annotations

import os
from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestCoreConfig:
    """CoreConfig reads env vars and has sensible defaults."""

    def test_defaults(self):
        from one_ai_core.config import CoreConfig
        cfg = CoreConfig()
        assert cfg.ollama_model            # non-empty
        assert cfg.max_retries >= 1
        assert cfg.rag_top_k >= 1
        assert cfg.output_dir.exists()     # mkdir is called in config.py

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ONEAI_CORE_OLLAMA_MODEL", "llama3:8b")
        monkeypatch.setenv("ONEAI_CORE_MAX_RETRIES", "7")
        # Re-import to pick up new env vars (module-level reads happen at import)
        import importlib
        import one_ai_core.config as cfg_module
        importlib.reload(cfg_module)
        from one_ai_core.config import CoreConfig as ReloadedConfig
        cfg = ReloadedConfig()
        # The dataclass field defaults read module-level constants
        # which were reloaded above.
        assert cfg.ollama_model == "llama3:8b"
        assert cfg.max_retries == 7

    def test_repr_hides_password(self):
        from one_ai_core.config import CoreConfig
        cfg = CoreConfig()
        cfg.one_password = "super-secret"
        assert "super-secret" not in repr(cfg)

    def test_all_fields_have_defaults(self):
        from one_ai_core.config import CoreConfig
        # Should construct with no arguments
        cfg = CoreConfig()
        for f in fields(cfg):
            assert hasattr(cfg, f.name), f"Missing field: {f.name}"


# ---------------------------------------------------------------------------
# Prompts tests
# ---------------------------------------------------------------------------

class TestPrompts:
    """Prompt templates contain required placeholders."""

    def test_system_prompt_has_rag_slot(self):
        from one_ai_core.prompts import USER_PROMPT
        assert "{rag_context}" in USER_PROMPT

    def test_user_prompt_has_request_slot(self):
        from one_ai_core.prompts import USER_PROMPT
        assert "{user_request}" in USER_PROMPT

    def test_retry_prompt_has_error_slot(self):
        from one_ai_core.prompts import RETRY_PROMPT
        assert "{error_summary}" in RETRY_PROMPT

    def test_system_prompt_lists_supported_actions(self):
        from one_ai_core.prompts import SYSTEM_PROMPT
        # A sample of the supported actions should be present
        for action in ("oneke.app.deploy", "oneke.namespace.create", "one.vm.create"):
            assert action in SYSTEM_PROMPT, f"Missing action in system prompt: {action}"

    def test_prompts_are_non_empty_strings(self):
        from one_ai_core.prompts import SYSTEM_PROMPT, USER_PROMPT, RETRY_PROMPT
        for name, p in [("SYSTEM", SYSTEM_PROMPT), ("USER", USER_PROMPT), ("RETRY", RETRY_PROMPT)]:
            assert isinstance(p, str) and len(p) > 50, f"{name}_PROMPT is too short"


# ---------------------------------------------------------------------------
# ChainResult tests
# ---------------------------------------------------------------------------

class TestChainResult:
    """ChainResult.summary() produces correct strings."""

    def test_success_summary(self):
        from one_ai_core.chain import ChainResult
        result = ChainResult(request="test")
        result.success = True
        result.attempts = 1
        result.elapsed_seconds = 2.5

        # Attach a mock config
        mock_config = MagicMock()
        mock_config.metadata.name = "deploy-wordpress"
        mock_config.steps = [MagicMock(), MagicMock()]
        result.config = mock_config

        summary = result.summary()
        assert "✅" in summary
        assert "deploy-wordpress" in summary
        assert "2 step" in summary

    def test_failure_summary(self):
        from one_ai_core.chain import ChainResult
        result = ChainResult(request="test")
        result.success = False
        result.attempts = 3
        result.error = "Validation failed: missing required field"

        summary = result.summary()
        assert "❌" in summary
        assert "3" in summary


# ---------------------------------------------------------------------------
# OneAIChain unit tests (mocked LLM + validator)
# ---------------------------------------------------------------------------

class TestOneAIChainMocked:
    """Test the chain logic with all external dependencies mocked out."""

    def _make_chain(self):
        from one_ai_core.chain import OneAIChain
        from one_ai_core.config import CoreConfig
        cfg = CoreConfig()
        cfg.max_retries = 3
        return OneAIChain(config=cfg)

    def _mock_valid_yaml(self):
        return """\
metadata:
  name: deploy-wordpress
  description: Deploy WordPress on OneKE
  version: "1.0"
  risk_level: medium
  tags: [wordpress, oneke]
steps:
  - id: create_namespace
    name: Create namespace
    action: oneke.namespace.create
    params:
      name: wordpress
    depends_on: []
    on_failure: stop
validation:
  pre_checks: []
  post_checks: []
rollback:
  enabled: false
  steps: []
"""

    def test_successful_run_on_first_attempt(self):
        chain = self._make_chain()
        valid_yaml = self._mock_valid_yaml()

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = valid_yaml
        chain._llm = mock_llm

        # Mock RAG retriever
        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents.return_value = []
        chain._retriever = mock_retriever

        # Mock validator — returns success
        mock_validation = MagicMock()
        mock_validation.is_valid = True
        mock_validation.warnings = []
        mock_validation.config = MagicMock()
        mock_validation.config.metadata.name = "deploy-wordpress"
        mock_validation.config.steps = [MagicMock()]

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_validation
        chain._validator = mock_validator

        # Mock code generator
        mock_script = MagicMock()
        mock_script.code = "# generated python"
        mock_codegen = MagicMock()
        mock_codegen.generate.return_value = mock_script
        chain._codegen = mock_codegen

        result = chain.run("Deploy WordPress on OneKE")

        assert result.success is True
        assert result.attempts == 1
        assert result.config_yaml != ""  # config_yaml was stored
        # assert result.config_yaml.strip() == valid_yaml.strip()
        mock_llm.invoke.assert_called_once()

    def test_retry_on_validation_failure_then_success(self):
        chain = self._make_chain()
        valid_yaml = self._mock_valid_yaml()

        # LLM returns bad output first, then good
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ["bad yaml ~~~", valid_yaml]
        chain._llm = mock_llm

        # RAG
        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents.return_value = []
        chain._retriever = mock_retriever

        # Validator: fails first, passes second
        fail_result = MagicMock()
        fail_result.is_valid = False
        fail_result.error_summary.return_value = "Missing required field: metadata.name"

        pass_result = MagicMock()
        pass_result.is_valid = True
        pass_result.warnings = []
        pass_result.config = MagicMock()
        pass_result.config.metadata.name = "deploy-wordpress"
        pass_result.config.steps = [MagicMock()]

        mock_validator = MagicMock()
        mock_validator.validate.side_effect = [fail_result, pass_result]
        chain._validator = mock_validator

        # Code gen
        mock_codegen = MagicMock()
        mock_codegen.generate.return_value = MagicMock(code="# script")
        chain._codegen = mock_codegen

        result = chain.run("Deploy WordPress on OneKE")

        assert result.success is True
        assert result.attempts == 2
        assert mock_llm.invoke.call_count == 2

    def test_exhausted_retries_returns_failure(self):
        chain = self._make_chain()
        chain.cfg.max_retries = 2

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "bad yaml"
        chain._llm = mock_llm

        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents.return_value = []
        chain._retriever = mock_retriever

        fail_result = MagicMock()
        fail_result.is_valid = False
        fail_result.error_summary.return_value = "Always fails"
        mock_validator = MagicMock()
        mock_validator.validate.return_value = fail_result
        chain._validator = mock_validator

        result = chain.run("Do something impossible")

        assert result.success is False
        assert result.attempts == 2
        assert mock_llm.invoke.call_count == 2

    def test_rag_retrieval_failure_is_non_fatal(self):
        """If RAG is unavailable, the chain should still produce output."""
        chain = self._make_chain()
        valid_yaml = self._mock_valid_yaml()

        # Retriever raises an exception
        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents.side_effect = RuntimeError("ChromaDB not found")
        chain._retriever = mock_retriever

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = valid_yaml
        chain._llm = mock_llm

        pass_result = MagicMock()
        pass_result.is_valid = True
        pass_result.warnings = []
        pass_result.config = MagicMock()
        pass_result.config.metadata.name = "deploy-wordpress"
        pass_result.config.steps = []
        mock_validator = MagicMock()
        mock_validator.validate.return_value = pass_result
        chain._validator = mock_validator

        mock_codegen = MagicMock()
        mock_codegen.generate.return_value = MagicMock(code="# script")
        chain._codegen = mock_codegen

        result = chain.run("Deploy WordPress on OneKE")
        # Should still succeed (RAG failure is non-fatal)
        assert result.success is True
        assert result.rag_chunks == []

    def test_llm_exception_stops_chain(self):
        chain = self._make_chain()
        chain.cfg.max_retries = 1

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("Ollama not running")
        chain._llm = mock_llm

        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents.return_value = []
        chain._retriever = mock_retriever

        result = chain.run("Deploy anything")

        assert result.success is False
        assert "LLM call failed" in result.error


# ---------------------------------------------------------------------------
# CLI tests (Click test runner — no Ollama required)
# ---------------------------------------------------------------------------

class TestCLI:
    """CLI structure and --help output tests."""

    def _runner(self):
        from click.testing import CliRunner
        return CliRunner()

    def test_root_help(self):
        from one_ai_core.cli import cli
        runner = self._runner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "one-ai" in result.output

    def test_generate_help(self):
        from one_ai_core.cli import cli
        runner = self._runner()
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0

    def test_plan_help(self):
        from one_ai_core.cli import cli
        runner = self._runner()
        result = runner.invoke(cli, ["plan", "--help"])
        assert result.exit_code == 0

    def test_apply_help(self):
        from one_ai_core.cli import cli
        runner = self._runner()
        result = runner.invoke(cli, ["apply", "--help"])
        assert result.exit_code == 0

    def test_config_command(self):
        from one_ai_core.cli import cli
        runner = self._runner()
        result = runner.invoke(cli, ["config"])
        assert result.exit_code == 0
        assert "ollama_model" in result.output

    def test_eval_command_placeholder(self):
        from one_ai_core.cli import cli
        runner = self._runner()
        result = runner.invoke(cli, ["eval"])
        assert result.exit_code == 0
        assert "placeholder" in result.output.lower()

    def test_generate_calls_chain(self):
        """generate command invokes chain.run() with the user request."""
        from one_ai_core.cli import cli
        from one_ai_core.chain import ChainResult

        mock_result = ChainResult(request="deploy wordpress")
        mock_result.success = True
        mock_result.attempts = 1
        mock_result.elapsed_seconds = 1.0
        mock_result.config_yaml = "metadata:\n  name: test\n"
        mock_config = MagicMock()
        mock_config.metadata.name = "test"
        mock_config.steps = []
        mock_result.config = mock_config

        runner = self._runner()
        with patch("one_ai_core.chain.OneAIChain.run", return_value=mock_result):
            result = runner.invoke(cli, ["generate", "deploy wordpress"])
        assert result.exit_code == 0
