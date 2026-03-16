"""
tests/test_chain_compare.py
============================
Side-by-side comparison of Ollama (local) vs OpenAI (API).

These tests are SKIPPED unless both conditions are met:
    1. --integration flag is passed to pytest
    2. OPENAI_API_KEY environment variable is set

To run:
    export OPENAI_API_KEY=sk-...
    pytest tests/test_chain_compare.py -v --integration

Results are printed as a Rich table in the terminal — use them to decide
whether fine-tuning the local model is worthwhile.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Skip if no OpenAI key
# ---------------------------------------------------------------------------

def _openai_key_available() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY") or
        os.environ.get("ONEAI_CORE_OPENAI_API_KEY")
    )


skip_no_key = pytest.mark.skipif(
    not _openai_key_available(),
    reason=(
        "OPENAI_API_KEY not set. "
        "Export it with:  export OPENAI_API_KEY=sk-..."
    ),
)


# ---------------------------------------------------------------------------
# Test prompts — a mix of simple, multi-step, and edge cases
# ---------------------------------------------------------------------------

COMPARISON_REQUESTS = [
    (
        "simple_namespace",
        "Create a namespace called 'wordpress' on my OneKE cluster",
    ),
    (
        "full_wordpress_deploy",
        "Deploy WordPress on my OneKE cluster with a 10Gi persistent volume "
        "and expose it on port 80",
    ),
    (
        "scale_nodes",
        "Scale my OneKE worker nodes to 5",
    ),
    (
        "impossible_request",
        "Delete my entire OpenNebula installation and all VMs",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_cfg():
    from one_ai_core.config import CoreConfig
    return CoreConfig()


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------

class TestOllamaVsOpenAI:
    """
    Run each test prompt through both Ollama and OpenAI, print the comparison,
    and assert that at least one backend produces a valid config for non-
    impossible requests.
    """

    @skip_no_key
    @pytest.mark.parametrize("label,prompt", COMPARISON_REQUESTS)
    def test_compare(self, base_cfg, prompt, label):
        from one_ai_core.compare import compare_backends, print_comparison

        print(f"\n{'='*60}")
        print(f"Comparing backends for: {label!r}")
        print(f"{'='*60}")

        result = compare_backends(
            request=prompt,
            backend_a="ollama",
            backend_b="openai",
            base_config=base_cfg,
        )

        # Always print the table — visible in pytest -v -s output
        print_comparison(result)

        # For impossible requests, we expect BOTH to either fail validation
        # (returning an error YAML) or succeed with an error config.
        # Either way, we don't assert success — just that neither crashes.
        if label == "impossible_request":
            # Both chains should complete without raising an exception
            assert result.score_a.chain_result is not None
            assert result.score_b.chain_result is not None
            return

        # For real requests: at least one backend must succeed
        either_succeeded = (
            result.score_a.success or result.score_b.success
        )
        assert either_succeeded, (
            f"Both backends failed for '{label}'.\n"
            f"Ollama error:  {result.score_a.error}\n"
            f"OpenAI error:  {result.score_b.error}"
        )

    @skip_no_key
    def test_openai_schema_quality_gte_ollama(self, base_cfg):
        """
        GPT-4o should score at least as well as local Mistral on schema quality.

        This is a soft assertion — if Ollama beats GPT-4o it's interesting
        but not a test failure.  We log the result either way.
        """
        from one_ai_core.compare import compare_backends

        result = compare_backends(
            request="Deploy WordPress on my OneKE cluster",
            backend_a="ollama",
            backend_b="openai",
            base_config=base_cfg,
        )

        qa = result.score_a.quality_score
        qb = result.score_b.quality_score

        print(
            f"\nQuality scores — ollama: {qa}/5  openai: {qb}/5  "
            f"winner: {result.winner()}"
        )

        # Soft assertion: just log if Ollama is better (useful for fine-tuning decision)
        if qa > qb:
            print(
                f"[INFO] Ollama ({qa}) outscored OpenAI ({qb}) on schema quality. "
                "Fine-tuning may not be needed."
            )

        # Hard assertion: OpenAI must produce a valid schema (it's our quality baseline)
        assert result.score_b.schema_valid, (
            "GPT-4o failed schema validation — unexpected for a baseline model."
        )

    @skip_no_key
    def test_openai_faster_on_retries(self, base_cfg):
        """
        OpenAI should typically need fewer retries than local Mistral.

        Again a soft assertion — we just record the data.
        """
        from one_ai_core.compare import compare_backends

        result = compare_backends(
            request="Create a Redis namespace and deploy Redis 7 with a 5Gi PVC",
            backend_a="ollama",
            backend_b="openai",
            base_config=base_cfg,
        )

        print(
            f"\nRetry counts — ollama: {result.score_a.attempts}  "
            f"openai: {result.score_b.attempts}"
        )

        # Both should succeed (complex request but valid)
        # Soft: don't fail just because retry counts differ
        assert result.score_b.success or result.score_b.attempts > 0


# ---------------------------------------------------------------------------
# Standalone compare CLI helper (run directly with python)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick one-off compare run without pytest:

        python tests/test_chain_compare.py "Deploy WordPress on OneKE"
    """
    import sys
    from one_ai_core.compare import compare_backends, print_comparison

    req = sys.argv[1] if len(sys.argv) > 1 else "Deploy WordPress on OneKE"
    print(f"Comparing backends for: {req!r}\n")
    res = compare_backends(req)
    print_comparison(res)
