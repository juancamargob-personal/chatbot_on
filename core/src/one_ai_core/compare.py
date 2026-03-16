"""
one_ai_core.compare
====================
Side-by-side comparison of two LLM backends on the same request.

Typical use
-----------
Run a request through both local Ollama (free) and OpenAI GPT-4o (API),
then score both outputs with the schema validator so you can objectively
see how they differ in quality, speed, and retry count.

    from one_ai_core.compare import compare_backends, print_comparison

    result = compare_backends(
        request="Deploy WordPress on my OneKE cluster",
        backend_a="ollama",   # local Mistral
        backend_b="openai",   # GPT-4o
    )
    print_comparison(result)

This is also what powers:
    one-ai eval --compare ollama,openai "Deploy WordPress on OneKE"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# ChainResult, OneAIChain, CoreConfig imported lazily inside compare_backends()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BackendScore:
    """Scoring breakdown for a single backend run."""
    backend: str            # e.g. "ollama/mistral:7b-instruct-v0.3-q4_K_M"
    success: bool
    attempts: int
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    # Schema-level scores (populated by _score_result)
    schema_valid: bool = False
    step_count: int = 0
    has_rollback: bool = False
    has_pre_checks: bool = False
    has_post_checks: bool = False

    # Raw chain result (for further inspection)
    chain_result: "ChainResult | None" = None

    @property
    def quality_score(self) -> int:
        """
        Simple 0-5 integer quality score based on schema completeness.

        Points awarded:
          1  — output passed schema validation
          1  — at least one step generated
          1  — rollback section present
          1  — pre-checks present
          1  — post-checks present
        """
        if not self.schema_valid:
            return 0
        return sum([
            self.schema_valid,
            self.step_count > 0,
            self.has_rollback,
            self.has_pre_checks,
            self.has_post_checks,
        ])


@dataclass
class CompareResult:
    """Side-by-side comparison of two backends on the same request."""
    request: str
    score_a: BackendScore
    score_b: BackendScore
    total_elapsed_seconds: float = 0.0

    def winner(self) -> str:
        """Return label of the better-scoring backend, or 'tie'."""
        qa = self.score_a.quality_score
        qb = self.score_b.quality_score
        if qa > qb:
            return self.score_a.backend
        if qb > qa:
            return self.score_b.backend
        # Tiebreak by speed (fewer seconds wins)
        if self.score_a.elapsed_seconds < self.score_b.elapsed_seconds:
            return f"{self.score_a.backend} (faster)"
        if self.score_b.elapsed_seconds < self.score_a.elapsed_seconds:
            return f"{self.score_b.backend} (faster)"
        return "tie"


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def _score_result(result: ChainResult, backend_label: str) -> BackendScore:
    """Convert a ChainResult into a BackendScore."""
    score = BackendScore(
        backend=backend_label,
        success=result.success,
        attempts=result.attempts,
        elapsed_seconds=result.elapsed_seconds,
        warnings=result.warnings,
        error=result.error,
        chain_result=result,
    )

    if not result.success or result.config is None:
        return score

    cfg = result.config
    score.schema_valid = True
    score.step_count = len(cfg.steps)

    # Check rollback
    try:
        score.has_rollback = bool(cfg.rollback and cfg.rollback.enabled)
    except AttributeError:
        score.has_rollback = False

    # Check pre/post validation checks
    try:
        score.has_pre_checks = bool(
            cfg.validation and cfg.validation.pre_checks
        )
        score.has_post_checks = bool(
            cfg.validation and cfg.validation.post_checks
        )
    except AttributeError:
        pass

    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_backends(
    request: str,
    backend_a: str = "ollama",
    backend_b: str = "openai",
    base_config=None,
) -> "CompareResult":
    """
    Run ``request`` through two backends and return a ``CompareResult``.

    Parameters
    ----------
    request:
        The natural-language infrastructure request to test.
    backend_a:
        First backend label: 'ollama' or 'openai'.
    backend_b:
        Second backend label: 'ollama' or 'openai'.
    base_config:
        Base ``CoreConfig`` to use.  A copy is made for each backend so
        settings like API keys are inherited from the environment.

    Returns
    -------
    CompareResult
    """
    from .chain import OneAIChain
    from .config import CoreConfig
    t0 = time.monotonic()
    base = base_config or CoreConfig()

    # --- Backend A ---
    cfg_a = CoreConfig(
        llm_backend=backend_a,
        # inherit all other settings from base
        ollama_base_url=base.ollama_base_url,
        ollama_model=base.ollama_model,
        ollama_temperature=base.ollama_temperature,
        ollama_timeout=base.ollama_timeout,
        openai_api_key=base.openai_api_key,
        openai_model=base.openai_model,
        openai_temperature=base.openai_temperature,
        openai_timeout=base.openai_timeout,
        max_retries=base.max_retries,
        rag_top_k=base.rag_top_k,
        rag_rerank=base.rag_rerank,
        output_dir=base.output_dir,
        one_endpoint=base.one_endpoint,
    )
    chain_a = OneAIChain(config=cfg_a)
    result_a = chain_a.run(request)
    score_a = _score_result(result_a, cfg_a.active_model)

    # --- Backend B ---
    cfg_b = CoreConfig(
        llm_backend=backend_b,
        ollama_base_url=base.ollama_base_url,
        ollama_model=base.ollama_model,
        ollama_temperature=base.ollama_temperature,
        ollama_timeout=base.ollama_timeout,
        openai_api_key=base.openai_api_key,
        openai_model=base.openai_model,
        openai_temperature=base.openai_temperature,
        openai_timeout=base.openai_timeout,
        max_retries=base.max_retries,
        rag_top_k=base.rag_top_k,
        rag_rerank=base.rag_rerank,
        output_dir=base.output_dir,
        one_endpoint=base.one_endpoint,
    )
    chain_b = OneAIChain(config=cfg_b)
    result_b = chain_b.run(request)
    score_b = _score_result(result_b, cfg_b.active_model)

    return CompareResult(
        request=request,
        score_a=score_a,
        score_b=score_b,
        total_elapsed_seconds=time.monotonic() - t0,
    )


def print_comparison(result: CompareResult) -> None:
    """
    Pretty-print a ``CompareResult`` to stdout using Rich.

    Falls back to plain text if Rich is not installed.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        _rich_print(result, Console())
    except ImportError:
        _plain_print(result)


def _rich_print(result: CompareResult, console) -> None:
    from rich.table import Table
    from rich.panel import Panel

    console.print(Panel(
        f"[bold]Request:[/bold] {result.request}",
        title="[bold cyan]Backend Comparison[/bold cyan]",
    ))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column(result.score_a.backend, justify="center")
    table.add_column(result.score_b.backend, justify="center")

    def _tick(val: bool) -> str:
        return "✅" if val else "❌"

    def _highlight(a_val, b_val, fmt=str):
        """Colour the better value green."""
        sa, sb = fmt(a_val), fmt(b_val)
        if a_val == b_val:
            return sa, sb
        # For booleans / scores: higher is better
        if isinstance(a_val, (int, float, bool)):
            if a_val > b_val:
                return f"[green]{sa}[/green]", sb
            else:
                return sa, f"[green]{sb}[/green]"
        return sa, sb

    qa, qb = result.score_a.quality_score, result.score_b.quality_score
    table.add_row("Quality score (0-5)", *_highlight(qa, qb))
    table.add_row("Schema valid",        _tick(result.score_a.schema_valid),
                                         _tick(result.score_b.schema_valid))
    table.add_row("Steps generated",     *_highlight(result.score_a.step_count,
                                                      result.score_b.step_count))
    table.add_row("Has rollback",        _tick(result.score_a.has_rollback),
                                         _tick(result.score_b.has_rollback))
    table.add_row("Has pre-checks",      _tick(result.score_a.has_pre_checks),
                                         _tick(result.score_b.has_pre_checks))
    table.add_row("Has post-checks",     _tick(result.score_a.has_post_checks),
                                         _tick(result.score_b.has_post_checks))
    table.add_row("Attempts needed",     *_highlight(
        result.score_a.attempts, result.score_b.attempts,
        # Fewer attempts is BETTER — invert the highlight logic
    ))
    ea = f"{result.score_a.elapsed_seconds:.1f}s"
    eb = f"{result.score_b.elapsed_seconds:.1f}s"
    table.add_row("Elapsed time",        ea, eb)

    if result.score_a.error:
        table.add_row("Error", f"[red]{result.score_a.error[:60]}[/red]", "")
    if result.score_b.error:
        table.add_row("Error", "", f"[red]{result.score_b.error[:60]}[/red]")

    console.print(table)
    console.print(
        f"\n[bold]Winner:[/bold] [green]{result.winner()}[/green]  "
        f"(total wall time: {result.total_elapsed_seconds:.1f}s)"
    )


def _plain_print(result: CompareResult) -> None:
    a, b = result.score_a, result.score_b
    print(f"\n=== Backend Comparison: {result.request!r} ===")
    print(f"{'Metric':<22}  {a.backend:<35}  {b.backend}")
    print("-" * 80)
    rows = [
        ("Quality (0-5)",   a.quality_score,     b.quality_score),
        ("Schema valid",    a.schema_valid,       b.schema_valid),
        ("Steps",           a.step_count,         b.step_count),
        ("Rollback",        a.has_rollback,       b.has_rollback),
        ("Pre-checks",      a.has_pre_checks,     b.has_pre_checks),
        ("Post-checks",     a.has_post_checks,    b.has_post_checks),
        ("Attempts",        a.attempts,           b.attempts),
        ("Time (s)",        f"{a.elapsed_seconds:.1f}", f"{b.elapsed_seconds:.1f}"),
    ]
    for label, va, vb in rows:
        print(f"{label:<22}  {str(va):<35}  {vb}")
    print(f"\nWinner: {result.winner()}")
