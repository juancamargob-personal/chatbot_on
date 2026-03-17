"""
one_ai_core.cli
================
Click-based CLI.  Installed as the ``one-ai`` command.

Commands
--------
one-ai generate <request>   Produce and print the YAML config
one-ai plan     <request>   Produce config + print the generated Python script
one-ai apply    <request>   Produce config + script + save both to disk
one-ai eval                 Run benchmark (placeholder — wired up in later phase)
one-ai config               Print active configuration values
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        level=level,
    )


def _get_chain(ctx: click.Context):
    """Build a OneAIChain from the context's config object."""
    from .chain import OneAIChain
    from .config import CoreConfig

    cfg = ctx.obj or CoreConfig()
    return OneAIChain(config=cfg)


def _print_result_header(result) -> None:
    """Print a status banner for a ChainResult."""
    if result.success:
        console.print(Panel(
            f"[bold green]{result.summary()}[/bold green]",
            title="[bold]one-ai[/bold]",
            border_style="green",
        ))
        if result.warnings:
            for w in result.warnings:
                console.print(f"  [yellow]⚠[/yellow]  {w}")
    else:
        console.print(Panel(
            f"[bold red]{result.summary()}[/bold red]",
            title="[bold]one-ai[/bold]",
            border_style="red",
        ))


def _print_rag_context(result) -> None:
    """Optionally show the retrieved doc chunks."""
    if not result.rag_chunks:
        return
    console.print(f"\n[dim]RAG: {len(result.rag_chunks)} chunk(s) retrieved[/dim]")
    for i, chunk in enumerate(result.rag_chunks, 1):
        console.print(f"  [dim]{i}. {chunk['source']}[/dim]")


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.option("--ollama-model", envvar="ONEAI_CORE_OLLAMA_MODEL", default=None,
              help="Override the Ollama model (e.g. mistral:7b-instruct-v0.3-q4_K_M).")
@click.option("--one-endpoint", envvar="ONEAI_CORE_ONE_ENDPOINT", default=None,
              help="OpenNebula RPC endpoint.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, ollama_model: str | None, one_endpoint: str | None):
    """
    one-ai — OpenNebula AI Configuration Assistant

    Translates natural-language infrastructure requests into validated,
    executable OpenNebula configurations.

    \b
    Examples:
      one-ai generate "Deploy WordPress on my OneKE cluster"
      one-ai plan    "Scale the worker nodes to 5"
      one-ai apply   "Create a Redis namespace and deploy Redis"
    """
    _setup_logging(verbose)

    from .config import CoreConfig
    cfg = CoreConfig()
    if ollama_model:
        cfg.ollama_model = ollama_model
    if one_endpoint:
        cfg.one_endpoint = one_endpoint
    cfg.verbose = verbose
    ctx.ensure_object(dict)
    ctx.obj = cfg


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("request")
@click.option("--show-rag", is_flag=True, default=False,
              help="Show which RAG chunks were retrieved.")
@click.pass_context
def generate(ctx: click.Context, request: str, show_rag: bool):
    """
    Generate a YAML config for REQUEST and print it to stdout.

    \b
    Example:
      one-ai generate "Deploy WordPress on my OneKE cluster"
    """
    chain = _get_chain(ctx)

    with console.status("[bold cyan]Retrieving docs + generating config…[/bold cyan]"):
        result = chain.run(request)

    _print_result_header(result)

    if show_rag:
        _print_rag_context(result)

    if result.success:
        console.print()
        console.print(Syntax(result.config_yaml, "yaml", theme="monokai", line_numbers=True))
    else:
        sys.exit(1)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("request")
@click.option("--show-rag", is_flag=True, default=False)
@click.pass_context
def plan(ctx: click.Context, request: str, show_rag: bool):
    """
    Generate config AND show the Python execution script (nothing is saved).

    \b
    Example:
      one-ai plan "Deploy WordPress on my OneKE cluster"
    """
    chain = _get_chain(ctx)

    with console.status("[bold cyan]Retrieving docs + generating plan…[/bold cyan]"):
        result = chain.run(request)

    _print_result_header(result)

    if show_rag:
        _print_rag_context(result)

    if not result.success:
        sys.exit(1)

    # Show YAML config
    console.print("\n[bold]── YAML Config ──[/bold]")
    console.print(Syntax(result.config_yaml, "yaml", theme="monokai", line_numbers=True))

    # Show generated script
    if result.script:
        console.print("\n[bold]── Generated Python Script ──[/bold]")
        console.print(Syntax(result.script.code, "python", theme="monokai", line_numbers=True))
    else:
        console.print("\n[yellow]Script generation not available.[/yellow]")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("request")
@click.option("--dry-run", is_flag=True, default=None,
              help="Show what would run without executing (overrides ONEAI_CORE_DRY_RUN_DEFAULT).")
@click.option("--output-dir", type=click.Path(), default=None,
              help="Directory to write config.yaml and script.py into.")
@click.option("--show-rag", is_flag=True, default=False)
@click.pass_context
def apply(ctx: click.Context, request: str, dry_run: bool | None,
          output_dir: str | None, show_rag: bool):
    """
    Generate config + script and SAVE them to disk.

    In a future release this will also invoke one-ai-agent to execute the
    script against your OpenNebula cluster (with an approval gate).

    \b
    Example:
      one-ai apply "Deploy WordPress on OneKE" --output-dir ./runs/wordpress
    """
    from .config import CoreConfig

    cfg: CoreConfig = ctx.obj or CoreConfig()

    # Resolve dry-run flag: CLI flag > config default
    effective_dry_run = dry_run if dry_run is not None else cfg.dry_run_default

    chain = _get_chain(ctx)

    with console.status("[bold cyan]Retrieving docs + generating config…[/bold cyan]"):
        result = chain.run(request)

    _print_result_header(result)

    if show_rag:
        _print_rag_context(result)

    if not result.success:
        sys.exit(1)

    # Determine output directory
    out = Path(output_dir) if output_dir else cfg.output_dir / (
(
            getattr(result.config.metadata, "name", None)
            or getattr(result.config.metadata, "description", "output")[:30].replace(" ", "-").lower()
        ) if result.config else "output"
    )
    out.mkdir(parents=True, exist_ok=True)

    # Save YAML config
    config_path = out / "config.yaml"
    config_path.write_text(result.config_yaml, encoding="utf-8")

    # Save Python script
    script_path = None
    if result.script:
        script_path = out / "script.py"
        script_path.write_text(result.script.code, encoding="utf-8")
        script_path.chmod(0o755)

    # Summary table
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("[bold]Config YAML[/bold]", f"[cyan]{config_path}[/cyan]")
    if script_path:
        table.add_row("[bold]Python script[/bold]", f"[cyan]{script_path}[/cyan]")
    table.add_row("[bold]Dry-run[/bold]", str(effective_dry_run))
    console.print("\n", table)

    if effective_dry_run:
        console.print(Panel(
            "[yellow]Dry-run mode: files saved but script NOT executed.[/yellow]\n"
            f"To execute manually:\n  python {script_path} --dry-run",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            "[dim]one-ai-agent (execution engine) is not yet implemented.\n"
            f"Run manually:  python {script_path}[/dim]",
            border_style="dim",
        ))


# ---------------------------------------------------------------------------
# eval  (placeholder — to be wired to one_ai_finetune benchmarks)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--compare", default="base,finetuned",
              help="Comma-separated model tags to compare.")
@click.pass_context
def eval(ctx: click.Context, compare: str):  # noqa: A001
    """
    Run evaluation benchmarks.

    Compares base+RAG vs fine-tuned+RAG model performance using the
    schema evaluator and LLM judge from one-ai-finetune.

    (Full implementation in a later phase.)
    """
    console.print(Panel(
        "[yellow]Eval command is a placeholder.[/yellow]\n"
        "It will run one_ai_finetune.eval benchmarks once the fine-tuning\n"
        "phase is complete.\n\n"
        f"Requested comparison: [bold]{compare}[/bold]",
        title="one-ai eval",
        border_style="yellow",
    ))


# ---------------------------------------------------------------------------
# config (show active settings)
# ---------------------------------------------------------------------------

@cli.command("config")
@click.pass_context
def show_config(ctx: click.Context):
    """Print the active configuration values (reads env vars)."""
    from .config import CoreConfig
    cfg: CoreConfig = ctx.obj or CoreConfig()

    table = Table(title="Active Configuration", show_header=True)
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value")

    table.add_row("ollama_model",      cfg.ollama_model)
    table.add_row("ollama_base_url",   cfg.ollama_base_url)
    table.add_row("ollama_temperature",str(cfg.ollama_temperature))
    table.add_row("ollama_timeout",    str(cfg.ollama_timeout))
    table.add_row("max_retries",       str(cfg.max_retries))
    table.add_row("rag_top_k",         str(cfg.rag_top_k))
    table.add_row("rag_rerank",        str(cfg.rag_rerank))
    table.add_row("output_dir",        str(cfg.output_dir))
    table.add_row("one_endpoint",      cfg.one_endpoint)
    table.add_row("dry_run_default",   str(cfg.dry_run_default))
    table.add_row("verbose",           str(cfg.verbose))

    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
