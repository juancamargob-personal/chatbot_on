"""
CLI for the one-ai-rag pipeline.

Commands:
    one-ai-rag scrape        Scrape OpenNebula docs
    one-ai-rag chunk         Chunk scraped docs
    one-ai-rag ingest        Embed and ingest chunks into ChromaDB
    one-ai-rag pipeline      Run the full scrape → chunk → ingest pipeline
    one-ai-rag query         Test a retrieval query interactively
    one-ai-rag stats         Show vector store statistics
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

console = Console()


def cmd_scrape(args):
    """Scrape OpenNebula documentation."""
    from one_ai_rag.scraper import DocScraper

    scraper = DocScraper()
    pages = scraper.scrape_all()
    scraper.save(pages)
    console.print(f"\n[bold green]Scraping complete: {len(pages)} pages[/bold green]")


def cmd_chunk(args):
    """Chunk previously scraped documentation."""
    from one_ai_rag.chunker import DocChunker
    from one_ai_rag.scraper import DocScraper

    console.print("Loading scraped pages...")
    pages = DocScraper.load()
    console.print(f"Loaded {len(pages)} pages")

    chunker = DocChunker()
    chunks = chunker.chunk_pages(pages)
    chunker.save(chunks)


def cmd_ingest(args):
    """Embed chunks and ingest into ChromaDB."""
    from one_ai_rag.chunker import DocChunker
    from one_ai_rag.embedder import create_embedder
    from one_ai_rag.store import VectorStore

    console.print("Loading chunks...")
    chunks = DocChunker.load()
    console.print(f"Loaded {len(chunks)} chunks")

    embedder = create_embedder(backend=args.embedding_backend)
    store = VectorStore()
    store.ingest(chunks, embedder, replace=args.replace)


def cmd_pipeline(args):
    """Run the full pipeline: scrape → chunk → embed → ingest."""
    from one_ai_rag.chunker import DocChunker
    from one_ai_rag.embedder import create_embedder
    from one_ai_rag.scraper import DocScraper
    from one_ai_rag.store import VectorStore

    # Step 1: Scrape
    console.print("\n[bold]Step 1/3: Scraping documentation[/bold]")
    scraper = DocScraper()
    pages = scraper.scrape_all()
    scraper.save(pages)

    # Step 2: Chunk
    console.print("\n[bold]Step 2/3: Chunking documents[/bold]")
    chunker = DocChunker()
    chunks = chunker.chunk_pages(pages)
    chunker.save(chunks)

    # Step 3: Embed + Ingest
    console.print("\n[bold]Step 3/3: Embedding and ingesting[/bold]")
    embedder = create_embedder(backend=args.embedding_backend)
    store = VectorStore()
    store.ingest(chunks, embedder, replace=args.replace)

    console.print("\n[bold green]Pipeline complete![/bold green]")
    _print_stats(store)


def cmd_query(args):
    """Test retrieval queries interactively."""
    from one_ai_rag.retriever import OneAIRetriever

    console.print("[cyan]Initialising retriever...[/cyan]")
    retriever = OneAIRetriever()

    if args.query:
        # Single query mode
        _run_query(retriever, args.query, args.section, args.top_k)
    else:
        # Interactive mode
        console.print("[bold]Interactive query mode. Type 'quit' to exit.[/bold]\n")
        while True:
            try:
                query = console.input("[bold cyan]Query>[/bold cyan] ")
            except (EOFError, KeyboardInterrupt):
                break
            if query.strip().lower() in ("quit", "exit", "q"):
                break
            if query.strip():
                _run_query(retriever, query, args.section, args.top_k)
                console.print()


def _run_query(retriever, query: str, section: str | None, top_k: int):
    """Execute a single retrieval query and display results."""
    chunks = retriever.retrieve(query, section_filter=section, top_k=top_k)

    if not chunks:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    table = Table(title=f"Results for: {query}")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=6)
    table.add_column("Section", width=12)
    table.add_column("Heading", width=30)
    table.add_column("Preview", width=60)

    for i, chunk in enumerate(chunks, 1):
        preview = chunk.content[:120].replace("\n", " ") + "..."
        table.add_row(
            str(i),
            f"{chunk.score:.3f}",
            chunk.section,
            chunk.heading_hierarchy[:30],
            preview,
        )

    console.print(table)

    # Show full context block
    console.print("\n[bold]Formatted context for LLM:[/bold]")
    context = retriever.get_context(query, section_filter=section)
    console.print(context[:2000])
    if len(context) > 2000:
        console.print(f"[dim]... ({len(context)} chars total)[/dim]")


def cmd_stats(args):
    """Show vector store statistics."""
    from one_ai_rag.store import VectorStore

    store = VectorStore()
    _print_stats(store)


def _print_stats(store):
    """Print vector store stats."""
    stats = store.get_stats()

    console.print(f"\n[bold]Vector Store Stats[/bold]")
    console.print(f"  Total chunks: {stats['total_chunks']}")
    console.print(f"  Code chunks:  {stats.get('code_chunks', 'N/A')}")

    if stats.get("sections"):
        console.print("  Sections:")
        for sec, count in sorted(stats["sections"].items()):
            console.print(f"    {sec}: {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="one-ai-rag: RAG pipeline for OpenNebula documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scrape
    subparsers.add_parser("scrape", help="Scrape OpenNebula documentation")

    # chunk
    subparsers.add_parser("chunk", help="Chunk scraped documentation")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Embed and ingest into ChromaDB")
    p_ingest.add_argument("--embedding-backend", default="local", choices=["local", "openai"])
    p_ingest.add_argument("--replace", action="store_true", help="Replace existing collection")

    # pipeline
    p_pipeline = subparsers.add_parser("pipeline", help="Run full scrape → chunk → ingest")
    p_pipeline.add_argument("--embedding-backend", default="local", choices=["local", "openai"])
    p_pipeline.add_argument("--replace", action="store_true", help="Replace existing collection")

    # query
    p_query = subparsers.add_parser("query", help="Test retrieval queries")
    p_query.add_argument("query", nargs="?", help="Query string (omit for interactive mode)")
    p_query.add_argument("--section", default=None, help="Filter by section (e.g. 'oneke')")
    p_query.add_argument("--top-k", type=int, default=6)

    # stats
    subparsers.add_parser("stats", help="Show vector store statistics")

    args = parser.parse_args()
    cmd_map = {
        "scrape": cmd_scrape,
        "chunk": cmd_chunk,
        "ingest": cmd_ingest,
        "pipeline": cmd_pipeline,
        "query": cmd_query,
        "stats": cmd_stats,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
