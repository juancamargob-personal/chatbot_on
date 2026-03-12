"""
Intelligent document chunker for OpenNebula documentation.

Splits scraped pages into semantically meaningful chunks suitable for
embedding and retrieval.  Key design decisions:

* Split on heading boundaries first (H2/H3 sections stay together)
* Keep code blocks intact — never split in the middle of a code fence
* Attach rich metadata to each chunk for filtered retrieval
* Configurable overlap for context continuity between chunks
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

from one_ai_rag.config import CHUNKS_DIR, settings
from one_ai_rag.scraper import ScrapedPage

console = Console()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DocChunk:
    """A single chunk of documentation ready for embedding."""

    chunk_id: str                   # Unique ID: "{url_hash}_{chunk_index}"
    content: str                    # The chunk text
    token_count: int                # Approximate token count

    # Provenance
    source_url: str
    source_title: str
    section: str                    # e.g. "oneke", "api"
    section_label: str

    # Semantic context
    heading_hierarchy: list[str]    # e.g. ["OneKE Guide", "Deploying Apps", "Helm Charts"]
    has_code: bool                  # Whether this chunk contains code blocks
    breadcrumb: list[str]

    # For ChromaDB metadata (must be flat primitives)
    def to_metadata(self) -> dict:
        return {
            "source_url": self.source_url,
            "source_title": self.source_title,
            "section": self.section,
            "section_label": self.section_label,
            "heading_hierarchy": " > ".join(self.heading_hierarchy),
            "has_code": self.has_code,
            "token_count": self.token_count,
            "chunk_id": self.chunk_id,
        }


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Fast approximate token count.

    Uses the rough heuristic of ~4 characters per token for English text,
    with a slight adjustment for code (which tokenises less efficiently).
    For exact counts, swap in tiktoken — but this is 100x faster and
    accurate enough for chunking decisions.
    """
    # Count code block characters separately (higher token density)
    code_pattern = re.compile(r"```.*?```", re.DOTALL)
    code_chars = sum(len(m.group()) for m in code_pattern.finditer(text))
    prose_chars = len(text) - code_chars

    return int(prose_chars / 4 + code_chars / 3.2)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class DocChunker:
    """
    Splits documentation pages into retrieval-friendly chunks.

    Strategy:
    1. Split page into top-level sections at H2 boundaries.
    2. If a section is within the target size, keep it as one chunk.
    3. If it's too large, sub-split at H3 boundaries.
    4. If still too large, split on paragraph boundaries.
    5. Code blocks are never split — they stay with their section.
    6. Tiny residual pieces are merged with neighbours.

    Usage:
        chunker = DocChunker()
        chunks = chunker.chunk_pages(pages)
        chunker.save(chunks)
    """

    def __init__(
        self,
        target_size: int = settings.chunk_size,
        overlap: int = settings.chunk_overlap,
        min_size: int = settings.min_chunk_size,
    ):
        self.target_size = target_size
        self.overlap = overlap
        self.min_size = min_size

    def chunk_pages(self, pages: list[ScrapedPage]) -> list[DocChunk]:
        """Chunk all pages and return a flat list of DocChunks."""
        all_chunks: list[DocChunk] = []

        for page in pages:
            page_chunks = self._chunk_page(page)
            all_chunks.extend(page_chunks)

        console.print(
            f"[green]Chunked {len(pages)} pages → {len(all_chunks)} chunks "
            f"(avg {sum(c.token_count for c in all_chunks) // max(len(all_chunks), 1)} tokens/chunk)[/green]"
        )
        return all_chunks

    def _chunk_page(self, page: ScrapedPage) -> list[DocChunk]:
        """Split a single page into chunks."""
        content = page.content
        if not content.strip():
            return []

        # Step 1: Split into sections at H2 boundaries
        sections = self._split_by_heading(content, level=2)

        raw_chunks: list[tuple[str, list[str]]] = []  # (text, heading_hierarchy)

        for heading, body in sections:
            section_text = f"## {heading}\n{body}" if heading else body
            section_tokens = estimate_tokens(section_text)
            hierarchy = [page.title]
            if heading:
                hierarchy.append(heading)

            if section_tokens <= self.target_size:
                raw_chunks.append((section_text, hierarchy))
            else:
                # Sub-split at H3 boundaries
                sub_sections = self._split_by_heading(body, level=3)
                for sub_heading, sub_body in sub_sections:
                    sub_text = f"### {sub_heading}\n{sub_body}" if sub_heading else sub_body
                    sub_hierarchy = hierarchy + ([sub_heading] if sub_heading else [])
                    sub_tokens = estimate_tokens(sub_text)

                    if sub_tokens <= self.target_size:
                        raw_chunks.append((sub_text, sub_hierarchy))
                    else:
                        # Last resort: split on paragraphs
                        para_chunks = self._split_by_paragraphs(sub_text, sub_hierarchy)
                        raw_chunks.extend(para_chunks)

        # Step 2: Merge tiny chunks with neighbours
        merged = self._merge_small_chunks(raw_chunks)

        # Step 3: Add overlap
        overlapped = self._add_overlap(merged)

        # Step 4: Build DocChunk objects
        doc_chunks = []
        for idx, (text, hierarchy) in enumerate(overlapped):
            has_code = "```" in text
            chunk = DocChunk(
                chunk_id=f"{page.url_hash}_{idx:03d}",
                content=text.strip(),
                token_count=estimate_tokens(text),
                source_url=page.url,
                source_title=page.title,
                section=page.section,
                section_label=page.section_label,
                heading_hierarchy=hierarchy,
                has_code=has_code,
                breadcrumb=page.breadcrumb,
            )
            doc_chunks.append(chunk)

        return doc_chunks

    def _split_by_heading(self, text: str, level: int) -> list[tuple[Optional[str], str]]:
        """
        Split text at heading boundaries.

        Returns: list of (heading_text_or_None, body_text) tuples.
        """
        pattern = re.compile(rf"^(#{{{level}}})\s+(.+)$", re.MULTILINE)
        matches = list(pattern.finditer(text))

        if not matches:
            return [(None, text)]

        sections: list[tuple[Optional[str], str]] = []

        # Content before first heading
        pre_content = text[: matches[0].start()].strip()
        if pre_content:
            sections.append((None, pre_content))

        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            sections.append((heading, body))

        return sections

    def _split_by_paragraphs(
        self, text: str, hierarchy: list[str]
    ) -> list[tuple[str, list[str]]]:
        """Split text into paragraph-based chunks that fit the target size."""
        # Split on double newlines, but keep code blocks intact
        blocks = self._split_preserving_code(text)
        chunks: list[tuple[str, list[str]]] = []
        current_parts: list[str] = []
        current_tokens = 0

        for block in blocks:
            block_tokens = estimate_tokens(block)

            if current_tokens + block_tokens > self.target_size and current_parts:
                # Flush current chunk
                chunks.append(("\n\n".join(current_parts), hierarchy))
                current_parts = []
                current_tokens = 0

            current_parts.append(block)
            current_tokens += block_tokens

        if current_parts:
            chunks.append(("\n\n".join(current_parts), hierarchy))

        return chunks

    def _split_preserving_code(self, text: str) -> list[str]:
        """
        Split text into blocks on double-newlines, but treat
        code fences (``` ... ```) as single unsplittable blocks.
        """
        blocks: list[str] = []
        in_code = False
        current: list[str] = []

        for line in text.split("\n"):
            if line.strip().startswith("```"):
                if in_code:
                    # End of code block
                    current.append(line)
                    blocks.append("\n".join(current))
                    current = []
                    in_code = False
                else:
                    # Start of code block — flush prose before it
                    if current:
                        prose = "\n".join(current).strip()
                        if prose:
                            # Split prose on double-newlines
                            for para in re.split(r"\n\s*\n", prose):
                                if para.strip():
                                    blocks.append(para.strip())
                        current = []
                    current.append(line)
                    in_code = True
            else:
                current.append(line)

        # Flush remainder
        if current:
            remainder = "\n".join(current).strip()
            if remainder:
                if in_code:
                    blocks.append(remainder)
                else:
                    for para in re.split(r"\n\s*\n", remainder):
                        if para.strip():
                            blocks.append(para.strip())

        return blocks

    def _merge_small_chunks(
        self, chunks: list[tuple[str, list[str]]]
    ) -> list[tuple[str, list[str]]]:
        """Merge chunks smaller than min_size with their neighbours."""
        if len(chunks) <= 1:
            return chunks

        merged: list[tuple[str, list[str]]] = []
        i = 0
        while i < len(chunks):
            text, hierarchy = chunks[i]

            if estimate_tokens(text) < self.min_size and i + 1 < len(chunks):
                # Merge with next
                next_text, next_hierarchy = chunks[i + 1]
                combined = text + "\n\n" + next_text
                # Use the more specific hierarchy
                best_h = next_hierarchy if len(next_hierarchy) > len(hierarchy) else hierarchy
                merged.append((combined, best_h))
                i += 2
            else:
                merged.append((text, hierarchy))
                i += 1

        return merged

    def _add_overlap(
        self, chunks: list[tuple[str, list[str]]]
    ) -> list[tuple[str, list[str]]]:
        """Add overlap from the end of the previous chunk to the start of each chunk."""
        if self.overlap <= 0 or len(chunks) <= 1:
            return chunks

        result: list[tuple[str, list[str]]] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_text, _ = chunks[i - 1]
            curr_text, curr_hierarchy = chunks[i]

            # Take the last ~overlap tokens from the previous chunk
            overlap_text = self._tail_tokens(prev_text, self.overlap)
            if overlap_text:
                combined = f"[...] {overlap_text}\n\n{curr_text}"
                result.append((combined, curr_hierarchy))
            else:
                result.append((curr_text, curr_hierarchy))

        return result

    def _tail_tokens(self, text: str, target_tokens: int) -> str:
        """Get approximately the last `target_tokens` tokens of text."""
        words = text.split()
        # ~1.3 words per token on average
        target_words = int(target_tokens * 1.3)
        if len(words) <= target_words:
            return ""
        return " ".join(words[-target_words:])

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save(self, chunks: list[DocChunk], output_dir: Optional[Path] = None) -> Path:
        """Save chunks to disk as JSONL for inspection and reuse."""
        out = Path(output_dir) if output_dir else CHUNKS_DIR
        out.mkdir(parents=True, exist_ok=True)

        chunks_path = out / "chunks.jsonl"
        with open(chunks_path, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

        # Summary stats
        stats = {
            "total_chunks": len(chunks),
            "total_tokens": sum(c.token_count for c in chunks),
            "avg_tokens": sum(c.token_count for c in chunks) // max(len(chunks), 1),
            "chunks_with_code": sum(1 for c in chunks if c.has_code),
            "sections": list(set(c.section for c in chunks)),
        }
        stats_path = out / "chunk_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2))

        console.print(f"[green]Saved {len(chunks)} chunks to {chunks_path}[/green]")
        console.print(f"  Total tokens: {stats['total_tokens']:,}")
        console.print(f"  Chunks with code: {stats['chunks_with_code']}")
        return out

    @staticmethod
    def load(input_dir: Optional[Path] = None) -> list[DocChunk]:
        """Load chunks from disk."""
        src = Path(input_dir) if input_dir else CHUNKS_DIR
        chunks_path = src / "chunks.jsonl"

        if not chunks_path.exists():
            raise FileNotFoundError(f"No chunks file at {chunks_path}")

        chunks = []
        with open(chunks_path) as f:
            for line in f:
                if line.strip():
                    chunks.append(DocChunk(**json.loads(line)))

        return chunks
