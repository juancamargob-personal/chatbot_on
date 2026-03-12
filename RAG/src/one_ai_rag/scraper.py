"""
OpenNebula documentation scraper.

Crawls the docs site section-by-section, extracts clean text content
while preserving code blocks, and stores raw pages with metadata for
downstream chunking.

Usage:
    scraper = DocScraper()
    pages = scraper.scrape_all()
    scraper.save(pages, output_dir="data/raw")
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from one_ai_rag.config import DOC_SOURCES, EXTRA_PAGES, RAW_DIR, settings

console = Console()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ScrapedPage:
    """A single scraped documentation page."""
    url: str
    title: str
    section: str            # e.g. "oneke", "api", "management_operations"
    section_label: str      # e.g. "OneKE / Kubernetes"
    content: str            # Clean text with preserved code blocks
    code_blocks: list[str]  # Extracted code blocks (for reference)
    headings: list[str]     # H1-H4 headings found on the page
    breadcrumb: list[str]   # Navigation breadcrumb if available
    url_hash: str = ""      # SHA256 of URL for deduplication

    def __post_init__(self):
        if not self.url_hash:
            self.url_hash = hashlib.sha256(self.url.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# HTML content extraction
# ---------------------------------------------------------------------------

class ContentExtractor:
    """Extracts clean documentation text from OpenNebula HTML pages."""

    # CSS selectors for the main content area
    # Ordered by specificity: try the most specific first
    CONTENT_SELECTORS = [
        "div.td-content",        # Hugo Docsy theme (OpenNebula 7.0+)
        "article.td-article",    # Hugo Docsy article wrapper
        "div.document",          # Sphinx default (older docs)
        "div[role='main']",      # Sphinx role-based
        "main",                  # HTML5 semantic
        "article",               # Generic article
        "div.body",              # Older Sphinx
        "div.rst-content",       # ReadTheDocs theme
    ]

    # Elements to strip from content
    STRIP_SELECTORS = [
        "nav",
        "div.sidebar",
        "div.sphinxsidebar",
        "div.related",
        "div.footer",
        "div.header",
        "script",
        "style",
        "a.headerlink",           # Sphinx permalink anchors
        "div.toctree-wrapper",    # TOC trees (we follow links instead)
        # Hugo/Docsy specific
        "div.td-toc",            # Table of contents sidebar
        "div.page-meta",         # Page metadata (edit links, dates)
        "div.taxonomy-terms",    # Tags section
        "a.td-heading-self-link",# Self-link anchors on headings
        "div.td-max-width-on-larger-screens.td-page-meta", # Page meta
    ]

    # Tags whose text should be preceded/followed by a space to prevent
    # "the<strong>Service</strong>item" → "theServiceitem"
    INLINE_SPACE_TAGS = {
        "strong", "b", "em", "i", "a", "span", "code",
        "mark", "abbr", "cite", "dfn", "kbd", "samp", "var",
    }

    def extract(self, html: str, url: str) -> tuple[str, list[str], list[str], list[str]]:
        """
        Extract clean content from HTML.

        Returns:
            (content_text, code_blocks, headings, breadcrumb)
        """
        soup = BeautifulSoup(html, "lxml")

        # Find the main content area
        content_el = None
        for selector in self.CONTENT_SELECTORS:
            content_el = soup.select_one(selector)
            if content_el:
                break

        if not content_el:
            # Fallback: use body
            content_el = soup.body or soup

        # Strip unwanted elements
        for selector in self.STRIP_SELECTORS:
            for el in content_el.select(selector):
                el.decompose()

        # Extract breadcrumb
        breadcrumb = self._extract_breadcrumb(soup)

        # Extract headings
        headings = []
        for tag in content_el.find_all(re.compile(r"^h[1-4]$")):
            text = tag.get_text(strip=True)
            if text:
                headings.append(text)

        # Extract code blocks before converting to text
        code_blocks = []
        for pre in content_el.find_all("pre"):
            code_text = pre.get_text()
            if code_text.strip():
                code_blocks.append(code_text.strip())

        # Convert to clean text
        content = self._element_to_text(content_el)

        return content, code_blocks, headings, breadcrumb

    def _extract_breadcrumb(self, soup: BeautifulSoup) -> list[str]:
        """Try to extract the navigation breadcrumb."""
        crumbs = []
        # Hugo/Docsy and Sphinx breadcrumb patterns
        for selector in [
            "nav.td-breadcrumbs li",     # Hugo Docsy
            "ol.breadcrumb li",          # Bootstrap-style
            "ul.wy-breadcrumbs li",      # Sphinx
            "div.related li",
            "nav.breadcrumb li",
        ]:
            els = soup.select(selector)
            if els:
                crumbs = [el.get_text(strip=True) for el in els if el.get_text(strip=True)]
                break
        return crumbs

    def _element_to_text(self, el: Tag) -> str:
        """
        Convert an HTML element to clean text, preserving structure.

        Uses recursive processing to avoid visiting children of
        already-handled elements (headings, code blocks), and adds
        proper spacing around inline elements like <strong>, <em>, <a>.
        """
        parts = []
        self._walk(el, parts)

        # Clean up whitespace
        text = "".join(parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        # Fix spaces before punctuation that spacing insertion may cause
        text = re.sub(r" ([.,;:!?)])", r"\1", text)
        return text.strip()

    def _walk(self, el, parts: list) -> None:
        """Recursively walk the DOM tree, building text parts."""
        for child in el.children:  # .children, NOT .descendants — avoids re-visiting
            if isinstance(child, NavigableString):
                text = str(child)
                # Collapse internal whitespace but preserve some spacing
                text = re.sub(r"\s+", " ", text)
                if text.strip():
                    parts.append(text)

            elif isinstance(child, Tag):
                # --- Block elements that we handle specially ---
                if child.name in ("h1", "h2", "h3", "h4"):
                    level = int(child.name[1])
                    prefix = "#" * level
                    heading_text = child.get_text(" ", strip=True)
                    parts.append(f"\n\n{prefix} {heading_text}\n")
                    # Do NOT recurse into heading children

                elif child.name == "pre":
                    code = child.get_text()
                    parts.append(f"\n```\n{code.strip()}\n```\n")
                    # Do NOT recurse into pre children

                elif child.name == "code" and child.parent.name != "pre":
                    # Inline code — add spaces around it
                    parts.append(f" `{child.get_text(strip=True)}` ")
                    # Do NOT recurse

                # --- Block elements that create line breaks ---
                elif child.name in ("p",):
                    parts.append("\n\n")
                    self._walk(child, parts)

                elif child.name == "div":
                    parts.append("\n")
                    self._walk(child, parts)

                elif child.name == "li":
                    parts.append("\n- ")
                    self._walk(child, parts)

                elif child.name == "br":
                    parts.append("\n")

                elif child.name in ("table",):
                    parts.append("\n")
                    self._walk(child, parts)
                    parts.append("\n")

                elif child.name in ("tr",):
                    parts.append("\n")
                    self._walk(child, parts)

                elif child.name in ("td", "th"):
                    parts.append(" | ")
                    self._walk(child, parts)

                elif child.name in ("dl",):
                    parts.append("\n")
                    self._walk(child, parts)

                elif child.name == "dt":
                    parts.append("\n**")
                    self._walk(child, parts)
                    parts.append("**: ")

                elif child.name == "dd":
                    self._walk(child, parts)
                    parts.append("\n")

                # --- Inline elements that need spacing ---
                elif child.name in self.INLINE_SPACE_TAGS:
                    parts.append(" ")
                    self._walk(child, parts)
                    parts.append(" ")

                # --- Everything else: just recurse ---
                else:
                    self._walk(child, parts)


# ---------------------------------------------------------------------------
# Link crawler
# ---------------------------------------------------------------------------

class LinkCrawler:
    """Discovers documentation pages within a section by following links."""

    def __init__(self, base_url: str, max_depth: int = 3, max_pages: int = 200):
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.base_path = urlparse(base_url).path.rstrip("/")
        self.max_depth = max_depth
        self.max_pages = max_pages

    def discover_links(self, html: str, current_url: str) -> list[str]:
        """Find all doc links on a page that belong to this section."""
        soup = BeautifulSoup(html, "lxml")
        links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Skip anchors, javascript, mailto
            if href.startswith(("#", "javascript:", "mailto:")):
                continue

            absolute = urljoin(current_url, href)

            # Strip fragment
            absolute = absolute.split("#")[0]

            # Normalize the URL path (resolve .. segments)
            parsed = urlparse(absolute)
            normalized_path = os.path.normpath(parsed.path)
            # normpath strips trailing slash; add it back for directories
            if not normalized_path.endswith("/") and "." not in normalized_path.split("/")[-1]:
                normalized_path += "/"
            absolute = parsed._replace(path=normalized_path).geturl()

            if self._is_valid_doc_link(absolute):
                links.append(absolute)

        return list(set(links))

    def _is_valid_doc_link(self, url: str) -> bool:
        """Check if a URL belongs to our documentation section."""
        parsed = urlparse(url)

        # Must be same domain
        if parsed.netloc != self.base_domain:
            return False

        # Normalize both paths for comparison
        url_path = parsed.path.rstrip("/")
        base_path = self.base_path.rstrip("/")

        # Must be under our section path
        if not url_path.startswith(base_path):
            return False

        # Exclude downloads, images, etc.
        excluded = (".png", ".jpg", ".jpeg", ".gif", ".svg",
                     ".pdf", ".zip", ".tar", ".gz", ".css", ".js",
                     ".xml", ".json", ".yaml", ".yml")
        if any(url_path.endswith(ext) for ext in excluded):
            return False

        return True


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

class DocScraper:
    """
    Scrapes OpenNebula documentation for RAG ingestion.

    Usage:
        scraper = DocScraper()
        pages = scraper.scrape_all()
        scraper.save(pages)
    """

    def __init__(self, request_settings: Optional[dict] = None):
        self.extractor = ContentExtractor()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.scrape_user_agent,
        })
        self.delay = settings.scrape_delay_seconds
        self.timeout = settings.scrape_timeout_seconds

    def scrape_all(self) -> list[ScrapedPage]:
        """Scrape all configured documentation sections."""
        all_pages: list[ScrapedPage] = []
        seen_urls: set[str] = set()

        console.print("[bold blue]Starting OpenNebula docs scrape[/bold blue]")

        for source in DOC_SOURCES:
            console.print(f"\n[yellow]Section: {source['label']}[/yellow]")
            section_pages = self._scrape_section(
                base_url=source["base_url"],
                section=source["section"],
                section_label=source["label"],
                max_depth=source["max_depth"],
                seen_urls=seen_urls,
            )
            all_pages.extend(section_pages)
            console.print(f"  Scraped {len(section_pages)} pages")

        # Scrape extra standalone pages
        if EXTRA_PAGES:
            console.print(f"\n[yellow]Extra pages ({len(EXTRA_PAGES)})[/yellow]")
            for url in EXTRA_PAGES:
                if url not in seen_urls:
                    page = self._scrape_page(url, section="extra", section_label="Reference")
                    if page:
                        all_pages.append(page)
                        seen_urls.add(url)

        console.print(f"\n[bold green]Total pages scraped: {len(all_pages)}[/bold green]")
        return all_pages

    def _scrape_section(
        self,
        base_url: str,
        section: str,
        section_label: str,
        max_depth: int,
        seen_urls: set[str],
    ) -> list[ScrapedPage]:
        """Breadth-first crawl of a documentation section."""
        crawler = LinkCrawler(
            base_url=base_url,
            max_depth=max_depth,
            max_pages=settings.max_pages_per_section,
        )

        pages: list[ScrapedPage] = []
        queue: list[tuple[str, int]] = [(base_url, 0)]  # (url, depth)
        visited: set[str] = set()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Crawling {section}...", total=None)

            while queue and len(pages) < settings.max_pages_per_section:
                url, depth = queue.pop(0)

                # Normalize URL for dedup (strip trailing slash inconsistencies)
                norm_url = url.rstrip("/") + "/"

                if norm_url in visited or norm_url in seen_urls:
                    continue
                visited.add(norm_url)
                seen_urls.add(norm_url)

                # Fetch once
                html = self._fetch(url)
                if html is None:
                    continue

                # Extract content
                page = self._parse_page(html, url, section, section_label)
                if page is not None:
                    pages.append(page)
                    progress.update(task, description=f"[{section}] {len(pages)} pages...")

                # Discover more links from same HTML (if not at max depth)
                if depth < max_depth:
                    new_links = crawler.discover_links(html, url)
                    for link in new_links:
                        link_norm = link.rstrip("/") + "/"
                        if link_norm not in visited:
                            queue.append((link, depth + 1))

        return pages

    def _parse_page(
        self, html: str, url: str, section: str, section_label: str
    ) -> Optional[ScrapedPage]:
        """Parse already-fetched HTML into a ScrapedPage."""
        try:
            content, code_blocks, headings, breadcrumb = self.extractor.extract(html, url)
        except Exception as e:
            console.print(f"  [red]Extract error: {url} — {e}[/red]")
            return None

        # Skip near-empty pages
        if len(content.strip()) < 50:
            return None

        # Derive title from first heading or URL
        title = headings[0] if headings else (
            url.rstrip("/").split("/")[-1].replace("_", " ").replace("-", " ").title()
        )

        return ScrapedPage(
            url=url,
            title=title,
            section=section,
            section_label=section_label,
            content=content,
            code_blocks=code_blocks,
            headings=headings,
            breadcrumb=breadcrumb,
        )

    def _scrape_page(
        self, url: str, section: str, section_label: str
    ) -> Optional[ScrapedPage]:
        """Scrape a single page: fetch + parse."""
        html = self._fetch(url)
        if html is None:
            return None
        return self._parse_page(html, url, section, section_label)

    def _fetch(self, url: str) -> Optional[str]:
        """Fetch a URL with polite delay and error handling."""
        try:
            time.sleep(self.delay)
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()

            # Only process HTML
            ct = resp.headers.get("content-type", "")
            if "text/html" not in ct:
                return None

            return resp.text
        except requests.RequestException as e:
            console.print(f"  [red]Fetch error: {url} — {e}[/red]")
            return None

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save(self, pages: list[ScrapedPage], output_dir: Optional[Path] = None) -> Path:
        """
        Save scraped pages to disk as JSON.

        Each page is saved individually (for incremental updates) plus
        a manifest file listing all pages.
        """
        out = Path(output_dir) if output_dir else RAW_DIR
        out.mkdir(parents=True, exist_ok=True)

        manifest = []
        for page in pages:
            filename = f"{page.section}_{page.url_hash}.json"
            filepath = out / filename
            filepath.write_text(json.dumps(asdict(page), indent=2, ensure_ascii=False))
            manifest.append({
                "filename": filename,
                "url": page.url,
                "title": page.title,
                "section": page.section,
            })

        manifest_path = out / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        console.print(f"[green]Saved {len(pages)} pages to {out}[/green]")
        return out

    @staticmethod
    def load(input_dir: Optional[Path] = None) -> list[ScrapedPage]:
        """Load previously scraped pages from disk."""
        src = Path(input_dir) if input_dir else RAW_DIR
        manifest_path = src / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest found at {manifest_path}")

        manifest = json.loads(manifest_path.read_text())
        pages = []
        for entry in manifest:
            filepath = src / entry["filename"]
            if filepath.exists():
                data = json.loads(filepath.read_text())
                pages.append(ScrapedPage(**data))

        return pages
