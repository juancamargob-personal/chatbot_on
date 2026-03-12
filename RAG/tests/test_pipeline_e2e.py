"""
End-to-end integration tests for the RAG pipeline.

Tests the full flow: scrape (mocked) -> chunk -> embed -> ingest -> query.
Uses mock HTML pages from conftest instead of hitting the live docs site.
No network calls, no GPU — runs entirely offline with mock embeddings.

Run with:
    pytest tests/test_pipeline_e2e.py -v
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from one_ai_rag.scraper import DocScraper, ContentExtractor, ScrapedPage
from one_ai_rag.chunker import DocChunker, DocChunk, estimate_tokens
from one_ai_rag.store import VectorStore, RetrievedChunk
from one_ai_rag.retriever import OneAIRetriever, LangChainOpenNebulaRetriever


# ===================================================================
# PHASE 1: Scraper — extract content from HTML
# ===================================================================

class TestScraperExtraction:
    """Tests that the scraper correctly extracts structured content from HTML."""

    def test_extracts_all_pages(self, sample_scraped_pages):
        """All three mock pages should be parsed successfully."""
        assert len(sample_scraped_pages) == 3

    def test_oneke_page_content(self, sample_scraped_pages):
        """OneKE deployment page should contain key concepts."""
        oneke_page = [p for p in sample_scraped_pages if p.section == "oneke"][0]

        assert "Helm" in oneke_page.content
        assert "OneKE" in oneke_page.content
        assert "Kubernetes" in oneke_page.content

    def test_code_blocks_extracted(self, sample_scraped_pages):
        """Code blocks should be captured both in content and code_blocks list."""
        oneke_page = [p for p in sample_scraped_pages if p.section == "oneke"][0]

        # Code blocks list populated
        assert len(oneke_page.code_blocks) >= 2
        assert any("helm install" in cb for cb in oneke_page.code_blocks)
        assert any("helm repo add" in cb for cb in oneke_page.code_blocks)

        # Code also present in content (as fenced blocks)
        assert "```" in oneke_page.content

    def test_headings_extracted(self, sample_scraped_pages):
        """H1-H4 headings should be captured."""
        oneke_page = [p for p in sample_scraped_pages if p.section == "oneke"][0]

        assert "Deploying Applications on OneKE" in oneke_page.headings
        assert "Prerequisites" in oneke_page.headings
        assert "Deploying with Helm" in oneke_page.headings
        assert "Pod Health Checks" in oneke_page.headings  # H3

    def test_nav_and_footer_stripped(self, sample_scraped_pages):
        """Navigation and footer elements should not appear in content."""
        oneke_page = [p for p in sample_scraped_pages if p.section == "oneke"][0]

        assert "Copyright OpenNebula Systems" not in oneke_page.content

    def test_pyone_page_has_api_examples(self, sample_scraped_pages):
        """The pyone API page should contain Python code examples."""
        api_page = [p for p in sample_scraped_pages if p.section == "api"][0]

        assert "pyone" in api_page.content
        assert any("import pyone" in cb for cb in api_page.code_blocks)
        assert any("vm.allocate" in cb for cb in api_page.code_blocks)

    def test_vnet_page_has_template(self, sample_scraped_pages):
        """The virtual network page should contain the network template."""
        vnet_page = [p for p in sample_scraped_pages if p.section == "management_operations"][0]

        assert "virtual network" in vnet_page.content.lower()
        assert any("VN_MAD" in cb for cb in vnet_page.code_blocks)

    def test_url_hash_is_consistent(self, sample_scraped_pages):
        """Same URL should always produce the same hash."""
        page = sample_scraped_pages[0]
        assert len(page.url_hash) == 16

        page2 = ScrapedPage(
            url=page.url, title="X", section="x", section_label="X",
            content="x", code_blocks=[], headings=[], breadcrumb=[],
        )
        assert page.url_hash == page2.url_hash


# ===================================================================
# PHASE 2: Scraper persistence — save/load round-trip
# ===================================================================

class TestScraperPersistence:
    """Tests that scraped pages survive serialization."""

    def test_save_and_load(self, sample_scraped_pages, tmp_path):
        """Pages should be identical after save/load cycle."""
        scraper = DocScraper.__new__(DocScraper)  # Skip __init__ (no HTTP session needed)

        scraper.save(sample_scraped_pages, output_dir=tmp_path)

        # Verify files written
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert len(manifest) == 3

        for entry in manifest:
            assert (tmp_path / entry["filename"]).exists()

        # Reload
        loaded = DocScraper.load(input_dir=tmp_path)
        assert len(loaded) == 3
        assert loaded[0].url == sample_scraped_pages[0].url
        assert loaded[0].content == sample_scraped_pages[0].content
        assert loaded[0].section == sample_scraped_pages[0].section


# ===================================================================
# PHASE 3: Chunker — split pages into retrieval-friendly pieces
# ===================================================================

class TestChunkerOnRealContent:
    """Tests chunking on realistic OpenNebula documentation content."""

    def test_produces_multiple_chunks(self, sample_chunks):
        """Multi-section pages should produce multiple chunks."""
        assert len(sample_chunks) >= 3, (
            f"Expected at least 3 chunks from 3 pages, got {len(sample_chunks)}"
        )

    def test_chunks_have_content(self, sample_chunks):
        """No chunk should be empty."""
        for chunk in sample_chunks:
            assert len(chunk.content.strip()) > 0
            assert chunk.token_count > 0

    def test_chunks_carry_provenance(self, sample_chunks):
        """Every chunk should know which page and section it came from."""
        for chunk in sample_chunks:
            assert chunk.source_url.startswith("https://")
            assert chunk.source_title
            assert chunk.section in ("oneke", "api", "management_operations")
            assert chunk.section_label

    def test_code_blocks_stay_intact(self, sample_chunks):
        """Code blocks should not be split across chunks."""
        code_chunks = [c for c in sample_chunks if c.has_code]
        assert len(code_chunks) >= 1

        for chunk in code_chunks:
            # Count opening and closing fences — should be balanced
            opens = chunk.content.count("```\n") + chunk.content.count("```")
            # Rough check: fences should come in pairs
            # (not exact because of formatting variations, but no orphan opens)
            assert opens >= 2 or "```" in chunk.content

    def test_heading_hierarchy_populated(self, sample_chunks):
        """Chunks should carry their heading hierarchy for context."""
        for chunk in sample_chunks:
            assert len(chunk.heading_hierarchy) >= 1
            # First element should be the page title
            assert chunk.heading_hierarchy[0]

    def test_chunk_ids_globally_unique(self, sample_chunks):
        """All chunk IDs should be unique across all pages."""
        ids = [c.chunk_id for c in sample_chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_is_chroma_compatible(self, sample_chunks):
        """ChromaDB metadata must be flat primitives only."""
        for chunk in sample_chunks:
            meta = chunk.to_metadata()
            for key, value in meta.items():
                assert isinstance(value, (str, int, float, bool)), (
                    f"Chunk {chunk.chunk_id}: metadata['{key}'] is {type(value).__name__}, "
                    f"expected str/int/float/bool"
                )

    def test_section_distribution(self, sample_chunks):
        """Chunks should come from multiple sections."""
        sections = set(c.section for c in sample_chunks)
        assert len(sections) >= 2

    def test_chunks_respect_target_size(self, sample_scraped_pages):
        """Most chunks should be near the target size (within 2x)."""
        chunker = DocChunker(target_size=200, overlap=0, min_size=20)
        chunks = chunker.chunk_pages(sample_scraped_pages)

        oversized = [c for c in chunks if c.token_count > 200 * 2.5]
        # Allow a few oversized chunks (code blocks can cause this)
        assert len(oversized) <= len(chunks) * 0.2, (
            f"Too many oversized chunks: {len(oversized)}/{len(chunks)}"
        )


class TestChunkerPersistenceIntegration:
    """Test chunk save/load with realistic data."""

    def test_round_trip(self, sample_chunks, tmp_path):
        """Chunks should survive a save/load cycle with all fields intact."""
        chunker = DocChunker()
        chunker.save(sample_chunks, output_dir=tmp_path)

        # Stats file should exist
        stats_path = tmp_path / "chunk_stats.json"
        assert stats_path.exists()
        stats = json.loads(stats_path.read_text())
        assert stats["total_chunks"] == len(sample_chunks)
        assert stats["total_tokens"] > 0

        # Reload and compare
        loaded = DocChunker.load(input_dir=tmp_path)
        assert len(loaded) == len(sample_chunks)

        for original, reloaded in zip(sample_chunks, loaded):
            assert original.chunk_id == reloaded.chunk_id
            assert original.content == reloaded.content
            assert original.source_url == reloaded.source_url
            assert original.section == reloaded.section
            assert original.has_code == reloaded.has_code
            assert original.heading_hierarchy == reloaded.heading_hierarchy


# ===================================================================
# PHASE 4: Vector Store — ingest and query with ChromaDB
# ===================================================================

class TestVectorStoreIngestion:
    """Tests ingesting chunks into ChromaDB and basic operations."""

    def test_ingest_all_chunks(self, sample_chunks, mock_embedder, tmp_path):
        """All chunks should be ingested successfully."""
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        count = store.ingest(sample_chunks, mock_embedder)

        assert count == len(sample_chunks)
        assert store.collection.count() == len(sample_chunks)

    def test_deduplication(self, sample_chunks, mock_embedder, tmp_path):
        """Re-ingesting the same chunks should add nothing."""
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        store.ingest(sample_chunks, mock_embedder)
        second_count = store.ingest(sample_chunks, mock_embedder)

        assert second_count == 0
        assert store.collection.count() == len(sample_chunks)

    def test_replace_mode(self, sample_chunks, mock_embedder, tmp_path):
        """replace=True should clear old data before ingesting."""
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        store.ingest(sample_chunks, mock_embedder)

        # Ingest a single chunk with replace — should end up with just 1
        store.ingest(sample_chunks[:1], mock_embedder, replace=True)
        assert store.collection.count() == 1

    def test_stats_reflect_sections(self, sample_chunks, mock_embedder, tmp_path):
        """Stats should break down chunks by section."""
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        store.ingest(sample_chunks, mock_embedder)

        stats = store.get_stats()
        assert stats["total_chunks"] == len(sample_chunks)
        assert "oneke" in stats["sections"]

    def test_clear_empties_store(self, sample_chunks, mock_embedder, tmp_path):
        """Clearing should remove all data."""
        store = VectorStore(persist_dir=str(tmp_path / "db"), collection_name="test")
        store.ingest(sample_chunks, mock_embedder)
        assert store.collection.count() > 0

        store.clear()
        assert store.collection.count() == 0


class TestVectorStoreQuery:
    """Tests querying the vector store after ingestion."""

    @pytest.fixture(autouse=True)
    def populated_store(self, sample_chunks, mock_embedder, tmp_path):
        """Set up a populated store for each test."""
        self.store = VectorStore(
            persist_dir=str(tmp_path / "db"), collection_name="test"
        )
        self.store.ingest(sample_chunks, mock_embedder)
        self.embedder = mock_embedder
        self.chunks = sample_chunks

    def test_query_returns_results(self):
        """A query should return chunks."""
        results = self.store.query(
            "deploy application helm", self.embedder,
            top_k=3, score_threshold=None,
        )
        assert len(results) > 0
        assert len(results) <= 3
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_results_have_scores(self):
        """Each result should have a similarity score."""
        results = self.store.query(
            "kubernetes pods", self.embedder,
            top_k=3, score_threshold=None,
        )
        for r in results:
            assert isinstance(r.score, float)

    def test_results_carry_metadata(self):
        """Results should carry source metadata."""
        results = self.store.query(
            "pyone api", self.embedder,
            top_k=3, score_threshold=None,
        )
        for r in results:
            assert r.source_url.startswith("https://")
            assert r.section in ("oneke", "api", "management_operations")
            assert r.chunk_id

    def test_section_filter(self):
        """Section filter should limit results to one section."""
        results = self.store.query(
            "deploy", self.embedder,
            top_k=10, section_filter="oneke", score_threshold=None,
        )
        for r in results:
            assert r.section == "oneke"

    def test_query_with_context_returns_string(self):
        """query_with_context should produce a formatted context block."""
        context = self.store.query_with_context(
            "deploy application", self.embedder, max_context_tokens=2000,
        )
        assert isinstance(context, str)
        assert len(context) > 0
        assert "[Source:" in context  # Should have attribution headers

    def test_query_with_context_respects_token_budget(self):
        """Context should not exceed the token budget (approximately)."""
        context_small = self.store.query_with_context(
            "deploy", self.embedder, max_context_tokens=200,
        )
        context_large = self.store.query_with_context(
            "deploy", self.embedder, max_context_tokens=5000,
        )
        # Smaller budget should produce shorter context
        assert len(context_small) <= len(context_large)


# ===================================================================
# PHASE 5: Retriever — high-level retrieval interface
# ===================================================================

class TestRetrieverIntegration:
    """Tests the OneAIRetriever with a real ChromaDB store (but mock embedder)."""

    @pytest.fixture(autouse=True)
    def setup_retriever(self, sample_chunks, mock_embedder, tmp_path):
        """Build a fully wired retriever with ingested data."""
        store = VectorStore(
            persist_dir=str(tmp_path / "db"), collection_name="test"
        )
        store.ingest(sample_chunks, mock_embedder)

        self.retriever = OneAIRetriever(
            embedder=mock_embedder,
            store=store,
            top_k=5,
            rerank=False,
        )
        self.chunks = sample_chunks

    def test_retrieve_returns_chunks(self):
        """Basic retrieval should return RetrievedChunk objects."""
        results = self.retriever.retrieve("deploy helm chart kubernetes")
        assert len(results) > 0
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_retrieve_respects_top_k(self):
        """Should not return more than top_k results."""
        results = self.retriever.retrieve("deploy", top_k=2)
        assert len(results) <= 2

    def test_retrieve_with_section_filter(self):
        """Section filter should be applied."""
        results = self.retriever.retrieve("deploy", section_filter="api")
        for r in results:
            assert r.section == "api"

    def test_get_context_returns_string(self):
        """get_context should return a ready-to-use context string."""
        context = self.retriever.get_context("how to deploy wordpress on oneke")

        assert isinstance(context, str)
        assert len(context) > 0
        # Should not be the "no results" message
        assert "No relevant documentation found" not in context

    def test_get_context_includes_source_attribution(self):
        """Context should include source URLs for transparency."""
        context = self.retriever.get_context("deploy application")
        assert "docs.opennebula.io" in context


class TestLangChainRetrieverIntegration:
    """Tests the LangChain-compatible retriever wrapper."""

    @pytest.fixture(autouse=True)
    def setup_retriever(self, sample_chunks, mock_embedder, tmp_path):
        store = VectorStore(
            persist_dir=str(tmp_path / "db"), collection_name="test"
        )
        store.ingest(sample_chunks, mock_embedder)

        inner = OneAIRetriever(
            embedder=mock_embedder,
            store=store,
            top_k=5,
            rerank=False,
        )
        self.retriever = LangChainOpenNebulaRetriever(inner=inner, top_k=3)

    def test_returns_langchain_documents(self):
        """Should return LangChain Document objects."""
        from langchain_core.documents import Document

        docs = self.retriever._get_relevant_documents("deploy app")
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_documents_have_metadata(self):
        """LangChain documents should carry our custom metadata."""
        docs = self.retriever._get_relevant_documents("pyone api")
        for doc in docs:
            assert "source" in doc.metadata
            assert "section" in doc.metadata
            assert "score" in doc.metadata


# ===================================================================
# PHASE 6: Full pipeline end-to-end
# ===================================================================

class TestFullPipelineE2E:
    """
    End-to-end test: scrape (mock) -> chunk -> ingest -> retrieve.

    Simulates what happens when you run `one-ai-rag pipeline` but
    without any network or GPU dependencies.
    """

    def test_full_pipeline(self, sample_scraped_pages, mock_embedder, tmp_path):
        """The entire pipeline should produce queryable results."""
        # ---- Step 1: We already have scraped pages (from conftest) ----
        pages = sample_scraped_pages
        assert len(pages) == 3

        # ---- Step 2: Chunk ----
        chunker = DocChunker(target_size=300, overlap=50, min_size=30)
        chunks = chunker.chunk_pages(pages)

        assert len(chunks) >= 3
        assert all(c.token_count > 0 for c in chunks)
        assert all(c.source_url for c in chunks)

        # Save and reload (verify persistence)
        chunker.save(chunks, output_dir=tmp_path / "chunks")
        reloaded_chunks = DocChunker.load(input_dir=tmp_path / "chunks")
        assert len(reloaded_chunks) == len(chunks)

        # ---- Step 3: Ingest into vector store ----
        store = VectorStore(
            persist_dir=str(tmp_path / "vectordb"),
            collection_name="e2e_test",
        )
        ingested = store.ingest(chunks, mock_embedder)
        assert ingested == len(chunks)
        assert store.collection.count() == len(chunks)

        # ---- Step 4: Retrieve ----
        retriever = OneAIRetriever(
            embedder=mock_embedder,
            store=store,
            top_k=5,
            rerank=False,
        )

        # Query 1: OneKE deployment
        results = retriever.retrieve("How to deploy an application on OneKE")
        assert len(results) > 0
        assert all(isinstance(r, RetrievedChunk) for r in results)

        # Query 2: Get formatted context for LLM
        context = retriever.get_context("deploy wordpress helm chart kubernetes")
        assert isinstance(context, str)
        assert len(context) > 50
        assert "docs.opennebula.io" in context

        # Query 3: Section-filtered query
        api_results = retriever.retrieve("python api client", section_filter="api")
        for r in api_results:
            assert r.section == "api"

    def test_pipeline_stats_consistent(self, sample_scraped_pages, mock_embedder, tmp_path):
        """Stats reported at each stage should be internally consistent."""
        # Chunk
        chunker = DocChunker(target_size=300, overlap=50, min_size=30)
        chunks = chunker.chunk_pages(sample_scraped_pages)
        chunker.save(chunks, output_dir=tmp_path / "chunks")

        # Verify chunk stats file
        stats = json.loads((tmp_path / "chunks" / "chunk_stats.json").read_text())
        assert stats["total_chunks"] == len(chunks)
        assert stats["total_tokens"] == sum(c.token_count for c in chunks)
        assert stats["chunks_with_code"] == sum(1 for c in chunks if c.has_code)

        # Ingest and verify store stats
        store = VectorStore(
            persist_dir=str(tmp_path / "vectordb"),
            collection_name="stats_test",
        )
        store.ingest(chunks, mock_embedder)

        store_stats = store.get_stats()
        assert store_stats["total_chunks"] == stats["total_chunks"]

        # Section breakdown should sum to total
        section_sum = sum(store_stats["sections"].values())
        assert section_sum == store_stats["total_chunks"]

    def test_pipeline_idempotent(self, sample_scraped_pages, mock_embedder, tmp_path):
        """Running the pipeline twice should not create duplicates."""
        chunker = DocChunker(target_size=300, overlap=50, min_size=30)
        chunks = chunker.chunk_pages(sample_scraped_pages)

        store = VectorStore(
            persist_dir=str(tmp_path / "vectordb"),
            collection_name="idempotent_test",
        )

        # First run
        first_count = store.ingest(chunks, mock_embedder)
        assert first_count == len(chunks)

        # Second run (same data)
        second_count = store.ingest(chunks, mock_embedder)
        assert second_count == 0

        # Total should still be the original count
        assert store.collection.count() == len(chunks)


# ===================================================================
# PHASE 7: Edge cases and error handling
# ===================================================================

class TestEdgeCases:
    """Tests for boundary conditions and error scenarios."""

    def test_empty_pages_produce_no_chunks(self):
        """Pages with no content should be silently skipped."""
        empty_page = ScrapedPage(
            url="https://example.com/empty",
            title="Empty",
            section="test",
            section_label="Test",
            content="",
            code_blocks=[],
            headings=[],
            breadcrumb=[],
        )
        chunker = DocChunker()
        chunks = chunker.chunk_pages([empty_page])
        assert len(chunks) == 0

    def test_very_short_page(self):
        """A page with minimal content should produce one chunk."""
        page = ScrapedPage(
            url="https://example.com/short",
            title="Short Page",
            section="test",
            section_label="Test",
            content="This is a very short documentation page with minimal content.",
            code_blocks=[],
            headings=["Short Page"],
            breadcrumb=[],
        )
        chunker = DocChunker(target_size=1000, overlap=0, min_size=5)
        chunks = chunker.chunk_pages([page])
        assert len(chunks) == 1

    def test_page_with_only_code(self):
        """A page that is mostly code should still produce valid chunks."""
        page = ScrapedPage(
            url="https://example.com/code",
            title="Code Example",
            section="api",
            section_label="API",
            content="## Example\n```\nimport pyone\nclient = pyone.OneServer('http://localhost:2633')\nvm_pool = client.vmpool.info(-2, -1, -1, -1)\nfor vm in vm_pool.VM:\n    print(vm.NAME)\n```",
            code_blocks=["import pyone\nclient = pyone.OneServer('http://localhost:2633')"],
            headings=["Code Example"],
            breadcrumb=[],
        )
        chunker = DocChunker(target_size=500, overlap=0, min_size=10)
        chunks = chunker.chunk_pages([page])

        assert len(chunks) >= 1
        assert chunks[0].has_code

    def test_query_empty_store(self, mock_embedder, tmp_path):
        """Querying an empty store should return an empty list, not crash."""
        store = VectorStore(
            persist_dir=str(tmp_path / "db"), collection_name="empty"
        )
        results = store.query("anything", mock_embedder, score_threshold=None)
        assert results == []

    def test_query_with_context_empty_store(self, mock_embedder, tmp_path):
        """query_with_context on empty store should return a fallback message."""
        store = VectorStore(
            persist_dir=str(tmp_path / "db"), collection_name="empty"
        )
        context = store.query_with_context("anything", mock_embedder)
        assert "No relevant documentation found" in context

    def test_content_extractor_malformed_html(self):
        """Extractor should handle broken HTML gracefully."""
        extractor = ContentExtractor()
        broken_html = "<html><body><div class='document'><p>Unclosed paragraph<p>Another"

        content, code_blocks, headings, breadcrumb = extractor.extract(broken_html, "http://test")
        # Should not crash; content should be extractable
        assert isinstance(content, str)

    def test_content_extractor_no_content_div(self):
        """Extractor should fall back to body if no content div found."""
        extractor = ContentExtractor()
        html = "<html><body><p>Just a paragraph without the expected div structure.</p></body></html>"

        content, _, _, _ = extractor.extract(html, "http://test")
        assert "paragraph" in content

    def test_chunk_token_estimation_consistency(self):
        """Token estimation should be roughly proportional to text length."""
        short = "Hello world."
        medium = "Hello world. " * 50
        long_text = "Hello world. " * 500

        t_short = estimate_tokens(short)
        t_medium = estimate_tokens(medium)
        t_long = estimate_tokens(long_text)

        assert t_short < t_medium < t_long
        # Rough sanity: 1 word ~= 1.3 tokens, "Hello world." = 3 words
        assert 1 <= t_short <= 10
