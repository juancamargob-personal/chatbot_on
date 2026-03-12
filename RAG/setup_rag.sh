#!/usr/bin/env bash
# ============================================================
# setup_rag.sh — Restructures flat files into proper package layout
#
# Run from inside your RAG directory:
#   chmod +x setup_rag.sh && ./setup_rag.sh
#
# After running, your directory will look like:
#   one-ai-rag/
#   ├── pyproject.toml
#   ├── README.md
#   ├── .gitignore
#   ├── data/
#   │   ├── raw/
#   │   ├── chunks/
#   │   └── vectordb/
#   ├── src/
#   │   └── one_ai_rag/
#   │       ├── __init__.py
#   │       ├── config.py
#   │       ├── scraper.py
#   │       ├── chunker.py
#   │       ├── embedder.py
#   │       ├── store.py
#   │       ├── retriever.py
#   │       └── cli.py
#   └── tests/
#       ├── __init__.py
#       ├── conftest.py
#       ├── test_pipeline_e2e.py
#       ├── test_scraper.py
#       ├── test_chunker.py
#       ├── test_embedder.py
#       ├── test_store.py
#       └── test_retriever.py
# ============================================================

set -e

echo "=== Setting up one-ai-rag project structure ==="

# Get the directory where this script is running
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "Working in: $PROJECT_DIR"

# --- Step 1: Create directory structure ---
echo "[1/4] Creating directories..."
mkdir -p src/one_ai_rag
mkdir -p tests
mkdir -p data/raw
mkdir -p data/chunks
mkdir -p data/vectordb

# --- Step 2: Move source files into src/one_ai_rag/ ---
echo "[2/4] Moving source files..."

SOURCE_FILES=(config.py scraper.py chunker.py embedder.py store.py retriever.py cli.py)
for f in "${SOURCE_FILES[@]}"; do
    if [ -f "$f" ]; then
        mv "$f" src/one_ai_rag/
        echo "  Moved $f -> src/one_ai_rag/$f"
    else
        echo "  WARNING: $f not found, skipping"
    fi
done

# --- Step 3: Move test files into tests/ ---
echo "[3/4] Moving test files..."

# Handle various naming conventions
if [ -f "test_pipelinee2e.py" ]; then
    mv test_pipelinee2e.py tests/test_pipeline_e2e.py
    echo "  Moved test_pipelinee2e.py -> tests/test_pipeline_e2e.py"
elif [ -f "test_pipeline_e2e.py" ]; then
    mv test_pipeline_e2e.py tests/test_pipeline_e2e.py
    echo "  Moved test_pipeline_e2e.py -> tests/test_pipeline_e2e.py"
fi

TEST_FILES=(test_scraper.py test_chunker.py test_embedder.py test_store.py test_retriever.py)
for f in "${TEST_FILES[@]}"; do
    if [ -f "$f" ]; then
        mv "$f" tests/
        echo "  Moved $f -> tests/$f"
    fi
done

if [ -f "conftest.py" ]; then
    mv conftest.py tests/
    echo "  Moved conftest.py -> tests/conftest.py"
fi

# --- Step 4: Create package files ---
echo "[4/4] Creating package files..."

# __init__.py for the package
cat > src/one_ai_rag/__init__.py << 'INITEOF'
"""
one-ai-rag: RAG pipeline for OpenNebula documentation.

Quickstart:
    from one_ai_rag import OneAIRetriever

    retriever = OneAIRetriever()
    context = retriever.get_context("How to deploy an app on OneKE")
    chunks  = retriever.retrieve("Helm chart deployment")

Full pipeline:
    from one_ai_rag import DocScraper, DocChunker, VectorStore, create_embedder

    # 1. Scrape docs
    scraper = DocScraper()
    pages = scraper.scrape_all()
    scraper.save(pages)

    # 2. Chunk
    chunker = DocChunker()
    chunks = chunker.chunk_pages(pages)
    chunker.save(chunks)

    # 3. Embed + ingest
    embedder = create_embedder("local")
    store = VectorStore()
    store.ingest(chunks, embedder)

    # 4. Query
    retriever = OneAIRetriever(embedder=embedder, store=store)
    results = retriever.retrieve("deploy WordPress")
"""

from one_ai_rag.chunker import DocChunk, DocChunker
from one_ai_rag.embedder import BaseEmbedder, LocalEmbedder, OpenAIEmbedder, create_embedder
from one_ai_rag.retriever import (
    LangChainOpenNebulaRetriever,
    OneAIRetriever,
    create_langchain_retriever,
)
from one_ai_rag.scraper import DocScraper, ScrapedPage
from one_ai_rag.store import RetrievedChunk, VectorStore

__all__ = [
    "DocScraper",
    "ScrapedPage",
    "DocChunker",
    "DocChunk",
    "BaseEmbedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "create_embedder",
    "VectorStore",
    "RetrievedChunk",
    "OneAIRetriever",
    "LangChainOpenNebulaRetriever",
    "create_langchain_retriever",
]
INITEOF

# __init__.py for tests
touch tests/__init__.py

# pyproject.toml
cat > pyproject.toml << 'TOMLEOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "one-ai-rag"
version = "0.1.0"
description = "RAG pipeline for OpenNebula documentation — scrape, chunk, embed, retrieve"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
    "beautifulsoup4>=4.12",
    "requests>=2.31",
    "lxml>=4.9",
    "sentence-transformers>=2.2",
    "chromadb>=0.4",
    "langchain>=0.1",
    "langchain-core>=0.1",
    "langchain-community>=0.1",
    "tiktoken>=0.5",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "rich>=13.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
openai = ["langchain-openai>=0.1", "openai>=1.0"]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff",
]

[project.scripts]
one-ai-rag = "one_ai_rag.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/one_ai_rag"]

[tool.ruff]
line-length = 100
target-version = "py311"
TOMLEOF

# .gitignore
cat > .gitignore << 'GIEOF'
# Data artifacts (generated by pipeline, not committed)
data/raw/*.json
data/chunks/*.jsonl
data/chunks/chunk_stats.json
data/vectordb/

# Python
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp

# Env
.env
GIEOF

# Clean up the setup script itself and any leftover archives
echo ""
echo "=== Structure created successfully ==="
echo ""
echo "Your project layout:"
find . -not -path './.git/*' -not -path './venv/*' -not -path './.venv/*' \
       -not -path './__pycache__/*' -not -name '*.pyc' \
       -not -path './data/vectordb/*' -not -name 'files.zip' \
       -not -name 'setup_rag.sh' \
       | head -40 | sort
echo ""
echo "=== Next steps ==="
echo "  1. pip install -e '.[dev]'"
echo "  2. one-ai-rag pipeline          # run full pipeline"
echo "  3. one-ai-rag query             # test queries interactively"
echo "  4. pytest tests/ -v             # run tests"
