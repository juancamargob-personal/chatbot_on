"""
conftest.py
===========
Pytest configuration for one-ai-core.

The ``--integration`` flag enables tests marked with ``@pytest.mark.integration``.
Without it (the default), all integration tests are skipped automatically.

Usage
-----
    pytest tests/               # runs only unit/smoke tests (fast, no services needed)
    pytest tests/ --integration # runs everything including live Ollama + OpenAI
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that require live Ollama, ChromaDB, and/or OpenAI API.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require live external services (Ollama, OpenAI, ChromaDB)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --integration is passed."""
    if config.getoption("--integration"):
        # --integration passed: run everything, don't skip
        return

    skip_integration = pytest.mark.skip(
        reason="Integration test — pass --integration to run"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
