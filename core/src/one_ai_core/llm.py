"""
one_ai_core.llm
================
LLM factory.

Returns the correct LangChain LLM object based on ``CoreConfig.llm_backend``.
Keeping this in its own module means chain.py never has to import both Ollama
and OpenAI — it just calls ``build_llm(cfg)`` and gets back a unified object.

Supported backends
------------------
ollama  — Local Ollama server (default).  Free, private, runs on your NUC.
          Requires: pip install langchain-ollama
          Model pulled with: ollama pull mistral:7b-instruct-v0.3-q4_K_M

openai  — OpenAI Chat API (GPT-4o by default).  Used for comparison / eval.
          Requires: pip install langchain-openai
          Key set via: export OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_llm(cfg):
    """
    Build and return a LangChain LLM for the given config.

    The returned object always supports ``.invoke(messages)`` where
    ``messages`` is a list of LangChain message objects (SystemMessage,
    HumanMessage, AIMessage).

    Parameters
    ----------
    cfg:
        Active ``CoreConfig``.

    Returns
    -------
    A LangChain BaseChatModel or BaseLLM instance.

    Raises
    ------
    ValueError
        If ``cfg.llm_backend`` is not one of the supported values.
    ImportError
        If the required package for the chosen backend is not installed.
    """
    backend = cfg.llm_backend.lower().strip()

    if backend == "ollama":
        return _build_ollama(cfg)
    elif backend == "openai":
        return _build_openai(cfg)
    else:
        raise ValueError(
            f"Unknown LLM backend: {cfg.llm_backend!r}. "
            "Valid options: 'ollama', 'openai'."
        )


# ---------------------------------------------------------------------------
# Backend builders
# ---------------------------------------------------------------------------

def _build_ollama(cfg):
    """
    Build a LangChain Ollama LLM.

    Tries ``langchain_ollama`` first (newer, dedicated package), falls back
    to ``langchain_community.llms.Ollama`` for older installs.
    """
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=cfg.ollama_temperature,
        )
        logger.debug(
            "Built ChatOllama: model=%s  base_url=%s",
            cfg.ollama_model,
            cfg.ollama_base_url,
        )
        return llm
    except ImportError:
        pass

    # Fallback: langchain_community
    try:
        from langchain_community.chat_models import ChatOllama as CommunityChatOllama
        llm = CommunityChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=cfg.ollama_temperature,
        )
        logger.debug("Built community ChatOllama (fallback)")
        return llm
    except ImportError:
        pass

    # Last resort: plain LLM (non-chat)
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=cfg.ollama_temperature,
        )
        logger.debug("Built plain Ollama LLM (last resort fallback)")
        return llm
    except ImportError as exc:
        raise ImportError(
            "Could not import any Ollama LangChain integration.\n"
            "Run: pip install langchain-ollama"
        ) from exc


def _build_openai(cfg):
    """
    Build a LangChain OpenAI chat model.

    Requires ``langchain-openai`` and a valid ``OPENAI_API_KEY``.
    """
    if not cfg.openai_api_key:
        raise ValueError(
            "OpenAI backend selected but no API key found.\n"
            "Set it with:  export OPENAI_API_KEY=sk-...\n"
            "Or:           export ONEAI_CORE_OPENAI_API_KEY=sk-..."
        )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "Could not import langchain_openai.\n"
            "Run: pip install langchain-openai"
        ) from exc

    llm = ChatOpenAI(
        model=cfg.openai_model,
        temperature=cfg.openai_temperature,
        timeout=cfg.openai_timeout,
        api_key=cfg.openai_api_key,
    )
    logger.debug("Built ChatOpenAI: model=%s", cfg.openai_model)
    return llm
