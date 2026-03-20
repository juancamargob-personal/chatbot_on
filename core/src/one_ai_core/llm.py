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

Fine-tuned mode
---------------
When ``cfg.is_finetuned`` is True, the Ollama backend uses ``OllamaLLM``
(which hits /api/generate) instead of ``ChatOllama`` (which hits /api/chat).

This is critical because the Modelfile TEMPLATE containing the ``<<SYS>>``
block only gets applied by the /api/generate endpoint.  The /api/chat
endpoint formats messages its own way, bypassing the TEMPLATE and causing
the fine-tuned model to fall back to base Mistral behavior.

OllamaLLM.invoke() accepts a plain string and returns a plain string.
ChatOllama.invoke() accepts messages and returns an AIMessage.
The chain handles both cases.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_llm(cfg):
    """
    Build and return a LangChain LLM for the given config.

    For base models:  returns ChatOllama (uses /api/chat).
    For fine-tuned:   returns OllamaLLM  (uses /api/generate, applies TEMPLATE).

    Parameters
    ----------
    cfg:
        Active ``CoreConfig``.

    Returns
    -------
    A LangChain BaseChatModel or BaseLLM instance.
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

    Fine-tuned models use OllamaLLM (/api/generate) so the Modelfile
    TEMPLATE with <<SYS>> is applied correctly.

    Base models use ChatOllama (/api/chat) for structured message handling
    with few-shot examples.
    """
    if cfg.is_finetuned:
        return _build_ollama_generate(cfg)
    else:
        return _build_ollama_chat(cfg)


def _build_ollama_generate(cfg):
    """
    Build OllamaLLM for fine-tuned models.

    Uses /api/generate endpoint which applies the Modelfile TEMPLATE.
    This ensures the <<SYS>> system prompt block is formatted correctly,
    matching the training format.

    OllamaLLM.invoke(prompt_string) -> string
    """
    try:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=cfg.ollama_temperature,
        )
        logger.debug(
            "Built OllamaLLM (generate endpoint) for fine-tuned model: %s",
            cfg.ollama_model,
        )
        return llm
    except ImportError as exc:
        raise ImportError(
            "Could not import langchain_ollama.OllamaLLM.\n"
            "Run: pip install langchain-ollama"
        ) from exc


def _build_ollama_chat(cfg):
    """
    Build ChatOllama for base models.

    Uses /api/chat endpoint for structured message handling (SystemMessage,
    HumanMessage, AIMessage) needed for few-shot prompting.

    ChatOllama.invoke(messages) -> AIMessage
    """
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=cfg.ollama_temperature,
        )
        logger.debug("Built ChatOllama (chat endpoint): model=%s", cfg.ollama_model)
        return llm
    except ImportError:
        pass

    # Last resort: langchain_community (will show deprecation warning)
    try:
        from langchain_community.chat_models import ChatOllama as CommunityChatOllama
        llm = CommunityChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=cfg.ollama_temperature,
        )
        logger.warning(
            "Using deprecated langchain_community.ChatOllama. "
            "Run: pip install -U langchain-ollama"
        )
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
