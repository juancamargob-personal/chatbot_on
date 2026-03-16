"""
one_ai_core.chain
==================
The core LangChain orchestration chain.

Flow
----
1. Retrieve relevant OpenNebula doc chunks from ChromaDB via one-ai-rag.
2. Build the prompt (system + RAG context + user request).
3. Call the local Ollama LLM.
4. Validate the raw output with one-ai-config's ConfigValidator.
5. If validation fails, feed the error summary back to the LLM and retry
   (up to ``config.max_retries`` times).
6. On success, generate the executable Python script via CodeGenerator.

Returns a ``ChainResult`` dataclass with the validated config, generated
script, and metadata about retries / warnings.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .prompts import SYSTEM_PROMPT, USER_PROMPT, RETRY_PROMPT

if TYPE_CHECKING:
    # Lazy imports so the module loads even if sibling packages aren't present
    from one_ai_config.schema.base import OneAIConfig
    from one_ai_config.codegen.generator import GeneratedScript

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChainResult:
    """Everything the chain produced for a single user request."""
    request: str

    # Happy path
    config: "OneAIConfig | None" = None
    config_yaml: str = ""
    script: "GeneratedScript | None" = None

    # Diagnostics
    attempts: int = 0
    warnings: list[str] = field(default_factory=list)
    rag_chunks: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    # Failure path
    success: bool = False
    error: str = ""

    def summary(self) -> str:
        if self.success:
            steps = len(self.config.steps) if self.config else 0
            return (
                f"✅  Generated config '{self.config.metadata.name}' "
                f"with {steps} step(s) in {self.attempts} attempt(s) "
                f"({self.elapsed_seconds:.1f}s)"
            )
        return f"❌  Failed after {self.attempts} attempt(s): {self.error}"


# ---------------------------------------------------------------------------
# Chain class
# ---------------------------------------------------------------------------

class OneAIChain:
    """
    Wires together RAG retrieval, the Ollama LLM, and the config validator.

    Parameters
    ----------
    config:
        ``CoreConfig`` instance.  Defaults to a fresh one (reads env vars).
    retriever:
        Optional pre-built ``LangChainOpenNebulaRetriever``.  If ``None``,
        the chain will try to import and build one from one-ai-rag.
    """

    def __init__(
        self,
        config: "CoreConfig | None" = None,
        retriever=None,
    ) -> None:
        if config is None:
            from .config import CoreConfig
            config = CoreConfig()
        self.cfg = config
        self._retriever = retriever
        self._llm = None          # lazy-init on first call
        self._validator = None    # lazy-init on first call
        self._codegen = None      # lazy-init on first call

    # ------------------------------------------------------------------
    # Lazy initialisation helpers
    # ------------------------------------------------------------------

    def _get_llm(self):
        """Build the LangChain LLM (cached after first call).

        The actual backend (Ollama or OpenAI) is chosen by cfg.llm_backend.
        See one_ai_core.llm.build_llm for details.
        """
        if self._llm is None:
            from .llm import build_llm
            self._llm = build_llm(self.cfg)
            logger.debug("Initialised LLM backend: %s", self.cfg.active_model)
        return self._llm

    def _get_retriever(self):
        """Build the RAG retriever (cached after first call)."""
        if self._retriever is None:
            try:
                from one_ai_rag.retriever import LangChainOpenNebulaRetriever
                self._retriever = LangChainOpenNebulaRetriever(
                    top_k=self.cfg.rag_top_k,
                    rerank=self.cfg.rag_rerank,
                )
                logger.debug("Initialised RAG retriever (top_k=%d)", self.cfg.rag_top_k)
            except Exception as exc:
                logger.warning("RAG retriever unavailable: %s — running without context", exc)
        return self._retriever

    def _get_validator(self):
        if self._validator is None:
            from one_ai_config.validator import ConfigValidator
            self._validator = ConfigValidator()
        return self._validator

    def _get_codegen(self):
        if self._codegen is None:
            from one_ai_config.codegen.generator import CodeGenerator
            self._codegen = CodeGenerator()
        return self._codegen

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------

    def _retrieve_context(self, request: str) -> tuple[str, list[dict]]:
        """
        Retrieve RAG chunks for ``request``.

        Returns ``(formatted_context_string, raw_chunks_list)``.
        Falls back to empty string if retriever is unavailable.
        """
        retriever = self._get_retriever()
        if retriever is None:
            return "(No documentation context available.)", []

        try:
            docs = retriever.get_relevant_documents(request)
            chunks = []
            parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "unknown")
                text = doc.page_content.strip()
                parts.append(f"[{i}] Source: {source}\n{text}")
                chunks.append({"source": source, "text": text})
            context = "\n\n---\n\n".join(parts) if parts else "(No relevant docs found.)"
            return context, chunks
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return "(RAG retrieval error.)", []

    # ------------------------------------------------------------------
    # LLM call helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        request: str,
        rag_context: str,
        conversation_history: list[dict],
    ) -> list[dict]:
        """
        Build the message list for the LLM.

        ``conversation_history`` holds previous (assistant, user) pairs for
        the retry loop.
        """
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(rag_context=rag_context)),
        ]
        # Replay previous turns (for retry: alternating assistant/user)
        for turn in conversation_history:
            if turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
            else:
                messages.append(HumanMessage(content=turn["content"]))

        messages.append(HumanMessage(content=USER_PROMPT.format(user_request=request)))
        return messages

    def _call_llm(self, messages: list) -> str:
        """Invoke the LLM and return the raw string response."""
        llm = self._get_llm()
        # LangChain's invoke() accepts a list of messages for chat models;
        # for plain LLMs we join them into a single string.
        try:
            response = llm.invoke(messages)
        except Exception:
            # Some older integrations need a plain string
            prompt_text = "\n\n".join(
                (m.content if hasattr(m, "content") else str(m)) for m in messages
            )
            response = llm.invoke(prompt_text)

        return response if isinstance(response, str) else response.content

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, request: str) -> ChainResult:
        """
        Execute the full pipeline for ``request``.

        Parameters
        ----------
        request:
            Natural-language infrastructure request from the user.

        Returns
        -------
        ChainResult
            Contains the validated config, generated script, and diagnostics.
        """
        t0 = time.monotonic()
        result = ChainResult(request=request)

        # 1. RAG retrieval (once per request, not per retry)
        rag_context, chunks = self._retrieve_context(request)
        result.rag_chunks = chunks
        logger.debug("Retrieved %d RAG chunks", len(chunks))

        validator = self._get_validator()
        conversation_history: list[dict] = []

        for attempt in range(1, self.cfg.max_retries + 1):
            result.attempts = attempt
            logger.info("Attempt %d / %d", attempt, self.cfg.max_retries)

            # 2. Build messages (first attempt = initial prompt; subsequent = retry prompt)
            if attempt == 1:
                messages = self._build_messages(request, rag_context, [])
            else:
                # Append retry instruction with validation errors to history
                last_error = conversation_history[-1]["content"]  # set below on failure
                retry_msg = RETRY_PROMPT.format(error_summary=last_error)
                conversation_history.append({"role": "user", "content": retry_msg})
                messages = self._build_messages(request, rag_context, conversation_history)

            # 3. LLM call
            try:
                raw_output = self._call_llm(messages)
                logger.debug("LLM raw output (attempt %d):\n%s", attempt, raw_output[:500])
            except Exception as exc:
                result.error = f"LLM call failed: {exc}"
                logger.error(result.error)
                break

            # Record assistant turn for potential next retry
            conversation_history.append({"role": "assistant", "content": raw_output})

            # 4. Validate
            validation = validator.validate(raw_output)

            if not validation.is_valid:
                error_summary = validation.error_summary()
                logger.warning("Validation failed (attempt %d):\n%s", attempt, error_summary)
                # Store error for the next retry prompt
                conversation_history.append({"role": "user", "content": error_summary})
                result.error = error_summary
                continue

            # 5. Validation passed — collect warnings
            result.warnings = validation.warnings
            result.config = validation.config
            result.config_yaml = raw_output

            # 6. Code generation
            try:
                codegen = self._get_codegen()
                result.script = codegen.generate(validation.config)
                logger.info("Code generation successful")
            except Exception as exc:
                logger.warning("Code generation failed (non-fatal): %s", exc)
                result.warnings.append(f"Code generation warning: {exc}")

            result.success = True
            break

        result.elapsed_seconds = time.monotonic() - t0
        logger.info(result.summary())
        return result
