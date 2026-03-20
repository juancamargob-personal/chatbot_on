"""
one_ai_core.chain
==================
The core LangChain orchestration chain.

Flow
----
1. Retrieve relevant OpenNebula doc chunks from ChromaDB via one-ai-rag.
2. Build the prompt (system + few-shot examples + RAG context + user request).
3. Call the local Ollama LLM.
4. Validate the raw output with one-ai-config's ConfigValidator.
5. If validation fails, feed the error summary back to the LLM and retry
   (up to ``config.max_retries`` times).
6. On success, generate the executable Python script via CodeGenerator.

Returns a ``ChainResult`` dataclass with the validated config, generated
script, and metadata about retries / warnings.

Fine-tuned mode
---------------
When ``config.is_finetuned`` is True, the chain builds a raw prompt string
(not LangChain messages) and sends it to OllamaLLM which uses the
/api/generate endpoint.  This is necessary because:

- The fine-tuned model requires the exact ``<<SYS>>`` prompt format used
  during QLoRA training.
- The Modelfile TEMPLATE applies this format, but only via /api/generate.
- ChatOllama uses /api/chat which formats messages differently, causing
  the model to fall back to base Mistral behavior.

The extract -> patch -> validate pipeline is identical for both modes.
"""

from __future__ import annotations

import logging
import re
import time
import yaml
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from .prompts import (
    SYSTEM_PROMPT, USER_PROMPT, RETRY_PROMPT,
    FEW_SHOT_USER, FEW_SHOT_ASSISTANT,
    FEW_SHOT_USER_2, FEW_SHOT_ASSISTANT_2,
    FEW_SHOT_USER_3, FEW_SHOT_ASSISTANT_3,
)

if TYPE_CHECKING:
    from one_ai_config.schema.base import OneAIConfig
    from one_ai_config.codegen.generator import GeneratedScript

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt used during fine-tuning (must match exactly)
# ---------------------------------------------------------------------------

FINETUNED_SYSTEM_PROMPT = """You are an OpenNebula infrastructure configuration assistant.
Given a natural language request, you produce a YAML configuration file that
describes the steps needed to fulfill the request on an OpenNebula / OneKE cluster.

Your output must be valid YAML following the OneAI configuration schema.
If the request is impossible or unsupported, produce an error configuration
explaining why and suggesting alternatives.

Always include:
- Proper metadata with description, risk level, and tags
- Ordered steps with dependencies
- Pre-checks and post-checks in the validation section
- Rollback steps for operations that modify the cluster"""

# Retry prompt for fine-tuned model
FINETUNED_RETRY_PROMPT = """Your previous output failed validation:

{error_summary}

Please produce a corrected YAML configuration. Fix the errors above and ensure all field names, step IDs, and action names match the OneAI schema exactly."""


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
            label = (
                getattr(self.config.metadata, "name", None)
                or getattr(self.config.metadata, "description", None)
                or self.request[:40]
            ) if self.config else self.request[:40]
            return (
                f"✅  Generated config '{label}' "
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
        self._llm = None
        self._validator = None
        self._codegen = None

    # ------------------------------------------------------------------
    # Lazy initialisation helpers
    # ------------------------------------------------------------------

    def _get_llm(self):
        if self._llm is None:
            from .llm import build_llm
            self._llm = build_llm(self.cfg)
            logger.debug("Initialised LLM backend: %s", self.cfg.active_model)
        return self._llm

    def _get_retriever(self):
        if self._retriever is None:
            try:
                from one_ai_rag.retriever import (
                    OneAIRetriever,
                    LangChainOpenNebulaRetriever,
                )
                from one_ai_rag.embedder import LocalEmbedder
                embedder = LocalEmbedder(
                    model_name=self.cfg.rag_embedding_model,
                )
                inner = OneAIRetriever(
                    embedder=embedder,
                    top_k=self.cfg.rag_top_k,
                    rerank=self.cfg.rag_rerank,
                )
                self._retriever = LangChainOpenNebulaRetriever(inner=inner)
                logger.debug(
                    "Initialised RAG retriever (top_k=%d, rerank=%s, model=%s)",
                    self.cfg.rag_top_k,
                    self.cfg.rag_rerank,
                    self.cfg.rag_embedding_model,
                )
            except Exception as exc:
                logger.warning(
                    "RAG retriever unavailable: %s — running without context", exc
                )
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
        if not self.cfg.rag_enabled:
            logger.debug("RAG disabled by config")
            return "(No documentation context available.)", []
        retriever = self._get_retriever()
        if retriever is None:
            return "(No documentation context available.)", []

        try:
            docs = retriever.invoke(request)
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
    ) -> Union[list, str]:
        """
        Build the prompt for the LLM.

        Returns
        -------
        list
            LangChain message objects for base model (ChatOllama, /api/chat).
        str
            Raw prompt string for fine-tuned model (OllamaLLM, /api/generate).
        """
        if self.cfg.is_finetuned:
            return self._build_prompt_finetuned(
                request, rag_context, conversation_history
            )

        # --- Base model: few-shot message list for ChatOllama ---
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=FEW_SHOT_USER),
            AIMessage(content=FEW_SHOT_ASSISTANT),
            HumanMessage(content=FEW_SHOT_USER_2),
            AIMessage(content=FEW_SHOT_ASSISTANT_2),
            HumanMessage(content=FEW_SHOT_USER_3),
            AIMessage(content=FEW_SHOT_ASSISTANT_3),
        ]

        for turn in conversation_history:
            if turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
            else:
                messages.append(HumanMessage(content=turn["content"]))

        messages.append(HumanMessage(content=USER_PROMPT.format(
            rag_context=rag_context,
            user_request=request,
        )))
        return messages

    def _build_prompt_finetuned(
        self,
        request: str,
        rag_context: str,
        conversation_history: list[dict],
    ) -> str:
        """
        Build a raw prompt string for the fine-tuned model.

        This is sent to OllamaLLM which hits /api/generate.  Ollama then
        applies the Modelfile TEMPLATE, which wraps the system and user
        content in the ``<<SYS>>`` / ``[INST]`` format that was used
        during training.

        The TEMPLATE expects two variables:
        - {{ .System }} -- filled from the Modelfile SYSTEM directive
        - {{ .Prompt }} -- filled from the prompt string we send here

        So we only need to send the user content (request + optional RAG
        context).  The system prompt and [INST]/<<SYS>> formatting are
        handled by the Modelfile.

        For retries, we prepend the conversation history to give the model
        context about what went wrong.
        """
        parts = []

        # Include retry history if present
        for turn in conversation_history:
            role = turn["role"]
            content = turn["content"]
            if role == "assistant":
                parts.append(f"[Previous attempt output]\n{content}")
            else:
                parts.append(content)

        # Add RAG context if available
        has_rag = rag_context and rag_context not in (
            "(No documentation context available.)",
            "(No relevant docs found.)",
            "(RAG retrieval error.)",
        )

        if has_rag:
            parts.append(
                f"OpenNebula documentation context (reference only -- "
                f"do NOT copy commands verbatim):\n\n{rag_context}\n\n---\n\n"
                f"Request: {request}"
            )
        else:
            parts.append(request)

        return "\n\n".join(parts)

    def _call_llm(self, prompt: Union[list, str]) -> str:
        """
        Call the LLM with either a message list or a raw prompt string.

        - Base model (ChatOllama):  prompt is a list of LangChain messages.
        - Fine-tuned (OllamaLLM):  prompt is a plain string.

        Both return a string.
        """
        llm = self._get_llm()

        if isinstance(prompt, str):
            # OllamaLLM -- /api/generate endpoint, returns string directly
            response = llm.invoke(prompt)
            return response if isinstance(response, str) else str(response)

        # ChatOllama -- /api/chat endpoint, returns AIMessage
        try:
            response = llm.invoke(prompt)
        except Exception:
            # Fallback: join messages into a single string
            prompt_text = "\n\n".join(
                (m.content if hasattr(m, "content") else str(m)) for m in prompt
            )
            response = llm.invoke(prompt_text)

        return response if isinstance(response, str) else response.content

    # ------------------------------------------------------------------
    # YAML extraction and patching
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_yaml(raw: str) -> str:
        """
        Extract clean YAML from an LLM response.

        Aggressively strips markdown fences, prose preambles, and trailing text.
        Works for both base and fine-tuned model outputs (both wrap in fences).
        """
        raw = raw.strip()

        # Step 1: Strip markdown fences (aggressive -- handles all edge cases)
        if "```" in raw:
            # Split on ``` and find the block that looks like YAML
            parts = raw.split("```")
            # parts[0] = before first fence, parts[1] = first block content,
            # parts[2] = after first close, etc.
            # Try each odd-indexed part (content inside fences)
            for i in range(1, len(parts), 2):
                content = parts[i].strip()
                # Remove language tag if present (yaml, yml, etc.)
                if content and content.split("\n")[0].strip().isalpha():
                    content = "\n".join(content.split("\n")[1:]).strip()
                if content:
                    raw = content
                    break

        # Step 2: Find first line starting with a known schema key
        schema_starts = ("metadata:", "steps:", "error:", "version:",
                         "validation:", "rollback:")
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith(schema_starts):
                raw = "\n".join(lines[i:]).strip()
                break

        # Step 3: Remove any remaining fences or trailing prose
        if "```" in raw:
            raw = raw[:raw.index("```")].strip()

        # Step 4: Remove YAML document separators (---) that cause
        # "expected a single document" errors. Keep only first document.
        cleaned_lines = []
        for line in raw.splitlines():
            if line.strip() == "---":
                break
            cleaned_lines.append(line)
        raw = "\n".join(cleaned_lines).strip()

        return raw

    @staticmethod
    def _patch_yaml(text: str, request: str) -> str:
        """
        Auto-fix common LLM omissions before validation.

        Mistral consistently drops 'description' from metadata and steps.
        Rather than waste retries on this, we patch it in from context.
        The fine-tuned model is much less likely to need patching, but
        this is a safe no-op when the fields are already present.
        """
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            return text  # unparseable -- let the validator report it

        if not isinstance(data, dict):
            return text

        changed = False

        # Patch metadata.description
        if "metadata" in data and isinstance(data["metadata"], dict):
            if "description" not in data["metadata"]:
                data["metadata"]["description"] = request[:100]
                changed = True
        elif "metadata" in data and isinstance(data["metadata"], str):
            # Mistral sometimes outputs metadata: "some string" instead of a dict
            data["metadata"] = {"description": data["metadata"] or request[:100]}
            changed = True
        elif "metadata" in data and data["metadata"] is None:
            data["metadata"] = {"description": request[:100]}
            changed = True
        elif "metadata" not in data and "steps" in data:
            data["metadata"] = {"description": request[:100]}
            changed = True

        # Patch step descriptions
        if "steps" in data and isinstance(data["steps"], list):
            for step in data["steps"]:
                if isinstance(step, dict) and "description" not in step:
                    action = step.get("action", "execute step")
                    params = step.get("params", {})
                    name = params.get("name", "") if isinstance(params, dict) else ""
                    step["description"] = f"{action} {name}".strip()
                    changed = True

        if changed:
            logger.debug("Auto-patched missing fields in LLM output")
            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        return text

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

        if self.cfg.is_finetuned:
            logger.info("Using fine-tuned model mode (OllamaLLM, no few-shot)")
        else:
            logger.info("Using base model mode (ChatOllama, few-shot examples)")

        # 1. RAG retrieval (once per request, not per retry)
        rag_context, chunks = self._retrieve_context(request)
        result.rag_chunks = chunks
        logger.debug("Retrieved %d RAG chunks", len(chunks))

        validator = self._get_validator()
        conversation_history: list[dict] = []

        # Pick the right retry prompt template
        retry_template = (
            FINETUNED_RETRY_PROMPT if self.cfg.is_finetuned else RETRY_PROMPT
        )

        for attempt in range(1, self.cfg.max_retries + 1):
            result.attempts = attempt
            logger.info("Attempt %d / %d", attempt, self.cfg.max_retries)

            # 2. Build prompt (messages list for base, string for fine-tuned)
            if attempt == 1:
                prompt = self._build_messages(request, rag_context, [])
            else:
                retry_msg = retry_template.format(error_summary=result.error)
                conversation_history.append({"role": "user", "content": retry_msg})
                prompt = self._build_messages(request, rag_context, conversation_history)

            # 3. LLM call
            try:
                raw_output = self._call_llm(prompt)
                logger.debug("LLM raw output (attempt %d):\n%s", attempt, raw_output[:500])
            except Exception as exc:
                result.error = f"LLM call failed: {exc}"
                logger.error(result.error)
                break

            conversation_history.append({"role": "assistant", "content": raw_output})

            # 4. Extract, patch, and validate
            clean_output = self._extract_yaml(raw_output)
            if clean_output != raw_output:
                logger.debug("Stripped prose/fences from LLM output before validation")

            # Auto-fix common omissions (safe no-op when fields already present)
            clean_output = self._patch_yaml(clean_output, request)

            validation = validator.validate(clean_output)

            if not validation.is_valid:
                error_summary = validation.error_summary()
                logger.warning("Validation failed (attempt %d):\n%s", attempt, error_summary)
                result.error = error_summary
                continue

            # 5. Validation passed
            result.warnings = validation.warnings
            result.config = validation.config
            result.config_yaml = clean_output

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
