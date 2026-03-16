"""
one_ai_core
===========
CLI orchestrator for the OpenNebula AI Configuration Assistant.

Wires together:
  one-ai-rag     (RAG retrieval)
  local Ollama   (LLM generation)
  one-ai-config  (validation + code generation)

Typical programmatic use::

    from one_ai_core.chain import OneAIChain
    from one_ai_core.config import CoreConfig

    cfg = CoreConfig()
    chain = OneAIChain(config=cfg)
    result = chain.run("Deploy WordPress on my OneKE cluster")

    if result.success:
        print(result.config_yaml)
        print(result.script.code)
"""

from .chain import ChainResult, OneAIChain
from .config import CoreConfig

__all__ = ["OneAIChain", "ChainResult", "CoreConfig"]
__version__ = "0.1.0"
