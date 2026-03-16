# one-ai-core

CLI orchestrator for the OpenNebula AI Configuration Assistant.

Wires together `one-ai-rag` (RAG retrieval), a local Ollama LLM, and
`one-ai-config` (validation + code generation) into a single usable CLI.

## Install

```bash
# From the repo root, with sibling packages already installed:
cd core
pip install -e . --no-deps

# Or, in a clean env, install everything at once:
pip install -e ".[full]"
```

## Usage

```bash
# Generate a YAML config
one-ai generate "Deploy WordPress on my OneKE cluster"

# Show the generated Python execution script (nothing saved)
one-ai plan "Deploy WordPress on my OneKE cluster"

# Generate + save config.yaml and script.py to disk
one-ai apply "Deploy WordPress on my OneKE cluster"

# Apply with explicit output directory
one-ai apply "Scale worker nodes to 5" --output-dir ./runs/scale

# Show active configuration (reads ONEAI_CORE_* env vars)
one-ai config
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ONEAI_CORE_OLLAMA_MODEL` | `mistral:7b-instruct-v0.3-q4_K_M` | Ollama model tag |
| `ONEAI_CORE_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `ONEAI_CORE_OLLAMA_TEMPERATURE` | `0.1` | LLM temperature |
| `ONEAI_CORE_MAX_RETRIES` | `3` | Validation retry attempts |
| `ONEAI_CORE_RAG_TOP_K` | `5` | RAG chunks per query |
| `ONEAI_CORE_RAG_RERANK` | `true` | Enable cross-encoder reranking |
| `ONEAI_CORE_OUTPUT_DIR` | `<repo-root>/output` | Where to save generated files |
| `ONEAI_CORE_ONE_ENDPOINT` | `http://localhost:2633/RPC2` | OpenNebula RPC endpoint |
| `ONEAI_CORE_DRY_RUN_DEFAULT` | `false` | Default dry-run mode for `apply` |

## Running tests

```bash
cd core
pytest tests/ -v
```

The smoke tests mock all external dependencies (Ollama, ChromaDB) so they
run without any services installed.

## Architecture

```
one-ai generate / plan / apply
          │
          ▼
    cli.py (Click)
          │
          ▼
    chain.py (OneAIChain)
    ├── RAG retrieval  ──► one-ai-rag LangChainOpenNebulaRetriever
    ├── LLM call       ──► Ollama (langchain-ollama)
    ├── Validation     ──► one-ai-config ConfigValidator
    ├── Retry loop     ──► up to max_retries on failure
    └── Code gen       ──► one-ai-config CodeGenerator
```
