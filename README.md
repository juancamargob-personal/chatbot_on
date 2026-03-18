# OpenNebula AI Configuration Assistant

An AI-powered system that translates natural-language infrastructure requests into validated, executable OpenNebula configurations.

```
User: "Deploy WordPress on my OneKE cluster with a 10Gi persistent volume"

    → RAG retrieval (OpenNebula docs)
    → LLM generation (Mistral 7B / GPT-4o)
    → Schema validation (Pydantic)
    → Code generation (Python script)
    → Execution with rollback (planned)
```

## What it does

You describe what you want in plain English. The system produces:

1. **Structured YAML config** — validated against a strict Pydantic schema with dependency tracking, rollback steps, and pre/post-checks
2. **Executable Python script** — ready-to-run code using `kubectl`, `helm`, and `pyone` to apply the config to your OpenNebula cluster
3. **Safety guardrails** — risk assessment, human approval gates, and automatic rollback on failure

## Architecture

```
┌─────────────────┐
│  User Request    │  "Deploy WordPress on OneKE"
└────────┬────────┘
         ▼
┌─────────────────┐
│  one-ai-rag     │  ChromaDB + sentence-transformers retrieval
│                 │  1223 chunks from docs.opennebula.io v7.0
└────────┬────────┘
         ▼
┌─────────────────┐
│  LLM            │  Local Ollama (Mistral 7B) or OpenAI GPT-4o
│                 │  Few-shot prompting + retry loop
└────────┬────────┘
         ▼
┌─────────────────┐
│  one-ai-config  │  Pydantic schema validation + code generation
│                 │  26 action types, dependency graph, rollback
└────────┬────────┘
         ▼
┌─────────────────┐
│  one-ai-core    │  CLI orchestrator: generate / plan / apply
│                 │  Wires RAG → LLM → validator → codegen
└────────┬────────┘
         ▼
┌─────────────────┐
│  one-ai-agent   │  Execution engine (planned)
│                 │  Dry-run, approval gate, rollback
└─────────────────┘
```

## Packages

| Package | Status | Description |
|---------|--------|-------------|
| `RAG/` | ✅ Complete | Scrapes OpenNebula v7.0 docs, chunks, embeds with `all-mpnet-base-v2`, stores in ChromaDB, retrieves with cross-encoder reranking |
| `config/` | ✅ Complete | Pydantic schema for YAML configs, 26 registered action types (OneKE + OpenNebula VM), Jinja2-free Python code generator |
| `core/` | ✅ Functional | CLI orchestrator with few-shot prompting, YAML extraction/patching, retry loop, backend comparison |
| `finetune/` | 🔧 In progress | QLoRA fine-tuning pipeline: synthetic data generation, training, schema evaluation, LLM judge |
| `agent/` | 📋 Planned | Step-by-step execution engine with dry-run, approval gates, and automatic rollback |

## Quick start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) with `mistral:7b-instruct-v0.3-q4_K_M` pulled
- ~4GB disk for ChromaDB + model weights

### Installation

```bash
git clone https://github.com/juancamargob-personal/chatbot_on.git
cd chatbot_on

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages (--no-deps avoids pip resolution hell)
cd RAG && pip install -e . --no-deps && cd ..
cd config && pip install -e . --no-deps && cd ..
cd core && pip install -e . --no-deps && cd ..

# Install dependencies from lockfile
pip install -r requirements-lock.txt

# Pull the LLM model
ollama pull mistral:7b-instruct-v0.3-q4_K_M
```

### Build the RAG index

```bash
one-ai-rag pipeline    # Scrapes, chunks, embeds, stores (~15 min first run)
one-ai-rag stats       # Verify: should show 1223 chunks
```

### Generate a config

```bash
# Generate YAML config only
one-ai generate "Deploy WordPress on my OneKE cluster"

# Generate config + show the Python execution script
one-ai plan "Deploy WordPress on my OneKE cluster"

# Generate config + script and save to disk
one-ai apply "Deploy WordPress on OneKE" --output-dir ./runs/wordpress

# Show current settings
one-ai config
```

### Run tests

```bash
# Smoke tests (fast, no external services needed)
cd core && pytest tests/test_core_smoke.py -v

# Integration tests (requires Ollama + ChromaDB)
cd core && pytest tests/ -v --integration

# Config package tests
cd config && pytest tests/ -v
```

## Supported actions

### OneKE / Kubernetes
| Action | Description |
|--------|-------------|
| `oneke.namespace.create/delete/list` | Kubernetes namespace management |
| `oneke.app.deploy/uninstall/upgrade` | Helm-based application lifecycle |
| `oneke.app.wait_ready/get_status/list` | Deployment monitoring |
| `oneke.service.get_endpoint/expose/list` | Service discovery and exposure |
| `oneke.storage.create_pvc/list_pvcs/delete_pvc` | Persistent volume management |
| `oneke.cluster.get_info/get_status/list_nodes/scale_nodes` | Cluster operations |

### OpenNebula VMs
| Action | Description |
|--------|-------------|
| `one.vm.create/delete` | VM lifecycle |
| `one.vm.poweroff/resume` | Power management |
| `one.vm.list/resize/snapshot_create` | VM operations |

## Example output

Input: `"Create a namespace called wordpress on my OneKE cluster"`

```yaml
metadata:
  description: Create wordpress namespace on the OneKE cluster
  risk_level: low
  tags: [namespace, oneke]
steps:
  - id: step_01
    action: oneke.namespace.create
    description: Create the wordpress namespace
    params:
      name: wordpress
    depends_on: []
    on_failure: abort
validation:
  pre_checks:
    - type: cluster_reachable
      target: oneke
      description: Verify the OneKE cluster is reachable
  post_checks:
    - type: namespace_exists
      target: wordpress
      description: Verify the wordpress namespace was created
rollback:
  enabled: false
  steps: []
```

## Technical decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM framework | LangChain | Ecosystem support, Ollama integration |
| Local LLM | Mistral 7B (Q4_K_M) | Runs on consumer GPU, decent instruction following |
| Prompting | Few-shot conversation turns | Small models learn from examples, not long instructions |
| Embeddings | `all-mpnet-base-v2` (768 dims) | Better retrieval quality than MiniLM |
| Vector store | ChromaDB (persistent, local) | Simple, no external services |
| Schema validation | Pydantic v2 | Types as documentation, fast validation |
| Fine-tuning | QLoRA (4-bit, rank 16) | Fits on single 16GB GPU |
| Config format | YAML (imperative, ordered steps) | K8s ecosystem alignment |

## Base model performance

With the base Mistral 7B model (no fine-tuning):

| Request type | Pass rate | Notes |
|-------------|-----------|-------|
| VM creation | ~90% | Consistently produces valid schema |
| Namespace creation | ~50% | Sometimes omits `steps` key entirely |
| Multi-step deploys | ~40% | May produce K8s manifests instead of custom schema |

The fine-tuning pipeline (`finetune/`) is designed to improve these rates to >95% using QLoRA training on synthetic data generated from the 5 gold examples.

## Project structure

```
chatbot_on/
├── RAG/                      one-ai-rag: retrieval pipeline
│   ├── src/one_ai_rag/       Scraper, chunker, embedder, store, retriever
│   ├── tests/                E2E + unit tests
│   └── data/                 Scraped docs, chunks, ChromaDB
├── config/                   one-ai-config: schema + code generator
│   ├── src/one_ai_config/    Schema (base + oneke), validator, codegen
│   └── tests/                43 tests
├── core/                     one-ai-core: CLI orchestrator
│   ├── src/one_ai_core/      Chain, prompts, LLM factory, CLI, compare
│   └── tests/                Smoke + integration tests
├── finetune/                 one-ai-finetune: training pipeline
│   ├── src/one_ai_finetune/  Synthetic data gen, QLoRA training, evaluation
│   ├── tests/                25 passed, 3 to fix
│   └── data/seed/            5 gold training examples
├── requirements-lock.txt     Frozen pip dependencies
└── PROJECT_HANDOFF.md        Detailed session handoff notes
```

## License

MIT
