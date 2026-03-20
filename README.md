# OpenNebula AI Configuration Assistant

An AI-powered system that translates natural language infrastructure requests into validated, executable OpenNebula configurations. Ask it to "Deploy WordPress on my OneKE cluster" and get a schema-validated YAML config + runnable Python script.

## Results

| Model | Schema Validation Pass Rate | Notes |
|-------|:--:|-------|
| Base Mistral 7B (no fine-tuning) | 0% | Produces raw kubectl/Helm commands |
| Fine-tuned v1 (300 training examples) | 93% | 1 failure pattern: deploy to existing namespace |
| **Fine-tuned v2 (406 training examples)** | **100%** | **All first attempt, 2-7s per request** |

Total cost: ~$0.06 in API calls (GPT-4o-mini for synthetic data) + 40 minutes of GPU time on an RTX A4000.

## How It Works

```
User: "Deploy WordPress on OneKE"
         │
         ▼
    ┌─ RAG ─┐         Retrieves relevant OpenNebula docs
    └───┬───┘
        ▼
  ┌─ LLM (local) ─┐   Fine-tuned Mistral 7B → structured YAML
  └───────┬────────┘
          ▼
  ┌─ Extract+Patch ┐   Strips markdown fences, patches missing fields
  └───────┬────────┘
          ▼
  ┌─ Validator ────┐   Pydantic schema validation (26 action types)
  └───────┬────────┘
          ▼
  ┌─ Code Generator ┐  Maps config → executable Python script
  └───────┬──────────┘
          ▼
    ┌─ CLI ─┐          generate / plan / apply
    └───────┘
```

## Quick Start

```bash
# Prerequisites: Python 3.10+, Ollama installed and running

# Clone and set up
git clone https://github.com/juancamargob-personal/chatbot_on.git
cd chatbot_on
python -m venv venv && source venv/bin/activate

# Install packages (use --no-deps to avoid dependency resolution issues)
cd RAG && pip install -e . --no-deps && cd ..
cd config && pip install -e . --no-deps && cd ..
cd core && pip install -e . --no-deps && cd ..

# Pull the base model (if fine-tuned model is not registered yet)
ollama pull mistral:7b-instruct-v0.3-q4_K_M

# Generate a config
export ONEAI_CORE_OLLAMA_MODEL=oneai-mistral:latest  # or mistral:7b-instruct-v0.3-q4_K_M
export ONEAI_CORE_RAG_ENABLED=false
one-ai generate "Deploy WordPress on my OneKE cluster"
```

## Example Output

**Request:** "Deploy Nginx using Helm on the default namespace"

```yaml
version: "1.0"
metadata:
  description: "Deploy Nginx on the default namespace via Helm"
  target_cluster: oneke-cluster
  estimated_duration: "5 minutes"
  risk_level: low
  tags: [nginx, helm, oneke]
steps:
  - id: step_01
    action: oneke.app.deploy
    description: "Deploy Nginx via Helm chart to default namespace"
    params:
      chart: nginx
      namespace: default
      release_name: nginx
      repo_url: https://charts.bitnami.com/bitnami
      create_namespace: false
    timeout_seconds: 300
  - id: step_02
    action: oneke.app.wait_ready
    description: "Wait for Nginx pods to be ready"
    params:
      namespace: default
      label_selector: "app.kubernetes.io/name=nginx"
      timeout_seconds: 300
    depends_on: [step_01]
validation:
  pre_checks:
    - type: cluster_reachable
      description: "Verify cluster connectivity"
    - type: namespace_exists
      description: "Check if default namespace is available"
  post_checks:
    - type: pods_running
      description: "Ensure Nginx pods are running"
rollback:
  steps:
    - id: step_90
      action: oneke.app.uninstall
      description: "Remove Nginx Helm release"
      params:
        release_name: nginx
        namespace: default
```

The system also generates a complete Python script that executes this config using kubectl, helm, and pyone.

## Project Structure

```
chatbot_on/
├── RAG/          ← Document retrieval (ChromaDB + sentence-transformers)
├── config/       ← YAML schema, Pydantic validation, code generation
├── core/         ← CLI orchestrator (LangChain + Ollama integration)
├── finetune/     ← QLoRA training pipeline + synthetic data generation
└── output/       ← Generated configs and scripts
```

### RAG Pipeline (`RAG/`)

Scrapes OpenNebula v7.0 docs, chunks them intelligently (heading boundaries, code blocks preserved), embeds with `all-mpnet-base-v2`, stores in ChromaDB, and retrieves with cross-encoder reranking.

- 1223 chunks covering 7 documentation sections
- Embedding: `sentence-transformers/all-mpnet-base-v2` (768 dims)
- Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Configuration Schema (`config/`)

Defines 26 action types across OpenNebula VMs and OneKE Kubernetes operations:

- **OneKE:** namespace (create/delete/list), app (deploy/uninstall/upgrade/list/wait_ready/get_status), service (get_endpoint/expose/list), storage (create_pvc/delete_pvc/list_pvcs), cluster (get_info/get_status/list_nodes/scale_nodes)
- **OpenNebula VMs:** create, delete, poweroff, resume, resize, snapshot_create, list

Each action has a Pydantic model with exact field names, types, and constraints. The validator checks YAML structure, action names, parameter types, step ID patterns, dependency cycles, and check type enums.

### Core Orchestrator (`core/`)

Wires RAG retrieval → LLM → post-processing → validation → code generation:

```bash
one-ai generate "..."   # YAML config only
one-ai plan "..."        # config + Python script
one-ai apply "..."       # save both to disk
```

Supports both base model (few-shot prompting via ChatOllama) and fine-tuned model (Modelfile TEMPLATE via OllamaLLM). The fine-tuned path skips few-shot examples — the model learned the schema directly.

### Fine-Tuning Pipeline (`finetune/`)

QLoRA training on Mistral 7B Instruct v0.3:

- **Data generation:** 11 gold seed examples → GPT-4o-mini expansion → 406 synthetic training pairs (100% schema validation rate)
- **Training:** 4-bit NF4 quantization, LoRA rank 16, all attention + MLP layers, 3 epochs, ~23 min on RTX A4000
- **Evaluation:** Automated schema validation + optional GPT-4o judge scoring

## Supported Request Types

| Category | Example Requests |
|----------|-----------------|
| Namespace ops | "Create a namespace called redis" |
| App deployment (new namespace) | "Deploy WordPress with a dedicated namespace and storage" |
| App deployment (existing namespace) | "Install Redis in the production namespace" |
| App management | "Remove the Redis deployment from production" |
| Service exposure | "Deploy Grafana and expose it on port 3000" |
| VM creation | "Create a VM with 4 CPUs and 8GB RAM using template ID 5" |
| VM operations | "Power off VM 42 and take a snapshot" |
| Cluster scaling | "Scale my OneKE cluster to 5 worker nodes" |
| Inventory | "List all VMs and apps on my cluster" |
| Error handling | "Deploy on AWS EKS" → produces error config with suggestion |

## Configuration

All settings use environment variables with `ONEAI_CORE_` prefix:

```bash
# LLM selection
export ONEAI_CORE_OLLAMA_MODEL=oneai-mistral:latest     # fine-tuned (recommended)
export ONEAI_CORE_OLLAMA_MODEL=mistral:7b-instruct-v0.3-q4_K_M  # base model

# RAG (disable for fine-tuned model)
export ONEAI_CORE_RAG_ENABLED=false

# Retry behavior
export ONEAI_CORE_MAX_RETRIES=3

# OpenNebula connection (for future agent execution)
export ONEAI_CORE_ONE_ENDPOINT=http://localhost:2633/RPC2
export ONEAI_CORE_ONE_USER=oneadmin
export ONEAI_CORE_ONE_PASSWORD=...
```

## Testing

```bash
# Config schema + validator + codegen (43+ tests)
cd config && pytest tests/ -v

# Core smoke tests (23 tests, no external services)
cd core && pytest tests/test_core_smoke.py -v

# Core integration tests (requires Ollama + ChromaDB)
cd core && pytest tests/ -v --integration

# Fine-tuning pipeline tests (28 tests)
cd finetune && pytest tests/ -v
```

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Local LLM | Mistral 7B Instruct v0.3 (Q4_K_M via Ollama) |
| Fine-tuning | QLoRA (PEFT + TRL) on RTX A4000 |
| Orchestration | LangChain |
| Embeddings | sentence-transformers/all-mpnet-base-v2 |
| Vector store | ChromaDB (persistent, local) |
| Schema validation | Pydantic v2 |
| Code generation | Python functions + Jinja2 template |
| Synthetic data | GPT-4o-mini via OpenAI API |
| Model serving | Ollama (GGUF, Q4_K_M quantization) |

## Infrastructure

This project is designed for and tested on a real OpenNebula environment:
- Two NUC servers (compute) + one orchestrator
- OneKE (OpenNebula's Kubernetes Engine) for container workloads
- NVIDIA RTX A4000 (16GB VRAM) for model training

## Roadmap

- [x] RAG pipeline (ChromaDB + sentence-transformers)
- [x] Configuration schema + Pydantic validation (26 actions)
- [x] Code generation (config → executable Python script)
- [x] CLI orchestrator (generate / plan / apply)
- [x] QLoRA fine-tuning (0% → 100% pass rate)
- [x] Ollama integration (LoRA → GGUF → Modelfile)
- [ ] Execution agent (dry-run, approval gates, rollback)
- [ ] Demo GIFs and benchmark visualizations

## License

MIT
