#!/usr/bin/env bash
set -e

echo "=== Setting up one-ai-finetune project structure ==="

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "Working in: $PROJECT_DIR"

# --- Create directories ---
echo "[1/4] Creating directories..."
mkdir -p src/one_ai_finetune/{data,training,eval,data_quality}
mkdir -p tests
mkdir -p data/{seed,synthetic,eval,processed}
mkdir -p models results

# --- Move source files ---
echo "[2/4] Moving source files..."

# Data module
for f in generate_synthetic.py format_dataset.py; do
    [ -f "$f" ] && mv "$f" src/one_ai_finetune/data/ && echo "  Moved $f -> src/one_ai_finetune/data/$f"
done

# Training module
for f in qlora_train.py; do
    [ -f "$f" ] && mv "$f" src/one_ai_finetune/training/ && echo "  Moved $f -> src/one_ai_finetune/training/$f"
done

# Eval module
for f in llm_judge.py schema_eval.py; do
    [ -f "$f" ] && mv "$f" src/one_ai_finetune/eval/ && echo "  Moved $f -> src/one_ai_finetune/eval/$f"
done

# Data quality module
for f in dedup.py; do
    [ -f "$f" ] && mv "$f" src/one_ai_finetune/data_quality/ && echo "  Moved $f -> src/one_ai_finetune/data_quality/$f"
done

# Seed data
for f in gold_examples.json; do
    [ -f "$f" ] && mv "$f" data/seed/ && echo "  Moved $f -> data/seed/$f"
done

# --- Move test files ---
echo "[3/4] Moving test files..."
for f in test_finetune_e2e.py; do
    [ -f "$f" ] && mv "$f" tests/ && echo "  Moved $f -> tests/$f"
done

# --- Create package files ---
echo "[4/4] Creating package files..."

# Package __init__.py files
touch src/one_ai_finetune/__init__.py
touch src/one_ai_finetune/data/__init__.py
touch src/one_ai_finetune/training/__init__.py
touch src/one_ai_finetune/eval/__init__.py
touch src/one_ai_finetune/data_quality/__init__.py
touch tests/__init__.py

# pyproject.toml
cat > pyproject.toml << 'TOMLEOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "one-ai-finetune"
version = "0.1.0"
description = "LoRA/QLoRA fine-tuning pipeline and evaluation for OpenNebula AI Assistant"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
    "openai>=1.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
train = [
    "torch>=2.1",
    "transformers>=4.36",
    "peft>=0.7",
    "trl>=0.7",
    "bitsandbytes>=0.41",
    "datasets>=2.14",
    "accelerate>=0.25",
]
wandb = ["wandb>=0.16"]
dev = [
    "pytest>=7.0",
    "ruff",
]

[tool.hatch.build.targets.wheel]
packages = ["src/one_ai_finetune"]

[tool.ruff]
line-length = 100
target-version = "py310"
TOMLEOF

echo ""
echo "=== Structure created ==="
find . -not -path './.git/*' -not -path './__pycache__/*' \
       -not -name '*.pyc' -not -name 'files.zip' \
       -not -name 'setup_finetune.sh' \
       | sort | head -35
echo ""
echo "=== Next steps ==="
echo "  1. source ~/Projects/chatbot_on/venv/bin/activate"
echo "  2. pip install -e . --no-deps"
echo "  3. pytest tests/ -v"
echo "  4. To generate synthetic data: export OPENAI_API_KEY=sk-..."
echo "     python -m one_ai_finetune.data.generate_synthetic"
