#!/usr/bin/env bash
# ============================================================
# setup_config.sh — Restructures flat files into proper package layout
#
# Run from inside your config directory:
#   chmod +x setup_config.sh && ./setup_config.sh
# ============================================================

set -e

echo "=== Setting up one-ai-config project structure ==="

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "Working in: $PROJECT_DIR"

# --- Create directory structure ---
echo "[1/4] Creating directories..."
mkdir -p src/one_ai_config/schema
mkdir -p src/one_ai_config/codegen/templates
mkdir -p tests

# --- Move source files ---
echo "[2/4] Moving source files..."

# Schema files (from RAG phase — these should already exist in the zip)
for f in base.py oneke.py; do
    if [ -f "$f" ]; then
        mv "$f" src/one_ai_config/schema/
        echo "  Moved $f -> src/one_ai_config/schema/$f"
    fi
done

# Validator
if [ -f "validator.py" ]; then
    mv validator.py src/one_ai_config/
    echo "  Moved validator.py -> src/one_ai_config/validator.py"
fi

# Code generator
if [ -f "generator.py" ]; then
    mv generator.py src/one_ai_config/codegen/
    echo "  Moved generator.py -> src/one_ai_config/codegen/generator.py"
fi

# Templates
for f in script.py.j2 actions.j2; do
    if [ -f "$f" ]; then
        mv "$f" src/one_ai_config/codegen/templates/
        echo "  Moved $f -> src/one_ai_config/codegen/templates/$f"
    fi
done

# --- Move test files ---
echo "[3/4] Moving test files..."
for f in test_config_e2e.py conftest.py; do
    if [ -f "$f" ]; then
        mv "$f" tests/
        echo "  Moved $f -> tests/$f"
    fi
done

# --- Create package files ---
echo "[4/4] Creating package files..."

# Schema __init__.py
cat > src/one_ai_config/schema/__init__.py << 'EOF'
"""Schema models for OneAI configuration."""
EOF

# Codegen __init__.py
cat > src/one_ai_config/codegen/__init__.py << 'EOF'
"""Code generation module — maps validated configs to executable Python scripts."""
from one_ai_config.codegen.generator import CodeGenerator, GeneratedScript
__all__ = ["CodeGenerator", "GeneratedScript"]
EOF

# Main package __init__.py
cat > src/one_ai_config/__init__.py << 'INITEOF'
"""
one-ai-config: Configuration schema, validator, and code generator.

Quickstart:
    from one_ai_config import parse_config, ConfigValidator, CodeGenerator

    validator = ConfigValidator()
    result = validator.validate(yaml_string)

    if result.is_valid:
        generator = CodeGenerator()
        script = generator.generate(result.config)
        script.save("deploy.py")
"""

from one_ai_config.schema.base import (
    OneAIConfig,
    ConfigStep,
    ConfigMetadata,
    ValidationConfig,
    ValidationCheck,
    RollbackConfig,
    ErrorResponse,
    RiskLevel,
    FailureStrategy,
    CheckType,
    SUPPORTED_ACTIONS,
    parse_config,
    config_to_yaml,
)
from one_ai_config.schema.oneke import (
    ACTION_PARAM_REGISTRY,
    validate_step_params,
)
from one_ai_config.validator import ConfigValidator, ValidationResult
from one_ai_config.codegen.generator import CodeGenerator, GeneratedScript

__all__ = [
    "OneAIConfig", "ConfigStep", "ConfigMetadata", "ValidationConfig",
    "ValidationCheck", "RollbackConfig", "ErrorResponse",
    "RiskLevel", "FailureStrategy", "CheckType", "SUPPORTED_ACTIONS",
    "parse_config", "config_to_yaml",
    "ACTION_PARAM_REGISTRY", "validate_step_params",
    "ConfigValidator", "ValidationResult",
    "CodeGenerator", "GeneratedScript",
]
INITEOF

# Tests __init__.py
touch tests/__init__.py

# pyproject.toml
cat > pyproject.toml << 'TOMLEOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "one-ai-config"
version = "0.1.0"
description = "Configuration schema, validator, and code generator for OpenNebula AI Assistant"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "jinja2>=3.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff",
]

[tool.hatch.build.targets.wheel]
packages = ["src/one_ai_config"]

[tool.hatch.build.targets.wheel.force-include]
"src/one_ai_config/codegen/templates" = "one_ai_config/codegen/templates"

[tool.ruff]
line-length = 100
target-version = "py310"
TOMLEOF

echo ""
echo "=== Structure created ==="
find . -not -path './.git/*' -not -path './__pycache__/*' \
       -not -name '*.pyc' -not -name 'files.zip' \
       -not -name 'setup_config.sh' \
       | sort | head -30
echo ""
echo "=== IMPORTANT ==="
echo "You also need the schema files (base.py, oneke.py, validator.py)."
echo "If they weren't in your zip, copy them from the outputs download:"
echo "  base.py     -> src/one_ai_config/schema/base.py"
echo "  oneke.py    -> src/one_ai_config/schema/oneke.py"
echo "  validator.py -> src/one_ai_config/validator.py"
echo ""
echo "=== Next steps ==="
echo "  1. Activate your venv: source ~/Projects/chatbot_on/venv/bin/activate"
echo "  2. pip install -e . --no-deps"
echo "  3. pytest tests/ -v"
