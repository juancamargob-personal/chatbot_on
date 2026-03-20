#!/usr/bin/env python3
"""
Project Health Check for chatbot_on
Run from: ~/Projects/chatbot_on/ with venv activated

Usage:
    python check_project_health.py
"""

import sys
import json
import subprocess
from pathlib import Path

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}")
    if detail and not condition:
        print(f"      → {detail}")
    return condition

def warn(label, detail=""):
    print(f"  {WARN} {label}")
    if detail:
        print(f"      → {detail}")

# ---------- Python & Environment ----------
section("1. Environment")
check("Python >= 3.10", sys.version_info >= (3, 10), f"Got {sys.version}")

root = Path.cwd()
check("Running from chatbot_on root", 
      (root / "RAG").is_dir() and (root / "config").is_dir(),
      f"CWD is {root}")

# ---------- Package imports ----------
section("2. Package Imports")

packages_ok = True
for pkg, label in [
    ("one_ai_rag", "one-ai-rag"),
    ("one_ai_config", "one-ai-config"),
    ("one_ai_core", "one-ai-core"),
    ("one_ai_finetune", "one-ai-finetune"),
]:
    try:
        __import__(pkg)
        check(f"{label} importable", True)
    except ImportError as e:
        check(f"{label} importable", False, str(e))
        packages_ok = False

# ---------- Config schema introspection ----------
section("3. Config Schema")
try:
    from one_ai_config.schema.base import OneAIConfig, ConfigMetadata, ConfigStep
    from one_ai_config.schema.oneke import ACTION_PARAM_REGISTRY

    check("OneAIConfig importable", True)
    check("ConfigMetadata has 'description' field", 
          "description" in ConfigMetadata.model_fields)
    check("ConfigMetadata has NO 'name' field",
          "name" not in ConfigMetadata.model_fields)

    actions = sorted(ACTION_PARAM_REGISTRY.keys())
    print(f"\n  Registered actions ({len(actions)}):")
    for a in actions:
        print(f"    - {a}")

    vm_actions = [a for a in actions if a.startswith("one.vm.")]
    oneke_actions = [a for a in actions if a.startswith("oneke.")]
    check(f"OneKE actions: {len(oneke_actions)}", len(oneke_actions) >= 19)
    check(f"VM actions: {len(vm_actions)}", len(vm_actions) >= 5)
except Exception as e:
    check("Config schema inspection", False, str(e))

# ---------- Validator ----------
section("4. Validator")
try:
    from one_ai_config.validator import ConfigValidator
    validator = ConfigValidator()

    # Test with a minimal valid config
    test_yaml = """
version: "1.0"
metadata:
  description: "Test config"
  risk_level: low
steps:
  - id: step_01
    action: oneke.namespace.create
    description: "Create test namespace"
    params:
      namespace: test
"""
    result = validator.validate(test_yaml)
    check("Validator accepts valid YAML", result.is_valid, 
          result.error_summary() if not result.is_valid else "")

    # Test that passing a dict fails gracefully
    try:
        validator.validate({"not": "a string"})
        check("Validator rejects dict input", False, "Should have raised an error")
    except (AttributeError, TypeError):
        check("Validator rejects dict input (raises error)", True)
except Exception as e:
    check("Validator tests", False, str(e))

# ---------- Code Generator ----------
section("5. Code Generator")
try:
    from one_ai_config.codegen.generator import CodeGenerator, GeneratedScript
    check("CodeGenerator importable", True)
    check("GeneratedScript has '.script' field",
          hasattr(GeneratedScript, "script") or "script" in GeneratedScript.__dataclass_fields__
          if hasattr(GeneratedScript, "__dataclass_fields__") 
          else hasattr(GeneratedScript, "script"))
except Exception as e:
    check("CodeGenerator inspection", False, str(e))

# ---------- Finetune package ----------
section("6. Finetune Package")
try:
    from one_ai_finetune.data.generate_synthetic import SyntheticDataGenerator
    check("SyntheticDataGenerator importable", True)

    # Check if response_format bug is present
    import inspect
    source = inspect.getsource(SyntheticDataGenerator._call_api)
    has_json_format = "response_format" in source
    if has_json_format:
        warn("generate_synthetic.py still has response_format={'type': 'json_object'}",
             "This forces GPT to wrap in objects, not arrays. Should be removed.")
    else:
        check("No response_format restriction in _call_api", True)
except Exception as e:
    check("Finetune import", False, str(e))

try:
    from one_ai_finetune.data_quality.dedup import DataQualityCleaner
    check("DataQualityCleaner importable", True)
except Exception as e:
    check("DataQualityCleaner import", False, str(e))

# ---------- Gold examples ----------
section("7. Gold Seed Examples")
gold_path = root / "finetune" / "data" / "seed" / "gold_examples.json"
if gold_path.exists():
    seeds = json.loads(gold_path.read_text())
    check(f"gold_examples.json exists ({len(seeds)} examples)", True)

    # Validate each seed
    try:
        from one_ai_config.validator import ConfigValidator
        validator = ConfigValidator()
        for i, seed in enumerate(seeds):
            result = validator.validate(seed.get("output", ""))
            status = "valid" if result.is_valid else f"INVALID: {result.error_summary()}"
            check(f"  Seed {i+1}: {seed['instruction'][:50]}... → {status}",
                  result.is_valid)
    except Exception as e:
        warn(f"Could not validate seeds: {e}")

    # Check for VM-focused seeds
    has_vm_seed = any("vm" in s.get("instruction", "").lower() or 
                      "virtual machine" in s.get("instruction", "").lower()
                      for s in seeds)
    if not has_vm_seed:
        warn("No VM-focused seed examples found",
             "Consider adding seeds for one.vm.* actions")
else:
    check("gold_examples.json exists", False, str(gold_path))

# ---------- Finetune tests ----------
section("8. Finetune Tests")
try:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "finetune/tests/", "-v", "--tb=short", "-q"],
        capture_output=True, text=True, timeout=60,
        cwd=str(root)
    )
    lines = result.stdout.strip().split("\n")
    # Show summary line
    for line in lines[-5:]:
        if "passed" in line or "failed" in line or "error" in line:
            print(f"  {line.strip()}")
    check("All finetune tests pass", result.returncode == 0,
          result.stdout[-200:] if result.returncode != 0 else "")
except Exception as e:
    warn(f"Could not run tests: {e}")

# ---------- Core tests ----------
section("9. Core Smoke Tests")
try:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "core/tests/test_core_smoke.py", "-v", 
         "--tb=short", "-q"],
        capture_output=True, text=True, timeout=60,
        cwd=str(root)
    )
    lines = result.stdout.strip().split("\n")
    for line in lines[-5:]:
        if "passed" in line or "failed" in line or "error" in line:
            print(f"  {line.strip()}")
    check("Core smoke tests pass", result.returncode == 0,
          result.stdout[-200:] if result.returncode != 0 else "")
except Exception as e:
    warn(f"Could not run tests: {e}")

# ---------- Ollama ----------
section("10. Ollama & LLM")
try:
    result = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        check("Ollama is running", True)
        models = result.stdout.strip()
        has_mistral = "mistral" in models.lower()
        check("Mistral model available", has_mistral, 
              "Run: ollama pull mistral:7b-instruct-v0.3-q4_K_M")
        # Print model list
        for line in models.split("\n")[:5]:
            print(f"    {line}")
    else:
        check("Ollama is running", False, "Start with: ollama serve")
except FileNotFoundError:
    check("Ollama installed", False, "Install from https://ollama.ai")

# ---------- GPU ----------
section("11. GPU & CUDA")
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        check("NVIDIA GPU detected", True)
        print(f"    {result.stdout.strip()}")
    else:
        check("NVIDIA GPU", False)
except FileNotFoundError:
    check("nvidia-smi available", False)

try:
    import torch
    check(f"PyTorch CUDA available: {torch.cuda.is_available()}", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"    Device: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    warn("PyTorch not installed")

# ---------- OpenAI ----------
section("12. OpenAI API Key")
import os
has_key = bool(os.getenv("OPENAI_API_KEY"))
check("OPENAI_API_KEY is set", has_key,
      "Set with: export OPENAI_API_KEY=sk-...")

# ---------- Summary ----------
section("Summary")
print("  Review any ✗ or ⚠ items above before proceeding.")
print("  Copy this output and share it so we can plan next steps.")
