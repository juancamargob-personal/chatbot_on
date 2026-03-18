#!/usr/bin/env python3
"""
Debug script for the chain → validator integration bug.

Run from the chatbot_on repo root with the venv activated:
    python debug_chain_validator.py

This will:
1. Test that the validator works standalone (should pass)
2. Monkey-patch the chain to log exactly what reaches the validator
3. Run a simple request and show the bytes that fail
"""
import sys
import yaml

# ─── Test 1: Standalone validator ───────────────────────────────────
print("=" * 60)
print("TEST 1: Standalone validator with known-good YAML")
print("=" * 60)

valid_yaml = """\
metadata:
  description: Create a test namespace
  risk_level: low
  tags: [test]
steps:
  - id: step_01
    action: oneke.namespace.create
    description: Create the wordpress namespace
    params:
      name: test
    depends_on: []
    on_failure: abort
validation:
  pre_checks: []
  post_checks: []
rollback:
  enabled: false
  steps: []
"""

from one_ai_config.validator import ConfigValidator
v = ConfigValidator()

# First check: does yaml.safe_load produce a dict?
parsed = yaml.safe_load(valid_yaml)
print(f"  yaml.safe_load type: {type(parsed).__name__}")
print(f"  Is dict: {isinstance(parsed, dict)}")
assert isinstance(parsed, dict), "YAML didn't parse as dict!"

# Second check: does the validator accept it?
result = v.validate(valid_yaml)
print(f"  Validator result: valid={result.is_valid}")
if not result.is_valid:
    print(f"  Errors: {result.error_summary()}")
print()


# ─── Test 2: Simulate what _extract_yaml produces ──────────────────
print("=" * 60)
print("TEST 2: Simulate _extract_yaml on typical Mistral output")
print("=" * 60)

from one_ai_core.chain import OneAIChain

# Typical Mistral outputs to test
test_outputs = [
    ("Plain YAML (no fences)", valid_yaml),
    ("Fenced YAML", f"Here is the config:\n\n```yaml\n{valid_yaml}\n```\n\nLet me explain..."),
    ("Fenced no language tag", f"```\n{valid_yaml}\n```"),
    ("Prose then YAML", f"Based on the docs, here's the configuration:\n{valid_yaml}"),
    ("Double-fenced", f"```yaml\n```yaml\n{valid_yaml}\n```\n```"),
]

for label, raw in test_outputs:
    cleaned = OneAIChain._extract_yaml(raw)
    parsed = yaml.safe_load(cleaned)
    is_dict = isinstance(parsed, dict)
    result = v.validate(cleaned)
    status = "✅" if result.is_valid else "❌"
    print(f"  {status} {label}")
    print(f"     yaml.safe_load type: {type(parsed).__name__}, is_dict: {is_dict}")
    print(f"     validator: valid={result.is_valid}")
    if not result.is_valid:
        print(f"     error: {result.error_summary()[:100]}")
    print()


# ─── Test 3: Intercept the real chain ──────────────────────────────
print("=" * 60)
print("TEST 3: Intercept validator.validate() in a real chain.run()")
print("=" * 60)

from one_ai_core.config import CoreConfig

cfg = CoreConfig()
chain = OneAIChain(config=cfg)

# Monkey-patch the validator to see exactly what it receives
real_validator = chain._get_validator()
original_validate = real_validator.validate

call_count = 0
def debug_validate(text):
    global call_count
    call_count += 1
    print(f"\n  --- validator.validate() call #{call_count} ---")
    print(f"  Input type: {type(text).__name__}")
    print(f"  Input length: {len(text)}")
    print(f"  First 80 chars: {repr(text[:80])}")
    print(f"  Last 40 chars: {repr(text[-40:])}")

    # Check what yaml.safe_load does with it
    try:
        parsed = yaml.safe_load(text)
        print(f"  yaml.safe_load type: {type(parsed).__name__}")
        if isinstance(parsed, str):
            print(f"  ⚠️  YAML parsed as STRING: {repr(parsed[:100])}")
            print(f"  This is the bug! The text is valid YAML but it's a scalar, not a mapping.")
    except Exception as e:
        print(f"  yaml.safe_load error: {e}")

    return original_validate(text)

real_validator.validate = debug_validate

print("  Running chain with: 'Create a namespace called wordpress on my OneKE cluster'")
print("  (This will call the real Ollama — may take 30-120s)")
print()

try:
    result = chain.run("Create a namespace called wordpress on my OneKE cluster")
    print(f"\n  Chain result: success={result.success}")
    print(f"  Attempts: {result.attempts}")
    if result.error:
        print(f"  Error: {result.error[:200]}")
    if result.success:
        print(f"  Config YAML (first 200 chars): {result.config_yaml[:200]}")
except Exception as e:
    print(f"\n  Chain raised exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DONE. Check the validator.validate() call logs above.")
print("If yaml.safe_load returns 'str' type, the _extract_yaml")
print("output needs fixing. If it returns 'dict', the bug is in")
print("the validator's internal processing.")
print("=" * 60)
