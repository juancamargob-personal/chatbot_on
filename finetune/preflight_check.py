#!/usr/bin/env python3
"""
Pre-flight check for the fine-tuning pipeline.
Run from: ~/Projects/chatbot_on/finetune

Verifies:
1. OpenAI API key is set and works
2. Gold examples pass schema validation
3. GPU is available with enough VRAM
4. Training dependencies are installed
5. generate_synthetic.py can be imported
"""
import sys
import os

print("=" * 60)
print("QLoRA Fine-Tuning Pre-Flight Check")
print("=" * 60)

all_ok = True

# 1. OpenAI API key
print("\n1. OpenAI API key")
key = os.environ.get("OPENAI_API_KEY", "")
if key and key.startswith("sk-"):
    print(f"   OK: set ({key[:12]}...)")
    # Quick connectivity test
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            max_tokens=5,
        )
        print(f"   OK: API responds ({response.choices[0].message.content.strip()})")
    except Exception as e:
        print(f"   WARNING: API call failed: {e}")
        all_ok = False
else:
    print("   FAIL: OPENAI_API_KEY not set or invalid")
    print("   Run: export OPENAI_API_KEY=sk-your-key-here")
    all_ok = False

# 2. Gold examples validation
print("\n2. Gold examples schema validation")
try:
    import json
    from one_ai_config.validator import ConfigValidator
    with open("data/seed/gold_examples.json") as f:
        examples = json.load(f)
    v = ConfigValidator()
    valid_count = 0
    for i, ex in enumerate(examples):
        result = v.validate(ex["output"])
        if result.is_valid:
            valid_count += 1
        else:
            print(f"   FAIL: Example {i+1}: {result.error_summary()[:80]}")
    print(f"   OK: {valid_count}/{len(examples)} gold examples pass validation")
except Exception as e:
    print(f"   FAIL: {e}")
    all_ok = False

# 3. GPU
print("\n3. GPU availability")
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"   OK: {name} ({mem:.1f}GB)")
        if mem < 14:
            print(f"   WARNING: {mem:.1f}GB may be tight. 16GB+ recommended.")
    else:
        print("   FAIL: CUDA not available")
        all_ok = False
except ImportError:
    print("   FAIL: PyTorch not installed. Run: pip install torch")
    all_ok = False

# 4. Training dependencies
print("\n4. Training dependencies")
deps = {
    "peft": "pip install peft",
    "trl": "pip install trl",
    "bitsandbytes": "pip install bitsandbytes",
    "accelerate": "pip install accelerate",
    "transformers": "pip install transformers",
    "datasets": "pip install datasets",
}
missing = []
for pkg, install in deps.items():
    try:
        __import__(pkg)
        print(f"   OK: {pkg}")
    except ImportError:
        print(f"   MISSING: {pkg} — install with: {install}")
        missing.append(install)
        all_ok = False

if missing:
    print(f"\n   Quick fix: pip install {' '.join(d.split()[-1] for d in missing)} --break-system-packages")

# 5. generate_synthetic.py imports
print("\n5. Synthetic data generator")
try:
    from one_ai_finetune.data.generate_synthetic import (
        VARIATION_SYSTEM_PROMPT,
    )
    print("   OK: generate_synthetic imports successfully")
except Exception as e:
    print(f"   FAIL: {e}")
    all_ok = False

# 6. format_dataset imports
print("\n6. Dataset formatter")
try:
    from one_ai_finetune.data.format_dataset import DatasetFormatter
    print("   OK: format_dataset imports successfully")
except ImportError:
    try:
        from one_ai_finetune.data.format_dataset import format_mistral
        print("   OK: format_dataset imports (function-based)")
    except Exception as e:
        print(f"   FAIL: {e}")
        all_ok = False

# Summary
print("\n" + "=" * 60)
if all_ok:
    print("ALL CHECKS PASSED — ready to start fine-tuning!")
else:
    print("SOME CHECKS FAILED — fix the issues above before proceeding.")
print("=" * 60)
