#!/usr/bin/env python3
"""
merge_lora_to_full_model.py
============================
Merges the QLoRA adapter into the base Mistral 7B model and saves
a full-weight HF model ready for GGUF conversion.

Usage:
    cd ~/Projects/chatbot_on
    python merge_lora_to_full_model.py

Output:
    finetune/models/oneai-mistral-7b-merged/

Memory: ~14GB RAM (loads in float16 on CPU). No GPU required.
Time:   ~3-5 minutes.
"""

import os
import sys
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
ADAPTER_DIR = REPO_ROOT / "finetune" / "models" / "oneai-mistral-7b-lora" / "final_adapter"
MERGED_DIR = REPO_ROOT / "finetune" / "models" / "oneai-mistral-7b-merged"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def main():
    # Sanity checks
    if not ADAPTER_DIR.exists():
        print(f"ERROR: Adapter directory not found: {ADAPTER_DIR}")
        sys.exit(1)

    adapter_config = ADAPTER_DIR / "adapter_config.json"
    adapter_weights = ADAPTER_DIR / "adapter_model.safetensors"
    if not adapter_config.exists() or not adapter_weights.exists():
        print(f"ERROR: Missing adapter files in {ADAPTER_DIR}")
        print(f"  adapter_config.json: {'OK' if adapter_config.exists() else 'MISSING'}")
        print(f"  adapter_model.safetensors: {'OK' if adapter_weights.exists() else 'MISSING'}")
        sys.exit(1)

    print(f"Adapter directory:  {ADAPTER_DIR}")
    print(f"Output directory:   {MERGED_DIR}")
    print(f"Base model:         {BASE_MODEL}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load base model in float16 on CPU
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading base model (float16, CPU)...")
    print("  This will use ~14GB RAM. Be patient (~2 min).")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",          # Keep on CPU — no VRAM needed
        low_cpu_mem_usage=True,    # Stream weights to reduce peak RAM
    )
    print("  Base model loaded.")

    # ------------------------------------------------------------------
    # Step 2: Load and merge LoRA adapter
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2: Loading LoRA adapter and merging...")
    print("=" * 60)

    from peft import PeftModel

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_DIR),
        torch_dtype=torch.float16,
    )
    print(f"  LoRA adapter loaded from {ADAPTER_DIR}")

    # Merge LoRA weights into the base model and discard the adapter
    model = model.merge_and_unload()
    print("  LoRA weights merged into base model.")

    # ------------------------------------------------------------------
    # Step 3: Load tokenizer from adapter (has the training config)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: Loading tokenizer...")
    print("=" * 60)

    # Try adapter tokenizer first, fall back to base
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
        print(f"  Tokenizer loaded from adapter: {ADAPTER_DIR}")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        print(f"  Tokenizer loaded from base model: {BASE_MODEL}")

    # ------------------------------------------------------------------
    # Step 4: Save merged model
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Step 4: Saving merged model to {MERGED_DIR}")
    print("  This writes ~14GB of safetensors files. ~2 min.")
    print("=" * 60)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))
    print("  Merged model saved.")

    # ------------------------------------------------------------------
    # Verify output
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Verification")
    print("=" * 60)
    total_size = sum(f.stat().st_size for f in MERGED_DIR.iterdir() if f.is_file())
    print(f"  Output files: {len(list(MERGED_DIR.iterdir()))}")
    print(f"  Total size:   {total_size / (1024**3):.1f} GB")
    print()

    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    safetensors = list(MERGED_DIR.glob("*.safetensors"))
    all_ok = all((MERGED_DIR / f).exists() for f in required) and len(safetensors) > 0

    if all_ok:
        print("  All required files present.")
        print()
        print("SUCCESS! Merged model ready for GGUF conversion.")
        print()
        print("Next step:")
        print("  See the GGUF conversion commands in the integration guide.")
    else:
        print("  WARNING: Some files may be missing. Check the output directory.")
        for f in required:
            status = "OK" if (MERGED_DIR / f).exists() else "MISSING"
            print(f"    {f}: {status}")
        print(f"    safetensors files: {len(safetensors)}")


if __name__ == "__main__":
    main()
