"""
Dataset formatter for fine-tuning.

Converts raw (instruction, output) pairs into the chat template format
that the base model expects, then splits into train/eval sets.

The format depends on which base model you're using:
- Mistral: [INST] ... [/INST]
- Llama 3: <|begin_of_text|><|start_header_id|>...
- ChatML: <|im_start|>...

This module handles all formats and produces JSONL files ready for
the SFTTrainer.

Usage:
    python -m one_ai_finetune.data.format_dataset \
        --input data/synthetic/synthetic_examples.json \
        --output_dir data/processed \
        --format mistral \
        --eval_split 0.1
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# System prompt (shared across all formats)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an OpenNebula infrastructure configuration assistant.
Given a natural language request, you produce a YAML configuration file that
describes the steps needed to fulfill the request on an OpenNebula / OneKE cluster.

Your output must be valid YAML following the OneAI configuration schema.
If the request is impossible or unsupported, produce an error configuration
explaining why and suggesting alternatives.

Always include:
- Proper metadata with description, risk level, and tags
- Ordered steps with dependencies
- Pre-checks and post-checks in the validation section
- Rollback steps for operations that modify the cluster"""


# ---------------------------------------------------------------------------
# Format functions
# ---------------------------------------------------------------------------

def format_mistral(instruction: str, output: str = "") -> str:
    """
    Format for Mistral Instruct models.

    Mistral uses: <s>[INST] {system}\n\n{user} [/INST] {assistant}</s>
    """
    if output:
        return (
            f"<s>[INST] {SYSTEM_PROMPT}\n\n"
            f"{instruction} [/INST]\n"
            f"```yaml\n{output}\n```</s>"
        )
    return (
        f"<s>[INST] {SYSTEM_PROMPT}\n\n"
        f"{instruction} [/INST]\n"
    )


def format_llama3(instruction: str, output: str = "") -> str:
    """
    Format for Llama 3 Instruct models.

    Llama 3 uses: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...
    """
    parts = [
        "<|begin_of_text|>",
        "<|start_header_id|>system<|end_header_id|>\n\n",
        f"{SYSTEM_PROMPT}<|eot_id|>",
        "<|start_header_id|>user<|end_header_id|>\n\n",
        f"{instruction}<|eot_id|>",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ]
    if output:
        parts.append(f"```yaml\n{output}\n```<|eot_id|>")
    return "".join(parts)


def format_chatml(instruction: str, output: str = "") -> str:
    """
    Format for ChatML-compatible models (e.g., some Qwen, Phi models).
    """
    parts = [
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n",
        f"<|im_start|>user\n{instruction}<|im_end|>\n",
        "<|im_start|>assistant\n",
    ]
    if output:
        parts.append(f"```yaml\n{output}\n```<|im_end|>")
    return "".join(parts)


FORMAT_FUNCTIONS = {
    "mistral": format_mistral,
    "llama3": format_llama3,
    "chatml": format_chatml,
}


# ---------------------------------------------------------------------------
# Dataset formatter
# ---------------------------------------------------------------------------

class DatasetFormatter:
    """
    Formats and splits training data.

    Usage:
        formatter = DatasetFormatter(format="mistral", eval_split=0.1)
        formatter.format_and_save(
            input_path="data/synthetic/synthetic_examples.json",
            output_dir="data/processed",
        )
    """

    def __init__(
        self,
        format: str = "mistral",
        eval_split: float = 0.1,
        seed: int = 42,
        max_seq_length: int = 4096,
    ):
        if format not in FORMAT_FUNCTIONS:
            raise ValueError(
                f"Unknown format '{format}'. Supported: {list(FORMAT_FUNCTIONS.keys())}"
            )
        self.format_fn = FORMAT_FUNCTIONS[format]
        self.format_name = format
        self.eval_split = eval_split
        self.seed = seed
        self.max_seq_length = max_seq_length

    def format_and_save(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        seed_path: Optional[str | Path] = None,
    ) -> dict:
        """
        Format examples and save train/eval splits as JSONL.

        Args:
            input_path: Path to synthetic_examples.json (or any JSON array)
            output_dir: Directory for train.jsonl and eval.jsonl
            seed_path: Optional path to seed examples to always include in train

        Returns:
            Dict with stats about the formatted dataset
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Load examples
        examples = json.loads(Path(input_path).read_text())
        print(f"Loaded {len(examples)} examples from {input_path}")

        # Optionally load seeds (always go to train, never eval)
        seed_examples = []
        if seed_path and Path(seed_path).exists():
            seed_examples = json.loads(Path(seed_path).read_text())
            print(f"Loaded {len(seed_examples)} seed examples (will be in train only)")

        # Format all examples
        formatted = []
        skipped = 0
        for ex in examples:
            instruction = ex.get("instruction", "")
            output = ex.get("output", "")
            if not instruction or not output:
                skipped += 1
                continue

            text = self.format_fn(instruction, output)

            # Rough token check (skip if way too long)
            approx_tokens = len(text) // 4
            if approx_tokens > self.max_seq_length:
                skipped += 1
                continue

            formatted.append({"text": text, "instruction": instruction})

        print(f"Formatted {len(formatted)} examples ({skipped} skipped)")

        # Format seed examples separately
        formatted_seeds = []
        for ex in seed_examples:
            text = self.format_fn(ex["instruction"], ex["output"])
            formatted_seeds.append({"text": text, "instruction": ex["instruction"]})

        # Split into train/eval
        random.seed(self.seed)
        random.shuffle(formatted)

        eval_size = int(len(formatted) * self.eval_split)
        eval_set = formatted[:eval_size]
        train_set = formatted[eval_size:]

        # Add seeds to train (they're gold standard, always train on them)
        train_set.extend(formatted_seeds)
        random.shuffle(train_set)

        # Save as JSONL
        train_path = out / "train.jsonl"
        eval_path = out / "eval.jsonl"

        self._save_jsonl(train_set, train_path)
        self._save_jsonl(eval_set, eval_path)

        stats = {
            "format": self.format_name,
            "total_examples": len(formatted) + len(formatted_seeds),
            "train_examples": len(train_set),
            "eval_examples": len(eval_set),
            "skipped": skipped,
            "eval_split": self.eval_split,
        }

        stats_path = out / "dataset_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2))

        print(f"\nDataset saved:")
        print(f"  Train: {len(train_set)} examples -> {train_path}")
        print(f"  Eval:  {len(eval_set)} examples -> {eval_path}")

        return stats

    def _save_jsonl(self, examples: list[dict], path: Path) -> None:
        """Save examples as JSONL (one JSON object per line)."""
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def format_single(self, instruction: str, output: str = "") -> str:
        """Format a single example (useful for inference)."""
        return self.format_fn(instruction, output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Format dataset for fine-tuning")
    parser.add_argument("--input", required=True, help="Path to examples JSON")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--seed_path", default="data/seed/gold_examples.json")
    parser.add_argument("--format", default="mistral", choices=FORMAT_FUNCTIONS.keys())
    parser.add_argument("--eval_split", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    args = parser.parse_args()

    formatter = DatasetFormatter(
        format=args.format,
        eval_split=args.eval_split,
        max_seq_length=args.max_seq_length,
    )
    formatter.format_and_save(
        input_path=args.input,
        output_dir=args.output_dir,
        seed_path=args.seed_path,
    )


if __name__ == "__main__":
    main()
