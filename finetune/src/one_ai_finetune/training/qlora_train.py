"""
QLoRA fine-tuning pipeline for OpenNebula AI Configuration Assistant.

Trains a LoRA adapter on a base model (Mistral 7B / Llama 3 8B) using
QLoRA (4-bit quantization) on curated (instruction → config YAML) pairs.

Usage:
    python -m one_ai_finetune.training.qlora_train \
        --base_model mistralai/Mistral-7B-Instruct-v0.3 \
        --dataset_path data/processed/train.jsonl \
        --output_dir models/mistral-7b-oneke-v1 \
        --epochs 3 \
        --batch_size 4
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QLoRAConfig:
    """Training configuration with sensible defaults for 7B models."""

    # Model
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_seq_length: int = 4096

    # QLoRA quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA adapter
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    fp16: bool = False
    bf16: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "one-ai-finetune"

    # Dataset
    dataset_path: str = "data/processed/train.jsonl"
    eval_dataset_path: Optional[str] = "data/processed/eval.jsonl"
    output_dir: str = "models/mistral-7b-oneke-v1"


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an OpenNebula infrastructure configuration assistant.
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


def format_prompt(instruction: str, output: str = "") -> str:
    """
    Format a training example into the chat template.

    Uses a consistent format that works with both Mistral and Llama instruct models.
    """
    if output:
        return (
            f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"{instruction} [/INST]\n"
            f"```yaml\n{output}\n```</s>"
        )
    else:
        # Inference mode — no output
        return (
            f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"{instruction} [/INST]\n"
        )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_training_data(path: str) -> Dataset:
    """
    Load and format training data from JSONL or JSON.

    Expected format per line:
    {"instruction": "...", "input": "", "output": "..."}
    """
    ext = Path(path).suffix

    if ext == ".jsonl":
        dataset = load_dataset("json", data_files=path, split="train")
    elif ext == ".json":
        with open(path) as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    else:
        raise ValueError(f"Unsupported dataset format: {ext}")

    # Format into prompts
    def format_example(example):
        example["text"] = format_prompt(
            instruction=example["instruction"],
            output=example["output"],
        )
        return example

    dataset = dataset.map(format_example)
    print(f"Loaded {len(dataset)} training examples")
    return dataset


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_model_and_tokenizer(config: QLoRAConfig):
    """Load the base model with QLoRA quantization and attach LoRA adapter."""

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )

    # Load base model
    print(f"Loading base model: {config.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Attach adapter
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: QLoRAConfig) -> str:
    """
    Run the full QLoRA training pipeline.

    Returns:
        Path to the saved LoRA adapter
    """
    # Setup
    model, tokenizer = setup_model_and_tokenizer(config)
    train_dataset = load_training_data(config.dataset_path)

    eval_dataset = None
    if config.eval_dataset_path and Path(config.eval_dataset_path).exists():
        eval_dataset = load_training_data(config.eval_dataset_path)

    # W&B setup
    report_to = "wandb" if config.use_wandb else "none"
    if config.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", config.wandb_project)

    # Training arguments
    # Training arguments (trl 0.29: SFTConfig replaces TrainingArguments)
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=2,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        save_total_limit=3,
        load_best_model_at_end=bool(eval_dataset),
        report_to=report_to,
        run_name=f"oneke-qlora-r{config.lora_r}-lr{config.learning_rate}",
        optim="paged_adamw_32bit",
        # SFT-specific (moved here in trl 0.29)
        dataset_text_field="text",
        max_length=config.max_seq_length,
        packing=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Base model: {config.base_model}")
    print(f"  LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Output: {config.output_dir}")
    print()

    trainer.train()

    # Save the adapter
    adapter_path = os.path.join(config.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved to: {adapter_path}")

    return adapter_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for OneAI")
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--dataset_path", default="data/processed/train.jsonl")
    parser.add_argument("--eval_dataset_path", default="data/processed/eval.jsonl")
    parser.add_argument("--output_dir", default="models/mistral-7b-oneke-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    config = QLoRAConfig(
        base_model=args.base_model,
        dataset_path=args.dataset_path,
        eval_dataset_path=args.eval_dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb,
    )

    train(config)


if __name__ == "__main__":
    main()
