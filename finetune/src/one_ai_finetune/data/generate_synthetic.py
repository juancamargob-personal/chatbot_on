"""
Synthetic training data generator for OneAI fine-tuning.

Uses a strong LLM (GPT-4o) to expand a small set of hand-written seed
examples into hundreds of diverse training pairs. This is the most
common approach when you don't have thousands of real user interactions.

The generation strategy:
1. Load the gold-standard seed examples
2. For each seed, ask GPT-4o to generate variations:
   - Rephrase the instruction 10 different ways
   - Combine multiple operations into complex requests
   - Generate edge cases (missing info, ambiguous requests)
   - Generate negative examples (impossible/unsupported requests)
3. Validate each generated config against the Pydantic schema
4. Save validated pairs as the synthetic dataset

Usage:
    python -m one_ai_finetune.data.generate_synthetic \
        --seed_file data/seed/gold_examples.json \
        --output_dir data/synthetic \
        --variations_per_seed 10
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

VARIATION_SYSTEM_PROMPT = """\
You are a training data generator for an OpenNebula infrastructure automation tool.

Given a seed example of (instruction, config_output), generate {n} new variations.
Each variation should have a different natural language instruction that would
produce a SIMILAR (not identical) configuration output.

Vary the instructions by:
- Different phrasing styles (casual, formal, technical, beginner-friendly)
- Different application names and parameters
- Different namespaces and cluster names
- Different levels of detail in the request
- Different user personas (sysadmin, developer, manager)

The output YAML must follow the OneAI configuration schema with:
- version, metadata (description, risk_level, tags)
- steps with id, action, description, params, depends_on
- validation with pre_checks and post_checks
- rollback steps

Supported actions:
- oneke.namespace.create/delete/list
- oneke.app.deploy/uninstall/upgrade/list/wait_ready/get_status
- oneke.service.get_endpoint/expose/list
- oneke.storage.create_pvc/list_pvcs/delete_pvc
- oneke.cluster.get_info/get_status/list_nodes/scale_nodes
- one.vm.create/delete/poweroff/resume/list

Respond with a JSON array of objects, each with "instruction" and "output" fields.
The "output" field should be valid YAML as a string.
Do NOT include any markdown formatting or code fences in your response.
Return ONLY the JSON array."""

NEGATIVE_SYSTEM_PROMPT = """\
You are generating NEGATIVE training examples for an OpenNebula infrastructure tool.

Generate {n} examples where the user makes a request that CANNOT be fulfilled.
Each example should have:
- An "instruction" that sounds reasonable but is impossible or unsupported
- An "output" that is a valid OneAI error config YAML

Reasons a request might be impossible:
- Requesting operations on unsupported platforms (AWS native, bare Docker)
- Requesting features OpenNebula doesn't have
- Contradictory requirements
- Operations that require manual intervention
- Missing critical information that can't be inferred

The error output YAML must include:
- version: "1.0"
- metadata with description and risk_level
- steps: [] (empty)
- error with is_error: true, reason, and suggestion
- empty validation and rollback sections

Respond with a JSON array of objects, each with "instruction" and "output" fields.
Return ONLY the JSON array, no markdown."""

COMPLEX_SYSTEM_PROMPT = """\
You are generating COMPLEX multi-step training examples for an OpenNebula tool.

Generate {n} examples that combine MULTIPLE operations into a single request.
For example:
- "Set up a full monitoring stack with Prometheus, Grafana, and AlertManager"
- "Create a staging environment with its own namespace, deploy the app, and expose it"
- "Scale up workers, deploy a new version, and run health checks"

Each must produce a valid OneAI config YAML with 3-6 steps, proper dependencies,
pre-checks, post-checks, and rollback.

Respond with a JSON array of objects, each with "instruction" and "output" fields.
Return ONLY the JSON array, no markdown."""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

@dataclass
class GenerationStats:
    """Statistics from a generation run."""
    total_generated: int = 0
    schema_valid: int = 0
    schema_invalid: int = 0
    duplicates_removed: int = 0
    api_calls: int = 0
    api_cost_estimate_usd: float = 0.0


class SyntheticDataGenerator:
    """
    Generates synthetic training data using GPT-4o.

    Usage:
        generator = SyntheticDataGenerator()
        examples = generator.generate_from_seeds("data/seed/gold_examples.json")
        generator.save(examples, "data/synthetic/")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        validate_schema: bool = True,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.validate_schema = validate_schema
        self.stats = GenerationStats()

    def generate_from_seeds(
        self,
        seed_file: str | Path,
        variations_per_seed: int = 10,
        num_negative: int = 20,
        num_complex: int = 15,
    ) -> list[dict]:
        """
        Generate a full synthetic dataset from seed examples.

        Args:
            seed_file: Path to gold_examples.json
            variations_per_seed: How many variations to generate per seed
            num_negative: Number of negative (error) examples to generate
            num_complex: Number of complex multi-step examples to generate

        Returns:
            List of {"instruction": ..., "output": ...} dicts
        """
        seeds = json.loads(Path(seed_file).read_text())
        all_examples = list(seeds)  # Start with the seeds themselves
        print(f"Loaded {len(seeds)} seed examples")

        # Phase 1: Generate variations of each seed
        print(f"\n--- Phase 1: Generating {variations_per_seed} variations per seed ---")
        for i, seed in enumerate(seeds):
            print(f"  Seed {i+1}/{len(seeds)}: {seed['instruction'][:60]}...")
            variations = self._generate_variations(seed, variations_per_seed)
            all_examples.extend(variations)
            print(f"    Generated {len(variations)} valid variations")

        # Phase 2: Generate negative examples
        print(f"\n--- Phase 2: Generating {num_negative} negative examples ---")
        negatives = self._generate_negatives(num_negative)
        all_examples.extend(negatives)
        print(f"  Generated {len(negatives)} negative examples")

        # Phase 3: Generate complex multi-step examples
        print(f"\n--- Phase 3: Generating {num_complex} complex examples ---")
        complex_examples = self._generate_complex(num_complex)
        all_examples.extend(complex_examples)
        print(f"  Generated {len(complex_examples)} complex examples")

        # Phase 4: Validate all against schema
        if self.validate_schema:
            print(f"\n--- Phase 4: Validating schema compliance ---")
            all_examples = self._validate_examples(all_examples)

        self.stats.total_generated = len(all_examples)
        print(f"\n=== Generation complete: {len(all_examples)} total examples ===")
        print(f"  Schema valid: {self.stats.schema_valid}")
        print(f"  Schema invalid (removed): {self.stats.schema_invalid}")
        print(f"  API calls made: {self.stats.api_calls}")

        return all_examples

    def _generate_variations(self, seed: dict, n: int) -> list[dict]:
        """Generate n variations of a single seed example."""
        prompt = VARIATION_SYSTEM_PROMPT.format(n=n)

        user_message = (
            f"SEED INSTRUCTION: {seed['instruction']}\n\n"
            f"SEED OUTPUT:\n{seed['output']}\n\n"
            f"Generate {n} variations. Respond with a JSON array only."
        )

        return self._call_api(prompt, user_message)

    def _generate_negatives(self, n: int) -> list[dict]:
        """Generate n negative/error examples."""
        prompt = NEGATIVE_SYSTEM_PROMPT.format(n=n)
        user_message = f"Generate {n} negative examples. Respond with a JSON array only."
        return self._call_api(prompt, user_message)

    def _generate_complex(self, n: int) -> list[dict]:
        """Generate n complex multi-step examples."""
        prompt = COMPLEX_SYSTEM_PROMPT.format(n=n)
        user_message = f"Generate {n} complex examples. Respond with a JSON array only."
        return self._call_api(prompt, user_message)

    def _call_api(self, system_prompt: str, user_message: str) -> list[dict]:
        """Make a GPT-4o API call and parse the JSON response."""
        self.stats.api_calls += 1

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.8,  # Higher temp for diversity
                max_tokens=4096,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content

            # Estimate cost (GPT-4o: ~$2.50/1M input, ~$10/1M output)
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            self.stats.api_cost_estimate_usd += (
                input_tokens * 2.5 / 1_000_000 + output_tokens * 10 / 1_000_000
            )

            # Parse JSON — handle both array and {"examples": [...]} formats
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                # Try common wrapper keys
                for key in ("examples", "variations", "data", "items"):
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                return []
            else:
                return []

        except Exception as e:
            print(f"    API call failed: {e}")
            return []

    def _validate_examples(self, examples: list[dict]) -> list[dict]:
        """Validate each example's output against the OneAI schema."""
        try:
            from one_ai_config.validator import ConfigValidator
            validator = ConfigValidator()
        except ImportError:
            print("  WARNING: one_ai_config not installed, skipping schema validation")
            self.stats.schema_valid = len(examples)
            return examples

        valid = []
        for ex in examples:
            result = validator.validate(ex.get("output", ""))
            if result.is_valid:
                valid.append(ex)
                self.stats.schema_valid += 1
            else:
                self.stats.schema_invalid += 1

        return valid

    def save(
        self,
        examples: list[dict],
        output_dir: str | Path,
        filename: str = "synthetic_examples.json",
    ) -> Path:
        """Save generated examples to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        path = out / filename
        path.write_text(json.dumps(examples, indent=2, ensure_ascii=False))

        # Also save stats
        stats_path = out / "generation_stats.json"
        stats_path.write_text(json.dumps(vars(self.stats), indent=2))

        print(f"Saved {len(examples)} examples to {path}")
        return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--seed_file", default="data/seed/gold_examples.json")
    parser.add_argument("--output_dir", default="data/synthetic")
    parser.add_argument("--variations_per_seed", type=int, default=10)
    parser.add_argument("--num_negative", type=int, default=20)
    parser.add_argument("--num_complex", type=int, default=15)
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()

    generator = SyntheticDataGenerator(model=args.model)
    examples = generator.generate_from_seeds(
        seed_file=args.seed_file,
        variations_per_seed=args.variations_per_seed,
        num_negative=args.num_negative,
        num_complex=args.num_complex,
    )
    generator.save(examples, args.output_dir)


if __name__ == "__main__":
    main()
