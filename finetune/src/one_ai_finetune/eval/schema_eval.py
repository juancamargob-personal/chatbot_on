"""
Automated schema compliance evaluation.

Runs a model's outputs through the OneAI config validator and reports
what percentage of outputs are valid YAML that passes schema validation.
This is the fastest, cheapest eval — no LLM API calls needed.

Usage:
    evaluator = SchemaEvaluator()
    results = evaluator.evaluate(model_outputs)
    print(results.summary())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SchemaEvalResult:
    """Result of schema compliance evaluation for a single example."""
    instruction: str
    is_valid_yaml: bool
    is_valid_schema: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SchemaEvalReport:
    """Aggregated results across all examples."""
    total: int
    yaml_valid: int
    schema_valid: int
    results: list[SchemaEvalResult]

    @property
    def yaml_rate(self) -> float:
        return self.yaml_valid / max(self.total, 1)

    @property
    def schema_rate(self) -> float:
        return self.schema_valid / max(self.total, 1)

    def summary(self) -> str:
        lines = [
            f"=== Schema Compliance Report ===",
            f"Total examples:     {self.total}",
            f"Valid YAML:         {self.yaml_valid}/{self.total} ({self.yaml_rate:.1%})",
            f"Valid schema:       {self.schema_valid}/{self.total} ({self.schema_rate:.1%})",
        ]

        # Show common errors
        if self.total > self.schema_valid:
            error_counts: dict[str, int] = {}
            for r in self.results:
                for e in r.errors:
                    # Simplify error for grouping
                    key = e.split("]")[0] + "]" if "]" in e else e[:80]
                    error_counts[key] = error_counts.get(key, 0) + 1

            lines.append("")
            lines.append("Most common errors:")
            for err, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  ({count}x) {err}")

        return "\n".join(lines)


class SchemaEvaluator:
    """
    Evaluates model outputs for schema compliance.

    Usage:
        evaluator = SchemaEvaluator()

        # From a list of (instruction, output) pairs
        report = evaluator.evaluate([
            {"instruction": "Deploy WordPress", "output": "version: ..."},
        ])

        # Or from a generate function
        report = evaluator.evaluate_model(
            eval_data=[{"instruction": "Deploy WordPress"}],
            generate_fn=my_model.generate,
        )
    """

    def __init__(self):
        try:
            from one_ai_config.validator import ConfigValidator
            self.validator = ConfigValidator()
        except ImportError:
            raise ImportError(
                "one_ai_config is required for schema evaluation. "
                "Install it with: pip install -e ../one-ai-config"
            )

    def evaluate(self, examples: list[dict]) -> SchemaEvalReport:
        """
        Evaluate a list of pre-generated (instruction, output) pairs.

        Args:
            examples: List of dicts with "instruction" and "output" keys

        Returns:
            SchemaEvalReport with per-example and aggregate results
        """
        results = []
        yaml_valid = 0
        schema_valid = 0

        for ex in examples:
            instruction = ex.get("instruction", "")
            output = ex.get("output", "")

            result = self._evaluate_single(instruction, output)
            results.append(result)

            if result.is_valid_yaml:
                yaml_valid += 1
            if result.is_valid_schema:
                schema_valid += 1

        return SchemaEvalReport(
            total=len(examples),
            yaml_valid=yaml_valid,
            schema_valid=schema_valid,
            results=results,
        )

    def evaluate_model(
        self,
        eval_data: list[dict],
        generate_fn,
    ) -> SchemaEvalReport:
        """
        Generate outputs using a model function and evaluate them.

        Args:
            eval_data: List of dicts with "instruction" keys
            generate_fn: Function(instruction: str) -> str that generates YAML

        Returns:
            SchemaEvalReport
        """
        examples = []
        for item in eval_data:
            instruction = item["instruction"]
            output = generate_fn(instruction)
            examples.append({"instruction": instruction, "output": output})

        return self.evaluate(examples)

    def _evaluate_single(self, instruction: str, output: str) -> SchemaEvalResult:
        """Evaluate a single output."""
        import yaml

        # Step 1: Check if it's valid YAML
        is_valid_yaml = True
        try:
            cleaned = output.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            yaml.safe_load(cleaned)
        except yaml.YAMLError:
            is_valid_yaml = False

        # Step 2: Check against Pydantic schema
        validation = self.validator.validate(output)

        return SchemaEvalResult(
            instruction=instruction,
            is_valid_yaml=is_valid_yaml,
            is_valid_schema=validation.is_valid,
            errors=validation.errors,
            warnings=validation.warnings,
        )
