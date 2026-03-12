"""
Data quality checks — deduplication, filtering, and quality scoring.

Ensures the training dataset is clean before fine-tuning:
- Removes exact and near-duplicate instructions
- Filters out examples that are too short or too long
- Validates that outputs parse as valid YAML
- Reports quality statistics

Usage:
    cleaner = DataQualityCleaner()
    clean_data = cleaner.clean(raw_examples)
    print(cleaner.report())
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class QualityReport:
    """Report from a data quality cleaning run."""
    input_count: int = 0
    output_count: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    too_short: int = 0
    too_long: int = 0
    invalid_yaml: int = 0
    empty_fields: int = 0

    def summary(self) -> str:
        removed = self.input_count - self.output_count
        lines = [
            f"=== Data Quality Report ===",
            f"Input:              {self.input_count}",
            f"Output:             {self.output_count}",
            f"Removed:            {removed}",
            f"  Exact duplicates: {self.exact_duplicates}",
            f"  Near duplicates:  {self.near_duplicates}",
            f"  Too short:        {self.too_short}",
            f"  Too long:         {self.too_long}",
            f"  Invalid YAML:     {self.invalid_yaml}",
            f"  Empty fields:     {self.empty_fields}",
        ]
        return "\n".join(lines)


class DataQualityCleaner:
    """
    Cleans and deduplicates training data.

    Usage:
        cleaner = DataQualityCleaner()
        clean = cleaner.clean(raw_examples)
        print(cleaner.report.summary())
    """

    def __init__(
        self,
        min_instruction_words: int = 3,
        max_instruction_words: int = 200,
        min_output_length: int = 20,
        max_output_length: int = 10000,
        similarity_threshold: float = 0.9,
    ):
        self.min_instruction_words = min_instruction_words
        self.max_instruction_words = max_instruction_words
        self.min_output_length = min_output_length
        self.max_output_length = max_output_length
        self.similarity_threshold = similarity_threshold
        self.report = QualityReport()

    def clean(self, examples: list[dict]) -> list[dict]:
        """
        Clean a dataset by removing duplicates and low-quality examples.

        Args:
            examples: List of {"instruction": ..., "output": ...} dicts

        Returns:
            Cleaned list
        """
        self.report = QualityReport(input_count=len(examples))

        # Step 1: Remove empty fields
        examples = [ex for ex in examples if self._has_fields(ex)]

        # Step 2: Remove exact duplicates (by instruction hash)
        examples = self._remove_exact_duplicates(examples)

        # Step 3: Remove near-duplicates (by normalized instruction)
        examples = self._remove_near_duplicates(examples)

        # Step 4: Filter by length
        examples = [ex for ex in examples if self._check_length(ex)]

        # Step 5: Validate YAML
        examples = [ex for ex in examples if self._check_yaml(ex)]

        self.report.output_count = len(examples)
        return examples

    def _has_fields(self, ex: dict) -> bool:
        """Check that required fields are non-empty."""
        if not ex.get("instruction", "").strip() or not ex.get("output", "").strip():
            self.report.empty_fields += 1
            return False
        return True

    def _remove_exact_duplicates(self, examples: list[dict]) -> list[dict]:
        """Remove examples with identical instruction text."""
        seen: set[str] = set()
        unique = []
        for ex in examples:
            key = hashlib.sha256(ex["instruction"].encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(ex)
            else:
                self.report.exact_duplicates += 1
        return unique

    def _remove_near_duplicates(self, examples: list[dict]) -> list[dict]:
        """
        Remove near-duplicate instructions using normalized text comparison.

        Normalizes by lowercasing, stripping punctuation, and sorting words.
        Two instructions that normalize to the same string are considered
        near-duplicates.
        """
        import re

        def normalize(text: str) -> str:
            text = text.lower()
            text = re.sub(r"[^\w\s]", "", text)
            words = sorted(text.split())
            return " ".join(words)

        seen: set[str] = set()
        unique = []
        for ex in examples:
            key = normalize(ex["instruction"])
            if key not in seen:
                seen.add(key)
                unique.append(ex)
            else:
                self.report.near_duplicates += 1
        return unique

    def _check_length(self, ex: dict) -> bool:
        """Check instruction and output length constraints."""
        instruction_words = len(ex["instruction"].split())
        output_len = len(ex["output"])

        if instruction_words < self.min_instruction_words:
            self.report.too_short += 1
            return False
        if instruction_words > self.max_instruction_words:
            self.report.too_long += 1
            return False
        if output_len < self.min_output_length:
            self.report.too_short += 1
            return False
        if output_len > self.max_output_length:
            self.report.too_long += 1
            return False
        return True

    def _check_yaml(self, ex: dict) -> bool:
        """Check that the output is parseable YAML."""
        import yaml
        try:
            output = ex["output"].strip()
            if output.startswith("```"):
                output = output.split("\n", 1)[1] if "\n" in output else output[3:]
            if output.endswith("```"):
                output = output[:-3]
            yaml.safe_load(output)
            return True
        except yaml.YAMLError:
            self.report.invalid_yaml += 1
            return False
