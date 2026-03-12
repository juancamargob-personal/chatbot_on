"""
LLM-as-Judge evaluation for OneAI configuration outputs.

Uses a strong LLM (GPT-4o) to evaluate generated configurations
across multiple quality dimensions. Supports three-way comparison
between base model, fine-tuned model, and fine-tuned + RAG.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from openai import OpenAI


class EvalDimension(str, Enum):
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    SAFETY = "safety"
    CLARITY = "clarity"
    SCHEMA_ADHERENCE = "schema_adherence"


@dataclass
class DimensionScore:
    dimension: EvalDimension
    score: int  # 1-5
    reasoning: str


@dataclass
class JudgeResult:
    instruction: str
    model_name: str
    scores: list[DimensionScore]
    overall_score: float
    overall_reasoning: str
    raw_judge_output: str

    @property
    def score_dict(self) -> dict[str, int]:
        return {s.dimension.value: s.score for s in self.scores}

    def avg_score(self) -> float:
        return sum(s.score for s in self.scores) / len(self.scores) if self.scores else 0.0


@dataclass
class ComparisonResult:
    instruction: str
    results: dict[str, JudgeResult]  # model_name → JudgeResult
    winner: Optional[str] = None


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for AI-generated infrastructure configurations.
You evaluate configurations generated for OpenNebula / OneKE clusters.

You must score each configuration on these dimensions (1-5 scale):

1. **Correctness** (1-5): Would this configuration achieve the stated goal if executed?
   - 5: Perfectly correct, all actions and params are right
   - 3: Mostly correct, minor issues that wouldn't block execution
   - 1: Fundamentally wrong actions or parameters

2. **Completeness** (1-5): Are all necessary steps included?
   - 5: All steps present, nothing missing
   - 3: Core steps present, some auxiliary steps missing
   - 1: Major steps missing, config would fail

3. **Safety** (1-5): Are pre-checks, rollback, and error handling adequate?
   - 5: Comprehensive pre-checks, rollback plan, proper failure strategies
   - 3: Basic checks present, rollback partially defined
   - 1: No safety measures at all

4. **Clarity** (1-5): Are descriptions and structure clear and helpful?
   - 5: Excellent descriptions, logical ordering, helpful metadata
   - 3: Adequate descriptions, reasonable structure
   - 1: Confusing, poorly described, hard to understand

5. **Schema Adherence** (1-5): Does the output follow the expected YAML schema?
   - 5: Perfect schema compliance
   - 3: Minor schema deviations
   - 1: Major schema violations

Respond ONLY with valid JSON in this exact format:
{
  "scores": {
    "correctness": {"score": N, "reasoning": "..."},
    "completeness": {"score": N, "reasoning": "..."},
    "safety": {"score": N, "reasoning": "..."},
    "clarity": {"score": N, "reasoning": "..."},
    "schema_adherence": {"score": N, "reasoning": "..."}
  },
  "overall_score": N.N,
  "overall_reasoning": "..."
}
"""

JUDGE_USER_TEMPLATE = """## Original Request
{instruction}

## Generated Configuration
```yaml
{output}
```

{reference_section}

Please evaluate the generated configuration."""

REFERENCE_SECTION_TEMPLATE = """## Reference Configuration (Gold Standard)
```yaml
{reference}
```
"""


class LLMJudge:
    """
    Evaluates LLM-generated configs using GPT-4o as judge.

    Usage:
        judge = LLMJudge()
        result = judge.evaluate(
            instruction="Deploy WordPress on OneKE",
            output=generated_yaml,
            model_name="mistral-7b-qlora",
            reference=gold_standard_yaml,  # optional
        )
        print(result.overall_score)
    """

    def __init__(
        self,
        judge_model: str = "gpt-4o",
        api_key: Optional[str] = None,
    ):
        self.judge_model = judge_model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def evaluate(
        self,
        instruction: str,
        output: str,
        model_name: str,
        reference: Optional[str] = None,
    ) -> JudgeResult:
        """
        Evaluate a single generated configuration.

        Args:
            instruction: The original natural language request
            output: The generated YAML configuration
            model_name: Name of the model that generated the output
            reference: Optional gold-standard config for comparison

        Returns:
            JudgeResult with dimensional scores and overall assessment
        """
        ref_section = ""
        if reference:
            ref_section = REFERENCE_SECTION_TEMPLATE.format(reference=reference)

        user_message = JUDGE_USER_TEMPLATE.format(
            instruction=instruction,
            output=output,
            reference_section=ref_section,
        )

        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,  # Low temp for consistent evaluation
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content
        parsed = json.loads(raw_output)

        scores = []
        for dim in EvalDimension:
            dim_data = parsed["scores"].get(dim.value, {})
            scores.append(DimensionScore(
                dimension=dim,
                score=dim_data.get("score", 0),
                reasoning=dim_data.get("reasoning", "No reasoning provided"),
            ))

        return JudgeResult(
            instruction=instruction,
            model_name=model_name,
            scores=scores,
            overall_score=parsed.get("overall_score", 0.0),
            overall_reasoning=parsed.get("overall_reasoning", ""),
            raw_judge_output=raw_output,
        )

    def compare_models(
        self,
        instruction: str,
        outputs: dict[str, str],  # model_name → generated yaml
        reference: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Compare outputs from multiple models on the same instruction.

        Args:
            instruction: The original request
            outputs: Dict mapping model names to their generated configs
            reference: Optional gold standard

        Returns:
            ComparisonResult with per-model scores and a winner
        """
        results: dict[str, JudgeResult] = {}
        for model_name, output in outputs.items():
            results[model_name] = self.evaluate(
                instruction=instruction,
                output=output,
                model_name=model_name,
                reference=reference,
            )

        # Determine winner by overall score
        winner = max(results.keys(), key=lambda m: results[m].overall_score)

        return ComparisonResult(
            instruction=instruction,
            results=results,
            winner=winner,
        )


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results across an evaluation dataset."""
    model_name: str
    total_examples: int
    avg_scores: dict[str, float]  # dimension → avg score
    overall_avg: float
    schema_compliance_rate: float  # from automated eval
    individual_results: list[JudgeResult]

    def summary(self) -> str:
        lines = [
            f"=== Benchmark Results: {self.model_name} ===",
            f"Examples evaluated: {self.total_examples}",
            f"Overall average: {self.overall_avg:.2f} / 5.0",
            f"Schema compliance: {self.schema_compliance_rate:.1%}",
            "",
            "Dimension scores:",
        ]
        for dim, score in self.avg_scores.items():
            bar = "█" * int(score) + "░" * (5 - int(score))
            lines.append(f"  {dim:20s} {bar} {score:.2f}")
        return "\n".join(lines)


class BenchmarkRunner:
    """
    Runs the full benchmark suite: automated schema checks + LLM judge.

    Usage:
        runner = BenchmarkRunner(judge=LLMJudge())
        results = runner.run_benchmark(
            model_name="mistral-7b-qlora",
            eval_dataset=dataset,
            generate_fn=my_model_generate,
        )
        print(results.summary())
    """

    def __init__(self, judge: LLMJudge):
        self.judge = judge

    def run_benchmark(
        self,
        model_name: str,
        eval_dataset: list[dict],
        generate_fn,  # Callable: instruction → yaml string
        reference_configs: Optional[dict[str, str]] = None,
    ) -> BenchmarkResults:
        """
        Run full evaluation on a dataset.

        Args:
            model_name: Name of the model being evaluated
            eval_dataset: List of dicts with 'instruction' and optionally 'output' (reference)
            generate_fn: Function that takes an instruction and returns generated YAML
            reference_configs: Optional dict mapping instruction → reference YAML
        """
        from one_ai_config.validator import ConfigValidator

        validator = ConfigValidator()
        individual_results: list[JudgeResult] = []
        schema_pass_count = 0

        for example in eval_dataset:
            instruction = example["instruction"]

            # Generate config
            generated = generate_fn(instruction)

            # Automated schema check
            validation = validator.validate(generated)
            if validation.is_valid:
                schema_pass_count += 1

            # LLM judge evaluation
            reference = None
            if reference_configs and instruction in reference_configs:
                reference = reference_configs[instruction]
            elif "output" in example:
                reference = example["output"]

            judge_result = self.judge.evaluate(
                instruction=instruction,
                output=generated,
                model_name=model_name,
                reference=reference,
            )
            individual_results.append(judge_result)

        # Aggregate scores
        dim_totals: dict[str, float] = {}
        for result in individual_results:
            for score in result.scores:
                dim_totals.setdefault(score.dimension.value, 0.0)
                dim_totals[score.dimension.value] += score.score

        n = len(individual_results) or 1
        avg_scores = {dim: total / n for dim, total in dim_totals.items()}
        overall_avg = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.0

        return BenchmarkResults(
            model_name=model_name,
            total_examples=len(eval_dataset),
            avg_scores=avg_scores,
            overall_avg=overall_avg,
            schema_compliance_rate=schema_pass_count / n,
            individual_results=individual_results,
        )

    def run_three_way_comparison(
        self,
        eval_dataset: list[dict],
        generate_fns: dict[str, callable],  # model_name → generate function
    ) -> dict[str, BenchmarkResults]:
        """
        Run the three-way comparison benchmark.

        Expected model names:
        - "base_with_rag": Base model + RAG (no fine-tuning)
        - "finetuned_with_rag": Fine-tuned model + RAG
        - "finetuned_no_rag": Fine-tuned model without RAG
        """
        all_results = {}
        for model_name, gen_fn in generate_fns.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {model_name}")
            print(f"{'='*60}")
            all_results[model_name] = self.run_benchmark(
                model_name=model_name,
                eval_dataset=eval_dataset,
                generate_fn=gen_fn,
            )
            print(all_results[model_name].summary())

        return all_results
