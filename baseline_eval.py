#!/usr/bin/env python3
"""
Baseline Evaluation for Mistral 7B (pre-fine-tuning).

Runs a set of test prompts through the current pipeline (RAG + Mistral 7B + 
extract/patch/validate) and records pass/fail rates by category.

This establishes the benchmark that QLoRA fine-tuning should beat.

Usage:
    cd ~/Projects/chatbot_on
    python baseline_eval.py
    python baseline_eval.py --runs 3          # Average over 3 runs for reliability
    python baseline_eval.py --output results/baseline_$(date +%Y%m%d).json
"""

from __future__ import annotations

import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Test prompts organized by category and expected difficulty
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    # Category: simple_namespace (single step, straightforward)
    {
        "id": "ns_01",
        "category": "simple_namespace",
        "prompt": "Create a namespace called staging on my OneKE cluster",
        "expected_actions": ["oneke.namespace.create"],
        "difficulty": "easy",
    },
    {
        "id": "ns_02",
        "category": "simple_namespace",
        "prompt": "Delete the test namespace from the cluster",
        "expected_actions": ["oneke.namespace.delete"],
        "difficulty": "easy",
    },
    {
        "id": "ns_03",
        "category": "simple_namespace",
        "prompt": "Show me all the namespaces on my Kubernetes cluster",
        "expected_actions": ["oneke.namespace.list"],
        "difficulty": "easy",
    },

    # Category: simple_app (1-2 steps, app operations)
    {
        "id": "app_01",
        "category": "simple_app",
        "prompt": "Deploy Nginx using Helm on the default namespace",
        "expected_actions": ["oneke.app.deploy"],
        "difficulty": "easy",
    },
    {
        "id": "app_02",
        "category": "simple_app",
        "prompt": "Remove the grafana deployment from the monitoring namespace",
        "expected_actions": ["oneke.app.uninstall"],
        "difficulty": "easy",
    },
    {
        "id": "app_03",
        "category": "simple_app",
        "prompt": "Check if the redis release is healthy in production",
        "expected_actions": ["oneke.app.get_status"],
        "difficulty": "easy",
    },

    # Category: simple_vm (VM operations)
    {
        "id": "vm_01",
        "category": "simple_vm",
        "prompt": "Create a new VM called api-server with 2 CPUs and 4GB RAM using the Ubuntu template",
        "expected_actions": ["one.vm.create"],
        "difficulty": "easy",
    },
    {
        "id": "vm_02",
        "category": "simple_vm",
        "prompt": "Power off the development VM called dev-box",
        "expected_actions": ["one.vm.poweroff"],
        "difficulty": "easy",
    },
    {
        "id": "vm_03",
        "category": "simple_vm",
        "prompt": "List all my virtual machines in OpenNebula",
        "expected_actions": ["one.vm.list"],
        "difficulty": "easy",
    },

    # Category: multi_step (2-4 steps with dependencies)
    {
        "id": "multi_01",
        "category": "multi_step",
        "prompt": "Deploy WordPress on my OneKE cluster with its own namespace",
        "expected_actions": ["oneke.namespace.create", "oneke.app.deploy"],
        "difficulty": "medium",
    },
    {
        "id": "multi_02",
        "category": "multi_step",
        "prompt": "Create a namespace called logs, deploy Elasticsearch there, and wait for it to be ready",
        "expected_actions": ["oneke.namespace.create", "oneke.app.deploy", "oneke.app.wait_ready"],
        "difficulty": "medium",
    },
    {
        "id": "multi_03",
        "category": "multi_step",
        "prompt": "Scale the cluster to 4 workers and then deploy a new Redis instance",
        "expected_actions": ["oneke.cluster.scale_nodes", "oneke.app.deploy"],
        "difficulty": "medium",
    },

    # Category: complex (3+ steps, storage, expose, multiple concerns)
    {
        "id": "complex_01",
        "category": "complex",
        "prompt": "Set up a monitoring stack: create a monitoring namespace, deploy Prometheus with 20Gi storage, and expose it on NodePort",
        "expected_actions": ["oneke.namespace.create", "oneke.storage.create_pvc", "oneke.app.deploy", "oneke.service.expose"],
        "difficulty": "hard",
    },
    {
        "id": "complex_02",
        "category": "complex",
        "prompt": "Create a staging environment: new namespace, deploy the app with persistent storage, wait until ready, then get the endpoint",
        "expected_actions": ["oneke.namespace.create", "oneke.storage.create_pvc", "oneke.app.deploy", "oneke.app.wait_ready", "oneke.service.get_endpoint"],
        "difficulty": "hard",
    },

    # Category: negative (should produce error responses)
    {
        "id": "neg_01",
        "category": "negative",
        "prompt": "Deploy my app to AWS Lambda using a serverless configuration",
        "expected_actions": [],
        "difficulty": "easy",
        "expect_error": True,
    },
    {
        "id": "neg_02",
        "category": "negative",
        "prompt": "Set up a Docker Swarm cluster on my bare metal servers",
        "expected_actions": [],
        "difficulty": "easy",
        "expect_error": True,
    },

    # Category: cluster_ops (cluster management)
    {
        "id": "cluster_01",
        "category": "cluster_ops",
        "prompt": "Show me the current status of my OneKE cluster and list all nodes",
        "expected_actions": ["oneke.cluster.get_status", "oneke.cluster.list_nodes"],
        "difficulty": "easy",
    },
    {
        "id": "cluster_02",
        "category": "cluster_ops",
        "prompt": "Get info about my cluster",
        "expected_actions": ["oneke.cluster.get_info"],
        "difficulty": "easy",
    },
]


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class PromptResult:
    prompt_id: str
    category: str
    difficulty: str
    prompt: str
    passed: bool
    is_valid_yaml: bool
    is_valid_schema: bool
    has_correct_actions: bool
    has_error_response: bool
    expected_actions: list
    actual_actions: list
    error_message: str = ""
    raw_output: str = ""
    latency_seconds: float = 0.0


@dataclass 
class EvalRun:
    run_number: int
    timestamp: str
    results: list = field(default_factory=list)

    @property
    def total(self):
        return len(self.results)

    @property
    def passed(self):
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self):
        return self.passed / self.total if self.total > 0 else 0

    def by_category(self) -> dict:
        cats = {}
        for r in self.results:
            if r.category not in cats:
                cats[r.category] = {"total": 0, "passed": 0}
            cats[r.category]["total"] += 1
            if r.passed:
                cats[r.category]["passed"] += 1
        for cat in cats:
            cats[cat]["pass_rate"] = (
                cats[cat]["passed"] / cats[cat]["total"]
                if cats[cat]["total"] > 0 else 0
            )
        return cats

    def by_difficulty(self) -> dict:
        diffs = {}
        for r in self.results:
            if r.difficulty not in diffs:
                diffs[r.difficulty] = {"total": 0, "passed": 0}
            diffs[r.difficulty]["total"] += 1
            if r.passed:
                diffs[r.difficulty]["passed"] += 1
        for d in diffs:
            diffs[d]["pass_rate"] = (
                diffs[d]["passed"] / diffs[d]["total"]
                if diffs[d]["total"] > 0 else 0
            )
        return diffs


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class BaselineEvaluator:
    """
    Evaluates Mistral 7B through the full OneAI pipeline.

    Uses OneAIChain (which includes RAG retrieval, few-shot prompting,
    extract, patch, and validate steps) — the same pipeline used in production.
    """

    def __init__(self, max_retries: int = 3):
        from one_ai_core.chain import OneAIChain
        from one_ai_core.config import CoreConfig

        config = CoreConfig()
        config.max_retries = max_retries
        self.chain = OneAIChain(config=config)

    def evaluate_prompt(self, test: dict) -> PromptResult:
        """Run a single test prompt through the pipeline."""
        prompt = test["prompt"]
        expected_actions = test.get("expected_actions", [])
        expect_error = test.get("expect_error", False)

        start = time.time()
        try:
            result = self.chain.run(prompt)
            latency = time.time() - start
        except Exception as e:
            latency = time.time() - start
            return PromptResult(
                prompt_id=test["id"],
                category=test["category"],
                difficulty=test["difficulty"],
                prompt=prompt,
                passed=False,
                is_valid_yaml=False,
                is_valid_schema=False,
                has_correct_actions=False,
                has_error_response=False,
                expected_actions=expected_actions,
                actual_actions=[],
                error_message=f"Pipeline exception: {e}",
                latency_seconds=latency,
            )

        # Check validation result
        is_valid = result.is_valid if hasattr(result, "is_valid") else False
        config_yaml = getattr(result, "config_yaml", "")
        config_obj = getattr(result, "config", None)
        error_msg = getattr(result, "error", "") or ""

        # Extract actual actions from the validated config
        actual_actions = []
        has_error_response = False
        if config_obj:
            if hasattr(config_obj, "steps") and config_obj.steps:
                actual_actions = [s.action for s in config_obj.steps]
            if hasattr(config_obj, "error") and config_obj.error:
                has_error_response = config_obj.error.is_error if hasattr(config_obj.error, "is_error") else False

        # Determine if actions match expectations
        if expect_error:
            # For negative examples, pass if we got an error response OR empty steps
            has_correct_actions = has_error_response or len(actual_actions) == 0
        else:
            # Check that all expected actions appear (order doesn't matter)
            has_correct_actions = all(
                any(ea == aa for aa in actual_actions)
                for ea in expected_actions
            ) if expected_actions else True

        # Overall pass criteria
        passed = is_valid and (
            (expect_error and has_correct_actions) or
            (not expect_error and has_correct_actions and len(actual_actions) > 0)
        )

        return PromptResult(
            prompt_id=test["id"],
            category=test["category"],
            difficulty=test["difficulty"],
            prompt=prompt,
            passed=passed,
            is_valid_yaml=bool(config_yaml),
            is_valid_schema=is_valid,
            has_correct_actions=has_correct_actions,
            has_error_response=has_error_response,
            expected_actions=expected_actions,
            actual_actions=actual_actions,
            error_message=error_msg[:200] if error_msg else "",
            raw_output=config_yaml[:500] if config_yaml else "",
            latency_seconds=latency,
        )

    def run_eval(self, prompts: list[dict] = None, run_number: int = 1) -> EvalRun:
        """Run evaluation on all test prompts."""
        if prompts is None:
            prompts = TEST_PROMPTS

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        run = EvalRun(run_number=run_number, timestamp=timestamp)

        print(f"\n{'='*60}")
        print(f"  Baseline Evaluation Run #{run_number}")
        print(f"  {timestamp}")
        print(f"  {len(prompts)} test prompts")
        print(f"{'='*60}")

        for i, test in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] {test['id']}: {test['prompt'][:60]}...")
            result = self.evaluate_prompt(test)
            run.results.append(result)

            status = "\033[92mPASS\033[0m" if result.passed else "\033[91mFAIL\033[0m"
            print(f"  {status} | valid={result.is_valid_schema} | "
                  f"actions={result.actual_actions} | "
                  f"{result.latency_seconds:.1f}s")
            if not result.passed and result.error_message:
                print(f"  Error: {result.error_message[:100]}")

        self._print_report(run)
        return run

    def _print_report(self, run: EvalRun):
        """Print a formatted evaluation report."""
        print(f"\n{'='*60}")
        print(f"  BASELINE RESULTS — Run #{run.run_number}")
        print(f"{'='*60}")
        print(f"  Overall: {run.passed}/{run.total} ({run.pass_rate:.0%})")

        print(f"\n  By Category:")
        for cat, data in sorted(run.by_category().items()):
            bar = "█" * int(data["pass_rate"] * 10) + "░" * (10 - int(data["pass_rate"] * 10))
            print(f"    {cat:<20} {bar} {data['passed']}/{data['total']} ({data['pass_rate']:.0%})")

        print(f"\n  By Difficulty:")
        for diff, data in sorted(run.by_difficulty().items()):
            bar = "█" * int(data["pass_rate"] * 10) + "░" * (10 - int(data["pass_rate"] * 10))
            print(f"    {diff:<20} {bar} {data['passed']}/{data['total']} ({data['pass_rate']:.0%})")

        # Show failures
        failures = [r for r in run.results if not r.passed]
        if failures:
            print(f"\n  Failed prompts:")
            for f in failures:
                print(f"    - [{f.prompt_id}] {f.prompt[:50]}...")
                if f.error_message:
                    print(f"      Reason: {f.error_message[:80]}")


def save_results(runs: list[EvalRun], output_path: str):
    """Save evaluation results to JSON."""
    data = {
        "eval_type": "baseline_mistral7b",
        "num_runs": len(runs),
        "prompts_per_run": len(TEST_PROMPTS),
        "runs": [],
        "aggregate": {},
    }

    all_pass_rates = []
    category_rates = {}

    for run in runs:
        run_data = {
            "run_number": run.run_number,
            "timestamp": run.timestamp,
            "pass_rate": run.pass_rate,
            "by_category": run.by_category(),
            "by_difficulty": run.by_difficulty(),
            "results": [asdict(r) for r in run.results],
        }
        data["runs"].append(run_data)
        all_pass_rates.append(run.pass_rate)

        for cat, cat_data in run.by_category().items():
            if cat not in category_rates:
                category_rates[cat] = []
            category_rates[cat].append(cat_data["pass_rate"])

    # Aggregate across runs
    data["aggregate"] = {
        "mean_pass_rate": sum(all_pass_rates) / len(all_pass_rates),
        "min_pass_rate": min(all_pass_rates),
        "max_pass_rate": max(all_pass_rates),
        "by_category": {
            cat: {
                "mean": sum(rates) / len(rates),
                "min": min(rates),
                "max": max(rates),
            }
            for cat, rates in category_rates.items()
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(data, indent=2, default=str))
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation for Mistral 7B")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of evaluation runs (average for reliability)")
    parser.add_argument("--output", default="results/baseline_eval.json",
                        help="Output path for results JSON")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries per prompt in the chain")
    args = parser.parse_args()

    evaluator = BaselineEvaluator(max_retries=args.max_retries)

    runs = []
    for i in range(args.runs):
        run = evaluator.run_eval(run_number=i + 1)
        runs.append(run)
        if i < args.runs - 1:
            print("\n--- Pausing 5s between runs ---")
            time.sleep(5)

    save_results(runs, args.output)

    # Final summary
    if len(runs) > 1:
        rates = [r.pass_rate for r in runs]
        print(f"\n{'='*60}")
        print(f"  AGGREGATE ACROSS {len(runs)} RUNS")
        print(f"  Mean pass rate: {sum(rates)/len(rates):.0%}")
        print(f"  Range: {min(rates):.0%} - {max(rates):.0%}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
