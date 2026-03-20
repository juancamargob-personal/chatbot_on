"""
Synthetic training data generator for OneAI fine-tuning.

Uses GPT-4o-mini to expand seed examples into diverse training pairs.
All prompts include SCHEMA_REFERENCE with exact field names, types, and
constraints from the actual Pydantic models.

Usage:
    python -m one_ai_finetune.data.generate_synthetic \
        --seed_file data/seed/gold_examples.json \
        --output_dir data/synthetic \
        --variations_per_seed 5 \
        --num_negative 5 \
        --num_complex 5 \
        --model gpt-4o-mini
"""

from __future__ import annotations

import json
import os
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# Schema reference — matches the actual Pydantic models exactly.
# Generated from ACTION_PARAM_REGISTRY introspection.
# ---------------------------------------------------------------------------

SCHEMA_REFERENCE = """\
=== ONEAI CONFIGURATION SCHEMA (MANDATORY — follow exactly) ===

Every output must be a YAML string with this structure:

version: "1.0"
metadata:
  description: "<what this config does>"          # REQUIRED (str)
  target_cluster: "<cluster name>"                # optional (str)
  estimated_duration: "<e.g. 5 minutes>"          # optional (str)
  risk_level: low|medium|high                     # optional (enum)
  tags: [<tag1>, <tag2>]                          # optional (list[str])
steps:
  - id: step_01                                   # REQUIRED, pattern: step_XX or step_XXX
    action: <action_name>                         # REQUIRED, from list below
    description: "<what this step does>"          # REQUIRED (str)
    params:                                       # action-specific (see below)
      key: value
    depends_on: [step_XX]                         # optional (list[str])
    on_failure: abort|rollback|continue|retry     # optional (enum)
    timeout_seconds: 300                          # optional (int)
    retry_count: 2                                # optional (int)
validation:
  pre_checks:
    - type: <check_type>                          # REQUIRED — see valid types below
      description: "<what to check>"              # str
  post_checks:
    - type: <check_type>
      description: "<what to check>"
rollback:
  steps:
    - id: step_90                                 # Use step_90, step_91, etc. for rollback
      action: <action_name>                       # Must use step_XX format, NOT rollback_XX
      description: "<undo description>"
      params:
        key: value

For ERROR/IMPOSSIBLE requests, use this structure:
version: "1.0"
metadata:
  description: "<what was requested>"
  risk_level: high
steps: []
error:
  is_error: true
  reason: "<why it cannot be done>"
  suggestion: "<what user should do instead>"

=== VALID CHECK TYPES (for pre_checks/post_checks) ===
cluster_reachable, namespace_available, namespace_exists, pods_running,
service_available, vm_exists, resource_available, helm_repo_available

Do NOT use "command", "http", or "port" — only the types listed above.

=== VALID ACTIONS AND THEIR EXACT PARAMS ===

OneKE Namespace:
  oneke.namespace.create
    name: str (REQUIRED)              ← NOT "namespace", the field is "name"
    labels: dict[str,str] (optional)
    annotations: dict[str,str] (optional)
  oneke.namespace.delete
    name: str (REQUIRED)              ← NOT "namespace", the field is "name"
  oneke.namespace.list
    label_selector: str (optional)

OneKE App (Helm):
  oneke.app.deploy
    chart: str (REQUIRED)
    namespace: str (REQUIRED)
    release_name: str (REQUIRED)
    values: dict (optional)
    version: str (optional)
    repo_url: str (optional)          ← NO "repo_name" field exists
    create_namespace: bool (optional, default true)
    wait: bool (optional, default true)
  oneke.app.uninstall
    release_name: str (REQUIRED)
    namespace: str (REQUIRED)
    keep_history: bool (optional, default false)
  oneke.app.upgrade
    chart: str (REQUIRED)
    release_name: str (REQUIRED)
    namespace: str (REQUIRED)
    values: dict (optional)
    version: str (optional)
    reuse_values: bool (optional, default true)
  oneke.app.list
    namespace: str (optional)
  oneke.app.wait_ready
    namespace: str (REQUIRED)
    label_selector: str (REQUIRED)    ← NOT "release_name"
    timeout_seconds: int (optional, default 300)
    expected_replicas: int (optional)
  oneke.app.get_status
    release_name: str (REQUIRED)
    namespace: str (REQUIRED)

OneKE Service:
  oneke.service.get_endpoint
    namespace: str (REQUIRED)
    service_name: str (REQUIRED)
  oneke.service.expose
    namespace: str (REQUIRED)
    deployment_name: str (REQUIRED)   ← NOT "service_name"
    port: int (REQUIRED)
    target_port: int (optional)
    service_type: str (optional, default "ClusterIP")  ← NOT "type"
    service_name: str (optional)
  oneke.service.list
    namespace: str (optional)

OneKE Storage:
  oneke.storage.create_pvc
    name: str (REQUIRED)
    namespace: str (REQUIRED)
    size: str (REQUIRED)              ← e.g. "10Gi"
    storage_class: str (optional)
    access_mode: str (optional, default "ReadWriteOnce")
  oneke.storage.delete_pvc
    name: str (REQUIRED)
    namespace: str (REQUIRED)
  oneke.storage.list_pvcs
    namespace: str (optional)

OneKE Cluster:
  oneke.cluster.get_info
    cluster_name: str (optional)
  oneke.cluster.get_status
    cluster_name: str (optional)
  oneke.cluster.list_nodes
    cluster_name: str (optional)
    role: str (optional)
  oneke.cluster.scale_nodes
    worker_count: int (REQUIRED)
    cluster_name: str (optional)
    node_template: str (optional)

OpenNebula VM — IMPORTANT: VM IDs are integers, not strings:
  one.vm.create
    template_id: int (REQUIRED)       ← integer ID, NOT a template name
    name: str (REQUIRED)              ← NOT "vm_name"
    cpu: float (optional)
    memory_mb: int (optional)         ← in megabytes, NOT "memory"
    hold: bool (optional, default false)
    extra_template: str (optional)
  one.vm.delete
    vm_id: int (REQUIRED)             ← integer ID
  one.vm.poweroff
    vm_id: int (REQUIRED)
    hard: bool (optional, default false)
  one.vm.resume
    vm_id: int (REQUIRED)
  one.vm.resize
    vm_id: int (REQUIRED)
    cpu: float (optional)
    memory_mb: int (optional)
    enforce: bool (optional, default false)
  one.vm.snapshot_create
    vm_id: int (REQUIRED)
    snapshot_name: str (optional, default "snapshot")
  one.vm.list
    filter_flag: int (optional)

=== CRITICAL RULES ===
1. Step IDs: step_01, step_02, ... (pattern ^step_\\d{2,3}$)
2. Rollback step IDs: step_90, step_91, ... (same pattern, NOT rollback_XX)
3. depends_on must reference existing step IDs
4. Use ONLY the action names and param names listed above
5. VM operations use integer IDs (template_id, vm_id), not string names
6. Namespace create/delete use "name", not "namespace"
7. service.expose uses "deployment_name" and "service_type", not "service_name" and "type"
8. app.wait_ready uses "label_selector", not "release_name"
9. Check types: only the 8 enum values listed above
10. Output must be valid YAML (proper indentation, no tabs)
"""


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
- Different application names and parameters (e.g. nginx, grafana, redis, mongodb)
- Different namespaces, cluster names, and VM IDs
- Different levels of detail in the request
- Different user personas (sysadmin, developer, manager)

Focus on OpenNebula and OneKE infrastructure operations. The tool manages:
- OneKE Kubernetes clusters running on OpenNebula
- Application deployment via Helm charts on those clusters
- OpenNebula virtual machines via pyone API

{schema}

Respond with a JSON array of objects, each with "instruction" and "output" fields.
The "output" field must be a valid YAML STRING (not a dict/object).
Do NOT include any markdown formatting or code fences in your response.
Return ONLY the raw JSON array, starting with [ and ending with ]."""

NEGATIVE_SYSTEM_PROMPT = """\
You are generating NEGATIVE training examples for an OpenNebula infrastructure tool.

Generate {n} examples where the user makes a request that CANNOT be fulfilled.
Each example should have:
- An "instruction" that sounds reasonable but is impossible or unsupported
- An "output" that is a valid OneAI error config YAML string

Reasons a request might be impossible:
- Requesting operations on unsupported platforms (AWS, Azure, GCP, bare Docker)
- Requesting features OpenNebula doesn't have (e.g. serverless functions)
- Contradictory requirements (e.g. "deploy on OneKE without Kubernetes")
- Operations requiring manual physical intervention
- Missing critical information that can't be inferred
- Trying to use actions that don't exist in the supported list

{schema}

Respond with a JSON array of objects, each with "instruction" and "output" fields.
The "output" must be a YAML string using the error format from the schema.
Return ONLY the raw JSON array, starting with [ and ending with ]."""

COMPLEX_SYSTEM_PROMPT = """\
You are generating COMPLEX multi-step training examples for an OpenNebula tool.

Generate {n} examples that combine MULTIPLE operations into a single request.
For example:
- "Set up a full monitoring stack with Prometheus and Grafana on my OneKE cluster"
- "Create a staging namespace, deploy the app with persistent storage, and expose it"
- "Scale up the cluster workers, deploy a new app version, and verify it's healthy"
- "Create a VM and then set up a namespace on the OneKE cluster"

Each must produce a valid OneAI config YAML with 3-6 steps, proper step dependencies
(depends_on), pre-checks using valid check types, post-checks, and rollback steps
using step_90+ IDs.

Focus on realistic OpenNebula/OneKE infrastructure workflows.

{schema}

Respond with a JSON array of objects, each with "instruction" and "output" fields.
The "output" must be a valid YAML string.
Return ONLY the raw JSON array, starting with [ and ending with ]."""


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
    api_errors: int = 0
    api_cost_estimate_usd: float = 0.0
    invalid_examples: list = field(default_factory=list)


class SyntheticDataGenerator:
    """
    Generates synthetic training data using GPT-4o-mini.

    Usage:
        generator = SyntheticDataGenerator(model="gpt-4o-mini")
        examples = generator.generate_from_seeds("data/seed/gold_examples.json")
        generator.save(examples, "data/synthetic/")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        validate_schema: bool = True,
        max_retries: int = 3,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.validate_schema = validate_schema
        self.max_retries = max_retries
        self.stats = GenerationStats()

    def generate_from_seeds(
        self,
        seed_file: str | Path,
        variations_per_seed: int = 10,
        num_negative: int = 20,
        num_complex: int = 15,
    ) -> list[dict]:
        """Generate a full synthetic dataset from seed examples."""
        seeds = json.loads(Path(seed_file).read_text())
        all_examples = list(seeds)  # Start with the seeds themselves
        print(f"Loaded {len(seeds)} seed examples")

        # Phase 1: Generate variations of each seed
        print(f"\n--- Phase 1: Generating {variations_per_seed} variations per seed ---")
        for i, seed in enumerate(seeds):
            print(f"  Seed {i+1}/{len(seeds)}: {seed['instruction'][:60]}...")
            variations = self._generate_variations(seed, variations_per_seed)
            variations = self._normalize_outputs(variations)
            all_examples.extend(variations)
            print(f"    Generated {len(variations)} variations")

        # Phase 2: Generate negative examples
        print(f"\n--- Phase 2: Generating {num_negative} negative examples ---")
        negatives = self._generate_negatives(num_negative)
        negatives = self._normalize_outputs(negatives)
        all_examples.extend(negatives)
        print(f"  Generated {len(negatives)} negative examples")

        # Phase 3: Generate complex multi-step examples
        print(f"\n--- Phase 3: Generating {num_complex} complex examples ---")
        complex_examples = self._generate_complex(num_complex)
        complex_examples = self._normalize_outputs(complex_examples)
        all_examples.extend(complex_examples)
        print(f"  Generated {len(complex_examples)} complex examples")

        # Phase 4: Validate all against schema
        if self.validate_schema:
            print(f"\n--- Phase 4: Validating schema compliance ---")
            all_examples = self._validate_examples(all_examples)

        self.stats.total_generated = len(all_examples)
        self._print_summary()

        return all_examples

    def _generate_variations(self, seed: dict, n: int) -> list[dict]:
        """Generate n variations of a single seed example."""
        prompt = VARIATION_SYSTEM_PROMPT.format(n=n, schema=SCHEMA_REFERENCE)

        user_message = (
            f"SEED INSTRUCTION: {seed['instruction']}\n\n"
            f"SEED OUTPUT:\n{seed['output']}\n\n"
            f"Generate exactly {n} variations. Each must use only valid actions "
            f"and exact param names from the schema. Pay special attention to:\n"
            f"- namespace.create/delete use 'name', not 'namespace'\n"
            f"- VM actions use integer vm_id/template_id\n"
            f"- app.wait_ready uses 'label_selector', not 'release_name'\n"
            f"- service.expose uses 'deployment_name' and 'service_type'\n"
            f"- Rollback step IDs use step_90+, not rollback_XX\n"
            f"Respond with a JSON array only."
        )

        return self._call_api(prompt, user_message)

    def _generate_negatives(self, n: int) -> list[dict]:
        """Generate n negative/error examples."""
        prompt = NEGATIVE_SYSTEM_PROMPT.format(n=n, schema=SCHEMA_REFERENCE)
        user_message = (
            f"Generate exactly {n} negative examples showing impossible or "
            f"unsupported requests. Each output must use the error YAML format "
            f"with is_error: true, reason, and suggestion. "
            f"Respond with a JSON array only."
        )
        return self._call_api(prompt, user_message)

    def _generate_complex(self, n: int) -> list[dict]:
        """Generate n complex multi-step examples."""
        prompt = COMPLEX_SYSTEM_PROMPT.format(n=n, schema=SCHEMA_REFERENCE)
        user_message = (
            f"Generate exactly {n} complex multi-step examples combining 3-6 "
            f"operations each. Use correct param names throughout. Remember:\n"
            f"- namespace.create uses 'name' param\n"
            f"- app.wait_ready uses 'label_selector' param\n"
            f"- rollback steps use step_90+ IDs\n"
            f"- check types must be from the valid enum list\n"
            f"Respond with a JSON array only."
        )
        return self._call_api(prompt, user_message)

    def _call_api(self, system_prompt: str, user_message: str) -> list[dict]:
        """
        Make a GPT API call and parse the JSON response.
        
        NOTE: No response_format parameter — we want raw JSON arrays.
        """
        for attempt in range(self.max_retries):
            self.stats.api_calls += 1

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.8,
                    max_tokens=8192,
                )

                raw = response.choices[0].message.content.strip()

                # Track cost (gpt-4o-mini: ~$0.15/1M input, ~$0.60/1M output)
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                self.stats.api_cost_estimate_usd += (
                    input_tokens * 0.15 / 1_000_000
                    + output_tokens * 0.60 / 1_000_000
                )

                # Clean up common GPT formatting issues
                raw = self._clean_json_response(raw)

                # Parse JSON
                parsed = json.loads(raw)

                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    for key in ("examples", "variations", "data", "items", "results"):
                        if key in parsed and isinstance(parsed[key], list):
                            return parsed[key]
                    print(f"    WARNING: Got dict with keys {list(parsed.keys())}, no array found")
                    return []
                else:
                    print(f"    WARNING: Unexpected type {type(parsed)}")
                    return []

            except json.JSONDecodeError as e:
                self.stats.api_errors += 1
                print(f"    JSON parse error (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

            except Exception as e:
                self.stats.api_errors += 1
                print(f"    API call failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

        return []

    def _clean_json_response(self, raw: str) -> str:
        """Strip markdown fences and other GPT formatting artifacts."""
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        return raw.strip()

    def _normalize_outputs(self, examples: list[dict]) -> list[dict]:
        """
        Ensure all 'output' fields are YAML strings, not dicts.
        The validator expects a string input.
        """
        normalized = []
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            if "instruction" not in ex or "output" not in ex:
                continue

            output = ex["output"]
            if isinstance(output, dict):
                try:
                    ex["output"] = yaml.dump(
                        output, default_flow_style=False, sort_keys=False
                    )
                except Exception:
                    continue
            elif not isinstance(output, str):
                continue

            normalized.append(ex)

        return normalized

    def _validate_examples(self, examples: list[dict]) -> list[dict]:
        """Validate each example's output against the OneAI schema."""
        try:
            from one_ai_config.validator import ConfigValidator
            validator = ConfigValidator()
        except ImportError:
            print("  WARNING: one_ai_config not installed, skipping validation")
            self.stats.schema_valid = len(examples)
            return examples

        valid = []
        for ex in examples:
            output = ex.get("output", "")
            if not isinstance(output, str):
                self.stats.schema_invalid += 1
                continue

            result = validator.validate(output)
            if result.is_valid:
                valid.append(ex)
                self.stats.schema_valid += 1
            else:
                self.stats.schema_invalid += 1
                self.stats.invalid_examples.append({
                    "instruction": ex.get("instruction", "")[:80],
                    "error": result.error_summary()[:200]
                })

        return valid

    def _print_summary(self):
        """Print generation statistics."""
        print(f"\n{'='*50}")
        print(f"  Generation Summary")
        print(f"{'='*50}")
        print(f"  Total examples: {self.stats.total_generated}")
        print(f"  Schema valid: {self.stats.schema_valid}")
        print(f"  Schema invalid (removed): {self.stats.schema_invalid}")
        pct = (self.stats.schema_valid / 
               (self.stats.schema_valid + self.stats.schema_invalid) * 100
               if (self.stats.schema_valid + self.stats.schema_invalid) > 0 else 0)
        print(f"  Validation rate: {pct:.0f}%")
        print(f"  API calls: {self.stats.api_calls}")
        print(f"  API errors: {self.stats.api_errors}")
        print(f"  Estimated cost: ${self.stats.api_cost_estimate_usd:.4f}")

        if self.stats.invalid_examples:
            print(f"\n  Sample invalid examples ({len(self.stats.invalid_examples)} total):")
            for inv in self.stats.invalid_examples[:5]:
                print(f"    - {inv['instruction']}")
                print(f"      Error: {inv['error'][:120]}")

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

        stats_dict = {k: v for k, v in vars(self.stats).items()}
        stats_path = out / "generation_stats.json"
        stats_path.write_text(json.dumps(stats_dict, indent=2, ensure_ascii=False))

        print(f"\nSaved {len(examples)} examples to {path}")
        print(f"Saved stats to {stats_path}")
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
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip schema validation")
    args = parser.parse_args()

    generator = SyntheticDataGenerator(
        model=args.model,
        validate_schema=not args.no_validate,
    )
    examples = generator.generate_from_seeds(
        seed_file=args.seed_file,
        variations_per_seed=args.variations_per_seed,
        num_negative=args.num_negative,
        num_complex=args.num_complex,
    )
    generator.save(examples, args.output_dir)


if __name__ == "__main__":
    main()
