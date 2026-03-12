"""
Tests for the one-ai-finetune pipeline.

Tests the components that can run without GPU or API keys:
- Dataset formatting (chat templates)
- Data quality cleaning (dedup, filtering)
- Schema evaluation (requires one-ai-config)
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from one_ai_finetune.data.format_dataset import (
    DatasetFormatter,
    format_mistral,
    format_llama3,
    format_chatml,
    SYSTEM_PROMPT,
)
from one_ai_finetune.data_quality.dedup import DataQualityCleaner, QualityReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_examples():
    """Minimal set of training examples."""
    return [
        {
            "instruction": "Deploy WordPress on my OneKE cluster",
            "output": 'version: "1.0"\nmetadata:\n  description: "Deploy WordPress"\n  risk_level: "low"\nsteps:\n  - id: "step_01"\n    action: "oneke.app.deploy"\n    description: "Deploy WordPress Helm chart"\n    params:\n      chart: "bitnami/wordpress"\n      namespace: "wordpress"\n      release_name: "wordpress"',
        },
        {
            "instruction": "Remove Redis from staging",
            "output": 'version: "1.0"\nmetadata:\n  description: "Remove Redis"\n  risk_level: "medium"\nsteps:\n  - id: "step_01"\n    action: "oneke.app.uninstall"\n    description: "Uninstall Redis"\n    params:\n      release_name: "redis"\n      namespace: "staging"',
        },
        {
            "instruction": "Scale workers to 5",
            "output": 'version: "1.0"\nmetadata:\n  description: "Scale workers"\n  risk_level: "medium"\nsteps:\n  - id: "step_01"\n    action: "oneke.cluster.scale_nodes"\n    description: "Scale workers to 5"\n    params:\n      worker_count: 5',
        },
    ]


@pytest.fixture
def sample_examples_file(sample_examples, tmp_path):
    """Save examples to a JSON file and return the path."""
    path = tmp_path / "examples.json"
    path.write_text(json.dumps(sample_examples))
    return path


# ===================================================================
# Chat template formatting
# ===================================================================

class TestFormatFunctions:

    def test_mistral_format_has_inst_tags(self):
        text = format_mistral("Deploy app", "version: 1.0")
        assert "[INST]" in text
        assert "[/INST]" in text
        assert "</s>" in text
        assert SYSTEM_PROMPT in text

    def test_mistral_format_includes_output(self):
        text = format_mistral("Deploy app", "version: 1.0")
        assert "version: 1.0" in text
        assert "```yaml" in text

    def test_mistral_format_inference_mode(self):
        """Without output, should end after [/INST] for generation."""
        text = format_mistral("Deploy app")
        assert "[INST]" in text
        assert "[/INST]" in text
        assert "</s>" not in text  # No closing tag in inference mode

    def test_llama3_format_has_headers(self):
        text = format_llama3("Deploy app", "version: 1.0")
        assert "<|begin_of_text|>" in text
        assert "<|start_header_id|>system<|end_header_id|>" in text
        assert "<|start_header_id|>user<|end_header_id|>" in text
        assert "<|start_header_id|>assistant<|end_header_id|>" in text

    def test_llama3_includes_output(self):
        text = format_llama3("Deploy app", "version: 1.0")
        assert "version: 1.0" in text
        assert "<|eot_id|>" in text

    def test_chatml_format_has_markers(self):
        text = format_chatml("Deploy app", "version: 1.0")
        assert "<|im_start|>system" in text
        assert "<|im_start|>user" in text
        assert "<|im_start|>assistant" in text
        assert "<|im_end|>" in text

    def test_all_formats_include_system_prompt(self):
        for fn in [format_mistral, format_llama3, format_chatml]:
            text = fn("Deploy app", "version: 1.0")
            assert "OpenNebula" in text
            assert "YAML" in text


class TestDatasetFormatter:

    def test_format_and_save(self, sample_examples_file, tmp_path):
        formatter = DatasetFormatter(format="mistral", eval_split=0.0)
        stats = formatter.format_and_save(
            input_path=sample_examples_file,
            output_dir=tmp_path / "processed",
        )

        assert stats["total_examples"] == 3
        assert stats["train_examples"] == 3
        assert stats["eval_examples"] == 0
        assert (tmp_path / "processed" / "train.jsonl").exists()
        assert (tmp_path / "processed" / "eval.jsonl").exists()

    def test_eval_split(self, tmp_path):
        """With enough examples, eval split should produce both sets."""
        examples = [
            {"instruction": f"Task {i}", "output": f'version: "1.0"\nmetadata:\n  description: "Task {i}"\n  risk_level: "low"\nsteps: []'}
            for i in range(20)
        ]
        path = tmp_path / "examples.json"
        path.write_text(json.dumps(examples))

        formatter = DatasetFormatter(format="mistral", eval_split=0.2)
        stats = formatter.format_and_save(
            input_path=path,
            output_dir=tmp_path / "processed",
        )

        assert stats["eval_examples"] == 4  # 20 * 0.2
        assert stats["train_examples"] == 16

    def test_jsonl_format(self, sample_examples_file, tmp_path):
        """Output should be valid JSONL."""
        formatter = DatasetFormatter(format="mistral", eval_split=0.0)
        formatter.format_and_save(
            input_path=sample_examples_file,
            output_dir=tmp_path / "processed",
        )

        train_path = tmp_path / "processed" / "train.jsonl"
        with open(train_path) as f:
            for line in f:
                data = json.loads(line)
                assert "text" in data
                assert "[INST]" in data["text"]

    def test_seed_examples_added(self, sample_examples_file, tmp_path):
        """Seed examples should be added to training set."""
        # Create minimal synthetic data
        synthetic = [
            {"instruction": "Synthetic task", "output": 'version: "1.0"\nmetadata:\n  description: "x"\n  risk_level: "low"\nsteps: []'}
        ]
        syn_path = tmp_path / "synthetic.json"
        syn_path.write_text(json.dumps(synthetic))

        formatter = DatasetFormatter(format="mistral", eval_split=0.0)
        stats = formatter.format_and_save(
            input_path=syn_path,
            output_dir=tmp_path / "processed",
            seed_path=sample_examples_file,
        )

        # 1 synthetic + 3 seeds = 4 total in train
        assert stats["train_examples"] == 4

    def test_skips_empty_examples(self, tmp_path):
        examples = [
            {"instruction": "Valid task", "output": 'version: "1.0"\nsteps: []'},
            {"instruction": "", "output": "something"},
            {"instruction": "Another", "output": ""},
        ]
        path = tmp_path / "examples.json"
        path.write_text(json.dumps(examples))

        formatter = DatasetFormatter(format="mistral", eval_split=0.0)
        stats = formatter.format_and_save(
            input_path=path,
            output_dir=tmp_path / "processed",
        )

        assert stats["total_examples"] == 1
        assert stats["skipped"] == 2

    def test_unknown_format_rejected(self):
        with pytest.raises(ValueError, match="Unknown format"):
            DatasetFormatter(format="nonexistent")

    def test_format_single(self):
        formatter = DatasetFormatter(format="mistral")
        text = formatter.format_single("Deploy app", "version: 1.0")
        assert "[INST]" in text
        assert "version: 1.0" in text


# ===================================================================
# Data quality
# ===================================================================

class TestDataQualityCleaner:

    def test_removes_exact_duplicates(self):
        data = [
            {"instruction": "Deploy WordPress", "output": "version: '1.0'\nsteps: []"},
            {"instruction": "Deploy WordPress", "output": "version: '1.0'\nsteps: []"},
            {"instruction": "Deploy Redis", "output": "version: '1.0'\nsteps: []"},
        ]
        cleaner = DataQualityCleaner()
        clean = cleaner.clean(data)
        assert len(clean) == 2
        assert cleaner.report.exact_duplicates == 1

    def test_removes_near_duplicates(self):
        data = [
            {"instruction": "Deploy WordPress on my cluster", "output": "version: '1.0'\nsteps: []"},
            {"instruction": "deploy wordpress on my cluster!", "output": "version: '1.0'\nsteps: []"},
            {"instruction": "Scale workers to 5", "output": "version: '1.0'\nsteps: []"},
        ]
        cleaner = DataQualityCleaner()
        clean = cleaner.clean(data)
        assert len(clean) == 2
        assert cleaner.report.near_duplicates == 1

    def test_removes_short_instructions(self):
        data = [
            {"instruction": "Hi", "output": "version: '1.0'\nsteps: []"},
            {"instruction": "Deploy WordPress application on OneKE", "output": "version: '1.0'\nsteps: []"},
        ]
        cleaner = DataQualityCleaner(min_instruction_words=3)
        clean = cleaner.clean(data)
        assert len(clean) == 1
        assert cleaner.report.too_short >= 1

    def test_removes_empty_fields(self):
        data = [
            {"instruction": "", "output": "version: '1.0'"},
            {"instruction": "Deploy app", "output": ""},
            {"instruction": "Deploy app", "output": "version: '1.0'\nsteps: []"},
        ]
        cleaner = DataQualityCleaner()
        clean = cleaner.clean(data)
        assert len(clean) == 1
        assert cleaner.report.empty_fields == 2

    def test_removes_invalid_yaml(self):
        data = [
            {"instruction": "Deploy app", "output": "version: '1.0'\nsteps: []"},
            {"instruction": "Bad output", "output": "not: valid: yaml: [[["},
        ]
        cleaner = DataQualityCleaner()
        clean = cleaner.clean(data)
        assert len(clean) == 1
        assert cleaner.report.invalid_yaml == 1

    def test_report_summary(self):
        data = [
            {"instruction": "Deploy app", "output": "version: '1.0'\nsteps: []"},
            {"instruction": "Deploy app", "output": "version: '1.0'\nsteps: []"},
        ]
        cleaner = DataQualityCleaner()
        cleaner.clean(data)
        summary = cleaner.report.summary()
        assert "Data Quality Report" in summary
        assert "Input:" in summary

    def test_clean_preserves_good_data(self, sample_examples):
        cleaner = DataQualityCleaner()
        clean = cleaner.clean(sample_examples)
        assert len(clean) == len(sample_examples)


# ===================================================================
# Schema evaluation
# ===================================================================

class TestSchemaEvaluator:
    """Tests that require one-ai-config to be installed."""

    @pytest.fixture(autouse=True)
    def skip_if_no_config(self):
        try:
            from one_ai_config.validator import ConfigValidator
        except ImportError:
            pytest.skip("one-ai-config not installed")

    def test_valid_output_passes(self):
        from one_ai_finetune.eval.schema_eval import SchemaEvaluator

        evaluator = SchemaEvaluator()
        examples = [{
            "instruction": "Deploy WordPress",
            "output": (
                'version: "1.0"\n'
                'metadata:\n'
                '  description: "Deploy WordPress on OneKE"\n'
                '  risk_level: "low"\n'
                'steps:\n'
                '  - id: "step_01"\n'
                '    action: "oneke.app.deploy"\n'
                '    description: "Deploy WordPress Helm chart on the cluster"\n'
                '    params:\n'
                '      chart: "bitnami/wordpress"\n'
                '      namespace: "wordpress"\n'
                '      release_name: "wordpress"\n'
                'validation:\n'
                '  pre_checks: []\n'
                '  post_checks: []\n'
                'rollback:\n'
                '  steps: []'
            ),
        }]
        report = evaluator.evaluate(examples)
        assert report.schema_rate == 1.0

    def test_invalid_yaml_detected(self):
        from one_ai_finetune.eval.schema_eval import SchemaEvaluator

        evaluator = SchemaEvaluator()
        examples = [{
            "instruction": "Deploy something",
            "output": "this is not yaml: [[[",
        }]
        report = evaluator.evaluate(examples)
        assert report.yaml_rate == 0.0
        assert report.schema_rate == 0.0

    def test_report_summary(self):
        from one_ai_finetune.eval.schema_eval import SchemaEvaluator

        evaluator = SchemaEvaluator()
        report = evaluator.evaluate([
            {"instruction": "Test", "output": "not: valid"},
        ])
        summary = report.summary()
        assert "Schema Compliance" in summary


# ===================================================================
# Training config (no GPU needed — just validate the config dataclass)
# ===================================================================

class TestQLoRAConfig:

    @pytest.fixture(autouse=True)
    def skip_if_no_gpu_deps(self):
        try:
            import torch
            import datasets
        except ImportError:
            pytest.skip("GPU training dependencies not installed (install with pip install -e '.[train]')")

    def test_default_config(self):
        from one_ai_finetune.training.qlora_train import QLoRAConfig
        config = QLoRAConfig()
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.epochs == 3
        assert config.load_in_4bit is True

    def test_custom_config(self):
        from one_ai_finetune.training.qlora_train import QLoRAConfig
        config = QLoRAConfig(lora_r=32, epochs=5, learning_rate=1e-4)
        assert config.lora_r == 32
        assert config.epochs == 5
        assert config.learning_rate == 1e-4

    def test_prompt_formatting(self):
        from one_ai_finetune.training.qlora_train import format_prompt
        text = format_prompt("Deploy app", "version: 1.0")
        assert "[INST]" in text
        assert "version: 1.0" in text
        assert "</s>" in text

    def test_prompt_inference_mode(self):
        from one_ai_finetune.training.qlora_train import format_prompt
        text = format_prompt("Deploy app")
        assert "[INST]" in text
        assert "[/INST]" in text
        assert "</s>" not in text
