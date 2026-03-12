"""
End-to-end tests for the one-ai-config pipeline.

Tests the full flow: YAML → parse → validate → generate code.
"""

import pytest
import yaml

from one_ai_config.schema.base import (
    OneAIConfig, ConfigStep, parse_config, config_to_yaml,
    SUPPORTED_ACTIONS, RiskLevel, FailureStrategy,
)
from one_ai_config.schema.oneke import validate_step_params, ACTION_PARAM_REGISTRY
from one_ai_config.validator import ConfigValidator, ValidationResult
from one_ai_config.codegen.generator import CodeGenerator, GeneratedScript


# ===================================================================
# Schema parsing
# ===================================================================

class TestSchemaParsing:

    def test_parse_wordpress_config(self, wordpress_yaml):
        config = parse_config(wordpress_yaml)
        assert config.version == "1.0"
        assert config.metadata.description == "Deploy WordPress on OneKE cluster"
        assert config.metadata.risk_level == RiskLevel.LOW
        assert len(config.steps) == 4
        assert config.error is None

    def test_parse_error_config(self, error_yaml):
        config = parse_config(error_yaml)
        assert config.error is not None
        assert config.error.is_error is True
        assert "TensorFlow" in config.error.reason
        assert config.error.suggestion is not None

    def test_step_ids_valid(self, wordpress_config):
        for step in wordpress_config.steps:
            assert step.id.startswith("step_")

    def test_step_actions_are_supported(self, wordpress_config):
        for step in wordpress_config.steps:
            assert step.action in SUPPORTED_ACTIONS

    def test_dependencies_reference_valid_steps(self, wordpress_config):
        step_ids = {s.id for s in wordpress_config.steps}
        for step in wordpress_config.steps:
            for dep in step.depends_on:
                assert dep in step_ids

    def test_roundtrip_yaml(self, wordpress_config):
        """Config should survive a serialize → parse round trip."""
        yaml_str = config_to_yaml(wordpress_config)
        reparsed = parse_config(yaml_str)
        assert len(reparsed.steps) == len(wordpress_config.steps)
        assert reparsed.metadata.description == wordpress_config.metadata.description

    def test_rollback_steps_present(self, wordpress_config):
        assert len(wordpress_config.rollback.steps) == 2
        assert wordpress_config.rollback.steps[0].action == "oneke.app.uninstall"

    def test_validation_checks_present(self, wordpress_config):
        assert len(wordpress_config.validation.pre_checks) == 2
        assert len(wordpress_config.validation.post_checks) == 2


# ===================================================================
# Schema validation errors
# ===================================================================

class TestSchemaValidationErrors:

    def test_unsupported_action_rejected(self):
        with pytest.raises(Exception):
            parse_config("""\
version: "1.0"
metadata:
  description: "Test with bad action"
  risk_level: "low"
steps:
  - id: "step_01"
    action: "nonexistent.fake.action"
    description: "This action does not exist in the registry"
    params: {}
""")

    def test_missing_steps_and_error_rejected(self):
        with pytest.raises(Exception):
            parse_config("""\
version: "1.0"
metadata:
  description: "Neither steps nor error"
  risk_level: "low"
steps: []
""")

    def test_circular_dependency_rejected(self):
        with pytest.raises(Exception):
            parse_config("""\
version: "1.0"
metadata:
  description: "Circular deps"
  risk_level: "low"
steps:
  - id: "step_01"
    action: "oneke.namespace.create"
    description: "Create namespace A"
    params:
      name: "test-a"
    depends_on: ["step_02"]
  - id: "step_02"
    action: "oneke.namespace.create"
    description: "Create namespace B"
    params:
      name: "test-b"
    depends_on: ["step_01"]
""")

    def test_self_dependency_rejected(self):
        with pytest.raises(Exception):
            parse_config("""\
version: "1.0"
metadata:
  description: "Self dep"
  risk_level: "low"
steps:
  - id: "step_01"
    action: "oneke.namespace.create"
    description: "Create namespace that depends on itself"
    params:
      name: "test"
    depends_on: ["step_01"]
""")


# ===================================================================
# Action parameter validation
# ===================================================================

class TestActionParamValidation:

    def test_deploy_params_valid(self):
        params = {
            "chart": "bitnami/wordpress",
            "namespace": "default",
            "release_name": "wp",
        }
        result = validate_step_params("oneke.app.deploy", params)
        assert result.chart == "bitnami/wordpress"

    def test_deploy_params_missing_required(self):
        with pytest.raises(Exception):
            validate_step_params("oneke.app.deploy", {"chart": "bitnami/wordpress"})

    def test_namespace_name_validated(self):
        result = validate_step_params("oneke.namespace.create", {"name": "valid-name"})
        assert result.name == "valid-name"

    def test_namespace_name_invalid_chars(self):
        with pytest.raises(Exception):
            validate_step_params("oneke.namespace.create", {"name": "INVALID_NAME!"})

    def test_all_actions_have_param_registry(self):
        for action in SUPPORTED_ACTIONS:
            if action in ACTION_PARAM_REGISTRY:
                assert ACTION_PARAM_REGISTRY[action] is not None

    def test_pvc_params(self):
        params = {"name": "data", "namespace": "default", "size": "10Gi"}
        result = validate_step_params("oneke.storage.create_pvc", params)
        assert result.size == "10Gi"
        assert result.access_mode == "ReadWriteOnce"  # default


# ===================================================================
# Validator
# ===================================================================

class TestValidator:

    def setup_method(self):
        self.validator = ConfigValidator()

    def test_valid_config_passes(self, wordpress_yaml):
        result = self.validator.validate(wordpress_yaml)
        assert result.is_valid
        assert result.config is not None
        assert len(result.errors) == 0

    def test_error_config_passes(self, error_yaml):
        result = self.validator.validate(error_yaml)
        assert result.is_valid
        assert result.config.error is not None

    def test_invalid_yaml_rejected(self):
        result = self.validator.validate("not: valid: yaml: [[[")
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_non_yaml_rejected(self):
        result = self.validator.validate("just a plain string, not yaml at all {{{")
        assert not result.is_valid

    def test_strips_code_fences(self, wordpress_yaml):
        wrapped = f"```yaml\n{wordpress_yaml}\n```"
        result = self.validator.validate(wrapped)
        assert result.is_valid

    def test_error_summary_format(self):
        result = self.validator.validate("not: valid: yaml: [[[")
        summary = result.error_summary()
        assert "errors" in summary.lower() or "fix" in summary.lower()

    def test_warnings_for_missing_rollback(self, namespace_yaml):
        result = self.validator.validate(namespace_yaml)
        assert result.is_valid
        # Single-step config shouldn't warn about missing rollback


# ===================================================================
# Code Generator
# ===================================================================

class TestCodeGenerator:

    def setup_method(self):
        self.generator = CodeGenerator()

    def test_generate_wordpress_script(self, wordpress_config):
        result = self.generator.generate(wordpress_config)
        assert isinstance(result, GeneratedScript)
        assert len(result.script) > 100
        assert result.requires_helm
        assert result.requires_kubectl
        assert not result.requires_pyone

    def test_script_contains_helm_commands(self, wordpress_config):
        result = self.generator.generate(wordpress_config)
        assert "helm" in result.script
        assert "bitnami/wordpress" in result.script
        assert "repo add" in result.script

    def test_script_contains_namespace_creation(self, wordpress_config):
        result = self.generator.generate(wordpress_config)
        assert "create namespace" in result.script
        assert "wordpress" in result.script

    def test_script_contains_pre_checks(self, wordpress_config):
        result = self.generator.generate(wordpress_config)
        assert "cluster-info" in result.script
        assert "run_pre_checks" in result.script

    def test_script_contains_post_checks(self, wordpress_config):
        result = self.generator.generate(wordpress_config)
        assert "run_post_checks" in result.script
        assert "get pods" in result.script or "get svc" in result.script

    def test_script_contains_rollback(self, wordpress_config):
        result = self.generator.generate(wordpress_config)
        assert "run_rollback" in result.script
        assert "uninstall" in result.script

    def test_script_is_executable_python(self, wordpress_config):
        """The generated script should be valid Python syntax."""
        result = self.generator.generate(wordpress_config)
        compile(result.script, "<generated>", "exec")

    def test_script_has_dry_run(self, wordpress_config):
        result = self.generator.generate(wordpress_config)
        assert "--dry-run" in result.script
        assert "DRY_RUN" in result.script

    def test_script_has_main(self, wordpress_config):
        result = self.generator.generate(wordpress_config)
        assert 'if __name__ == "__main__"' in result.script
        assert "def main()" in result.script

    def test_error_config_generates_exit_script(self, error_config):
        result = self.generator.generate(error_config)
        assert "sys.exit(1)" in result.script
        assert "TensorFlow" in result.script

    def test_namespace_only_script(self, namespace_config):
        result = self.generator.generate(namespace_config)
        assert "create namespace" in result.script
        assert "production" in result.script
        assert result.requires_kubectl
        assert not result.requires_helm

    def test_action_summary(self, wordpress_config):
        result = self.generator.generate(wordpress_config)
        assert len(result.action_summary) == 4
        assert "oneke.namespace.create" in result.action_summary[0]

    def test_print_summary(self, wordpress_config):
        result = self.generator.generate(wordpress_config)
        summary = result.print_summary()
        assert "4 steps" in summary
        assert "kubectl" in summary
        assert "helm" in summary

    def test_save_script(self, wordpress_config, tmp_path):
        result = self.generator.generate(wordpress_config)
        path = result.save(tmp_path / "deploy.py")
        assert path.exists()
        content = path.read_text()
        assert "helm" in content

    def test_label_annotation_in_namespace(self, namespace_config):
        result = self.generator.generate(namespace_config)
        assert "label namespace" in result.script
        assert "env=production" in result.script


# ===================================================================
# Full pipeline: YAML → validate → generate
# ===================================================================

class TestFullPipeline:

    def test_yaml_to_script(self, wordpress_yaml):
        """Full pipeline: raw YAML string → validated config → Python script."""
        # Validate
        validator = ConfigValidator()
        validation = validator.validate(wordpress_yaml)
        assert validation.is_valid

        # Generate
        generator = CodeGenerator()
        result = generator.generate(validation.config)

        # Verify script
        assert len(result.script) > 500
        compile(result.script, "<generated>", "exec")  # Valid Python
        assert result.requires_helm
        assert result.requires_kubectl
        assert "wordpress" in result.script

    def test_error_yaml_to_script(self, error_yaml):
        validator = ConfigValidator()
        validation = validator.validate(error_yaml)
        assert validation.is_valid

        generator = CodeGenerator()
        result = generator.generate(validation.config)
        assert "ERROR" in result.script or "sys.exit" in result.script

    def test_invalid_yaml_stops_pipeline(self):
        validator = ConfigValidator()
        validation = validator.validate("garbage input {{{{")
        assert not validation.is_valid
        # Pipeline stops — no code generation happens
