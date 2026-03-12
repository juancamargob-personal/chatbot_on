"""
Configuration validator for LLM output.

This module validates the raw YAML/JSON output from the LLM:
1. Parses the text into a dict
2. Validates against the Pydantic schema
3. Validates action-specific parameters
4. Returns either a validated config or a list of errors for LLM retry
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml
from pydantic import ValidationError

from one_ai_config.schema.base import OneAIConfig
from one_ai_config.schema.oneke import validate_step_params


@dataclass
class ValidationResult:
    """Result of config validation."""
    is_valid: bool
    config: OneAIConfig | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error_summary(self) -> str:
        """Format errors for feeding back to the LLM."""
        if not self.errors:
            return "No errors."
        lines = ["The configuration has the following errors:"]
        for i, err in enumerate(self.errors, 1):
            lines.append(f"  {i}. {err}")
        lines.append("")
        lines.append("Please fix these errors and regenerate the configuration.")
        return "\n".join(lines)


class ConfigValidator:
    """
    Validates LLM-generated configuration output.

    Usage:
        validator = ConfigValidator()
        result = validator.validate(raw_yaml_string)
        if result.is_valid:
            config = result.config
        else:
            # Feed result.error_summary() back to LLM for retry
            print(result.error_summary())
    """

    def validate(self, raw_output: str) -> ValidationResult:
        """
        Full validation pipeline.

        Args:
            raw_output: Raw string output from the LLM (YAML expected)

        Returns:
            ValidationResult with either a valid config or error details
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Step 1: Strip markdown code fences if present
        cleaned = self._strip_code_fences(raw_output)

        # Step 2: Parse YAML
        try:
            data = yaml.safe_load(cleaned)
        except yaml.YAMLError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid YAML: {e}"],
            )

        if not isinstance(data, dict):
            return ValidationResult(
                is_valid=False,
                errors=["Output must be a YAML mapping (dict), got: " + type(data).__name__],
            )

        # Step 3: Validate against Pydantic schema
        try:
            config = OneAIConfig.model_validate(data)
        except ValidationError as e:
            for err in e.errors():
                loc = " → ".join(str(x) for x in err["loc"])
                errors.append(f"[{loc}] {err['msg']}")
            return ValidationResult(is_valid=False, errors=errors)

        # Step 4: If it's an error response, it's valid (LLM correctly refused)
        if config.error is not None:
            return ValidationResult(is_valid=True, config=config)

        # Step 5: Validate action-specific parameters
        for step in config.steps:
            try:
                validate_step_params(step.action, step.params)
            except Exception as e:
                errors.append(f"Step '{step.id}' ({step.action}): {e}")

        # Step 6: Check for warnings (non-blocking)
        warnings.extend(self._check_warnings(config))

        if errors:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        return ValidationResult(is_valid=True, config=config, warnings=warnings)

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences that LLMs sometimes wrap output in."""
        text = text.strip()
        if text.startswith("```yaml"):
            text = text[7:]
        elif text.startswith("```yml"):
            text = text[6:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _check_warnings(self, config: OneAIConfig) -> list[str]:
        """Generate non-blocking warnings about the config."""
        warnings = []

        # Warn if no rollback defined for non-trivial configs
        if len(config.steps) > 1 and not config.rollback.steps:
            warnings.append(
                "No rollback steps defined. Consider adding rollback for multi-step configs."
            )

        # Warn if no pre-checks
        if not config.validation.pre_checks:
            warnings.append(
                "No pre-checks defined. Consider adding cluster reachability checks."
            )

        # Warn about high-risk operations without explicit approval
        destructive_actions = {"oneke.namespace.delete", "one.vm.delete", "oneke.app.uninstall"}
        for step in config.steps:
            if step.action in destructive_actions:
                if config.metadata.risk_level not in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                    warnings.append(
                        f"Step '{step.id}' uses destructive action '{step.action}' "
                        f"but risk_level is '{config.metadata.risk_level}'. "
                        f"Consider setting risk_level to 'high' or 'critical'."
                    )

        return warnings


# Import needed for warning checks
from one_ai_config.schema.base import RiskLevel
