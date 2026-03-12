"""
Code generator — transforms validated OneAI configs into runnable Python scripts.

Takes a validated OneAIConfig and produces a complete, executable Python script
that uses kubectl/helm/pyone to carry out the defined steps.

Usage:
    from one_ai_config.codegen import CodeGenerator

    generator = CodeGenerator()
    result = generator.generate(config)
    print(result.script)        # Full Python script
    result.save("deploy.py")    # Save to file
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from one_ai_config.schema.base import OneAIConfig, ConfigStep
from one_ai_config.codegen.actions import ACTION_CODE_REGISTRY


# ---------------------------------------------------------------------------
# Templates directory
# ---------------------------------------------------------------------------

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Actions that require pyone (OpenNebula Python bindings)
PYONE_ACTIONS = {
    "one.vm.create", "one.vm.delete", "one.vm.poweroff",
    "one.vm.resume", "one.vm.list", "one.vm.resize",
    "one.vm.snapshot_create",
    "oneke.cluster.scale_nodes",
}


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class GeneratedScript:
    """The output of code generation."""
    script: str
    config: OneAIConfig
    action_summary: list[str]
    requires_pyone: bool = False
    requires_helm: bool = False
    requires_kubectl: bool = False
    warnings: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        """Save the script to a file and make it executable."""
        p = Path(path)
        p.write_text(self.script)
        p.chmod(0o755)
        return p

    def print_summary(self) -> str:
        """Human-readable summary of what the script will do."""
        lines = [
            f"Generated script: {len(self.config.steps)} steps",
            f"Description: {self.config.metadata.description}",
            f"Risk level: {self.config.metadata.risk_level.value}",
            "",
            "Steps:",
        ]
        for i, summary in enumerate(self.action_summary, 1):
            lines.append(f"  {i}. {summary}")

        if self.requires_pyone or self.requires_helm or self.requires_kubectl:
            lines.append("")
            lines.append("Requirements:")
            if self.requires_kubectl:
                lines.append("  - kubectl (configured with cluster access)")
            if self.requires_helm:
                lines.append("  - helm 3.x")
            if self.requires_pyone:
                lines.append("  - pyone (pip install pyone)")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠️  {w}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CodeGenerator:
    """
    Generates executable Python scripts from validated OneAI configs.

    Usage:
        generator = CodeGenerator()
        result = generator.generate(config)
        result.save("deploy.py")
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def generate(self, config: OneAIConfig) -> GeneratedScript:
        """Generate a complete Python script from a validated config."""
        if config.error:
            return self._generate_error_script(config)

        warnings: list[str] = []
        action_summary: list[str] = []

        # Determine requirements
        requires_pyone = any(s.action in PYONE_ACTIONS for s in config.steps)
        requires_helm = any("app." in s.action for s in config.steps)
        requires_kubectl = any(s.action.startswith("oneke.") for s in config.steps)

        # Generate code for each step
        steps_with_code = []
        for step in config.steps:
            code = self._render_action(step)
            steps_with_code.append({
                "id": step.id,
                "action": step.action,
                "description": step.description,
                "generated_code": code,
            })
            action_summary.append(f"[{step.action}] {step.description}")

        # Generate code for rollback steps
        rollback_with_code = {"steps": []}
        for step in config.rollback.steps:
            code = self._render_action(step)
            rollback_with_code["steps"].append({
                "id": step.id,
                "action": step.action,
                "description": step.description,
                "generated_code": code,
            })

        # Render the full script template
        script_template = self.env.get_template("script.py.j2")
        script = script_template.render(
            metadata={
                "description": config.metadata.description,
                "target_cluster": config.metadata.target_cluster or "default",
                "risk_level": config.metadata.risk_level.value,
                "estimated_duration": config.metadata.estimated_duration or "unknown",
            },
            steps=steps_with_code,
            validation=config.validation,
            rollback=rollback_with_code,
            has_pyone=requires_pyone,
        )

        # Clean up double blank lines
        while "\n\n\n" in script:
            script = script.replace("\n\n\n", "\n\n")

        return GeneratedScript(
            script=script,
            config=config,
            action_summary=action_summary,
            requires_pyone=requires_pyone,
            requires_helm=requires_helm,
            requires_kubectl=requires_kubectl,
            warnings=warnings,
        )

    def _render_action(self, step: ConfigStep) -> str:
        """Render the code snippet for a single action step."""
        code_fn = ACTION_CODE_REGISTRY.get(step.action)
        if code_fn is None:
            return f'log("Action {step.action} is not yet supported for code generation", "WARN")'

        try:
            return code_fn(step.params, step)
        except Exception as e:
            return f'log("Code generation failed for {step.action}: {e}", "ERROR")'

    def _generate_error_script(self, config: OneAIConfig) -> GeneratedScript:
        """Generate a script that just prints the error message."""
        error = config.error
        suggestion_line = ""
        if error.suggestion:
            suggestion_line = f'\nprint("Suggestion: {error.suggestion}")'

        script = (
            '#!/usr/bin/env python3\n'
            '"""OneAI Configuration — Request could not be fulfilled."""\n'
            'import sys\n\n'
            f'print("ERROR: {error.reason}")\n'
            f'{suggestion_line}\n'
            'sys.exit(1)\n'
        )

        return GeneratedScript(
            script=script,
            config=config,
            action_summary=[f"ERROR: {error.reason}"],
            warnings=[error.reason],
        )
