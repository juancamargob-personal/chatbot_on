"""
Core configuration schema for OpenNebula AI Configuration Assistant.

This module defines the Pydantic models that represent the configuration
output produced by the LLM. Every config must validate against these models
before it can be passed to the code generator or executor.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FailureStrategy(str, Enum):
    ROLLBACK = "rollback"
    ABORT = "abort"
    CONTINUE = "continue"
    RETRY = "retry"


class CheckType(str, Enum):
    CLUSTER_REACHABLE = "cluster_reachable"
    NAMESPACE_AVAILABLE = "namespace_available"
    NAMESPACE_EXISTS = "namespace_exists"
    PODS_RUNNING = "pods_running"
    SERVICE_AVAILABLE = "service_available"
    VM_EXISTS = "vm_exists"
    RESOURCE_AVAILABLE = "resource_available"
    HELM_REPO_AVAILABLE = "helm_repo_available"


# ---------------------------------------------------------------------------
# Action Registry — defines all supported action types
# ---------------------------------------------------------------------------

class ActionCategory(str, Enum):
    """Top-level action categories."""
    ONEKE_CLUSTER = "oneke.cluster"
    ONEKE_APP = "oneke.app"
    ONEKE_SERVICE = "oneke.service"
    ONEKE_NAMESPACE = "oneke.namespace"
    ONEKE_STORAGE = "oneke.storage"
    ONE_VM = "one.vm"
    ONE_TEMPLATE = "one.template"
    ONE_VNET = "one.vnet"
    ONE_IMAGE = "one.image"


# Full action names that are currently supported
SUPPORTED_ACTIONS: set[str] = {
    # OneKE Cluster operations
    "oneke.cluster.get_info",
    "oneke.cluster.list_nodes",
    "oneke.cluster.get_status",
    "oneke.cluster.scale_nodes",

    # OneKE Application operations
    "oneke.app.deploy",
    "oneke.app.uninstall",
    "oneke.app.upgrade",
    "oneke.app.list",
    "oneke.app.wait_ready",
    "oneke.app.get_status",

    # OneKE Service operations
    "oneke.service.get_endpoint",
    "oneke.service.expose",
    "oneke.service.list",

    # OneKE Namespace operations
    "oneke.namespace.create",
    "oneke.namespace.delete",
    "oneke.namespace.list",

    # OneKE Storage operations
    "oneke.storage.create_pvc",
    "oneke.storage.list_pvcs",
    "oneke.storage.delete_pvc",

    # OpenNebula VM operations (Phase 2+)
    "one.vm.create",
    "one.vm.delete",
    "one.vm.poweroff",
    "one.vm.resume",
    "one.vm.list",
    "one.vm.resize",
    "one.vm.snapshot_create",
}


# ---------------------------------------------------------------------------
# Schema Models
# ---------------------------------------------------------------------------

class StepParams(BaseModel):
    """
    Flexible parameter container for action steps.
    Each action type expects different params — the validator in the
    action registry enforces per-action constraints.
    """
    model_config = {"extra": "allow"}


class ValidationCheck(BaseModel):
    """A pre- or post-execution validation check."""
    type: CheckType = Field(description="Type of validation check to perform")
    target: Optional[str] = Field(default=None, description="Target resource (e.g., cluster name)")
    namespace: Optional[str] = Field(default=None, description="Kubernetes namespace")
    service_name: Optional[str] = Field(default=None, description="Service name to check")
    expected_count: Optional[int] = Field(default=None, description="Expected resource count")
    timeout_seconds: int = Field(default=60, description="Timeout for the check")
    label_selector: Optional[str] = Field(default=None, description="K8s label selector")


class ConfigStep(BaseModel):
    """A single executable step in the configuration."""
    id: str = Field(
        description="Unique step identifier (e.g., 'step_01')",
        pattern=r"^step_\d{2,3}$",
    )
    action: str = Field(description="Action to perform (e.g., 'oneke.app.deploy')")
    description: str = Field(
        description="Human-readable description of what this step does",
        min_length=10,
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of step IDs that must complete before this step",
    )
    on_failure: FailureStrategy = Field(
        default=FailureStrategy.ABORT,
        description="What to do if this step fails",
    )
    timeout_seconds: int = Field(
        default=120,
        description="Max seconds to wait for step completion",
        ge=1,
        le=3600,
    )
    retry_count: int = Field(
        default=0,
        description="Number of times to retry on failure",
        ge=0,
        le=5,
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        if v not in SUPPORTED_ACTIONS:
            raise ValueError(
                f"Unsupported action '{v}'. "
                f"Supported actions: {sorted(SUPPORTED_ACTIONS)}"
            )
        return v


class RollbackConfig(BaseModel):
    """Rollback steps to execute if the main pipeline fails."""
    steps: list[ConfigStep] = Field(
        default_factory=list,
        description="Ordered list of rollback steps",
    )


class ValidationConfig(BaseModel):
    """Pre- and post-execution validation checks."""
    pre_checks: list[ValidationCheck] = Field(
        default_factory=list,
        description="Checks to run before execution",
    )
    post_checks: list[ValidationCheck] = Field(
        default_factory=list,
        description="Checks to run after execution",
    )


class ConfigMetadata(BaseModel):
    """Metadata about the generated configuration."""
    description: str = Field(
        description="Human-readable description of what this config does",
        min_length=10,
    )
    target_cluster: Optional[str] = Field(
        default=None,
        description="Target OpenNebula/OneKE cluster name",
    )
    estimated_duration: Optional[str] = Field(
        default=None,
        description="Estimated execution time (e.g., '5-10 minutes')",
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW,
        description="Risk level of the operations",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization (e.g., ['oneke', 'wordpress', 'deployment'])",
    )


class ErrorResponse(BaseModel):
    """Response when the request cannot be fulfilled."""
    is_error: bool = Field(default=True)
    reason: str = Field(description="Why the request cannot be fulfilled")
    suggestion: Optional[str] = Field(
        default=None,
        description="Alternative approach or clarification needed",
    )


class OneAIConfig(BaseModel):
    """
    Root configuration model.

    This is the top-level schema that the LLM must produce.
    It can be either a valid configuration (steps present)
    or an error response (error field present).
    """
    version: str = Field(
        default="1.0",
        description="Schema version",
        pattern=r"^\d+\.\d+$",
    )
    metadata: ConfigMetadata = Field(description="Configuration metadata")
    steps: list[ConfigStep] = Field(
        default_factory=list,
        description="Ordered list of execution steps",
    )
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Pre- and post-execution checks",
    )
    rollback: RollbackConfig = Field(
        default_factory=RollbackConfig,
        description="Rollback plan for failure recovery",
    )
    error: Optional[ErrorResponse] = Field(
        default=None,
        description="Present when the request cannot be fulfilled",
    )

    @model_validator(mode="after")
    def validate_steps_or_error(self) -> "OneAIConfig":
        """Ensure we have either steps or an error, not neither."""
        if not self.steps and not self.error:
            raise ValueError(
                "Configuration must have either 'steps' (valid config) "
                "or 'error' (explanation of why request is impossible)"
            )
        return self

    @model_validator(mode="after")
    def validate_dependency_graph(self) -> "OneAIConfig":
        """Ensure dependency references are valid and acyclic."""
        step_ids = {s.id for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    raise ValueError(
                        f"Step '{step.id}' depends on '{dep}' which does not exist"
                    )
                if dep == step.id:
                    raise ValueError(
                        f"Step '{step.id}' cannot depend on itself"
                    )

        # Check for cycles using topological sort
        if self.steps:
            self._check_cycles(step_ids)
        return self

    def _check_cycles(self, step_ids: set[str]) -> None:
        """Detect cycles in the dependency graph."""
        adj: dict[str, list[str]] = {s.id: s.depends_on for s in self.steps}
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            in_stack.add(node)
            for neighbor in adj.get(node, []):
                if neighbor in in_stack:
                    return True  # Cycle detected
                if neighbor not in visited and dfs(neighbor):
                    return True
            in_stack.discard(node)
            return False

        for node in step_ids:
            if node not in visited:
                if dfs(node):
                    raise ValueError("Dependency graph contains a cycle")


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def parse_config(yaml_str: str) -> OneAIConfig:
    """Parse a YAML string into a validated OneAIConfig."""
    import yaml

    data = yaml.safe_load(yaml_str)
    return OneAIConfig.model_validate(data)


def config_to_yaml(config: OneAIConfig) -> str:
    """Serialize a OneAIConfig back to YAML."""
    import yaml

    # mode="json" converts enums to their string values
    return yaml.dump(
        config.model_dump(exclude_none=True, mode="json"),
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
