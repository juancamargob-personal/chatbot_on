"""
OneKE-specific action parameter schemas.

Each action type has a Pydantic model that validates the params dict
for that action. The ActionRegistry maps action names to their param models.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# OneKE Cluster Actions
# ---------------------------------------------------------------------------

class ClusterGetInfoParams(BaseModel):
    """Params for oneke.cluster.get_info"""
    cluster_name: Optional[str] = Field(
        default=None, description="Cluster name (uses default if omitted)"
    )


class ClusterListNodesParams(BaseModel):
    """Params for oneke.cluster.list_nodes"""
    cluster_name: Optional[str] = None
    role: Optional[str] = Field(
        default=None, description="Filter by role: 'master', 'worker', or None for all"
    )


class ClusterScaleNodesParams(BaseModel):
    """Params for oneke.cluster.scale_nodes"""
    cluster_name: Optional[str] = None
    worker_count: int = Field(ge=1, le=100, description="Desired number of worker nodes")
    node_template: Optional[str] = Field(
        default=None, description="VM template to use for new nodes"
    )


# ---------------------------------------------------------------------------
# OneKE App Actions
# ---------------------------------------------------------------------------

class AppDeployParams(BaseModel):
    """Params for oneke.app.deploy"""
    chart: str = Field(description="Helm chart reference (e.g., 'bitnami/wordpress')")
    namespace: str = Field(description="Target Kubernetes namespace")
    release_name: str = Field(description="Helm release name")
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="Helm values to override (nested dict)",
    )
    version: Optional[str] = Field(
        default=None, description="Chart version to deploy"
    )
    repo_url: Optional[str] = Field(
        default=None, description="Helm repo URL if not already added"
    )
    create_namespace: bool = Field(
        default=True, description="Create namespace if it doesn't exist"
    )
    wait: bool = Field(
        default=True, description="Wait for resources to be ready"
    )


class AppUninstallParams(BaseModel):
    """Params for oneke.app.uninstall"""
    release_name: str = Field(description="Helm release name to uninstall")
    namespace: str = Field(description="Kubernetes namespace")
    keep_history: bool = Field(default=False)


class AppUpgradeParams(BaseModel):
    """Params for oneke.app.upgrade"""
    chart: str = Field(description="Helm chart reference")
    release_name: str = Field(description="Existing Helm release name")
    namespace: str = Field(description="Kubernetes namespace")
    values: dict[str, Any] = Field(default_factory=dict)
    version: Optional[str] = None
    reuse_values: bool = Field(
        default=True, description="Reuse existing values and merge with new ones"
    )


class AppListParams(BaseModel):
    """Params for oneke.app.list"""
    namespace: Optional[str] = Field(
        default=None, description="Filter by namespace (None for all)"
    )


class AppWaitReadyParams(BaseModel):
    """Params for oneke.app.wait_ready"""
    namespace: str = Field(description="Kubernetes namespace")
    label_selector: str = Field(description="Label selector for pods to watch")
    timeout_seconds: int = Field(default=300, ge=30, le=1800)
    expected_replicas: Optional[int] = Field(
        default=None, description="Expected number of ready pods"
    )


class AppGetStatusParams(BaseModel):
    """Params for oneke.app.get_status"""
    release_name: str
    namespace: str


# ---------------------------------------------------------------------------
# OneKE Service Actions
# ---------------------------------------------------------------------------

class ServiceGetEndpointParams(BaseModel):
    """Params for oneke.service.get_endpoint"""
    namespace: str
    service_name: str


class ServiceExposeParams(BaseModel):
    """Params for oneke.service.expose"""
    namespace: str
    deployment_name: str
    port: int = Field(ge=1, le=65535)
    target_port: Optional[int] = Field(default=None, ge=1, le=65535)
    service_type: str = Field(
        default="ClusterIP",
        description="Service type: ClusterIP, NodePort, LoadBalancer",
    )
    service_name: Optional[str] = None


class ServiceListParams(BaseModel):
    """Params for oneke.service.list"""
    namespace: Optional[str] = None


# ---------------------------------------------------------------------------
# OneKE Namespace Actions
# ---------------------------------------------------------------------------

class NamespaceCreateParams(BaseModel):
    """Params for oneke.namespace.create"""
    name: str = Field(description="Namespace name", pattern=r"^[a-z0-9][a-z0-9\-]{0,62}$")
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)


class NamespaceDeleteParams(BaseModel):
    """Params for oneke.namespace.delete"""
    name: str


class NamespaceListParams(BaseModel):
    """Params for oneke.namespace.list"""
    label_selector: Optional[str] = None


# ---------------------------------------------------------------------------
# OneKE Storage Actions
# ---------------------------------------------------------------------------

class StorageCreatePVCParams(BaseModel):
    """Params for oneke.storage.create_pvc"""
    name: str = Field(description="PVC name")
    namespace: str
    size: str = Field(description="Storage size (e.g., '10Gi', '500Mi')")
    storage_class: Optional[str] = Field(
        default=None, description="Storage class name"
    )
    access_mode: str = Field(
        default="ReadWriteOnce",
        description="Access mode: ReadWriteOnce, ReadOnlyMany, ReadWriteMany",
    )


class StorageListPVCsParams(BaseModel):
    """Params for oneke.storage.list_pvcs"""
    namespace: Optional[str] = None


class StorageDeletePVCParams(BaseModel):
    """Params for oneke.storage.delete_pvc"""
    name: str
    namespace: str


# ---------------------------------------------------------------------------
# OpenNebula VM Actions
# ---------------------------------------------------------------------------

class VmCreateParams(BaseModel):
    """Params for one.vm.create"""
    template_id: int = Field(description="OpenNebula VM template ID")
    name: str = Field(description="Name for the new VM instance")
    cpu: Optional[float] = Field(default=None, ge=0.1, description="Number of CPUs")
    memory_mb: Optional[int] = Field(default=None, ge=128, description="Memory in MB")
    hold: bool = Field(default=False, description="Create VM on hold instead of pending")
    extra_template: Optional[str] = Field(
        default=None,
        description="Extra template attributes to merge (attribute=value format)",
    )


class VmDeleteParams(BaseModel):
    """Params for one.vm.delete"""
    vm_id: int = Field(description="OpenNebula VM ID to delete")


class VmPoweroffParams(BaseModel):
    """Params for one.vm.poweroff"""
    vm_id: int = Field(description="OpenNebula VM ID to power off")
    hard: bool = Field(default=False, description="Hard poweroff (skip ACPI signal)")


class VmResumeParams(BaseModel):
    """Params for one.vm.resume"""
    vm_id: int = Field(description="OpenNebula VM ID to resume")


class VmListParams(BaseModel):
    """Params for one.vm.list"""
    filter_flag: Optional[int] = Field(
        default=None,
        description="Filter: -2=all, -1=mine, >=0=specific user ID",
    )


class VmResizeParams(BaseModel):
    """Params for one.vm.resize"""
    vm_id: int = Field(description="OpenNebula VM ID to resize")
    cpu: Optional[float] = Field(default=None, ge=0.1)
    memory_mb: Optional[int] = Field(default=None, ge=128)
    enforce: bool = Field(
        default=False,
        description="If true, enforce capacity checks",
    )


class VmSnapshotCreateParams(BaseModel):
    """Params for one.vm.snapshot_create"""
    vm_id: int = Field(description="OpenNebula VM ID")
    snapshot_name: str = Field(default="snapshot", description="Name for the snapshot")


# ---------------------------------------------------------------------------
# Action Registry
# ---------------------------------------------------------------------------

ACTION_PARAM_REGISTRY: dict[str, type[BaseModel]] = {
    # Cluster
    "oneke.cluster.get_info": ClusterGetInfoParams,
    "oneke.cluster.list_nodes": ClusterListNodesParams,
    "oneke.cluster.get_status": ClusterGetInfoParams,
    "oneke.cluster.scale_nodes": ClusterScaleNodesParams,

    # App
    "oneke.app.deploy": AppDeployParams,
    "oneke.app.uninstall": AppUninstallParams,
    "oneke.app.upgrade": AppUpgradeParams,
    "oneke.app.list": AppListParams,
    "oneke.app.wait_ready": AppWaitReadyParams,
    "oneke.app.get_status": AppGetStatusParams,

    # Service
    "oneke.service.get_endpoint": ServiceGetEndpointParams,
    "oneke.service.expose": ServiceExposeParams,
    "oneke.service.list": ServiceListParams,

    # Namespace
    "oneke.namespace.create": NamespaceCreateParams,
    "oneke.namespace.delete": NamespaceDeleteParams,
    "oneke.namespace.list": NamespaceListParams,

    # Storage
    "oneke.storage.create_pvc": StorageCreatePVCParams,
    "oneke.storage.list_pvcs": StorageListPVCsParams,
    "oneke.storage.delete_pvc": StorageDeletePVCParams,

    # OpenNebula VMs
    "one.vm.create": VmCreateParams,
    "one.vm.delete": VmDeleteParams,
    "one.vm.poweroff": VmPoweroffParams,
    "one.vm.resume": VmResumeParams,
    "one.vm.list": VmListParams,
    "one.vm.resize": VmResizeParams,
    "one.vm.snapshot_create": VmSnapshotCreateParams,
}


def validate_step_params(action: str, params: dict) -> BaseModel:
    """
    Validate step params against the action-specific schema.

    Args:
        action: The action name (e.g., 'oneke.app.deploy')
        params: The params dict from the config step

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If action is unknown or params are invalid
    """
    model_cls = ACTION_PARAM_REGISTRY.get(action)
    if model_cls is None:
        raise ValueError(f"No parameter schema registered for action '{action}'")
    return model_cls.model_validate(params)
