"""
Action code snippets — maps each action type to a Python code string.

Each function takes (params: dict, step: ConfigStep) and returns
a string of Python code to be inserted into the generated script.
"""

from __future__ import annotations

import json
from typing import Any

from one_ai_config.schema.base import ConfigStep


# ---------------------------------------------------------------------------
# OneKE Namespace
# ---------------------------------------------------------------------------

def oneke_namespace_create(params: dict, step: ConfigStep) -> str:
    lines = [f'kubectl("create namespace {params["name"]}")']
    for key, val in params.get("labels", {}).items():
        lines.append(f'kubectl("label namespace {params["name"]} {key}={val}")')
    for key, val in params.get("annotations", {}).items():
        lines.append(f'kubectl("annotate namespace {params["name"]} {key}={val}")')
    return "\n".join(lines)


def oneke_namespace_delete(params: dict, step: ConfigStep) -> str:
    return f'kubectl("delete namespace {params["name"]} --ignore-not-found")'


def oneke_namespace_list(params: dict, step: ConfigStep) -> str:
    cmd = "get namespaces"
    if params.get("label_selector"):
        cmd += f" -l {params['label_selector']}"
    return f'kubectl("{cmd}")'


# ---------------------------------------------------------------------------
# OneKE App (Helm)
# ---------------------------------------------------------------------------

def oneke_app_deploy(params: dict, step: ConfigStep) -> str:
    lines = []
    chart = params["chart"]
    repo_name = chart.split("/")[0] if "/" in chart else None

    if params.get("repo_url") and repo_name:
        lines.append(f'# Add Helm repository')
        lines.append(f'helm("repo add {repo_name} {params["repo_url"]}")')
        lines.append(f'helm("repo update")')
        lines.append("")

    lines.append("# Deploy the Helm chart")
    parts = [f'install {params["release_name"]} {chart}']
    parts.append(f'--namespace {params["namespace"]}')
    if params.get("create_namespace", True):
        parts.append("--create-namespace")
    if params.get("version"):
        parts.append(f'--version {params["version"]}')
    if params.get("wait", True):
        parts.append("--wait")
        parts.append(f"--timeout {step.timeout_seconds}s")

    if params.get("values"):
        lines.append("import tempfile, yaml, os")
        lines.append(f"values = {json.dumps(params['values'], indent=4)}")
        lines.append('with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:')
        lines.append("    yaml.dump(values, f)")
        lines.append("    values_file = f.name")
        cmd = " ".join(parts)
        lines.append(f'helm("{cmd}" + f" -f {{values_file}}")')
        lines.append("os.unlink(values_file)")
    else:
        cmd = " ".join(parts)
        lines.append(f'helm("{cmd}")')

    return "\n".join(lines)


def oneke_app_uninstall(params: dict, step: ConfigStep) -> str:
    cmd = f'uninstall {params["release_name"]} --namespace {params["namespace"]}'
    if params.get("keep_history"):
        cmd += " --keep-history"
    return f'helm("{cmd}")'


def oneke_app_upgrade(params: dict, step: ConfigStep) -> str:
    lines = []
    parts = [f'upgrade {params["release_name"]} {params["chart"]}']
    parts.append(f'--namespace {params["namespace"]}')
    if params.get("version"):
        parts.append(f'--version {params["version"]}')
    if params.get("reuse_values", True):
        parts.append("--reuse-values")
    parts.append("--wait")

    if params.get("values"):
        lines.append("import tempfile, yaml, os")
        lines.append(f"values = {json.dumps(params['values'], indent=4)}")
        lines.append('with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:')
        lines.append("    yaml.dump(values, f)")
        lines.append("    values_file = f.name")
        cmd = " ".join(parts)
        lines.append(f'helm("{cmd}" + f" -f {{values_file}}")')
        lines.append("os.unlink(values_file)")
    else:
        cmd = " ".join(parts)
        lines.append(f'helm("{cmd}")')

    return "\n".join(lines)


def oneke_app_list(params: dict, step: ConfigStep) -> str:
    if params.get("namespace"):
        return f'helm("list --namespace {params["namespace"]}")'
    return 'helm("list --all-namespaces")'


def oneke_app_wait_ready(params: dict, step: ConfigStep) -> str:
    timeout = params.get("timeout_seconds", 300)
    return (
        f'log("Waiting for pods to be ready (timeout: {timeout}s)...", "INFO")\n'
        f'kubectl("wait --for=condition=ready pod -l {params["label_selector"]} '
        f'-n {params["namespace"]} --timeout={timeout}s")'
    )


def oneke_app_get_status(params: dict, step: ConfigStep) -> str:
    return f'helm("status {params["release_name"]} --namespace {params["namespace"]}")'


# ---------------------------------------------------------------------------
# OneKE Service
# ---------------------------------------------------------------------------

def oneke_service_get_endpoint(params: dict, step: ConfigStep) -> str:
    return (
        f'result = kubectl("get svc {params["service_name"]} -n {params["namespace"]} '
        f"""-o jsonpath='{{.status.loadBalancer.ingress[0].ip}}'", capture=True)\n"""
        f'if result.stdout.strip():\n'
        f'    log(f"Endpoint: {{result.stdout.strip()}}", "OK")\n'
        f'else:\n'
        f'    log("No external IP assigned yet. Service may still be provisioning.", "WARN")'
    )


def oneke_service_expose(params: dict, step: ConfigStep) -> str:
    parts = [f'expose deployment {params["deployment_name"]}']
    parts.append(f'--port={params["port"]}')
    if params.get("target_port"):
        parts.append(f'--target-port={params["target_port"]}')
    parts.append(f'--type={params.get("service_type", "ClusterIP")}')
    parts.append(f'-n {params["namespace"]}')
    if params.get("service_name"):
        parts.append(f'--name={params["service_name"]}')
    cmd = " ".join(parts)
    return f'kubectl("{cmd}")'


def oneke_service_list(params: dict, step: ConfigStep) -> str:
    if params.get("namespace"):
        return f'kubectl("get svc -n {params["namespace"]}")'
    return 'kubectl("get svc --all-namespaces")'


# ---------------------------------------------------------------------------
# OneKE Storage
# ---------------------------------------------------------------------------

def oneke_storage_create_pvc(params: dict, step: ConfigStep) -> str:
    pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": params["name"], "namespace": params["namespace"]},
        "spec": {
            "accessModes": [params.get("access_mode", "ReadWriteOnce")],
            "resources": {"requests": {"storage": params["size"]}},
        },
    }
    if params.get("storage_class"):
        pvc["spec"]["storageClassName"] = params["storage_class"]

    return (
        f"import tempfile, json, os\n"
        f"pvc_manifest = {json.dumps(pvc, indent=4)}\n"
        f'with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:\n'
        f"    json.dump(pvc_manifest, f)\n"
        f"    manifest_file = f.name\n"
        f'kubectl(f"apply -f {{manifest_file}}")\n'
        f"os.unlink(manifest_file)"
    )


def oneke_storage_list_pvcs(params: dict, step: ConfigStep) -> str:
    if params.get("namespace"):
        return f'kubectl("get pvc -n {params["namespace"]}")'
    return 'kubectl("get pvc --all-namespaces")'


def oneke_storage_delete_pvc(params: dict, step: ConfigStep) -> str:
    return f'kubectl("delete pvc {params["name"]} -n {params["namespace"]} --ignore-not-found")'


# ---------------------------------------------------------------------------
# OneKE Cluster
# ---------------------------------------------------------------------------

def oneke_cluster_get_info(params: dict, step: ConfigStep) -> str:
    return 'kubectl("cluster-info")'


def oneke_cluster_get_status(params: dict, step: ConfigStep) -> str:
    return (
        'kubectl("get nodes -o wide")\n'
        'kubectl("get pods --all-namespaces --field-selector=status.phase!=Running", check=False)'
    )


def oneke_cluster_list_nodes(params: dict, step: ConfigStep) -> str:
    cmd = "get nodes -o wide"
    if params.get("role"):
        cmd += f" -l node-role.kubernetes.io/{params['role']}"
    return f'kubectl("{cmd}")'


def oneke_cluster_scale_nodes(params: dict, step: ConfigStep) -> str:
    cluster = params.get("cluster_name", "default")
    count = params["worker_count"]
    return (
        f'log("Scaling workers to {count} nodes via OneFlow...", "INFO")\n'
        f'run_cmd(["oneflow", "scale", "{cluster}", "worker", "{count}"])'
    )


# ---------------------------------------------------------------------------
# OpenNebula VM (pyone)
# ---------------------------------------------------------------------------

def one_vm_create(params: dict, step: ConfigStep) -> str:
    template = params.get("template", 'NAME="vm-from-oneai"')
    # Escape for Python string
    template_escaped = template.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'client = get_one_client()\n'
        f'template_str = "{template_escaped}"\n'
        f'vm_id = client.vm.allocate(template_str)\n'
        f'log(f"VM created with ID: {{vm_id}}", "OK")'
    )


def one_vm_delete(params: dict, step: ConfigStep) -> str:
    return (
        f'client = get_one_client()\n'
        f'client.vm.action("terminate-hard", {params["vm_id"]})\n'
        f'log("VM {params["vm_id"]} terminated", "OK")'
    )


def one_vm_poweroff(params: dict, step: ConfigStep) -> str:
    return (
        f'client = get_one_client()\n'
        f'client.vm.action("poweroff", {params["vm_id"]})\n'
        f'log("VM {params["vm_id"]} powered off", "OK")'
    )


def one_vm_resume(params: dict, step: ConfigStep) -> str:
    return (
        f'client = get_one_client()\n'
        f'client.vm.action("resume", {params["vm_id"]})\n'
        f'log("VM {params["vm_id"]} resumed", "OK")'
    )


def one_vm_list(params: dict, step: ConfigStep) -> str:
    return (
        'client = get_one_client()\n'
        'vm_pool = client.vmpool.info(-2, -1, -1, -1)\n'
        'for vm in vm_pool.VM:\n'
        '    log(f"VM {vm.ID}: {vm.NAME} — state {vm.STATE}", "INFO")'
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ACTION_CODE_REGISTRY: dict[str, callable] = {
    "oneke.namespace.create":    oneke_namespace_create,
    "oneke.namespace.delete":    oneke_namespace_delete,
    "oneke.namespace.list":      oneke_namespace_list,
    "oneke.app.deploy":          oneke_app_deploy,
    "oneke.app.uninstall":       oneke_app_uninstall,
    "oneke.app.upgrade":         oneke_app_upgrade,
    "oneke.app.list":            oneke_app_list,
    "oneke.app.wait_ready":      oneke_app_wait_ready,
    "oneke.app.get_status":      oneke_app_get_status,
    "oneke.service.get_endpoint": oneke_service_get_endpoint,
    "oneke.service.expose":      oneke_service_expose,
    "oneke.service.list":        oneke_service_list,
    "oneke.storage.create_pvc":  oneke_storage_create_pvc,
    "oneke.storage.list_pvcs":   oneke_storage_list_pvcs,
    "oneke.storage.delete_pvc":  oneke_storage_delete_pvc,
    "oneke.cluster.get_info":    oneke_cluster_get_info,
    "oneke.cluster.get_status":  oneke_cluster_get_status,
    "oneke.cluster.list_nodes":  oneke_cluster_list_nodes,
    "oneke.cluster.scale_nodes": oneke_cluster_scale_nodes,
    "one.vm.create":             one_vm_create,
    "one.vm.delete":             one_vm_delete,
    "one.vm.poweroff":           one_vm_poweroff,
    "one.vm.resume":             one_vm_resume,
    "one.vm.list":               one_vm_list,
}
