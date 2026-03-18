"""
one_ai_core.prompts
====================
Prompt templates for the LLM. Uses few-shot conversation turns
to teach Mistral 7B the exact output format by demonstration.
"""

SYSTEM_PROMPT = """\
You are an OpenNebula infrastructure automation assistant.
You output a custom YAML configuration schema. You NEVER output shell commands, \
kubectl commands, Kubernetes manifests, or prose explanations.

RULES:
- Your response is RAW YAML only. First line must be "metadata:" or "error:".
- No markdown fences. No prose. No code blocks. Just YAML.
- Use ONLY these actions in steps:
  oneke.namespace.create (params: name)
  oneke.namespace.delete (params: name)
  oneke.namespace.list
  oneke.app.deploy (params: chart, namespace, release_name, values, version)
  oneke.app.uninstall (params: release_name, namespace)
  oneke.app.upgrade (params: release_name, namespace, chart, values)
  oneke.app.list (params: namespace)
  oneke.app.wait_ready (params: release_name, namespace, timeout_seconds)
  oneke.app.get_status (params: release_name, namespace)
  oneke.service.get_endpoint (params: service_name, namespace)
  oneke.service.expose (params: deployment_name, namespace, port, type)
  oneke.service.list (params: namespace)
  oneke.storage.create_pvc (params: name, namespace, storage_class, size_gi)
  oneke.storage.list_pvcs (params: namespace)
  oneke.storage.delete_pvc (params: name, namespace)
  oneke.cluster.get_info
  oneke.cluster.get_status
  oneke.cluster.list_nodes
  oneke.cluster.scale_nodes (params: count)
  one.vm.create (params: template_id, name, cpu, memory_mb)
  one.vm.delete (params: vm_id)
  one.vm.poweroff (params: vm_id)
  one.vm.resume (params: vm_id)
  one.vm.list
- step id format: step_01, step_02, etc.
- on_failure must be: abort | rollback | continue | retry
- metadata.description is REQUIRED."""


FEW_SHOT_USER = "Request: Create a namespace called redis on my OneKE cluster"

FEW_SHOT_ASSISTANT = """\
metadata:
  description: Create redis namespace on the OneKE cluster
  risk_level: low
  tags: [namespace, oneke]
steps:
  - id: step_01
    action: oneke.namespace.create
    description: Create the redis namespace
    params:
      name: redis
    depends_on: []
    on_failure: abort
validation:
  pre_checks:
    - type: cluster_reachable
      target: oneke
      description: Verify the OneKE cluster is reachable
  post_checks:
    - type: namespace_exists
      target: redis
      description: Verify the redis namespace was created
rollback:
  enabled: false
  steps: []"""


FEW_SHOT_USER_2 = "Request: Deploy Nginx on my OneKE cluster in the web namespace with a 5Gi volume"

FEW_SHOT_ASSISTANT_2 = """\
metadata:
  description: Deploy Nginx on OneKE with persistent storage in web namespace
  risk_level: medium
  tags: [nginx, oneke, helm, storage]
steps:
  - id: step_01
    action: oneke.namespace.create
    description: Create the web namespace
    params:
      name: web
    depends_on: []
    on_failure: abort
  - id: step_02
    action: oneke.storage.create_pvc
    description: Create a 5Gi persistent volume claim for Nginx
    params:
      name: nginx-data
      namespace: web
      storage_class: longhorn
      size_gi: 5
    depends_on: [step_01]
    on_failure: rollback
  - id: step_03
    action: oneke.app.deploy
    description: Deploy Nginx via Helm chart
    params:
      chart: bitnami/nginx
      namespace: web
      release_name: nginx
      values:
        persistence:
          existingClaim: nginx-data
    depends_on: [step_02]
    on_failure: rollback
  - id: step_04
    action: oneke.app.wait_ready
    description: Wait for Nginx pods to become ready
    params:
      release_name: nginx
      namespace: web
      timeout_seconds: 300
    depends_on: [step_03]
    on_failure: abort
validation:
  pre_checks:
    - type: cluster_reachable
      target: oneke
      description: Verify the OneKE cluster is reachable
  post_checks:
    - type: deployment_ready
      target: nginx
      description: Verify Nginx deployment is running
rollback:
  enabled: true
  steps:
    - id: step_r01
      action: oneke.app.uninstall
      description: Remove Nginx Helm release
      params:
        release_name: nginx
        namespace: web
      depends_on: []
      on_failure: continue
    - id: step_r02
      action: oneke.storage.delete_pvc
      description: Remove the persistent volume claim
      params:
        name: nginx-data
        namespace: web
      depends_on: [step_r01]
      on_failure: continue"""


USER_PROMPT = """\
Relevant OpenNebula documentation (for context only — do NOT copy commands from here):
---
{rag_context}
---

Request: {user_request}

Output ONLY the YAML configuration. First line must be "metadata:"."""


RETRY_PROMPT = """\
Your previous YAML output failed validation:

{error_summary}

Output ONLY the corrected YAML. First line must be "metadata:". No prose, no commands."""


FEW_SHOT_USER_3 = "Request: Create a new virtual machine from template ID 5 named test-vm"

FEW_SHOT_ASSISTANT_3 = """\
metadata:
  description: Create a new VM named test-vm from template 5
  risk_level: medium
  tags: [vm, opennebula]
steps:
  - id: step_01
    action: one.vm.create
    description: Instantiate a new VM from template ID 5
    params:
      template_id: 5
      name: test-vm
    depends_on: []
    on_failure: abort
validation:
  pre_checks: []
  post_checks: []
rollback:
  enabled: false
  steps: []"""
