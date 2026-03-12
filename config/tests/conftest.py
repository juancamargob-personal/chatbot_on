"""Shared fixtures for one-ai-config tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


WORDPRESS_DEPLOY_YAML = """\
version: "1.0"
metadata:
  description: "Deploy WordPress on OneKE cluster"
  target_cluster: "default"
  estimated_duration: "5-10 minutes"
  risk_level: "low"
  tags: ["oneke", "wordpress"]

steps:
  - id: "step_01"
    action: "oneke.namespace.create"
    description: "Create the wordpress namespace for the deployment"
    params:
      name: "wordpress"
      labels:
        app: "wordpress"
    on_failure: "abort"

  - id: "step_02"
    action: "oneke.app.deploy"
    description: "Deploy WordPress using the Bitnami Helm chart"
    params:
      chart: "bitnami/wordpress"
      namespace: "wordpress"
      release_name: "wordpress"
      repo_url: "https://charts.bitnami.com/bitnami"
      create_namespace: false
      wait: true
      values:
        wordpressUsername: "admin"
        service:
          type: "LoadBalancer"
        persistence:
          enabled: true
          size: "10Gi"
    depends_on: ["step_01"]
    on_failure: "rollback"
    timeout_seconds: 300

  - id: "step_03"
    action: "oneke.app.wait_ready"
    description: "Wait for WordPress pods to be running"
    params:
      namespace: "wordpress"
      label_selector: "app.kubernetes.io/name=wordpress"
      timeout_seconds: 300
    depends_on: ["step_02"]

  - id: "step_04"
    action: "oneke.service.get_endpoint"
    description: "Get the WordPress external endpoint"
    params:
      namespace: "wordpress"
      service_name: "wordpress"
    depends_on: ["step_03"]

validation:
  pre_checks:
    - type: "cluster_reachable"
      target: "default"
    - type: "namespace_available"
      namespace: "wordpress"
  post_checks:
    - type: "pods_running"
      namespace: "wordpress"
      label_selector: "app.kubernetes.io/name=wordpress"
      expected_count: 1
    - type: "service_available"
      namespace: "wordpress"
      service_name: "wordpress"

rollback:
  steps:
    - id: "step_01"
      action: "oneke.app.uninstall"
      description: "Remove the WordPress Helm release"
      params:
        release_name: "wordpress"
        namespace: "wordpress"
    - id: "step_02"
      action: "oneke.namespace.delete"
      description: "Remove the wordpress namespace"
      params:
        name: "wordpress"
      depends_on: ["step_01"]
"""


ERROR_RESPONSE_YAML = """\
version: "1.0"
metadata:
  description: "Request cannot be fulfilled"
  risk_level: "low"
steps: []
error:
  is_error: true
  reason: "TensorFlow Serving on bare VMs is not supported by this tool."
  suggestion: "Use OneKE to deploy TensorFlow Serving as a Kubernetes deployment."
validation:
  pre_checks: []
  post_checks: []
rollback:
  steps: []
"""


SIMPLE_NAMESPACE_YAML = """\
version: "1.0"
metadata:
  description: "Create a production namespace"
  risk_level: "low"
steps:
  - id: "step_01"
    action: "oneke.namespace.create"
    description: "Create the production namespace with labels"
    params:
      name: "production"
      labels:
        env: "production"
        team: "platform"
validation:
  pre_checks:
    - type: "cluster_reachable"
  post_checks: []
rollback:
  steps: []
"""


@pytest.fixture
def wordpress_yaml():
    return WORDPRESS_DEPLOY_YAML


@pytest.fixture
def error_yaml():
    return ERROR_RESPONSE_YAML


@pytest.fixture
def namespace_yaml():
    return SIMPLE_NAMESPACE_YAML


@pytest.fixture
def wordpress_config():
    from one_ai_config.schema.base import parse_config
    return parse_config(WORDPRESS_DEPLOY_YAML)


@pytest.fixture
def error_config():
    from one_ai_config.schema.base import parse_config
    return parse_config(ERROR_RESPONSE_YAML)


@pytest.fixture
def namespace_config():
    from one_ai_config.schema.base import parse_config
    return parse_config(SIMPLE_NAMESPACE_YAML)
