"""
Pytest configuration and shared fixtures for one-ai-rag tests.

Provides reusable mock data (HTML pages, scraped pages, chunks, embedders)
so every test module can focus on its own assertions without boilerplate.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure the src directory is on the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from one_ai_rag.scraper import ScrapedPage, ContentExtractor
from one_ai_rag.chunker import DocChunk


# ---------------------------------------------------------------------------
# Mock HTML pages (simulating docs.opennebula.io)
# ---------------------------------------------------------------------------

MOCK_ONEKE_DEPLOY_HTML = """
<html><head><title>OneKE Application Deployment</title></head>
<body>
<nav><a href="/">Home</a> &gt; <a href="/stable/">Docs</a></nav>
<div class="document">
    <h1>Deploying Applications on OneKE</h1>
    <p>OneKE is OpenNebula's Kubernetes Engine. It allows you to deploy
    containerized applications using Helm charts on Kubernetes clusters
    managed by OpenNebula.</p>

    <h2>Prerequisites</h2>
    <p>Before deploying applications, ensure you have:</p>
    <ul>
        <li>A running OneKE cluster with at least one worker node</li>
        <li>kubectl configured to access the cluster</li>
        <li>Helm 3.x installed on your local machine</li>
    </ul>

    <h2>Deploying with Helm</h2>
    <p>OneKE supports standard Helm chart deployments. To deploy an application:</p>
    <pre>helm repo add bitnami https://charts.bitnami.com/bitnami
helm install my-app bitnami/wordpress --namespace my-namespace --create-namespace</pre>
    <p>You can customize the deployment by providing a values file:</p>
    <pre>helm install my-app bitnami/wordpress -f custom-values.yaml</pre>

    <h2>Monitoring Deployments</h2>
    <p>Check the status of your deployment using kubectl:</p>
    <pre>kubectl get pods -n my-namespace
kubectl get svc -n my-namespace</pre>

    <h3>Pod Health Checks</h3>
    <p>OneKE configures liveness and readiness probes automatically for
    most Helm chart deployments. You can verify probe status with:</p>
    <pre>kubectl describe pod my-app-0 -n my-namespace</pre>

    <h2>Scaling Applications</h2>
    <p>Scale your application replicas using Helm upgrade:</p>
    <pre>helm upgrade my-app bitnami/wordpress --set replicaCount=3</pre>
    <p>Or directly with kubectl:</p>
    <pre>kubectl scale deployment my-app --replicas=3 -n my-namespace</pre>
</div>
<div class="footer">Copyright OpenNebula Systems</div>
</body></html>
"""

MOCK_PYONE_API_HTML = """
<html><head><title>Python API (pyone)</title></head>
<body>
<div class="document">
    <h1>Python Bindings for OpenNebula (pyone)</h1>
    <p>pyone is the official Python library for the OpenNebula XML-RPC API.
    It provides a Pythonic interface to manage VMs, templates, virtual
    networks, images, and other OpenNebula resources.</p>

    <h2>Installation</h2>
    <pre>pip install pyone</pre>

    <h2>Connection Setup</h2>
    <p>Create a server connection using your OpenNebula credentials:</p>
    <pre>import pyone
client = pyone.OneServer(
    "http://localhost:2633/RPC2",
    session="user:password"
)</pre>

    <h2>VM Management</h2>
    <p>Create a VM from a template:</p>
    <pre>vm_id = client.vm.allocate(template_string)
client.vm.action("resume", vm_id)</pre>
    <p>List all VMs:</p>
    <pre>vm_pool = client.vmpool.info(-2, -1, -1, -1)
for vm in vm_pool.VM:
    print(f"VM {vm.ID}: {vm.NAME} - {vm.STATE}")</pre>

    <h2>Template Management</h2>
    <p>Templates define the configuration for VMs. You can instantiate
    a template to create a VM with predefined settings.</p>
    <pre>template_id = client.template.allocate(template_string)
vm_id = client.template.instantiate(template_id, "my-vm")</pre>
</div>
</body></html>
"""

MOCK_VNET_HTML = """
<html><head><title>Virtual Networks</title></head>
<body>
<div class="document">
    <h1>Virtual Network Management</h1>
    <p>OpenNebula virtual networks provide L2 and L3 connectivity for VMs.
    Networks can be configured with VLAN tagging, IP address management,
    and security groups.</p>

    <h2>Creating a Virtual Network</h2>
    <p>Define a network template and allocate it:</p>
    <pre>vnet_template = '''
NAME = "my-network"
VN_MAD = "bridge"
BRIDGE = "br0"
AR = [
    TYPE = "IP4",
    IP = "192.168.1.100",
    SIZE = "50"
]
'''
vnet_id = client.vn.allocate(vnet_template)</pre>

    <h2>Address Ranges</h2>
    <p>Each virtual network can have multiple address ranges (ARs) that
    define pools of IP addresses or MAC addresses available for VMs.</p>
</div>
</body></html>
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_html_pages():
    """Return a dict of URL -> HTML for simulating HTTP fetches."""
    return {
        "https://docs.opennebula.io/stable/provision_clusters/oneke/deploy.html": MOCK_ONEKE_DEPLOY_HTML,
        "https://docs.opennebula.io/stable/integration_and_development/pyone.html": MOCK_PYONE_API_HTML,
        "https://docs.opennebula.io/stable/management_and_operations/vnet.html": MOCK_VNET_HTML,
    }


@pytest.fixture
def sample_scraped_pages():
    """Pre-built ScrapedPage objects from the mock HTML."""
    extractor = ContentExtractor()
    pages = []

    for url, html, section, label in [
        (
            "https://docs.opennebula.io/stable/provision_clusters/oneke/deploy.html",
            MOCK_ONEKE_DEPLOY_HTML,
            "oneke",
            "OneKE / Kubernetes",
        ),
        (
            "https://docs.opennebula.io/stable/integration_and_development/pyone.html",
            MOCK_PYONE_API_HTML,
            "api",
            "API & Integration",
        ),
        (
            "https://docs.opennebula.io/stable/management_and_operations/vnet.html",
            MOCK_VNET_HTML,
            "management_operations",
            "Management & Operations",
        ),
    ]:
        content, code_blocks, headings, breadcrumb = extractor.extract(html, url)
        pages.append(ScrapedPage(
            url=url,
            title=headings[0] if headings else "Unknown",
            section=section,
            section_label=label,
            content=content,
            code_blocks=code_blocks,
            headings=headings,
            breadcrumb=breadcrumb,
        ))

    return pages


@pytest.fixture
def sample_chunks(sample_scraped_pages):
    """Chunks generated from the sample scraped pages."""
    from one_ai_rag.chunker import DocChunker

    chunker = DocChunker(target_size=300, overlap=50, min_size=30)
    return chunker.chunk_pages(sample_scraped_pages)


@pytest.fixture
def mock_embedder():
    """
    A deterministic mock embedder that produces consistent 384-dim vectors.

    Uses a hash-based approach so the same text always gets the same vector,
    which makes retrieval tests predictable.
    """
    embedder = MagicMock()
    embedder.dimension = 384

    def _hash_embed(text):
        """Deterministic embedding: hash-based with normalization."""
        rng = np.random.RandomState(hash(text) % (2**32))
        vec = rng.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec.tolist()

    embedder.embed_texts.side_effect = lambda texts: [_hash_embed(t) for t in texts]
    embedder.embed_query.side_effect = _hash_embed

    return embedder
