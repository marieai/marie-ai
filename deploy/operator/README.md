# Marie Operator

A Kubernetes operator for deploying and managing Marie-AI clusters on Kubernetes.

## Overview

The Marie Operator provides Kubernetes-native management of Marie-AI document processing clusters. It follows the operator pattern inspired by [KubeRay](https://github.com/ray-project/kuberay) and provides:

- **MarieCluster CRD**: Manages Marie-AI clusters including server and executor pools
- **MarieJob CRD**: Manages document processing jobs submitted to Marie clusters
- **GPU Support**: Built-in support for GPU executor pools with scale-to-zero
- **Multi-pool Executors**: Support for different executor types (OCR, NER, Classifier)

## Custom Resource Definitions

### MarieCluster

A MarieCluster represents a complete Marie-AI deployment with:
- A server component (gateway + scheduler)
- Multiple executor groups (with GPU/CPU support)

```yaml
apiVersion: marie.ai/v1alpha1
kind: MarieCluster
metadata:
  name: my-cluster
spec:
  serverSpec:
    replicas: 1
    template:
      spec:
        containers:
          - name: marie
            image: marieai/marie:3.0-cuda
            # ... container config
  executorGroupSpecs:
    - groupName: ocr-gpu
      replicas: 2
      gpu: true
      template:
        spec:
          containers:
            - name: marie
              image: marieai/marie:3.0-cuda
              resources:
                limits:
                  nvidia.com/gpu: "1"
```

### MarieJob

A MarieJob represents a document processing job:

```yaml
apiVersion: marie.ai/v1alpha1
kind: MarieJob
metadata:
  name: ocr-batch-job
spec:
  clusterRef: my-cluster
  jobType: ocr
  input:
    uri: "s3://bucket/documents/"
  output:
    uri: "s3://bucket/results/"
  sla:
    softDeadline: "30m"
```

## Installation

### Prerequisites

- Kubernetes 1.25+
- kubectl configured to access your cluster
- (Optional) GPU nodes with NVIDIA device plugin for GPU workloads

### Using Kustomize

```bash
# Install CRDs
kubectl apply -k config/crd

# Deploy the operator
kubectl apply -k config/default
```

### Using Helm (coming soon)

```bash
helm repo add marie https://charts.marieai.co
helm install marie-operator marie/marie-operator
```

## Building from Source

```bash
# Build the operator binary
make build

# Build the Docker image
make docker-build IMG=marieai/marie-operator:latest

# Push the image
make docker-push IMG=marieai/marie-operator:latest

# Generate CRDs
make manifests
```

## Development

### Running Locally

```bash
# Install CRDs
make install

# Run the operator locally
make run
```

### Running Tests

```bash
make test
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Marie Operator                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐       │
│  │  MarieCluster        │    │  MarieJob            │       │
│  │  Controller          │    │  Controller          │       │
│  └──────────┬───────────┘    └──────────┬───────────┘       │
│             │                           │                    │
│  ┌──────────▼───────────────────────────▼───────────┐       │
│  │              Common Utilities                     │       │
│  │  ┌─────────┐ ┌─────────┐ ┌───────────────────┐   │       │
│  │  │ Pod     │ │ Service │ │ Association       │   │       │
│  │  │ Builder │ │ Builder │ │ Options           │   │       │
│  │  └─────────┘ └─────────┘ └───────────────────┘   │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Controller Reconciliation Flow

The MarieCluster controller follows a chain-of-responsibility pattern:

1. `reconcileServerService` - Creates/updates the server service
2. `reconcileServerHeadlessService` - Creates headless service for DNS
3. `reconcileServerDeployment` - Creates/updates server deployment
4. `reconcileExecutorServices` - Creates services for executor groups
5. `reconcileExecutorDeployments` - Creates/updates executor deployments
6. `reconcileIngress` - Creates ingress if enabled

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LEADER_ELECTION` | Enable leader election | `false` |
| `METRICS_BIND_ADDRESS` | Metrics server address | `:8080` |
| `HEALTH_PROBE_BIND_ADDRESS` | Health probe address | `:8081` |

## License

Apache License 2.0
