---
sidebar_position: 5
---

# Scaling

Scale Marie-AI deployments to handle varying workloads. This guide covers horizontal scaling, autoscaling, scale-to-zero for GPU workloads, and resource management.

## Overview

Marie-AI supports multiple scaling strategies:

| Strategy | Use Case | Components |
|----------|----------|------------|
| Horizontal scaling | Increase throughput | Gateway, Executors |
| Vertical scaling | Increase per-pod capacity | All components |
| Autoscaling (HPA) | Dynamic workload adaptation | Server, Executors |
| Scale-to-zero | Cost optimization for GPUs | GPU Executors |

```text
                     Scaling Strategies
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Horizontal                  Vertical                      │
│   ┌─────┐ ┌─────┐ ┌─────┐    ┌───────────────┐             │
│   │ Pod │ │ Pod │ │ Pod │    │    Pod        │             │
│   │ 1   │ │ 2   │ │ 3   │    │  (larger)     │             │
│   └─────┘ └─────┘ └─────┘    │  CPU: 8       │             │
│                               │  Mem: 32Gi   │             │
│   Add more pods               └───────────────┘             │
│                               Increase resources            │
│                                                             │
│   HPA (Auto)                  Scale-to-Zero                │
│   ┌─────┐ ┌─────┐            ┌─────┐                       │
│   │ Pod │ │ Pod │    ←       │ GPU │  → 0 pods when idle   │
│   └─────┘ └─────┘            │ Pod │                       │
│      ↑      ↓                └─────┘                       │
│   Scales based on metrics    Cost savings                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Horizontal scaling

### Manual replica configuration

Set replica counts directly in your deployment configuration.

**Docker Compose:**

```yaml
services:
  marie-gateway:
    image: marieai/marie:3.0-cuda
    deploy:
      replicas: 2

  marie-executor-extract:
    image: marieai/marie:3.0-cuda
    deploy:
      replicas: 4
```

**Kubernetes Deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: marie-executor
spec:
  replicas: 4
  selector:
    matchLabels:
      app: marie-executor
  template:
    # ... pod template
```

**Helm values:**

```yaml
server:
  replicas: 2

executor:
  pools:
    - name: cpu-general
      replicas: 4
```

### Gateway scaling

The Gateway handles request routing and job scheduling. Scale based on:
- Request throughput
- Number of concurrent connections
- Job submission rate

```yaml
server:
  replicas: 3
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"
```

### Executor scaling

Executors perform the actual document processing. Scale based on:
- Queue depth
- Processing time requirements
- GPU availability

```yaml
executor:
  pools:
    # CPU pool for lightweight tasks
    - name: cpu-general
      replicas: 4
      gpu: false
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"

    # GPU pool for ML-intensive tasks
    - name: gpu-ocr
      replicas: 2
      gpu: true
      resources:
        requests:
          nvidia.com/gpu: 1
```

## Horizontal Pod Autoscaler (HPA)

HPA automatically scales pods based on observed metrics.

### Enabling HPA

Enable autoscaling in Helm values:

```yaml
server:
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 5
    targetCPUUtilizationPercentage: 80
```

### HPA manifest

The generated HPA resource:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: marie-server
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: marie-server
  minReplicas: 1
  maxReplicas: 5
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 80
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### Custom metrics

Scale based on custom metrics from Prometheus:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: marie-executor
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: marie-executor
  minReplicas: 1
  maxReplicas: 10
  metrics:
    # Scale based on job queue depth
    - type: External
      external:
        metric:
          name: marie_jobs_pending
        target:
          type: AverageValue
          averageValue: "10"
    # Scale based on executor slot utilization
    - type: External
      external:
        metric:
          name: marie_executor_slots_used
        target:
          type: AverageValue
          averageValue: "3"
```

### Scaling behavior

Configure scale-up and scale-down behavior:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
```

## Scale-to-zero for GPU workloads

GPU resources are expensive. Scale-to-zero reduces costs during idle periods.

### Cluster Autoscaler integration

Configure GPU executors for scale-to-zero:

```yaml
executor:
  pools:
    - name: gpu-ocr
      enabled: true
      replicas: 1
      gpu: true

      scaleToZero:
        enabled: true
        annotations:
          # Allow cluster autoscaler to evict the pod
          cluster-autoscaler.kubernetes.io/safe-to-evict: "true"

      resources:
        requests:
          nvidia.com/gpu: 1
        limits:
          nvidia.com/gpu: 1
```

### KEDA for event-driven scaling

Use KEDA for more sophisticated scale-to-zero:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: marie-gpu-executor
spec:
  scaleTargetRef:
    name: marie-gpu-executor
  minReplicaCount: 0  # Scale to zero
  maxReplicaCount: 4
  cooldownPeriod: 300  # Wait 5 min before scaling down
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: marie_jobs_pending
        threshold: "1"
        query: |
          sum(marie_jobs_pending{executor="gpu-ocr"})
```

### Cold start considerations

When scaling from zero, account for startup time:

| Component | Typical Startup Time |
|-----------|---------------------|
| CPU Executor | 10-30 seconds |
| GPU Executor | 30-120 seconds |
| GPU Executor (model loading) | 60-300 seconds |

Mitigate cold starts:
1. Use startup probes with appropriate thresholds
2. Pre-pull images on GPU nodes
3. Cache models in persistent volumes
4. Consider keeping minimum 1 replica during business hours

```yaml
probes:
  startup:
    initialDelaySeconds: 30
    periodSeconds: 10
    failureThreshold: 60  # Allow up to 10 minutes for GPU startup
```

## Resource management

### CPU and memory

Set appropriate resource requests and limits:

```yaml
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
  limits:
    cpu: "4"
    memory: "8Gi"
```

**Guidelines:**

| Component | CPU Request | Memory Request | Notes |
|-----------|-------------|----------------|-------|
| Gateway | 1-2 cores | 2-4 Gi | Scales with connections |
| CPU Executor | 2-4 cores | 4-8 Gi | Scales with throughput |
| GPU Executor | 4-8 cores | 16-32 Gi | GPU memory separate |

### GPU allocation

Request NVIDIA GPUs through Kubernetes:

```yaml
resources:
  requests:
    nvidia.com/gpu: 1
  limits:
    nvidia.com/gpu: 1
```

**GPU node selection:**

```yaml
# GKE
nodeSelector:
  cloud.google.com/gke-accelerator: nvidia-tesla-t4

# EKS
nodeSelector:
  node.kubernetes.io/instance-type: g4dn.xlarge

# Generic
nodeSelector:
  gpu: "true"
```

**GPU tolerations:**

```yaml
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

### Quality of Service (QoS)

Configure QoS classes through resource settings:

| QoS Class | Configuration | Use Case |
|-----------|---------------|----------|
| Guaranteed | requests = limits | Production GPU workloads |
| Burstable | requests < limits | General executors |
| BestEffort | No resources | Development only |

**Guaranteed QoS for GPU:**

```yaml
resources:
  requests:
    cpu: "4"
    memory: "16Gi"
    nvidia.com/gpu: 1
  limits:
    cpu: "4"
    memory: "16Gi"
    nvidia.com/gpu: 1
```

## Load balancing

### Gateway load balancing

The Gateway distributes work across executors using capacity-aware scheduling:

```text
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
│                           │                                 │
│              ┌────────────┼────────────┐                    │
│              ▼            ▼            ▼                    │
│         ┌────────┐  ┌────────┐  ┌────────┐                 │
│         │Gateway │  │Gateway │  │Gateway │                 │
│         │   1    │  │   2    │  │   3    │                 │
│         └───┬────┘  └───┬────┘  └───┬────┘                 │
│             │           │           │                       │
│             └───────────┼───────────┘                       │
│                         │                                   │
│              ┌──────────┴──────────┐                        │
│              │  Capacity Manager   │                        │
│              └──────────┬──────────┘                        │
│                         │                                   │
│         ┌───────────────┼───────────────┐                   │
│         ▼               ▼               ▼                   │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐              │
│    │Executor │    │Executor │    │Executor │              │
│    │ slots:4 │    │ slots:4 │    │ slots:4 │              │
│    └─────────┘    └─────────┘    └─────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### Service configuration

Configure Kubernetes services for load balancing:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: marie-gateway
spec:
  type: ClusterIP
  selector:
    app: marie-gateway
  ports:
    - name: grpc
      port: 52000
      targetPort: 52000
    - name: http
      port: 8080
      targetPort: 8080
  sessionAffinity: None  # Round-robin by default
```

### Ingress configuration

For external access with load balancing:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: marie-ingress
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  ingressClassName: nginx
  rules:
    - host: marie.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: marie-gateway
                port:
                  number: 8080
```

## Capacity monitoring

### Checking current capacity

```bash
# Via REST API
curl http://localhost:54322/api/capacity
```

Response:

```json
{
  "status": "OK",
  "result": {
    "slots": [
      {
        "name": "extract",
        "capacity": 4,
        "target": 4,
        "used": 2,
        "available": 2,
        "holders": ["executor-1", "executor-2"]
      }
    ],
    "totals": {
      "capacity": 12,
      "used": 5,
      "available": 7
    }
  }
}
```

### Capacity metrics

Monitor capacity through Prometheus:

```promql
# Available slots
marie_executor_slots_available

# Utilization percentage
marie_executor_slots_used / (marie_executor_slots_used + marie_executor_slots_available) * 100

# Pending jobs per available slot (pressure indicator)
marie_jobs_pending / marie_executor_slots_available
```

## Scaling recommendations

### By workload type

| Workload | Gateway Replicas | CPU Executors | GPU Executors |
|----------|------------------|---------------|---------------|
| Development | 1 | 1 | 0-1 |
| Small production | 2 | 2-4 | 1-2 |
| Medium production | 3 | 4-8 | 2-4 |
| Large production | 3-5 | 8-16 | 4-8 |

### By throughput

| Documents/hour | Configuration |
|----------------|---------------|
| < 100 | 1 gateway, 2 executors |
| 100-1,000 | 2 gateways, 4-8 executors |
| 1,000-10,000 | 3 gateways, 8-16 executors |
| > 10,000 | Custom sizing required |

### Best practices

1. **Start small**: Begin with minimal replicas and scale up based on observed metrics

2. **Monitor before scaling**: Use observability tools to identify bottlenecks

3. **Set resource limits**: Prevent runaway pods from affecting cluster stability

4. **Use pod disruption budgets**: Ensure availability during updates

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: marie-gateway-pdb
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: marie-gateway
```

5. **Configure anti-affinity**: Spread pods across nodes

```yaml
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              app: marie-executor
          topologyKey: kubernetes.io/hostname
```

## Next steps

- [Observability](./observability.md) - Monitor scaling metrics
- [Kubernetes deployment](./kubernetes.md) - Production Kubernetes setup
- [Configuration](../job-management/configuration.md) - Scheduler configuration
