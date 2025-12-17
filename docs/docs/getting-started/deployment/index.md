---
sidebar_position: 1
---

# Deployment

Deploy Marie-AI to production using Docker, Kubernetes, or other infrastructure.

## Deployment options

| Option | Best for | Complexity |
| ------ | -------- | ---------- |
| [Docker](./docker.md) | Single-node deployments, development | Low |
| [Kubernetes](./kubernetes.md) | Production clusters, scaling | Medium-High |
| [Control plane](./control-plane.md) | Multi-node orchestration | High |

## Quick start

### Docker (single node)

```bash
docker run -d --gpus all \
  -p 54321:54321 \
  -v /path/to/config:/config \
  marieai/marie:latest
```

See the [Docker deployment guide](./docker.md) for complete setup instructions.

### Kubernetes

Deploy using Helm:

```bash
# Add the Marie-AI Helm repository
helm repo add marie https://marieai.github.io/charts

# Install Marie-AI
helm install marie marie/marie \
  --set postgresql.auth.password=your-password
```

See the [Kubernetes deployment guide](./kubernetes.md) for complete setup instructions.

## Operations

### Observability

Monitor and troubleshoot your deployment:

| Guide | Description |
|-------|-------------|
| [Observability](./observability.md) | Prometheus, Grafana, Loki, and Jaeger setup |
| [Troubleshooting](./troubleshooting.md) | Common issues and solutions |

### Scaling

Scale your deployment to handle increased workloads:

| Guide | Description |
|-------|-------------|
| [Scaling](./scaling.md) | HPA, scale-to-zero, resource management |

### Security

Secure your deployment:

| Guide | Description |
|-------|-------------|
| [Security](./security.md) | Authentication, RBAC, network policies |

## Architecture overview

```text
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Gateway Layer                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │Gateway 1│  │Gateway 2│  │Gateway 3│                     │
│  └────┬────┘  └────┬────┘  └────┬────┘                     │
└───────┼────────────┼────────────┼───────────────────────────┘
        │            │            │
┌───────▼────────────▼────────────▼───────────────────────────┐
│                   Executor Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │CPU Pool  │  │CPU Pool  │  │GPU Pool  │  │GPU Pool  │   │
│  │(extract) │  │(classify)│  │ (OCR)    │  │ (NER)    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
        │            │            │            │
┌───────▼────────────▼────────────▼────────────▼──────────────┐
│                   Data Layer                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │PostgreSQL│  │  etcd    │  │  MinIO   │  │ RabbitMQ │   │
│  │(jobs/kv) │  │(discovery)│ │(storage) │  │ (queue)  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Deployment checklist

### Development

- [ ] Deploy with Docker Compose
- [ ] Configure basic authentication
- [ ] Set up local storage

### Staging

- [ ] Deploy to Kubernetes
- [ ] Configure monitoring (Prometheus/Grafana)
- [ ] Set up log aggregation (Loki)
- [ ] Test scaling behavior
- [ ] Configure network policies

### Production

- [ ] Enable high availability (multiple replicas)
- [ ] Configure autoscaling (HPA)
- [ ] Set up alerting
- [ ] Configure TLS everywhere
- [ ] Enable audit logging
- [ ] Set up backup procedures
- [ ] Document runbooks

## Next steps

- [Configuration](../configuration/config.md) - Configure Marie-AI settings
- [Job management](../job-management/index.md) - Understand the job scheduler
