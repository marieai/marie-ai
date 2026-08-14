# Marie-AI All-in-One Deployment

## Overview

Single `docker-compose.allinone.yml` deployment bundling the entire Marie-AI infrastructure:

- **PostgreSQL** (DocumentDB) -- Document storage and metadata
- **MinIO S3** -- Object storage for documents and artifacts
- **RabbitMQ** -- Message broker for async task processing
- **etcd** -- Distributed coordination and configuration
- **ClickHouse** -- Column-oriented analytics database
- **Gitea** -- Self-hosted Git service (used by marie-studio)
- **LiteLLM** -- OpenAI-compatible provider gateway for AI model routing, fallback, budgets, and rate limits
- **HyperDX** -- Observability UI (logs, traces, metrics)
- **Gateway** -- Marie-AI HTTP/gRPC gateway
- **Extract executors** -- GPU-enabled document extraction servers
- **Log Collector** -- OpenTelemetry-based Docker container log collection

All services are orchestrated with declarative `depends_on` health checks, replacing the imperative `bootstrap-marie.sh` script.

## LLM Dispatch Boundary

Marie Gateway can run the LLM Dispatch Runtime for executor-originated LLM calls. Dispatch uses Valkey for live request/reply transport and calls one configured OpenAI-compatible backend URL.

Use LiteLLM, OpenRouter, vLLM, or another OpenAI-compatible backend for provider-level policy:

- provider fallback chains
- model/provider routing by tenant, cost, latency, region, or capacity
- provider budgets and rate limits

Marie Dispatch owns executor ingress, producer liveness, in-flight state, dispatch retry/timeout behavior, circuit-breaker state, and backpressure visibility.

## Suitable For

- Development and testing
- Demos and evaluations
- Single-server deployments
- Quick prototyping

**Not recommended for multi-node production deployments.** See [Production Deployment](#production-deployment) for alternatives.

## Quick Start

```bash
cd marie-ai

# Create the Docker network
docker network create --driver=bridge marie_default

# Copy environment template
cp docker/allinone/.env.example config/.env
# Edit config/.env with your settings

# Start everything
./docker/allinone/start.sh full

# Or just infrastructure (no gateway/extract)
./docker/allinone/start.sh infra-only
```

Verify services are running:

```bash
docker compose -f Dockerfiles/docker-compose.allinone.yml ps
```

## Profiles

| Profile | Command | Services |
|---------|---------|----------|
| `infra-only` | `./docker/allinone/start.sh infra-only` | PostgreSQL, MinIO, RabbitMQ, etcd |
| `observability` | `./docker/allinone/start.sh observability` | + ClickHouse, HyperDX, Log Collector |
| `application` | `./docker/allinone/start.sh application` | + Gitea, LiteLLM, Gateway, Extract |
| `full` | `./docker/allinone/start.sh full` | All services |
| `gpu` | `./docker/allinone/start.sh gpu` | All services + GPU-enabled Extract |

Services start in tiered order via `depends_on` conditions:

| Tier | Services | Waits For |
|------|----------|-----------|
| 0 | psql, s3server, rabbitmq, etcd | (nothing) |
| 1 | mc-setup, db-init | psql healthy, s3 healthy |
| 2 | clickhouse, gitea, litellm | db-init completed |
| 3 | hyperdx, log-collector | clickhouse healthy |
| 4 | gateway | all infrastructure healthy |
| 5 | extract (GPU) | gateway healthy |

## Persisting Data

Named volumes store all persistent data:

| Volume | Contents | Notes |
|--------|----------|-------|
| `psql_data` | PostgreSQL databases | Marie-AI, Gitea, LiteLLM, Mem0 |
| `/mnt/data/s3` (bind mount) | MinIO object storage | Documents, artifacts |
| `etcd_data` | etcd key-value store | Cluster coordination |
| `rabbitmq_data` | RabbitMQ queues | In-flight messages |
| `clickhouse_data` | ClickHouse tables | Analytics, observability |
| `gitea_data` | Git repositories | Source code, configs |
| `hyperdx_data` | HyperDX state | Dashboards, saved searches |

### Backup

```bash
# Stop services before backing up
./docker/allinone/stop.sh

# Back up PostgreSQL
docker run --rm -v marie_psql_data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/psql_data.tar.gz -C /data .

# Back up MinIO
tar czf backups/s3_data.tar.gz -C /mnt/data/s3 .

# Back up all named volumes
for vol in psql_data etcd_data rabbitmq_data clickhouse_data gitea_data hyperdx_data; do
  docker run --rm -v "marie_${vol}:/data" -v "$(pwd)/backups:/backup" \
    alpine tar czf "/backup/${vol}.tar.gz" -C /data .
done

# Restart
./docker/allinone/start.sh full
```

## GPU Support

Extract executors require NVIDIA GPUs with CUDA support for document extraction inference.

```bash
# Requires NVIDIA Container Toolkit
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
./docker/allinone/start.sh gpu
```

The `gpu` profile enables NVIDIA device passthrough:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu, compute, utility]
    limits:
      memory: 32G
```

Without a GPU, use the `application` profile which runs Extract in CPU mode, or omit Extract entirely with `infra-only` or `observability`.

## Customizing Ports

Override default ports via environment variables in `config/.env`:

| Variable | Default | Service |
|----------|---------|---------|
| `CLICKHOUSE_HTTP_PORT` | `8123` | ClickHouse HTTP API |
| `CLICKHOUSE_NATIVE_PORT` | `9000` | ClickHouse native protocol |
| `CLICKHOUSE_MYSQL_PORT` | `9004` | ClickHouse MySQL wire protocol |
| `HYPERDX_UI_PORT` | `8080` | HyperDX web UI |
| `HYPERDX_API_PORT` | `8002` | HyperDX API |
| `OTEL_GRPC_PORT` | `4317` | OTLP gRPC receiver |
| `OTEL_HTTP_PORT` | `4318` | OTLP HTTP receiver |
| `GITEA_HTTP_PORT` | `3001` | Gitea web UI |
| `GITEA_SSH_PORT` | `2222` | Gitea SSH |
| `LITELLM_PORT` | `4000` | LiteLLM proxy |
| `FLUENTD_PORT` | `24225` | Fluentd forward receiver |

Gateway (port 51000 HTTP, 52000 gRPC) and Extract use `network_mode: host` and bind directly to host ports.

## Observability

HyperDX provides a unified observability UI for logs, traces, and metrics.

| Endpoint | URL | Protocol |
|----------|-----|----------|
| HyperDX UI | `http://localhost:8080` | HTTP |
| OTLP gRPC | `localhost:4317` | gRPC |
| OTLP HTTP | `localhost:4318` | HTTP |

The log collector automatically ingests Docker container logs (JSON format) from all running containers, parses Marie-AI log fields (job_id, status, duration, component), and forwards them to HyperDX.

Default HyperDX credentials (set in `.env`):

```
HYPERDX_ADMIN_EMAIL=marie@marie.local
HYPERDX_ADMIN_PASSWORD=MarieAI@2026!
```

See the [Log Collector guide](log-collector.md) for details on collected fields and search examples.

## Using the Log Collector Standalone

Run just the log-collector against any external OTLP-compatible backend:

```bash
docker run -d --name log-collector \
  -v /var/lib/docker/containers:/var/lib/docker/containers:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  -v ./config/clickstack/log-collector-config.yml:/etc/otel/config.yml:ro \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://your-otlp-backend:4318 \
  -e DEPLOYMENT_ENV=production \
  --user 0:0 \
  otel/opentelemetry-collector-contrib:0.114.0 \
  --config=/etc/otel/config.yml
```

The collector must run as root (UID 0) to read Docker container log files from `/var/lib/docker/containers`.

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| **Global** | | |
| `DATA_DIR` | `/mnt/data/marie-ai` | Base data directory for config, models, and caches |
| `STACK_NAME` | `dev` | Deployment environment name |
| `PROJECT_NAME` | `marie` | Docker Compose project name |
| `MARIE_NETWORK` | `marie_default` | Docker network name |
| **PostgreSQL** | | |
| `POSTGRES_VERSION` | `16` | PostgreSQL major version |
| `POSTGRES_USER` | `postgres` | PostgreSQL superuser name |
| `POSTGRES_PASSWORD` | *(required)* | PostgreSQL superuser password |
| `POSTGRES_HOSTNAME` | `marie-psql-server` | PostgreSQL container hostname |
| **MinIO S3** | | |
| `MINIO_ROOT_USER` | *(required)* | MinIO root admin user |
| `MINIO_ROOT_PASSWORD` | *(required)* | MinIO root admin password |
| `MARIE_ACCESS_KEY` | `MARIEACCESSKEY` | Application-level S3 access key |
| `MARIE_SECRET_ACCESS_KEY` | `MARIESECRETACCESSKEY` | Application-level S3 secret key |
| `S3_ENDPOINT_URL` | `http://marie-s3-server:8000` | S3 endpoint URL |
| `S3_BUCKET_NAME` | `marie` | Default S3 bucket |
| `S3_REGION` | `us-east-1` | S3 region |
| **RabbitMQ** | | |
| `RABBIT_MQ_HOSTNAME` | `marie-rabbitmq` | RabbitMQ container hostname |
| `RABBIT_MQ_USERNAME` | `marie` | RabbitMQ username |
| `RABBIT_MQ_PASSWORD` | *(required)* | RabbitMQ password |
| `RABBIT_MQ_PORT` | `5672` | RabbitMQ AMQP port |
| **Gateway** | | |
| `GATEWAY_IMAGE_TAG` | `5.0.2-cpu` | Gateway Docker image tag |
| `GATEWAY_CONFIG_TAG` | `marie-gateway-4.0.0.yml` | Gateway config file name |
| `GATEWAY_LOG_LEVEL` | `DEBUG` | Gateway log verbosity |
| **Extract Executor** | | |
| `EXTRACT_IMAGE_TAG` | `5.0.2-cuda` | Extract Docker image tag |
| `EXTRACT_CONFIG_TAG` | `marie-extract-4.0.0.yml` | Extract config file name |
| `EXTRACT_LOG_LEVEL` | `DEBUG` | Extract log verbosity |
| `EXTRACT_REPLICAS` | `1` | Number of Extract replicas |
| **ClickHouse** | | |
| `CLICKHOUSE_HTTP_PORT` | `8123` | HTTP API port |
| `CLICKHOUSE_NATIVE_PORT` | `9000` | Native protocol port |
| `CLICKHOUSE_USER` | `default` | ClickHouse username |
| `CLICKHOUSE_PASSWORD` | *(empty)* | ClickHouse password |
| `CLICKHOUSE_DB` | `marie` | Default database |
| `CLICKHOUSE_MAX_MEMORY` | `4000000000` | Max memory usage (bytes) |
| `CLICKHOUSE_MEMORY_LIMIT` | `8G` | Docker memory limit |
| `CLICKHOUSE_MEMORY_RESERVATION` | `2G` | Docker memory reservation |
| **Gitea** | | |
| `GITEA_HTTP_PORT` | `3001` | Gitea web UI port |
| `GITEA_SSH_PORT` | `2222` | Gitea SSH port |
| `GITEA_DOMAIN` | `localhost` | Gitea domain name |
| `GITEA_ROOT_URL` | `http://localhost:3001` | Gitea external URL |
| `GITEA_DB_NAME` | `gitea` | Gitea database name |
| `GITEA_INSTALL_LOCK` | `true` | Skip installation wizard |
| `GITEA_ADMIN_USER` | `marie` | Initial admin username |
| `GITEA_ADMIN_PASSWORD` | *(required)* | Initial admin password |
| `GITEA_ADMIN_EMAIL` | `marie@marie.local` | Initial admin email |
| **LiteLLM** | | |
| `LITELLM_MASTER_KEY` | `sk-1234` | LiteLLM API master key |
| `LITELLM_SALT_KEY` | `sk-1234` | LiteLLM encryption salt |
| `LITELLM_LOG` | `INFO` | LiteLLM log level |
| `LITELLM_PORT` | `4000` | LiteLLM proxy port |
| **ClickStack / Observability** | | |
| `HYPERDX_UI_PORT` | `8080` | HyperDX web UI port |
| `HYPERDX_LOG_LEVEL` | `info` | HyperDX log level |
| `HYPERDX_MEMORY_LIMIT` | `4G` | HyperDX Docker memory limit |
| `HYPERDX_MEMORY_RESERVATION` | `1G` | HyperDX Docker memory reservation |
| `HYPERDX_ADMIN_EMAIL` | `marie@marie.local` | HyperDX admin email |
| `HYPERDX_ADMIN_PASSWORD` | *(required)* | HyperDX admin password (12+ chars, mixed case, number, special) |
| `OTEL_GRPC_PORT` | `4317` | OTLP gRPC port |
| `OTEL_HTTP_PORT` | `4318` | OTLP HTTP port |
| `LOG_COLLECTOR_MEMORY_LIMIT` | `512M` | Log collector memory limit |
| `LOG_COLLECTOR_MEMORY_RESERVATION` | `128M` | Log collector memory reservation |
| **Mem0** | | |
| `MEM0_DB_NAME` | `mem0` | Mem0 database name |
| `MEM0_COLLECTION_NAME` | `memories` | Mem0 collection name |
| `OPENAI_BASE_URL` | `http://localhost:4000/v1` | LiteLLM endpoint for Mem0 SDK |
| `OPENAI_API_KEY` | `sk-1234` | API key for LiteLLM proxy |

## Production Deployment

> **Warning:** The all-in-one deployment is NOT recommended for production use. It runs all services on a single host with no redundancy or horizontal scaling.

For production environments:

- Use individual compose files from `Dockerfiles/` for service isolation
- Deploy with Kubernetes using the Helm charts in `deploy/helm/`
- Use the operator in `deploy/operator/` for automated management
- Run PostgreSQL, MinIO, and RabbitMQ as managed services or dedicated clusters
- Separate GPU nodes for Extract executors from infrastructure nodes

## Troubleshooting

### Network not found

```
Error: network marie_default not found
```

Create the network before starting:

```bash
docker network create --driver=bridge marie_default
```

### GPU not detected

```
Error: could not select device driver "nvidia" with capabilities: [[gpu compute utility]]
```

Install the NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Database initialization failures

If `db-init` fails, check PostgreSQL is healthy:

```bash
docker logs marie-psql-server
docker exec marie-psql-server pg_isready
```

To re-run initialization:

```bash
docker compose -f Dockerfiles/docker-compose.allinone.yml restart db-init
```

### Port conflicts

If a port is already in use, override it in `config/.env`:

```bash
# Example: ClickHouse HTTP port conflict with another service
CLICKHOUSE_HTTP_PORT=18123
```

Then restart:

```bash
./docker/allinone/stop.sh
./docker/allinone/start.sh full
```

### ClickHouse out of memory

ClickHouse requires at least 2 GB RAM. Adjust memory limits in `.env`:

```bash
CLICKHOUSE_MEMORY_LIMIT=4G
CLICKHOUSE_MEMORY_RESERVATION=1G
CLICKHOUSE_MAX_MEMORY=2000000000
```

### HyperDX slow to start

HyperDX includes embedded ClickHouse and MongoDB. Allow up to 2 minutes for initial startup:

```bash
docker logs -f marie-hyperdx
```

The health check has a `start_period` of 120 seconds.

### Viewing service logs

```bash
# All services
docker compose -f Dockerfiles/docker-compose.allinone.yml logs -f

# Specific service
docker logs -f marie-psql-server
docker logs -f marie-hyperdx
docker logs -f marie-log-collector
docker logs -f marieai-gateway
```
