# Marie-AI Docker Infrastructure

This documentation describes the Docker Compose infrastructure for Marie-AI, including all services, networking, and deployment procedures.

## Table of Contents

- [Network Architecture](#network-architecture)
- [Infrastructure Services](#infrastructure-services)
- [Quick Start](#quick-start)
- [Service Details](#service-details)
- [Environment Configuration](#environment-configuration)
- [Annotator Services](#annotator-services)

---

## Network Architecture

All Marie-AI services run on a shared Docker bridge network called `marie_default`. This enables container-to-container communication using service names as hostnames.

### Network Setup

Create the network before starting any services:

```bash
docker network create --driver=bridge marie_default
```

### Service Connectivity

| Service | Container Name | Internal Hostname | Ports |
|---------|---------------|-------------------|-------|
| PostgreSQL | marie-psql-server | `marie-psql-server` | 5432 |
| ClickHouse | marie-clickhouse | `marie-clickhouse` | 8123, 9000, 9004 |
| RabbitMQ | marie-rabbitmq | `marie-rabbitmq` | 5672, 15672 |
| MinIO (S3) | marie-s3-server | `marie-s3-server` | 8000, 8001 |
| etcd | etcd-single | `etcd-single` | 2379, 2380 |
| Gitea | marie-gitea | `marie-gitea` | 3001, 2222 |
| LiteLLM | marie-litellm | `marie-litellm` | 4000 |

Services communicate internally using container names (e.g., `marie-psql-server:5432`).

---

## Infrastructure Services

| Compose File | Service | Description |
|--------------|---------|-------------|
| `docker-compose.storage.yml` | PostgreSQL | Primary database (DocumentDB compatible) |
| `docker-compose.clickhouse.yml` | ClickHouse | Analytics/metrics database |
| `docker-compose.rabbitmq.yml` | RabbitMQ | Message queue |
| `docker-compose.s3.yml` | MinIO | S3-compatible object storage |
| `docker-compose.etcd.yml` | etcd | Distributed key-value store |
| `docker-compose.gitea.yml` | Gitea | Git hosting (used by marie-studio) |
| `docker-compose.litellm.yml` | LiteLLM | LLM proxy/router |

---

## Quick Start

### 1. Create the Network

```bash
docker network create --driver=bridge marie_default
```

### 2. Start Core Infrastructure

```bash
# Start all infrastructure services
docker compose --env-file ./config/.env -f ./Dockerfiles/docker-compose.storage.yml --project-directory . up -d
docker compose --env-file ./config/.env -f ./Dockerfiles/docker-compose.clickhouse.yml --project-directory . up -d
docker compose --env-file ./config/.env -f ./Dockerfiles/docker-compose.rabbitmq.yml --project-directory . up -d
docker compose --env-file ./config/.env -f ./Dockerfiles/docker-compose.s3.yml --project-directory . up -d
docker compose --env-file ./config/.env -f ./Dockerfiles/docker-compose.etcd.yml --project-directory . up -d
docker compose --env-file ./config/.env -f ./Dockerfiles/docker-compose.gitea.yml --project-directory . up -d
docker compose --env-file ./config/.env -f ./Dockerfiles/docker-compose.litellm.yml --project-directory . up -d
```

### 3. Verify Services

```bash
# Check all containers are running
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Test connectivity
docker exec marie-gitea nc -zv marie-psql-server 5432
docker exec marie-gitea nc -zv marie-rabbitmq 5672
docker exec marie-gitea nc -zv marie-s3-server 8000
```

---

## Service Details

### PostgreSQL (docker-compose.storage.yml)

DocumentDB-compatible PostgreSQL with extensions for document storage.

```bash
# Start
docker compose --env-file ./config/.env -f ./Dockerfiles/docker-compose.storage.yml --project-directory . up -d

# Connect
docker exec -it marie-psql-server psql -U postgres

# Create database for Gitea
docker exec marie-psql-server psql -U postgres -c "CREATE DATABASE gitea;"
```

### ClickHouse (docker-compose.clickhouse.yml)

Column-oriented analytics database for metrics and logging.

```bash
# Start
docker compose --env-file ./config/.env -f ./Dockerfiles/docker-compose.clickhouse.yml --project-directory . up -d

# Connect via CLI
docker exec -it marie-clickhouse clickhouse-client

# Test HTTP API
curl 'http://localhost:8123/?query=SELECT%20version()'
```

### RabbitMQ (docker-compose.rabbitmq.yml)

Message broker for async task processing.

```bash
# Start
docker compose --env-file ./config/.env -f ./Dockerfiles/docker-compose.rabbitmq.yml --project-directory . up -d

# Management UI: http://localhost:15672 (guest/guest)
```

### MinIO S3 (docker-compose.s3.yml)

S3-compatible object storage.

```bash
# Start
docker compose --env-file ./config/.env -f ./Dockerfiles/docker-compose.s3.yml --project-directory . up -d

# Console UI: http://localhost:8001
# S3 API: http://localhost:8000
```

### Gitea (docker-compose.gitea.yml)

Self-hosted Git service for marie-studio workflows.

**Prerequisites:** PostgreSQL must be running with `gitea` database created.

```bash
# Create gitea database first
docker exec marie-psql-server psql -U postgres -c "CREATE DATABASE gitea;"

# Start Gitea
docker compose --env-file ./config/.env -f ./Dockerfiles/docker-compose.gitea.yml --project-directory . up -d

# Web UI: http://localhost:3001
# SSH: ssh://git@localhost:2222
```

### LiteLLM (docker-compose.litellm.yml)

LLM proxy for routing requests to various AI providers.

```bash
# Start
docker compose --env-file ./config/.env -f ./Dockerfiles/docker-compose.litellm.yml --project-directory . up -d

# API: http://localhost:4000
```

---

## Environment Configuration

All services read configuration from `./config/.env`. Key variables:

```bash
# PostgreSQL
POSTGRES_USER=postgres
POSTGRES_PASSWORD=123456
POSTGRES_HOST=marie-psql-server      # Container hostname for internal access

# RabbitMQ
RABBIT_MQ_HOSTNAME=marie-rabbitmq    # Container hostname for internal access
RABBIT_MQ_USERNAME=guest
RABBIT_MQ_PASSWORD=guest

# S3/MinIO
S3_ENDPOINT_URL=http://marie-s3-server:8000  # Container hostname for internal access
MINIO_ROOT_USER=marieadmin
MINIO_ROOT_PASSWORD=marietopsecret

# ClickHouse
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
CLICKHOUSE_DB=marie

# Gitea
GITEA_HTTP_PORT=3001
GITEA_DB_NAME=gitea
```

**Important:** For container-to-container communication, use container names (e.g., `marie-psql-server`), not `localhost`.

---

## G5 Annotator Services with Docker Compose

### Overview

This documentation describes how to use the `docker-compose.g5-annotators.yml` file to deploy and manage the `G5 Annotator` and `G5 LLM Annotator` services. These services are essential components of the Marie-AI platform for advanced document analysis. This setup is designed for a production-like environment that leverages GPU acceleration.

### Prerequisites

Before you begin, ensure your system meets the following requirements:

1.  **System Requirements**:
    *   **Operating System**: Linux (x86\_64)
    *   **GPU**: An NVIDIA GPU with CUDA support is required as the services are configured to use a CUDA-enabled image.
    *   **Storage**: Sufficient disk space for Docker images and Marie-AI models (at least 100GB recommended).

2.  **Software Dependencies**:
    *   **Docker Engine**
    *   **Docker Compose**
    *   **NVIDIA Container Toolkit**: Required for GPU support in Docker.

3.  **Directory and Model Setup**:
    *   The Marie-AI repository should be cloned to your local machine.
    *   The required directory structure and symbolic links must be in place as described in the main `bootstrap.md` guide. Specifically, ensure the `/mnt/data/marie-ai` directory exists and contains the `config` and `model_zoo` directories.

### File Location

Place the `docker-compose.g5-annotators.yml` file inside the `Dockerfiles/` directory of your `marie-ai` project. This keeps it organized with the other Docker-related configuration files.

### How to Use

Follow these steps to manage the G5 annotator services:

#### 1. Starting the Services

To start both the `annotator-g5` and `annotator-g5-llm` services, open a terminal, navigate to the root of the `marie-ai` project, and run the following command:

```shell script
docker compose -f ./Dockerfiles/docker-compose.g5-annotators.yml up -d
```


*   The `-f` flag specifies the path to your compose file.
*   The `-d` flag runs the containers in detached mode, meaning they will run in the background.

#### 2. Monitoring the Services

You can view the logs for each service to monitor its status and check for any errors.

To view the logs for the G5 annotator:

```shell script
docker logs -f marie-annotator-g5-server
```


To view the logs for the G5 LLM annotator:

```shell script
docker logs -f marie-annotator-g5-llm-server
```


*   The `-f` flag follows the log output in real-time. Press `Ctrl+C` to exit.

To check the status of all running containers:

```shell script
docker ps
```


#### 3. Stopping the Services

To stop the G5 annotator services, run the following command from the project root:

```shell script
docker compose -f ./Dockerfiles/docker-compose.g5-annotators.yml down
```


This command will gracefully stop and remove the containers defined in the file.

***