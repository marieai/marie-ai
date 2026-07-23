# Marie-AI System Bootstrap Documentation

---

## Overview

The Marie-AI System Bootstrap script (`bootstrap-marie.sh`) is a comprehensive deployment automation tool designed to streamline the deployment of the Marie-AI document processing platform. This script orchestrates the deployment of both infrastructure services and application components in a containerized environment using Docker Compose.

### Key Features

- **Automated Infrastructure Deployment**: Deploys essential services including RabbitMQ, Valkey, MinIO, ETCD, and PostgreSQL
- **Application Service Management**: Manages Gateway and Extract Executor services
- **Kubernetes Smoke Bootstrap**: Delegates to the Helm-based local k3d/kind bootstrap, including optional Argo CD for sandboxes
- **Flexible Deployment Options**: Supports infrastructure-only, services-only, or complete deployments
- **Health Monitoring**: Includes comprehensive health checks and status reporting
- **Environment Validation**: Validates system requirements and configuration files
- **Cleanup Management**: Provides options for clean deployment and service cleanup

### System Components

| Component | Purpose | Port(s) |
|-----------|---------|---------|
| **Marie Gateway** | API Gateway and request routing | 51000 (gRPC), 52000 (HTTP) |
| **Extract Executor** | Document processing and OCR | 8080 |
| **RabbitMQ** | Message queue and task distribution | 15672 (Management), 5672 (AMQP) |
| **Valkey** | Shared request/reply store for queued LLM execution | 6379 |
| **MinIO** | S3-compatible object storage | 9000 (API), 9001 (Console) |
| **ETCD** | Service discovery and configuration | 2379 (Client), 2380 (Peer) |
| **PostgreSQL** | Document database | 5432 |

---

## Prerequisites

### System Requirements

- **Operating System**: Linux (x86_64)
- **Memory**: Minimum 8GB RAM (16GB recommended for production)
- **Storage**: Minimum 100GB free disk space

### Software Dependencies

- **Docker Engine**: >= 20.10.0
- **Docker Compose**: >= 2.0.0
- **kubectl**, **helm**, and **k3d** or **kind** for Kubernetes bootstrap mode

### Hardware Requirements

- **GPU Support**: NVIDIA GPU with CUDA support (for Extract Executor)
- **CPU**: Multi-core processor (4+ cores recommended)

### Verify Prerequisites

```bash
docker --version
docker compose version

# Check system resources
free -h
df -h

# Check GPU availability 
nvidia-smi
```

## Quick Start

### 1. Clone the Repository
First, clone the Marie-AI repository from the `develop` branch:

```shell
git clone -b develop git@github.com:marieai/marie-ai.git
cd marie-ai
```

### 2. Setup Directory Structure
```shell
# Create base directory
sudo mkdir -p /mnt/data/marie-ai
sudo chown $USER:$USER /mnt/data/marie-ai

# Create symbolic links
ln -sf ~/dev/marieai/marie-ai/config /mnt/data/marie-ai/config
ln -sf ~/dev/marieai/marie-ai/model_zoo /mnt/data/marie-ai/model_zoo
```

**Expected Directory Structure:**

```markdown
┌── /mnt/data/marie-ai   v3.12.3(marie-3.12)    
└─λ tree -d
.
├── config -> /home/greg/dev/marieai/marie-ai/config
└── model_zoo -> /home/greg/dev/marieai/marie-ai/model_zoo/
```

### 3. Download Required Models

At this point we need to download the required models for Extract(OCR/Bounding Boxes)
You can always use the default models provider by Microsoft from [https://github.com/microsoft/unilm](https://github.com/microsoft/unilm).

``` markdown
┌── marie-ai/config/zoo on  develop [@14 !10 +3 ?2 ]  v3.12.3(marie-3.12) 35 hours ago    
└─λ tree
.
└── unilm
    └── dit
        ├── object_detection
        │  └── document_boundary
        │      ├── Base-RCNN-FPN.yaml
        │      ├── cascade
        │      │  ├── cascade_dit_base.yaml
        │      │  └── cascade_dit_large.yaml
        │      ├── maskrcnn
        │      │  ├── maskrcnn_dit_base.yaml
        │      │  └── maskrcnn_dit_large.yaml
        │      └── prod.yaml -> ./maskrcnn/maskrcnn_dit_base.yaml
        └── text_detection
            ├── Base-RCNN-FPN.yaml
            ├── mask_rcnn_dit_base.yaml
            ├── mask_rcnn_dit_large.yaml
            └── mask_rcnn_dit_prod.yaml  (THIS IS WHAT APPLICATION USES)
```

I have my model in the following directory but you can always relocate-it, just update the file.
(All paths relative to `model_zoo` dir)

```shell
# View model configuration
cat /mnt/data/marie-ai/config/zoo/unilm/dit/text_detection/mask_rcnn_dit_prod.yaml
WEIGHTS: "unilm/dit/text_detection/tuned-4000-LARGE/model_final.pth"
```
After downloading models, you should have a directory that has at least the following:

```markdown
/mnt/data/marie-ai/model_zoo/trocr
/mnt/data/marie-ai/model_zoo/unilm
```

### 4. Deployment

Execute the bootstrap script:

```shell
./bootstrap-marie.sh --infrastructure-only --no-litellm
```
The system uses for configuration: `/mnt/data/marie-ai/config/.env.dev`

* Compose files in `./Dockerfiles/`:

  | File                          | Description                  |
  | ----------------------------- | ---------------------------- |
  | docker-compose.storage.yml    | Storage backend              |
  | docker-compose.monitoring.yml | Monitoring stack             |
  | docker-compose.s3.yml         | MinIO S3 storage             |
  | docker-compose.rabbitmq.yml   | RabbitMQ                     |
  | docker-compose.valkey.yml     | Valkey for queued LLM work   |
  | docker-compose.etcd.yml       | ETCD cluster                 |
  | docker-compose.gateway.yml    | Gateway API (optional)       |
  | docker-compose.extract.yml    | Extract Executors (optional) |

If you plan to enable queued `BatchProcessor` execution in the bootstrapped stack, add these environment variables to your deployment env file.

Gateway and processors do not play the same role:

- `marie-gateway`
  - owns the `LLM Dispatch Runtime`
  - auto-starts the Valkey-backed dispatcher when `LLM_QUEUE_ENABLED=true`
  - consumes queued requests and executes them against the configured OpenAI-compatible backend URL
  - owns dispatch-layer liveness, timeout, retry, circuit-breaker, and backpressure behavior
- extract / annotator processors
  - act as queue producers
  - submit canonical completion calls into Valkey
  - do not need to run the dispatcher thread themselves

The configured OpenAI-compatible backend may be LiteLLM, OpenRouter, vLLM, or a hosted provider endpoint. Provider fallback chains, model/provider routing, budgets, and provider rate limits should be configured in that backend gateway. Marie's LLM Dispatch Runtime is the executor ingress and dispatch lifecycle layer, not the provider-routing policy layer.

Minimum gateway configuration:

```shell
LLM_QUEUE_ENABLED=true
LLM_QUEUE_VALKEY_URL=redis://localhost:6379/0
LLM_QUEUE_POOL_ID=default
LLM_QUEUE_MAX_INLINE_PAYLOAD_BYTES=16777216
OPENAI_API_KEY=EMPTY
OPENAI_API_BASE=http://localhost:4000/v1
```

Minimum processor configuration:

```shell
LLM_QUEUE_ENABLED=true
LLM_QUEUE_VALKEY_URL=redis://localhost:6379/0
LLM_QUEUE_POOL_ID=default
LLM_QUEUE_MAX_INLINE_PAYLOAD_BYTES=16777216
OPENAI_API_KEY=EMPTY
OPENAI_API_BASE=http://localhost:4000/v1
```

Notes:

- For container-to-container networking, replace `localhost` with the service name, for example `redis://marie-valkey:6379/0`.
- `OPENAI_API_BASE` may also be provided as `OPENAI_BASE_URL`; point it at the provider gateway/backend that should execute the OpenAI-compatible request.
- The default inline payload limit is `16777216` bytes (`16 MiB`) because multimodal requests include serialized image content.
- If `LLM_QUEUE_ENABLED=true` on the gateway but `LLM_QUEUE_VALKEY_URL` or `OPENAI_API_KEY` is missing, the gateway now fails fast at startup instead of booting without a consumer runtime.
- Runtime Fabric-scoped history requires the gateway dispatcher to set `LLM_QUEUE_FABRIC_GROUP_ID` and/or `LLM_QUEUE_GATEWAY_ID`; processors do not need these unless they also run a dispatcher.

---

Execute the service bootstrap script:

```shell
./bootstrap-marie.sh --services-only
```

### Kubernetes + Argo CD

The sandbox/snapshot control plane uses Argo CD as the GitOps reconciler. For a local k3d or kind cluster,
run the Kubernetes bootstrap mode from the same top-level script:

```shell
./bootstrap-marie.sh --k8s
```

Install Marie and Argo CD together for sandbox smoke testing:

```shell
./bootstrap-marie.sh --with-argocd
```

Use kind or pin an Argo CD version when needed:

```shell
./bootstrap-marie.sh --k8s --k8s-provider kind
./bootstrap-marie.sh --with-argocd --argocd-version v2.13.0
```

This delegates to `deploy/bootstrap.sh`, which installs the Marie Helm chart and, when enabled, installs
Argo CD into `argocd` by default. Before any cluster changes, the script prints a Kubernetes deployment
configuration summary covering the provider, cluster, namespace, Helm chart, images, smoke options, and
Argo CD settings.

## Usage

```markdown
========================================
    Marie-AI System Bootstrap
========================================
Unknown option --
Usage: ./bootstrap-marie.sh [options]

Options:
  --k8s, --helm         Bootstrap Marie-AI on local Kubernetes via deploy/bootstrap.sh
  --k8s-provider NAME   Kubernetes provider for --k8s: k3d or kind (default: k3d)
  --with-argocd         Bootstrap Kubernetes and install Argo CD for sandboxes
  --no-argocd           Disable Argo CD install when INSTALL_ARGOCD=true is set
  --argocd-namespace NS Argo CD namespace for --with-argocd (default: argocd)
  --argocd-version TAG  Argo CD install manifest tag/channel (default: stable)
  --verify              Verify the currently running Compose stack and exit
  --no-verify           Skip post-bootstrap verification
  --stop-all            Stop and remove all Marie-AI services and containers
  --no-gateway          Skip gateway deployment
  --no-extract          Skip extract executor deployment
  --no-infrastructure   Skip infrastructure services (includes LiteLLM and Valkey)
  --no-litellm          Skip LiteLLM proxy deployment
  --infrastructure-only Deploy only infrastructure services (includes LiteLLM and Valkey)
  --services-only       Deploy only Marie application services (gateway + extract)
  --litellm-only        Deploy only LiteLLM proxy (with required infrastructure)
  -h, --help           Show this help message

Service Categories:
  Infrastructure: Storage, Message Queue, LLM Queue Store, Service Discovery, LLM Proxy
  Application:    Gateway, Extract Executors

Examples:
  ./bootstrap-marie.sh                    # Deploy everything
  ./bootstrap-marie.sh --k8s              # Bootstrap local k3d cluster with Marie Helm chart
  ./bootstrap-marie.sh --with-argocd      # Bootstrap local k3d cluster with Marie + Argo CD
  ./bootstrap-marie.sh --verify           # Verify existing containers and login-capable UIs
  ./bootstrap-marie.sh --stop-all         # Stop all services and cleanup
  ./bootstrap-marie.sh --infrastructure-only  # Deploy infrastructure + LiteLLM
  ./bootstrap-marie.sh --services-only        # Deploy only gateway + extract
  ./bootstrap-marie.sh --no-extract           # Deploy infrastructure + gateway only
  ./bootstrap-marie.sh --litellm-only         # Deploy minimal infrastructure + LiteLLM
```

### Examples

* **Deploy everything**:

  ```bash
  ./bootstrap.sh
  ```

* **Deploy only infrastructure**:

  ```bash
  ./bootstrap.sh --infrastructure-only
  ```

* **Deploy only services**:

  ```bash
  ./bootstrap.sh --services-only
  ```

---

## Service Endpoints

| Service              | URL                                                              |
| -------------------- | ---------------------------------------------------------------- |
| RabbitMQ Management  | [http://localhost:15672](http://localhost:15672) (configured env credentials; default `marie/mariepassword`) |
| Valkey LLM Queue     | `redis://localhost:6379/0`                                       |
| MinIO S3 API         | [http://localhost:8000](http://localhost:8000)                   |
| MinIO Console        | [http://localhost:8001](http://localhost:8001)                   |
| Monitoring (Grafana) | [http://localhost:3000](http://localhost:3000)                   |
| HTTP Gateway         | [http://localhost:51000](http://localhost:51000)                 |
| GRPC Gateway         | grpc://localhost:52000                                           |
| Extract Executor     | [http://localhost:8080](http://localhost:8080)                   |

> **Note:** Services depend on deployment options.

---

##  Installation and output (Infrastructure)

```markdown
./bootstrap-marie.sh --infrastructure-only --no-litellm
========================================
    Marie-AI System Bootstrap
========================================
Deployment Configuration:
  Infrastructure: true
    ├── Storage (MinIO): true
    ├── Message Queue (RabbitMQ): true
    ├── LLM Queue Store (Valkey): true
    ├── Service Discovery (etcd): true
    └── LLM Proxy (LiteLLM): false
  Application Services:
    ├── Gateway: false
    └── Extract Executors: false

✅ Environment file found: ./config/.env.dev
✅ All required compose files found.

Starting Marie-AI system bootstrap...
✅ Environment loaded from ./config/.env.dev
🔧 Stage 1: Starting infrastructure services...
Starting infrastructure services with host networking...
[+] Running 15/15
 ✔ Volume "marie-infrastructure_rabbitmq_data"                            Created                                                                                                                                                        0.0s 
 ✔ Volume "marie-infrastructure_rabbitmq_log"                             Created                                                                                                                                                        0.0s 
 ✔ Volume "marie-infrastructure_valkey_data"                              Created                                                                                                                                                        0.0s 
 ✔ Volume "marie-infrastructure_etcd_data"                                Created                                                                                                                                                        0.0s 
 ✔ Volume "marie-infrastructure_psql_data"                                Created                                                                                                                                                        0.0s 
 ✔ Volume "marie-infrastructure_mc-config"                                Created                                                                                                                                                        0.0s 
 ✔ Container marie-rabbitmq                                               Started                                                                                                                                                        0.4s 
 ✔ Container marie-valkey                                                 Started                                                                                                                                                        0.4s 
 ✔ Container marie-psql-server                                            Started                                                                                                                                                        0.4s 
 ✔ Container etcd-single                                                  Started                                                                                                                                                        0.4s 
 ✔ Container marie-s3-server                                              Healthy                                                                                                                                                       30.9s 
 ! etcd-single Published ports are discarded when using host network mode                                                                                                                                                                0.0s 
 ! s3server Published ports are discarded when using host network mode                                                                                                                                                                   0.0s 
 ✔ Container marie-mc-setup                                               Started                                                                                                                                                       30.9s 
 ! rabbitmq Published ports are discarded when using host network mode                                                                                                                                                                   0.0s 
 ! psql Published ports are discarded when using host network mode                                                                                                                                                                       0.0s 
⏳ Waiting for infrastructure services to be healthy (excluding setup containers)...
[+] Running 5/5
 ✔ Container etcd-single        Healthy                                                                                                                                                                                                  0.5s 
 ✔ Container marie-s3-server    Healthy                                                                                                                                                                                                  0.5s 
 ✔ Container marie-rabbitmq     Healthy                                                                                                                                                                                                  0.5s 
 ✔ Container marie-valkey       Healthy                                                                                                                                                                                                  0.5s 
 ✔ Container marie-psql-server  Healthy                                                                                                                                                                                                  0.5s 
Checking MinIO setup completion...
✅ MinIO setup completed successfully
✅ Infrastructure services are ready
🚀 Stage 2: Starting application services...
No application services configured to start

🎉 Marie-AI system started successfully!

Services status:
Infrastructure Services:
NAME                IMAGE                                             COMMAND                  SERVICE       CREATED          STATUS                    PORTS
etcd-single         quay.io/coreos/etcd:v3.7.0                        "/usr/local/bin/etcd…"   etcd-single   31 seconds ago   Up 31 seconds (healthy)
marie-psql-server   ghcr.io/ferretdb/postgres-documentdb:17-0.103.0   "docker-entrypoint.s…"   psql          31 seconds ago   Up 31 seconds             
marie-rabbitmq      rabbitmq:3-management-alpine                      "docker-entrypoint.s…"   rabbitmq      31 seconds ago   Up 31 seconds             
marie-s3-server     minio/minio:latest                                "/usr/bin/docker-ent…"   s3server      31 seconds ago   Up 31 seconds (healthy)   
marie-valkey        valkey/valkey:8-alpine                            "docker-entrypoint.s…"   valkey        31 seconds ago   Up 31 seconds (healthy)   

🔗 Service Endpoints:
Infrastructure Services:
  🐰 RabbitMQ Management: http://localhost:15672 (guest/guest)
  🧠 Valkey LLM Queue: redis://localhost:6379/0
  💾 MinIO S3 API: http://localhost:9000 (marieadmin/marietopsecret)
  💾 MinIO Console: http://localhost:9001 (marieadmin/marietopsecret)
  📊 Monitoring: http://localhost:3000
  🗄️  etcd: http://localhost:2379

========================================
Bootstrap completed successfully!
========================================
```


## Services Installation and Output
```markdown
┌── marie-ai on  develop [@14 !8 +3 ?2 ] ⬢ v16.16.0   v3.12.3(marie-3.12) 34 hours ago    
└─λ ./bootstrap-marie.sh --services-only
========================================
    Marie-AI System Bootstrap
========================================
Deployment Configuration:
  Infrastructure: false
    ├── Storage (MinIO): false
    ├── Message Queue (RabbitMQ): false
    ├── LLM Queue Store (Valkey): false
    ├── Service Discovery (etcd): false
    └── LLM Proxy (LiteLLM): false
  Application Services:
    ├── Gateway: true
    └── Extract Executors: true

✅ Environment file found: ./config/.env.dev
✅ All required compose files found.

Starting Marie-AI system bootstrap...
✅ Environment loaded from ./config/.env.dev
🚀 Stage 2: Starting application services...
Starting application services with host networking...
[+] Running 2/2
 ✔ Container marieai-dev-server  Started                                                                                                                                                                                                 0.6s 
 ✔ Container marieai-gateway     Started                                                                                                                                                                                                 0.3s 

🎉 Marie-AI system started successfully!

Services status:

Application Services:
NAME                 IMAGE                             COMMAND                  SERVICE                  CREATED        STATUS                                     PORTS
marieai-dev-server   marieai/marie:5.0.0-cuda          "marie server --star…"   marie-extract-executor   1 second ago   Up Less than a second
marieai-gateway      marieai/marie-gateway:5.0.0-cpu   "marie gateway --use…"   marie-gateway            1 second ago   Up Less than a second (health: starting)

🔗 Service Endpoints:
Application Services:
  🌐 HTTP Gateway: http://localhost:51000
  🔌 GRPC Gateway: grpc://localhost:52000
  🔍 Extract Executor: http://localhost:8080

========================================
Bootstrap completed successfully!
========================================
```

## Verify running containers (Expected at least to have following)

```markdown
┌── marie-ai on  develop [@14 !10 +3 ?2 ] ⬢ v16.16.0   v3.12.3(marie-3.12) 34 hours ago    
└─λ docker ps
CONTAINER ID   IMAGE                                             COMMAND                  CREATED              STATUS                        PORTS     NAMES
7c999f7e4b00   marieai/marie:5.0.0-cuda                          "marie server --star…"   About a minute ago   Up About a minute                       marieai-dev-server
31acc1cc4ec0   marieai/marie-gateway:5.0.0-cpu                   "marie gateway --use…"   About a minute ago   Up About a minute (healthy)             marieai-gateway
60921ce11677   ghcr.io/ferretdb/postgres-documentdb:17-0.103.0   "docker-entrypoint.s…"   20 minutes ago       Up 20 minutes                           marie-psql-server
afe2b9aad84c   minio/minio:latest                                "/usr/bin/docker-ent…"   20 minutes ago       Up 20 minutes (healthy)                 marie-s3-server
5a4f81dcf644   rabbitmq:3-management-alpine                      "docker-entrypoint.s…"   20 minutes ago       Up 20 minutes                           marie-rabbitmq
db5950b54663   quay.io/coreos/etcd:v3.7.0                        "/usr/local/bin/etcd…"   20 minutes ago       Up 20 minutes (healthy)                 etcd-single
```

## Gateway verification
```shell
docker logs marieai-gateway  --follow
```

```markdown
INFO   gateway@ 7 Setting up MarieServerGateway                                                                                                                  [07/09/25 09:03:17]
INFO   marie@ 7 Loading env file from /etc/marie/config/.env                                                                                                     [07/09/25 09:03:17]
INFO   gateway@ 7 Debugging information:                                                                                                                                            
INFO   gateway@ 7 __model_path__ = /etc/marie/model_zoo                                                                                                                             
INFO   gateway@ 7 __config_dir__ = /etc/marie/config                                                                                                                                
INFO   gateway@ 7 __marie_home__ = /root/.marie                                                                                                                                     
INFO   gateway@ 7 __cache_path__ = /root/.cache/marie                                                                                                                               
INFO   gateway@ 7 yml_config = /etc/marie/config/service/extract/marie-gateway-4.0.0.yml                                                                                            
INFO   gateway@ 7 env_file = /etc/marie/config/.env                                                                                                                                 
...
INFO   gateway@ 7 Gateway started                                                                                                                                [07/09/25 09:03:17]
INFO   gateway@ 7 Waiting for ready_event with a timeout of 5 seconds                                                                                                               
INFO   gateway@ 7 Time remaining: 5 seconds                                                                                                                                         
INFO   gateway@ 7 Time remaining: 4 seconds                                                                                                                      [07/09/25 09:03:18]
INFO   gateway@ 7 Time remaining: 3 seconds                                                                                                                      [07/09/25 09:03:19]
INFO   gateway@ 7 Time remaining: 2 seconds                                                                                                                      [07/09/25 09:03:20]
INFO   gateway@ 7 Time remaining: 1 seconds                                                                                                                      [07/09/25 09:03:21]
WARNI… gateway@ 7 Timeout waiting for ready_event, starting scheduler anyway                                                                                     [07/09/25 09:03:22]
INFO   marie@ 7 Starting job scheduling agent                                                                                                                    [07/09/25 09:03:22]
INFO   marie@ 7 Tables installed: None                                                                                                                                              
INFO   marie@ 7 Wrote locked query to: /tmp/marie/psql/locked_query_20250709_090322.sql                                                                          [07/09/25 09:03:22]
INFO   marie@ 7 Create queue: gen5_extract                                                                                                                                          
INFO   marie@ 7 Create queue: extract                                                                                                                                               
INFO   marie@ 7 Create queue: classify                                                                                                                                              
INFO   marie@ 7 Create queue: load                                                                                                                                                  
INFO   marie@ 7 Create queue: transform                                                                                                                                             
INFO   marie@ 7 🔄  Scheduler Heartbeat                                                                                                                                             
INFO   marie@ 7   🧭  Mode              : serial                                                                                                                                    
INFO   marie@ 7   📦  Queue Size        : 1                                                                                                                                         
INFO   marie@ 7   ⚙️   Available Slots                                                                                                                                               
     ⚙️  Available Slots     
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Slot Type        ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ extract_executor │   1   │
└──────────────────┴───────┘
INFO   marie@ 7   🧠  Active DAGs        : 0                                                                                                                                        
                                                                 📊 Consolidated Job States for All Queues                                                                  
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Queue            ┃     Created      ┃      Retry       ┃      Active      ┃    Completed     ┃     Expired      ┃    Cancelled     ┃      Failed      ┃       All        ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ No Data          │        0         │        0         │        0         │        0         │        0         │        0         │        0         │        0         │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

## Extract Executor Verification

```shell
docker logs marieai-dev-server --follow
```

```markdown

 ██████   ██████   █████████   ███████████   █████ ██████████              █████████   █████      /\   /\   
░░██████ ██████   ███░░░░░███ ░░███░░░░░███ ░░███ ░░███░░░░░█             ███░░░░░███ ░░███      //\\_//\\     ____
 ░███░█████░███  ░███    ░███  ░███    ░███  ░███  ░███  █ ░             ░███    ░███  ░███      \_     _/    /   /
 ░███░░███ ░███  ░███████████  ░██████████   ░███  ░██████    ██████████ ░███████████  ░███       / * * \    /^^^]
 ░███ ░░░  ░███  ░███░░░░░███  ░███░░░░░███  ░███  ░███░░█   ░░░░░░░░░░  ░███░░░░░███  ░███       \_\O/_/    [   ] 
 ░███      ░███  ░███    ░███  ░███    ░███  ░███  ░███ ░   █            ░███    ░███  ░███        /   \_    [   /
 █████     █████ █████   █████ █████   █████ █████ ██████████            █████   █████ █████       \     \_  /  /
░░░░░     ░░░░░ ░░░░░   ░░░░░ ░░░░░   ░░░░░ ░░░░░ ░░░░░░░░░░            ░░░░░   ░░░░░ ░░░░░        [ [ /  \/ _/


/opt/venv/bin/marie server --start --uses                               
/etc/marie/config/service/extract/marie-extract-4.0.0.yml               
╭──────────┬───────────────────────────────────────────────────────────╮
│ Argument │ Value                                                     │
├──────────┼───────────────────────────────────────────────────────────┤
│      cli │ server                                                    │
│  ctl-cli │ None                                                      │
│      env │ None                                                      │
│ env-file │ None                                                      │
│    purge │ False                                                     │
│    start │ True                                                      │
│   status │ all                                                       │
│     uses │ /etc/marie/config/service/extract/marie-extract-4.0.0.yml │
╰──────────┴───────────────────────────────────────────────────────────╯
INFO   marie@ 7 Starting marie server : 4.0.0                                                                                                                    [07/09/25 09:03:17]
INFO   marie@ 7 Debugging information:                                                                                                                                              
INFO   marie@ 7 __model_path__ = /etc/marie/model_zoo                                                                                                                               
INFO   marie@ 7 __config_dir__ = /etc/marie/config                                                                                                                                  
INFO   marie@ 7 __marie_home__ = /root/.marie                                                                                                                                       
INFO   marie@ 7 __cache_path__ = /root/.cache/marie                                                                                                                                 
INFO   marie@ 7 yml_config = /etc/marie/config/service/extract/marie-extract-4.0.0.yml                                                                                              
INFO   marie@ 7 env = None                                                                                                                                                          
INFO   marie@ 7 CONTEXT.gpu_device_count = 1                                                                                                                                        
INFO   marie@ 7 Loading env file from /etc/marie/config/.env  


──────────────────────────────────────────────────────────────────────────── 🎉 Flow is ready to serve! ────────────────────────────────────────────────────────────────────────────
╭────────────── 🔗 Endpoint ───────────────╮
│  ⛓      Protocol                   GRPC  │
│  🏠        Local          0.0.0.0:49330  │
│  🔒      Private     192.168.1.21:49330  │
│  🌍       Public    72.198.17.215:49330  │
╰──────────────────────────────────────────╯
╭──────────── 💎 Deployment Nodes ────────────╮
│  🔒  extract_executor/rep-0  0.0.0.0:56842  │
│  🔒           gateway/rep-0  0.0.0.0:49330  │
╰─────────────────────────────────────────────╯
INFO   marie@ 7 Setting up service discovery ETCD ...                                                                                                            [07/09/25 09:03:24]
INFO   marie@ 7 Deployments addresses: {'extract_executor': ['grpc://0.0.0.0:56842']}                                                                                               
INFO   marie@ 7 Deployments ctrl_address: 192.168.1.21:49330                                                                                                                        
```
