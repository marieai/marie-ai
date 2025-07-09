# Marie-AI System Bootstrap Documentation

---

## Overview

The Marie-AI System Bootstrap script (`bootstrap-marie.sh`) is a comprehensive deployment automation tool designed to streamline the deployment of the Marie-AI document processing platform. This script orchestrates the deployment of both infrastructure services and application components in a containerized environment using Docker Compose.

### Key Features

- **Automated Infrastructure Deployment**: Deploys essential services including RabbitMQ, MinIO, ETCD, and PostgreSQL
- **Application Service Management**: Manages Gateway and Extract Executor services
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
  | docker-compose.etcd.yml       | ETCD cluster                 |
  | docker-compose.gateway.yml    | Gateway API (optional)       |
  | docker-compose.extract.yml    | Extract Executors (optional) |

---

Execute the service bootstrap script:

```shell
./bootstrap-marie.sh --services-only
```

## Usage

```markdown
========================================
    Marie-AI System Bootstrap
========================================
Unknown option --
Usage: ./bootstrap-marie.sh [options]

Options:
  --stop-all            Stop and remove all Marie-AI services and containers
  --no-gateway          Skip gateway deployment
  --no-extract          Skip extract executor deployment
  --no-infrastructure   Skip infrastructure services (includes LiteLLM)
  --no-litellm          Skip LiteLLM proxy deployment
  --infrastructure-only Deploy only infrastructure services (includes LiteLLM)
  --services-only       Deploy only Marie application services (gateway + extract)
  --litellm-only        Deploy only LiteLLM proxy (with required infrastructure)
  -h, --help           Show this help message

Service Categories:
  Infrastructure: Storage, Message Queue, Service Discovery, LLM Proxy
  Application:    Gateway, Extract Executors

Examples:
  ./bootstrap-marie.sh                    # Deploy everything
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
| RabbitMQ Management  | [http://localhost:15672](http://localhost:15672) (`guest/guest`) |
| MinIO Console        | [http://localhost:8001](http://localhost:8001)                   |
| Monitoring (Grafana) | [http://localhost:3000](http://localhost:3000)                   |
| HTTP Gateway         | [http://localhost:52000](http://localhost:52000)                 |
| GRPC Gateway         | grpc://localhost:51000                                           |
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
[+] Running 14/14
 ✔ Volume "marie-infrastructure_rabbitmq_data"                            Created                                                                                                                                                        0.0s 
 ✔ Volume "marie-infrastructure_rabbitmq_log"                             Created                                                                                                                                                        0.0s 
 ✔ Volume "marie-infrastructure_etcd_data"                                Created                                                                                                                                                        0.0s 
 ✔ Volume "marie-infrastructure_psql_data"                                Created                                                                                                                                                        0.0s 
 ✔ Volume "marie-infrastructure_mc-config"                                Created                                                                                                                                                        0.0s 
 ✔ Container marie-rabbitmq                                               Started                                                                                                                                                        0.4s 
 ✔ Container marie-psql-server                                            Started                                                                                                                                                        0.4s 
 ✔ Container etcd-single                                                  Started                                                                                                                                                        0.4s 
 ✔ Container marie-s3-server                                              Healthy                                                                                                                                                       30.9s 
 ! etcd-single Published ports are discarded when using host network mode                                                                                                                                                                0.0s 
 ! s3server Published ports are discarded when using host network mode                                                                                                                                                                   0.0s 
 ✔ Container marie-mc-setup                                               Started                                                                                                                                                       30.9s 
 ! rabbitmq Published ports are discarded when using host network mode                                                                                                                                                                   0.0s 
 ! psql Published ports are discarded when using host network mode                                                                                                                                                                       0.0s 
⏳ Waiting for infrastructure services to be healthy (excluding setup containers)...
[+] Running 4/4
 ✔ Container etcd-single        Healthy                                                                                                                                                                                                  0.5s 
 ✔ Container marie-s3-server    Healthy                                                                                                                                                                                                  0.5s 
 ✔ Container marie-rabbitmq     Healthy                                                                                                                                                                                                  0.5s 
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
etcd-single         quay.io/coreos/etcd:v3.6.1                        "/usr/local/bin/etcd…"   etcd-single   31 seconds ago   Up 31 seconds (healthy)   
marie-psql-server   ghcr.io/ferretdb/postgres-documentdb:17-0.103.0   "docker-entrypoint.s…"   psql          31 seconds ago   Up 31 seconds             
marie-rabbitmq      rabbitmq:3-management-alpine                      "docker-entrypoint.s…"   rabbitmq      31 seconds ago   Up 31 seconds             
marie-s3-server     minio/minio:latest                                "/usr/bin/docker-ent…"   s3server      31 seconds ago   Up 31 seconds (healthy)   

🔗 Service Endpoints:
Infrastructure Services:
  🐰 RabbitMQ Management: http://localhost:15672 (guest/guest)
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
marieai-dev-server   marieai/marie:4.0.0-cuda          "marie server --star…"   marie-extract-executor   1 second ago   Up Less than a second                      
marieai-gateway      marieai/marie-gateway:4.0.0-cpu   "marie gateway --use…"   marie-gateway            1 second ago   Up Less than a second (health: starting)   

🔗 Service Endpoints:
Application Services:
  🌐 HTTP Gateway: http://localhost:52000
  🔌 GRPC Gateway: grpc://localhost:51000
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
7c999f7e4b00   marieai/marie:4.0.0-cuda                          "marie server --star…"   About a minute ago   Up About a minute                       marieai-dev-server
31acc1cc4ec0   marieai/marie-gateway:4.0.0-cpu                   "marie gateway --use…"   About a minute ago   Up About a minute (healthy)             marieai-gateway
60921ce11677   ghcr.io/ferretdb/postgres-documentdb:17-0.103.0   "docker-entrypoint.s…"   20 minutes ago       Up 20 minutes                           marie-psql-server
afe2b9aad84c   minio/minio:latest                                "/usr/bin/docker-ent…"   20 minutes ago       Up 20 minutes (healthy)                 marie-s3-server
5a4f81dcf644   rabbitmq:3-management-alpine                      "docker-entrypoint.s…"   20 minutes ago       Up 20 minutes                           marie-rabbitmq
db5950b54663   quay.io/coreos/etcd:v3.6.1                        "/usr/local/bin/etcd…"   20 minutes ago       Up 20 minutes (healthy)                 etcd-single
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
