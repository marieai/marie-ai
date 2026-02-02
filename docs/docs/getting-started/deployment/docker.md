---
sidebar_position: 2
---

# Docker - Single node
Deployment single node via docker container

## User and permission setup
The container is setup with `app-svc` account so for that we will setup same account in the host system.

Setting up user, for more info visit [Manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)

```shell
sudo groupadd -r app-svc -g 433
sudo useradd -u 431 --comment 'app-svc' --create-home app-svc  --shell /usr/sbin/nologin
sudo usermod -aG docker app-svc
```

Directory structure, this is more of a convention than requirement.

```shell
sudo mkdir -p /opt/containers/apps/marie-ai
sudo mkdir -p /opt/containers/config/marie-ai
```

Change permissions to `app` and `config`

```shell
cd /opt/containers
sudo chown app-svc:app-svc apps/ -R
sudo chown app-svc:app-svc config/ -R
```

Now that we have our directory and permissions setup we can move on and setup the container.
The easiest way to manage container is by checking out the [Marie-AI project](https://github.com/gregbugaj/marie-ai.git)

```sh
sudo su app-svc

git clone https://github.com/gregbugaj/marie-ai.git
```

## Basic Configuration

:::warning Container Configuration

This is a single node setup up without control-plane.

Complete configuration setup can be found under [Configuration](/docs/category/configuration).

:::

Before we are able to start the container we need to configure few components. 


### Directory layout
This is the initial entrypoint for `models` and `config`, this can be changed by modifying `run.sh` Directories will be mapped as volumes when the container is created.

***TODO : this needs to be configurable via ENV variables***

```shell
/mnt/data/marie-a
/opt/logs/marie-icr
```


#### Single-GPU

On a single GPU the directory structure should loook like following

```
/mnt/data/marie-ai/
├── config
│   ├── consul
│   ├── executors
│   ├── ocr
│   └── traefik
│       └── log
└── model_zoo
```

#### Multi-GPU 

If the system support muli-gpus then we can configure the system to take advanate of that. Configuration is almost identical to single gpu. 
For each GPU we adappend GPU-ID to the config directory names

```
/mnt/data/marie-ai/
├── config-0
│   ├── consul
│   ├── ocr
│   └── traefik
├── config-1
│   ├── consul
│   ├── ocr
│   └── traefik
├── config-2
│   ├── consul
│   ├── ocr
│   └── traefik
├── config-3
│   ├── consul
│   ├── ocr
│   └── traefik
├── config-template
│   ├── consul
│   ├── ocr
│   └── traefik
└── model_zoo
```

### Required configuration

The easiest way to initialize the configs is by copying them from the project `config` directory.

```
/mnt/data/marie-ai/config/marie.yml
```

## Application Logs with Docker
There are couple ways that we can log from within a container to outside world.

### Named volumes
First option(default) is to use a `docker volume` and create new named volume called `marie_logs`

:::info

This is preferred option for single node setups.

:::

```shell
sudo mkdir -p /var/log/marie-ai
sudo chown app-svc:app-svc /var/log/marie-ai -R

docker volume create --driver local --name marie_logs --opt type=none --opt device=/var/log/marie-ai --opt o=uid=root,gid=root --opt o=bind
```

When we list the docker volumes we should see `marie_logs` in the output.

```shell
$ docker volume ls
DRIVER    VOLUME NAME
local     marie_cache
local     marie_logs
```

After the volume is created it can be mapped in the docker with the docker `-v` flag.
Application will log by default to `/home/app-svc/logs` directory, this can be changed by modifying `resources/logging.default.yml`

```shell
-v marie_logs:/home/app-svc/logs
```

We can inspect the volume and display the mount points via `docker volume inspect marie_logs`.

```shell
$ docker volume inspect marie_logs 
[
    {
        "CreatedAt": "2022-10-28T23:11:05Z",
        "Driver": "local",
        "Labels": {},
        "Mountpoint": "/var/lib/docker/volumes/marie_logs/_data",
        "Name": "marie_logs",
        "Options": {
            "device": "/var/log/marie-ai",
            "o": "bind",
            "type": "none"
        },
        "Scope": "local"
    }
]

```

```shell
$ ls -lt /var/log/marie-ai/*
-rw------- 1 root root    66 Oct 31 16:22 /var/log/marie-ai/ssh-agent-stderr---supervisor-ee3f8v_8.log
-rw-r--r-- 1  431  433 45654 Oct 31 16:22 /var/log/marie-ai/marie-2022-10-31T16:17:53.299405.log
-rw------- 1 root root    75 Oct 31 16:17 /var/log/marie-ai/ssh-agent-stdout---supervisor-gann9r5z.log
```

### Bind Mount

:::info

Preferred method for multi-node setup. This allows for better log segregation by GPU.

:::

### Permission setup : UID/GID
`uid` is a number associated with a user account and `gid` is a number associated with a group
The log directory needs to have the UID/GID of `431:433`

```shell
$ id
uid=431(app-svc) gid=433(app-svc) groups=433(app-svc),998(docker)
```

IDs are coming the user setup.

`431` : User ID given to `app-svc` account
`433` : Group ID given to `app-svc` account

```shell
sudo mkdir -p /var/log/marie-ai
sudo chown app-svc:app-svc /var/log/marie-ai -R
```

On a 4 GPU system the log directory should look like following, this includes couple log files.

```shell
$ tree /var/log/marie-ai
/var/log/marie-ai/
├── 0
│   ├── marie-2022-10-31T19:55:40.712174.log
├── 1
├── 2
├── 3
│   ├── marie-2022-10-31T19:59:40.286707.log
└── all
```

After the directory s created it can be mapped in the docker with the docker `-v` flag.
Application will log by default to `/home/app-svc/logs` directory, this can be changed by modifying `resources/logging.default.yml`

The `$GPUS` variable will be set from shell script while executed.

```shell
-v /var/log/marie-ai/$GPUS:/home/app-svc/logs
```

## Container management

To manage a container we have a set of scripts located in `marie-ai/docker-util`

:::info Changing container version

To change container version edit `id` file, it should contain a specific version that we like to use [Docker Registry](https://docs.docker.com/registry/).

```
gregbugaj/marie-icr:2.4-cuda
```

This could also include your specific registry as well.

```
localhost:5000/gregbugaj/marie-icr:2.4-cuda
```

:::


### Update container to specific version

```shell
cd marie-ai/docker-util
./update.sh
```

### Starting container

Start single `marie-ai` container in interactive mode with **ALL GPUS** assigned to single instance

```
./run-interactive-gpu.sh
```

You should see initial output :

```
Starting interactive/dev container : marie-icr
GPU_COUNT : 4
GPUS Selected  > all
CONTAINER_NAME > marie-icr
CONTAINER_ID   > 
CONFIG         > config
PORT           > 6000
CONFIG_DIR     > /mnt/data/marie-ai/config
```

### Manually starting container

Console mode

```shell
docker run --gpus all --rm -it --name=marieai --network=host \
  -e JINA_LOG_LEVEL=debug \
  -e MARIE_DEFAULT_MOUNT='/etc/marie' \
  -v /mnt/data/marie-ai/config:/etc/marie/config:ro \
  -v /mnt/data/marie-ai/model_zoo:/etc/marie/model_zoo:rw \
  marieai/marie:4.0.0-cuda server --uses /etc/marie/config/service/marie.yml

```

Daemon mode
```shell
docker run --gpus all --name=marieai -d --network=host \
  -e JINA_LOG_LEVEL=debug \
  -e MARIE_DEFAULT_MOUNT='/etc/marie' \
  -v /mnt/data/marie-ai/config:/etc/marie/config:ro \
  -v /mnt/data/marie-ai/model_zoo:/etc/marie/model_zoo:rw \
  marieai/marie:4.0.0-cuda server --uses /etc/marie/config/service/marie.yml
```

## Docker Compose Deployments

Marie-AI provides modular Docker Compose files for running infrastructure services. These are located in `Dockerfiles/` directory.

:::tip Isolated Testing with Vagrant
To test Docker Compose deployments without affecting your local environment, use the [Vagrant testing environment](./vagrant-testing.md). This creates an isolated VM with all dependencies pre-installed.

```bash
# Deploy in isolated VM
./bootstrap-marie.sh --vagrant

# SSH into VM for debugging
./bootstrap-marie.sh --vagrant-ssh
```
:::

### Available Compose Files

| File | Services | Description |
|------|----------|-------------|
| `docker-compose.yml` | Prometheus, Grafana, Loki, RabbitMQ | Core monitoring stack |
| `docker-compose.storage.yml` | PostgreSQL | Document database (FerretDB) |
| `docker-compose.s3.yml` | MinIO | S3-compatible storage |
| `docker-compose.etcd.yml` | etcd | Service discovery |
| `docker-compose.clickhouse.yml` | ClickHouse | Analytics database (marie-ai) |
| `docker-compose.gitea.yml` | Gitea | Git service (marie-studio) |

### Quick Start

```bash
# Create the docker network (required for all services)
docker network create --driver=bridge marie_default

# Start PostgreSQL (shared by all services)
docker compose --env-file ./config/.env \
  -f ./Dockerfiles/docker-compose.storage.yml \
  --project-directory . up -d

# Start ClickHouse (analytics)
docker compose --env-file ./config/.env \
  -f ./Dockerfiles/docker-compose.clickhouse.yml \
  --project-directory . up -d

# Start Gitea (Git service)
docker compose --env-file ./config/.env \
  -f ./Dockerfiles/docker-compose.gitea.yml \
  --project-directory . up -d
```

### ClickHouse

ClickHouse provides high-performance columnar analytics for marie-ai metrics and job analytics.

```bash
# Start ClickHouse
docker compose --env-file ./config/.env \
  -f ./Dockerfiles/docker-compose.clickhouse.yml \
  --project-directory . up -d

# View logs
docker logs -f marie-clickhouse

# Connect via CLI
docker exec -it marie-clickhouse clickhouse-client

# Test HTTP API
curl 'http://localhost:8123/?query=SELECT%20version()'

# Access Play UI
# Open http://localhost:8123/play
```

| Port | Protocol | Description |
|------|----------|-------------|
| 8123 | HTTP | HTTP API and Play UI |
| 9000 | TCP | Native protocol |
| 9004 | TCP | MySQL wire protocol |

### Gitea

Gitea provides self-hosted Git service for marie-studio integration.

```bash
# Start Gitea (requires PostgreSQL running)
docker compose --env-file ./config/.env \
  -f ./Dockerfiles/docker-compose.storage.yml \
  -f ./Dockerfiles/docker-compose.gitea.yml \
  --project-directory . up -d

# View logs
docker logs -f marie-gitea

# Access Web UI
# Open http://localhost:3001
# Complete installation wizard on first access
```

| Port | Protocol | Description |
|------|----------|-------------|
| 3001 | HTTP | Web UI |
| 2222 | SSH | Git SSH access |

:::note First-time Setup
On first access, Gitea displays an installation wizard. Configure the database connection (PostgreSQL is pre-configured via environment variables) and create an admin account.
:::

### Full Stack Deployment

Start all services together:

```bash
# Create network
docker network create --driver=bridge marie_default

# Start all infrastructure
docker compose --env-file ./config/.env \
  -f ./Dockerfiles/docker-compose.storage.yml \
  -f ./Dockerfiles/docker-compose.s3.yml \
  -f ./Dockerfiles/docker-compose.rabbitmq.yml \
  -f ./Dockerfiles/docker-compose.clickhouse.yml \
  -f ./Dockerfiles/docker-compose.gitea.yml \
  --project-directory . up -d

# Check all services
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Environment Variables

Key environment variables in `config/.env.dev`:

```bash
# PostgreSQL (shared)
POSTGRES_USER=postgres
POSTGRES_PASSWORD=123456

# ClickHouse
CLICKHOUSE_HTTP_PORT=8123
CLICKHOUSE_NATIVE_PORT=9000
CLICKHOUSE_DB=marie

# Gitea
GITEA_HTTP_PORT=3001
GITEA_SSH_PORT=2222
GITEA_DB_NAME=gitea
```

### Stopping Services

```bash
# Stop individual service
docker compose -f ./Dockerfiles/docker-compose.clickhouse.yml down

# Stop and remove volumes (full reset)
docker compose -f ./Dockerfiles/docker-compose.clickhouse.yml down --volumes
docker volume rm marie_clickhouse_data marie_clickhouse_logs --force
```

---

### References
[Docker volumes](https://docs.docker.com/storage/volumes/)
[Bind Mounts](https://docs.docker.com/storage/bind-mounts/)
[ClickHouse Docker](https://hub.docker.com/r/clickhouse/clickhouse-server/)
[Gitea Docker Installation](https://docs.gitea.com/installation/install-with-docker)
