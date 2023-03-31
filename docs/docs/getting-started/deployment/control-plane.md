---
sidebar_position: 2
---

# Control Plane
Control plane is responsible for orchestrating communication between nodes in the cluster.

## Docker Compose
Quickest way to bootstrap control plane is via `docker compose`.


* Install new version of [docker](https://docs.docker.com/engine/install/ubuntu/#set-up-the-repository)
* Install new version of [docker compose cli plugin](https://docs.docker.com/compose/install/)


### User and permission setup
The container is setup with `app-svc` account so for that we will setup same account in the host system.

Setting up user, for more info visit [Manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)

`uid` is a number associated with a user account and `gid` is a number associated with a group
Assigned ID that are mapped from within the container to outside world.

`431` : UID
`433` : GUI

```shell
sudo groupadd -r app-svc -g 433
sudo useradd -u 431 --comment 'app-svc' --create-home app-svc  --shell /usr/sbin/nologin -g app-svc
sudo usermod -aG docker app-svc
```

You can verify the user’s UID, using the id command:
```shell
$ id -u app-svc
```

Directory structure, this is more of a convention than requirement.

```sh
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

```shell
sudo su app-svc
git clone https://github.com/marieai/marie-ai
```

### Networking setup
The configuration uses custom bridge networks called `public` which we will create first.

```shell
docker network create --driver=bridge public

# If you receive following error, enable IPv4 forwarding
# WARNING: IPv4 forwarding is disabled. Networking will not work.
sysctl net.ipv4.conf.all.forwarding=1
```

### Starting and stopping Control Plane
All service could be started with single command and run on the same host, however, for production setup it is recommended to run storage service on separate host.

#### Controller(Traefik, Consul, RabbitMQ, Prometheus, Alertmanager, Loki, Grafana) 

```shell
docker compose  --env-file ./config/.env.prod -f docker-compose.yml --project-directory . up  --build --remove-orphans
```

#### Storage
```shell
docker compose  --env-file ./config/.env.prod -f docker-compose.s3.yml -f docker-compose.storage.yml --project-directory . up  --build --remove-orphans
```
After S3 is up and running we can mount via `s3fs` and test it out.

```shell
s3fs marie /mnt/s3-marie -o passwd_file=${HOME}/.passwd-s3fs -o url=http://127.0.0.1:8000 -o use_path_request_style
```

Starting and stopping specific services
```shell
docker compose down \ 
docker compose -f docker-compose.yml -f docker-compose.storage.yml \
--project-directory . up loki consul-server grafana prometheus traefik whoami  --build  --remove-orphans
```

## Kubernetes 
[Translate a Docker Compose File to Kubernetes Resources](https://kubernetes.io/docs/tasks/configure-pod-container/translate-compose-kubernetes/)


## Services
There are number of services that made up control plane. 

:::info SSH Port Forwarding

Some admin service can only be accesses via `localhost` host. We will use SSH forwarding to allow this from our local machine to control-plane.

Replace `ops-001` with the name of the control plane server. 
Alternatively you can edit `/etc/hosts` and add custom hostname.

```shell
172.0.0.1   ops-001.marie-ai.com
172.0.0.1   ops-001
```

```shell
ssh -vnT -N -L 15672:ops-001:15672 -L 8500:ops-001:8500 -L 5000:ops-001:5000  -L 7777:ops-001:7777 -L 9090:ops-001:9090 -L 3000:ops-001:3000 -L 3100:ops-001:3100 -L 9093:ops-001:9093 ops-001
```

[Explain](https://explainshell.com/explain?cmd=ssh+-N+-L+8500%3Aops-001%3A8500+-L+7777%3Aops-001%3A7777+-L+9090%3Aops-001%3A9090+-L+3000%3Aops-001%3A3000+ops-001)

:::


| Service      | Endpoint                                                                         | Description                                                                                                                                                 |
|--------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Consul       | http://localhost:8500/ui/                                                        | Consul is a distributed, highly available, and data center aware solution to connect and configure applications across dynamic, distributed infrastructure. |
| Traefik      | http://traefik.localhost:7777/dashboard/#/ http://traefik.localhost:7777/metrics | Traefik is a modern HTTP reverse proxy and load balancer that makes deploying microservices easy.                                                           |
| RabbitMQ     | http://localhost:15672                                                           | RabbitMQ message broker                                                                                                                                     |
| Grafana      | http://localhost:3000/                                                           | The open and composable observability and data visualization platform.                                                                                      |
| Prometheus   | http://localhost:9090/                                                           | The Prometheus monitoring system and time series database.                                                                                                  |
| Alertmanager | http://localhost:9093/                                                           | The Alertmanager handles alerts sent by client applications such as the Prometheus server.                                                                  |
| Loki         | http://localhost:3100/ready                                                      | Loki is a horizontally scalable, highly available, multi-tenant log aggregation system inspired by Prometheus                                               |


Traefik - Service endpoints can be changed in the configs but by default they are as follow: 

| Traefik - Service | Endpoint                                   |
|-------------------|--------------------------------------------|
| Dashboard         | http://traefik.localhost:7777/dashboard/#/ |
| traefik           | http://traefik.localhost:7777/             |
| traefik-debug     | http://traefik.localhost:7000/             |
| Service endpoint  | http://traefik.localhost:5000/             |



:::warning Grafana Data Sources / Loki

When configuring Grafana Loki Datasource make sure to use the public `IP` or `hostname` and not loopback ip(127.0.0.1/localhost)
in the HTTP URL field.

Example 
`http://ops-001:3100`

:::


## Logging queries

```sql
{job="marie-ai"} |= `` | json | line_format `{{.msg}}`
```

```sql
{job="marie-ai"} |= `` | json | levelname = `ERROR` | line_format `{{.msg}}`
```