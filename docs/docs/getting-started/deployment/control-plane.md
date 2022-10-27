---
sidebar_position: 2
---

# Control Plane
Control plane is responsible for orchestrating communication between nodes in the cluster.

## Docker Compose
Quickest way to bootstrap control plane is via `docker compose`.
Install new version of [docker compose cli plugin](https://docs.docker.com/compose/install/)


### User and permission setup
The container is setup with `app-svc` account so for that we will setup same account in the host system.

Setting up user, for more info visit [Manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)

```
sudo useradd --comment 'app-svc' --create-home app-svc --shell /bin/bash
sudo usermod -aG docker app-svc
```

Directory structure, this is more of a convention than requirement.

```sh
sudo mkdir -p /opt/containers/apps/marie-ai
sudo mkdir -p /opt/containers/config/marie-ai
```

Change permissions to `app` and `config`

```
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


### Networking setup

The configuration uses custom bridge networks called `public` which we will create first

```shell
docker network create --driver=bridge public

# If you receive following error, enable IPv4 forwarding
# WARNING: IPv4 forwarding is disabled. Networking will not work.
sysctl net.ipv4.conf.all.forwarding=1
```
### Starting and stopping

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
```shell
ssh -N -L 8500:ops-001:8500 -L 7777:ops-001:7777 -L 9090:ops-001:9090 -L 3000:ops-001:3000 ops-001
```

[Explain](https://explainshell.com/explain?cmd=ssh+-N+-L+8500%3Aops-001%3A8500+-L+7777%3Aops-001%3A7777+-L+9090%3Aops-001%3A9090+-L+3000%3Aops-001%3A3000+ops-001)

:::


|Service|Endpoint|Description|
|---|--|-----------------------------------------------------|
|Consul|http://localhost:8500/ui/| Consul is a distributed, highly available, and data center aware solution to connect and configure applications across dynamic, distributed infrastructure.  |
|Traefik| http://traefik.localhost:7777/dashboard http://traefik.localhost:7777/metrics| Traefik is a modern HTTP reverse proxy and load balancer that makes deploying microservices easy.  |
|Grafana|#| The open and composable observability and data visualization platform.|
|Prometheus|http://localhost:9090/| The Prometheus monitoring system and time series database.|
|Loki|#| Loki is a horizontally scalable, highly available, multi-tenant log aggregation system inspired by Prometheus|

```
http://localhost:8500/ui/dc1/services
http://traefik.localhost:7777/metrics
http://traefik.localhost:7777/dashboard/#/http/routers
http://localhost:7777/ping
http://localhost:3000/?orgId=1
http://localhost:9090/targets?search=
```

### Metrics
* [metrics - cadvisor](http://localhost:8077/metrics)  http://localhost:8077/metrics
* [metrics - DCGM](http://localhost:9400/metrics)   http://localhost:9400/metrics
* [metrics - node-exporter](http://localhost:9400/metrics) http://localhost:9400/metrics

### Monitoring
* [Promtail UI](http://localhost:9080/targets)
* [Alertmanager UI](http://localhost:9093/#/status)