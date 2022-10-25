---
sidebar_position: 2
---

# Control Plane
Control plane is responsible for orchestrating communication between nodes in the cluster.

## Services
There are number of services that made up control plane. 

#### Consul

Consul is a distributed, highly available, and data center aware solution to connect and configure applications across dynamic, distributed infrastructure.

#### Traefik
Traefik is a modern HTTP reverse proxy and load balancer that makes deploying microservices easy.

#### Grafana
The open and composable observability and data visualization platform.

#### Prometheus
The Prometheus monitoring system and time series database.

#### Loki
Loki is a horizontally scalable, highly available, multi-tenant log aggregation system inspired by Prometheus


## Docker Compose
Quickest way to bootstrap control plane is via `docker compose`.
Install new version of [docker compose cli plugin](https://docs.docker.com/compose/install/)

### Networking setup

The configuration uses custom bridge networks called `public` which we 

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

## Endpoints

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