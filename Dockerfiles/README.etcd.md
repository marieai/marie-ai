# README for etcd Cluster Deployment

This document describes how to deploy a secure etcd cluster using Docker Compose and Ansible, including steps to generate TLS certificates and configure single-node or multi-node setups.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Repository Layout](#repository-layout)
3. [Certificate Generation](#certificate-generation)
4. [Single-Node Deployment (Docker Compose)](#single-node-deployment-docker-compose)
5. [Multi-Node Cluster Deployment](#multi-node-cluster-deployment)

   * [Docker Compose Cluster](#docker-compose-cluster)
   * [Ansible Automation](#ansible-automation)
6. [Directory Structure](#directory-structure)
7. [Commands Reference](#commands-reference)
8. [Troubleshooting](#troubleshooting)
9. [Cleanup](#cleanup)

---

## Prerequisites

* Docker Engine (>= 20.10)
* Docker Compose plugin (>= 2.x)
* Ansible (>= 2.9)
* OpenSSL (for cert generation)
* Git (to clone this repository)

Ensure you have network connectivity between all etcd nodes on ports **2379** (client) and **2380** (peer).

---

## Repository Layout

```
.
├── inventory.yml                 # Ansible inventory
├── site.yml                      # Ansible site playbook
├── docker-compose.etcd.yml            # Single-node compose file
├── cluster-compose.yml.j2        # Jinja template for multi-node compose
├── README.etcd.md                # This README file
├── roles/
│   └── etcd_cluster/
│       ├── defaults/main.yml     # Role defaults
│       ├── files/
│       │   └── certs/
│       │       └── generate-etcd-certs.sh  # Cert generation script
│       ├── handlers/main.yml
│       ├── tasks/main.yml
│       └── templates/docker-compose.etcd.yml.j2
└── data/                         # Local data directory for Docker volumes
```

---

## Certificate Generation

A single shared etcd certificate (with SANs for all nodes) and CA is generated via the supplied script.

1. **Make the script executable**:

   ```bash
   chmod +x roles/etcd_cluster/files/certs/generate-etcd-certs.sh
   ```

2. **Run the script**:

   ```bash
   ./roles/etcd_cluster/files/certs/generate-etcd-certs.sh \
     roles/etcd_cluster/files/certs/out \
     etcd1.example.com etcd2.example.com etcd3.example.com
   ```

3. The output directory (`.../certs/out`) will contain:

   * `ca.pem`, `ca.key`
   * `server.pem`, `server.key`
   * `peer.pem`, `peer.key`

These files are then distributed to each host by Ansible.

---

## Single-Node Deployment (Docker Compose)

Use this setup for testing or development:

1. Place generated certificates under `certs/`:

   ```
   certs/
   ├── ca.pem
   ├── server.pem
   ├── server-key.pem
   ├── peer.pem
   └── peer-key.pem
   ```
2. Create a `data/` directory for etcd data:

   ```bash
   mkdir -p data
   ```
3. Launch the container:

   ```bash
   docker-compose -f docker-compose.etcd.yml up -d
   ```
4. Verify health:

   ```bash
   docker-compose exec etcd-single etcdctl \
     --endpoints=https://127.0.0.1:2379 \
     --cacert=/etc/etcd/certs/ca.pem \
     --cert=/etc/etcd/certs/server.pem \
     --key=/etc/etcd/certs/server-key.pem \
     endpoint health --cluster
   ```

---

## Multi-Node Cluster Deployment

### Docker Compose Cluster

1. Ensure each node has:

   * `data/<hostname>` directory mounted
   * `certs/` directory with TLS assets
2. Use `cluster-compose.yml.j2` to generate `docker-compose.etcd.yml` (via Ansible or manually).
3. Start the cluster:

   ```bash
   docker-compose -f docker-compose.etcd.yml up -d
   ```
4. Check health on any node:

   ```bash
   docker exec etcd1 etcdctl endpoint health --cluster \
     --cacert=/etc/etcd/certs/ca.pem --cert=/etc/etcd/certs/server.pem --key=/etc/etcd/certs/server-key.pem
   ```

### Ansible Automation

The `etcd_cluster` role will:

1. Install Docker & Compose plugin
2. Generate and distribute certs
3. Create directories
4. Render and deploy `docker-compose.etcd.yml`
5. Perform health checks

Run with:

```bash
ansible-playbook -i inventory.yml site.yml
```

Role variables can be adjusted in `roles/etcd_cluster/defaults/main.yml`.

---

## Directory Structure

* **data/**: etcd data volumes
* **certs/**: TLS assets
* **roles/etcd\_cluster/**: Ansible role
* **inventory.yml**, **site.yml**: Ansible orchestration
* **docker-compose.etcd.yml**: Single-node stack
* **cluster-compose.yml.j2**: Template for multi-node

---

## Commands Reference

| Action                    | Command                                           |
| ------------------------- | ------------------------------------------------- |
| Generate certs            | `generate-etcd-certs.sh out ${NODES...}`          |
| Start single-node         | `docker-compose -f docker-compose.etcd.yml up -d` |
| Start multi-node (manual) | `docker-compose -f docker-compose.etcd.yml up -d` |
| Ansible deploy            | `ansible-playbook -i inventory.yml site.yml`      |
| Check endpoint health     | `etcdctl endpoint health --cluster [params]`      |

---

## Troubleshooting

* **Error: "unhealthy" on healthcheck**

  * Verify SAN DNS names/IPs match node hostnames.
  * Check certificate validity: `openssl x509 -noout -text -in server.pem`.
* **Cluster won’t form quorum**

  * Ensure peer URLs reachable (ports 2380).
  * Confirm `ETCD_INITIAL_CLUSTER` matches service aliases.
* **Permissions issues**

  * Certs must be readable by root (mode 600).

---

## Cleanup

To remove containers and volumes:

```bash
docker-compose down -v
rm -rf data/*
```

For Ansible-managed hosts, you can remove the `data_dir` and `certs_dest` folders manually or via an additional teardown playbook.

---
