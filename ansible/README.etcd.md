# ETCD Cluster Deployment Guide (Marie AI)

## Overview

This document describes how to deploy, manage, and validate a 3-node ETCD cluster using Ansible and Docker, as used within the **Marie AI** platform. The deployment supports both **non-TLS** and **TLS-secured** configurations with optional shared certificates for simplicity.

---

## 🔧 Prerequisites

- Ansible installed (`ansible-core >= 2.12`)
- Docker installed on all target hosts
- Inventory populated with ETCD node IPs
- Vault password configured via `vault.txt` or env
- `community.docker` collection installed:

```bash
ansible-galaxy collection install community.docker
```

---

## 🔐 TLS Setup (Optional)

To use shared TLS certs for all nodes:

```bash
cd roles/etcd_cluster/files/certs
./generate-etcd-certs.sh
```

This creates:
- `ca.pem`
- `etcd.pem`
- `etcd-key.pem`

Set the following in `group_vars/etcd_cluster.yml`:

```yaml
enable_tls: true
shared_cert: true
etcd_cert_dir: /etc/etcd/certs
```

---

## 📁 Inventory Structure

```yaml
etcd_cluster:
  hosts:
    etcd1:
      ansible_host: "10.0.0.11"
    etcd2:
      ansible_host: "10.0.0.12"
    etcd3:
      ansible_host: "10.0.0.13"
```

---

## 🗂 Playbooks

| File | Description |
|------|-------------|
| `playbook/etcd-cluster.yml` | Deploys ETCD containers (TLS-aware) |
| `playbook/etcd-cleanup.yml` | Stops and removes all ETCD containers |
| `playbook/etcd-health.yml`  | Runs `etcdctl endpoint health` across nodes |

---

## 🚀 Deploy Cluster

```bash
./play-etcd-cluster.sh
```

---

## 🔁 Clean Up Containers

```bash
./play-etcd-cleanup.sh
```

---

## ✅ Health Check

```bash
./play-etcd-health.sh
```

If `enable_tls` is enabled, the `etcdctl` command will auto-adjust to use `--cert`, `--key`, and `--cacert` as needed.

---

## 🛡️ Security Notes

| Mode        | Comment |
|-------------|---------|
| Non-TLS     | Suitable for dev/internal-only |
| TLS (shared)| Good compromise for trusted clusters |
| TLS (per-node certs) | Recommended for hardened deployments (future-ready) |

---

## 📦 File Layout Summary

```
inventories/
├── hosts.yml
│
group_vars/
└── etcd_cluster.yml

playbook/
├── etcd-cluster.yml
├── etcd-cleanup.yml
└── etcd-health.yml

roles/
└── etcd_cluster/
    ├── tasks/main.yml
    ├── defaults/main.yml
    └── files/certs/
        ├── generate-etcd-certs.sh
        ├── ca.pem
        ├── etcd.pem
        └── etcd-key.pem

play-etcd-cluster.sh
play-etcd-cleanup.sh
play-etcd-health.sh
```

---

## 👥 Maintainers

For issues or enhancements, contact the Marie AI Infrastructure team.