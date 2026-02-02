---
sidebar_position: 7
---

# Vagrant Testing Environment

Test Docker Compose deployments in an isolated VM without affecting your local Docker setup.

## Overview

The Vagrant testing environment provides:
- Full VM isolation from local Docker containers
- **Multi-instance support** - run up to 9 VMs simultaneously
- Pre-configured Ubuntu 24.04 LTS with Docker CE
- Port forwarding with unique offsets per instance
- Easy integration with `bootstrap-marie.sh`

## Prerequisites

- **VirtualBox** >= 7.0 (or libvirt on Linux)
- **Vagrant** >= 2.4.0
- ~10GB disk space
- 8GB RAM recommended

### Installation

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="macos" label="macOS" default>

```bash
brew install --cask virtualbox vagrant
```

</TabItem>
<TabItem value="ubuntu" label="Ubuntu/Debian">

```bash
sudo apt-get update
sudo apt-get install virtualbox vagrant
```

</TabItem>
<TabItem value="fedora" label="Fedora/RHEL">

```bash
sudo dnf install VirtualBox vagrant
```

</TabItem>
</Tabs>

## Quick Start

### Via bootstrap-marie.sh

```bash
# Start VM and deploy full stack (instance 1)
./bootstrap-marie.sh --vagrant

# Deploy in a specific instance
./bootstrap-marie.sh --vagrant --instance=2

# Deploy infrastructure only
./bootstrap-marie.sh --vagrant --infrastructure-only

# List all running instances
./bootstrap-marie.sh --vagrant-list

# Check VM status
./bootstrap-marie.sh --vagrant-status
./bootstrap-marie.sh --vagrant-status --instance=2

# SSH into VM for debugging
./bootstrap-marie.sh --vagrant-ssh
./bootstrap-marie.sh --vagrant-ssh --instance=2

# Destroy VM when done
./bootstrap-marie.sh --vagrant-down
./bootstrap-marie.sh --vagrant-down --instance=2
```

### Direct Vagrant Commands

```bash
cd vagrant/

# Start VM (provisions Docker on first run)
vagrant up

# SSH into VM
vagrant ssh

# Inside VM, deploy services
cd /home/vagrant/marie
./bootstrap-marie.sh

# Stop VM (preserves data)
vagrant halt

# Destroy VM completely
vagrant destroy
```

## Port Mapping

Each instance uses a unique port offset:
- **Instance 1**: +10000 offset
- **Instance 2**: +11000 offset
- **Instance 3**: +12000 offset

### Instance 1 Ports (Default)

| Service | Container Port | Local Docker | Instance 1 |
|---------|---------------|--------------|------------|
| PostgreSQL | 5432 | 5432 | 15432 |
| ClickHouse HTTP | 8123 | 8123 | 18123 |
| RabbitMQ Management | 15672 | 15672 | 25672 |
| MinIO Console | 9002 | 9002 | 19002 |
| Gitea HTTP | 3001 | 3001 | 13001 |
| LiteLLM | 4000 | 4000 | 14000 |
| Gateway HTTP | 52000 | 52000 | 62000 |

### Multi-Instance Quick Reference

| Service | Instance 1 | Instance 2 | Instance 3 |
|---------|-----------|-----------|-----------|
| PostgreSQL | 15432 | 16432 | 17432 |
| ClickHouse | 18123 | 19123 | 20123 |
| RabbitMQ Mgmt | 25672 | 26672 | 27672 |
| Gateway HTTP | 62000 | 63000 | 64000 |
| VM IP | 192.168.56.11 | 192.168.56.12 | 192.168.56.13 |

### Accessing Services

```bash
# Instance 1 services
curl http://localhost:18123  # ClickHouse in Instance 1
psql -h localhost -p 15432   # PostgreSQL in Instance 1

# Instance 2 services
curl http://localhost:19123  # ClickHouse in Instance 2
psql -h localhost -p 16432   # PostgreSQL in Instance 2

# Local services (unchanged)
curl http://localhost:8123   # Local ClickHouse
psql -h localhost -p 5432    # Local PostgreSQL

# Direct VM access (via private network)
curl http://192.168.56.11:8123  # Instance 1 via IP
curl http://192.168.56.12:8123  # Instance 2 via IP
```

## Configuration

### VM Resources

Set environment variables before starting the VM:

```bash
# Increase resources for larger deployments
export VAGRANT_MEMORY=16384  # RAM in MB (default: 8192)
export VAGRANT_CPUS=8        # CPU cores (default: 4)

./bootstrap-marie.sh --vagrant-up
```

### Test Environments

Pre-configured environment files are available in `vagrant/envs/`:

| File | Description | Use Case |
|------|-------------|----------|
| `test-default.env` | Standard configuration | General testing |
| `test-minimal.env` | Reduced resources | Quick infrastructure tests |
| `test-full.env` | Full stack with observability | Complete integration testing |

```bash
# Use a specific environment
cp vagrant/envs/test-full.env config/.env.vagrant
ENV_FILE=./config/.env.vagrant ./bootstrap-marie.sh --vagrant
```

## Common Operations

### Sync Files

Files are synced automatically on `vagrant up`. To manually sync changes:

```bash
cd vagrant/
vagrant rsync
```

### View Logs

```bash
# View service logs inside VM
./bootstrap-marie.sh --vagrant-ssh
docker logs -f marie-psql-server
docker logs -f marie-clickhouse
```

Or directly:

```bash
vagrant ssh -c "docker logs marie-psql-server"
```

### Test Configuration Changes

1. Modify compose files or configs locally
2. Sync changes to VM: `cd vagrant && vagrant rsync`
3. SSH in and redeploy: `vagrant ssh -c "cd /home/vagrant/marie && ./bootstrap-marie.sh"`

## Troubleshooting

### VM Won't Start

```bash
# Check VirtualBox service
sudo systemctl status vboxdrv

# Restart VirtualBox kernel modules
sudo systemctl restart vboxdrv
```

### Port Conflicts

```bash
# Check what's using a port
lsof -i :15432

# Modify port in Vagrantfile and reload
vim vagrant/Vagrantfile
cd vagrant && vagrant reload
```

### Docker Issues in VM

```bash
./bootstrap-marie.sh --vagrant-ssh

# Inside VM:
sudo systemctl status docker
docker ps -a
docker logs <container_name>

# Restart Docker if needed
sudo systemctl restart docker
```

### Sync Issues

```bash
cd vagrant/

# Force re-sync
vagrant rsync

# Re-provision VM
vagrant provision
```

### Clean Start

```bash
# Destroy and recreate VM
./bootstrap-marie.sh --vagrant-down
./bootstrap-marie.sh --vagrant
```

## Use Cases

### Testing Deployment Changes

```bash
# Make changes to compose files
vim Dockerfiles/docker-compose.clickhouse.yml

# Test in isolated VM
./bootstrap-marie.sh --vagrant --infrastructure-only

# Verify services
vagrant ssh -c "docker ps"
curl http://localhost:18123/?query=SELECT+version()
```

### Parallel Development

Run both local and VM environments simultaneously:

```bash
# Terminal 1: Local development
./bootstrap-marie.sh --infrastructure-only
curl http://localhost:8123  # Local ClickHouse

# Terminal 2: VM testing
./bootstrap-marie.sh --vagrant --infrastructure-only
curl http://localhost:18123  # VM ClickHouse
```

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
jobs:
  test-deployment:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Vagrant
        run: |
          wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
          echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
          sudo apt-get update && sudo apt-get install vagrant virtualbox
      - name: Test Deployment
        run: |
          ./bootstrap-marie.sh --vagrant --infrastructure-only
          ./bootstrap-marie.sh --vagrant-status
```

## Comparison: Local vs Vagrant

| Aspect | Local Docker | Vagrant VM |
|--------|-------------|------------|
| Isolation | Shared daemon | Complete VM |
| Performance | Native | ~10-20% overhead |
| Disk space | Containers only | ~10GB + containers |
| Cleanup | `docker compose down` | `vagrant destroy` |
| Port conflicts | Possible | No (offset ports) |
| Best for | Development | Testing, CI |

## Security Notes

:::warning Test Credentials
Default credentials in test environments are for testing only. Never use in production.
:::

- The Vagrant VM uses a private network (192.168.56.x)
- Port forwarding exposes services only on localhost (127.0.0.1)
- Destroy VM after testing to remove all data

## Next Steps

- [Docker deployment](./docker.md) - Single node production deployment
- [Kubernetes](./kubernetes.md) - Cluster deployment
- [Observability](./observability.md) - Monitoring setup
