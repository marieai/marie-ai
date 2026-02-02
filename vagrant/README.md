# Marie-AI Vagrant Test Environment

This directory contains Vagrant configuration for creating isolated VMs to test Docker Compose deployments without affecting your local development environment.

## Overview

The Vagrant VM provides:
- Full VM isolation from local Docker containers
- **Multi-instance support** - run up to 9 VMs simultaneously
- Pre-configured Ubuntu 24.04 LTS with Docker CE
- Port forwarding with unique offsets per instance
- Synced folders for compose files and configs
- Easy integration with `bootstrap-marie.sh`

## Prerequisites

- **Vagrant** >= 2.4.0
- **VM Provider**: VirtualBox or libvirt (auto-detected)
- ~10GB disk space for VM
- 8GB RAM recommended (configurable)

### Installation

The bootstrap script automatically detects KVM and chooses the appropriate provider.

**macOS (VirtualBox):**
```bash
brew install --cask virtualbox vagrant
```

**Ubuntu/Debian with KVM (recommended for Linux):**
```bash
# Install Vagrant
sudo apt-get install -y vagrant

# Install libvirt and dependencies
sudo apt-get install -y libvirt-dev libvirt-daemon-system libvirt-clients qemu-kvm

# Add user to libvirt group
sudo usermod -aG libvirt $USER
newgrp libvirt  # or log out and back in

# Install vagrant-libvirt plugin
vagrant plugin install vagrant-libvirt
```

**Ubuntu/Debian with VirtualBox:**
```bash
# Note: VirtualBox cannot run if KVM modules are loaded
sudo apt-get install -y virtualbox vagrant
```

**Fedora/RHEL with libvirt:**
```bash
sudo dnf install -y vagrant libvirt-devel libvirt-daemon libvirt-client qemu-kvm
sudo usermod -aG libvirt $USER
newgrp libvirt
vagrant plugin install vagrant-libvirt
```

### Provider Auto-Detection

The bootstrap script automatically:
1. Detects if KVM kernel modules are loaded
2. Checks for required libvirt dependencies
3. Verifies vagrant-libvirt plugin is installed
4. Provides specific installation instructions if anything is missing

You can also force a specific provider:
```bash
VAGRANT_DEFAULT_PROVIDER=libvirt ./bootstrap-marie.sh --vagrant
VAGRANT_DEFAULT_PROVIDER=virtualbox ./bootstrap-marie.sh --vagrant
```

## Quick Start

### Option 1: Via bootstrap-marie.sh (Recommended)

```bash
# Start VM and deploy full stack (instance 1)
./bootstrap-marie.sh --vagrant

# Deploy in a specific instance (e.g., instance 2)
./bootstrap-marie.sh --vagrant --instance=2

# Deploy infrastructure only
./bootstrap-marie.sh --vagrant --infrastructure-only

# List all running instances
./bootstrap-marie.sh --vagrant-list

# Check VM status
./bootstrap-marie.sh --vagrant-status
./bootstrap-marie.sh --vagrant-status --instance=2

# SSH into VM
./bootstrap-marie.sh --vagrant-ssh
./bootstrap-marie.sh --vagrant-ssh --instance=2

# Destroy VM
./bootstrap-marie.sh --vagrant-down
./bootstrap-marie.sh --vagrant-down --instance=2
```

### Option 2: Direct Vagrant Commands

```bash
cd vagrant/

# Start instance 1 (default)
vagrant up

# Start instance 2
VAGRANT_INSTANCE=2 vagrant up

# SSH into instance 2
VAGRANT_INSTANCE=2 vagrant ssh

# Inside VM, run bootstrap
cd /home/vagrant/marie
./bootstrap-marie.sh

# Stop VM (preserves data)
VAGRANT_INSTANCE=2 vagrant halt

# Destroy VM completely
VAGRANT_INSTANCE=2 vagrant destroy
```

## Configuration

### VM Resources

Set environment variables before running `vagrant up`:

```bash
export VAGRANT_MEMORY=16384  # RAM in MB (default: 8192)
export VAGRANT_CPUS=8        # CPU cores (default: 4)
vagrant up
```

### Base Box Selection

The default box is `cloud-image/ubuntu-24.04`. You can override this:

```bash
# Use Bento box
VAGRANT_BOX=bento/ubuntu-24.04 vagrant up

# Use official Ubuntu box
VAGRANT_BOX=ubuntu/noble64 vagrant up
```

Available options:
- `cloud-image/ubuntu-24.04` (default) - Official Ubuntu cloud images
- `bento/ubuntu-24.04` - Bento project, well-maintained
- `ubuntu/noble64` - Official Canonical box

### Test Environments

Pre-configured environment files are available in `envs/`:

| File | Description |
|------|-------------|
| `test-default.env` | Standard test configuration |
| `test-minimal.env` | Infrastructure only, reduced resources |
| `test-full.env` | Full stack with ClickStack observability |

To use a specific environment:

```bash
# Copy to config directory
cp vagrant/envs/test-full.env config/.env.vagrant

# Deploy with that environment
ENV_FILE=./config/.env.vagrant ./bootstrap-marie.sh --vagrant
```

## Port Mapping Reference

Each instance uses a unique port offset to avoid conflicts:
- **Instance 1**: +10000 offset (ports 15432, 18123, etc.)
- **Instance 2**: +11000 offset (ports 16432, 19123, etc.)
- **Instance 3**: +12000 offset (ports 17432, 20123, etc.)

### Instance 1 Port Mapping (Default)

| Service | Container Port | Local (Bare Metal) | Instance 1 |
|---------|---------------|-------------------|------------|
| PostgreSQL | 5432 | 5432 | 15432 |
| ClickHouse HTTP | 8123 | 8123 | 18123 |
| ClickHouse Native | 9000 | 9000 | 19000 |
| RabbitMQ AMQP | 5672 | 5672 | 15672 |
| RabbitMQ Management | 15672 | 15672 | 25672 |
| MinIO API | 9001 | 9001 | 19001 |
| MinIO Console | 9002 | 9002 | 19002 |
| etcd | 2379 | 2379 | 12379 |
| Gitea HTTP | 3001 | 3001 | 13001 |
| Gitea SSH | 2222 | 2222 | 12222 |
| LiteLLM | 4000 | 4000 | 14000 |
| Gateway HTTP | 52000 | 52000 | 62000 |
| Gateway gRPC | 51000 | 51000 | 61000 |

### Multi-Instance Quick Reference

| Service | Instance 1 | Instance 2 | Instance 3 |
|---------|-----------|-----------|-----------|
| PostgreSQL | 15432 | 16432 | 17432 |
| ClickHouse | 18123 | 19123 | 20123 |
| RabbitMQ Mgmt | 25672 | 26672 | 27672 |
| MinIO Console | 19002 | 20002 | 21002 |
| Gitea | 13001 | 14001 | 15001 |
| Gateway HTTP | 62000 | 63000 | 64000 |
| VM IP | 192.168.56.11 | 192.168.56.12 | 192.168.56.13 |

### Accessing Services

```bash
# Instance 1 services (via forwarded ports)
curl http://localhost:18123  # ClickHouse in Instance 1
psql -h localhost -p 15432   # PostgreSQL in Instance 1

# Instance 2 services
curl http://localhost:19123  # ClickHouse in Instance 2
psql -h localhost -p 16432   # PostgreSQL in Instance 2

# Local services (unchanged)
curl http://localhost:8123   # Local ClickHouse
psql -h localhost -p 5432    # Local PostgreSQL

# Direct VM access (via private network)
curl http://192.168.56.11:8123  # ClickHouse via Instance 1 IP
curl http://192.168.56.12:8123  # ClickHouse via Instance 2 IP
```

## Directory Structure

```
vagrant/
├── Vagrantfile              # Main Vagrant configuration
├── README.md                # This file
├── provision/
│   ├── bootstrap.sh         # VM provisioning script
│   └── install-docker.sh    # Standalone Docker installation
└── envs/
    ├── test-default.env     # Default test environment
    ├── test-minimal.env     # Minimal configuration
    └── test-full.env        # Full stack configuration
```

## Common Operations

### Sync Files to VM

Files are synced on `vagrant up` and `vagrant provision`. To manually sync:

```bash
vagrant rsync
```

### View VM Logs

```bash
# SSH into VM and view Docker logs
vagrant ssh -c "docker logs marie-psql-server"
vagrant ssh -c "docker logs -f marie-gateway"
```

### Clean Up

```bash
# Stop VM (preserves data)
vagrant halt

# Destroy VM and all data
vagrant destroy -f

# Remove VirtualBox VM files
rm -rf ~/.vagrant.d/boxes/generic-VAGRANTSLASH-ubuntu2404/
```

### Troubleshooting

**VM won't start:**
```bash
# Check VirtualBox service
sudo systemctl status vboxdrv

# Restart VirtualBox
sudo systemctl restart vboxdrv
```

**Port conflicts:**
```bash
# Check what's using a port
lsof -i :15432

# Modify port in Vagrantfile and reload
vagrant reload
```

**Sync issues:**
```bash
# Force re-sync
vagrant rsync

# Re-provision VM
vagrant provision
```

**Docker issues in VM:**
```bash
# SSH and check Docker
vagrant ssh
sudo systemctl status docker
docker ps -a
docker logs <container_name>
```

## Comparison: Local vs Vagrant

| Aspect | Local Docker | Vagrant VM |
|--------|-------------|------------|
| Isolation | Shared Docker daemon | Complete VM isolation |
| Performance | Native speed | ~10-20% overhead |
| Disk space | Containers only | ~10GB for VM + containers |
| Cleanup | `docker compose down` | `vagrant destroy` |
| Port conflicts | Yes, if same ports | No, offset ports |
| Use case | Development | Testing, CI, validation |

## Integration with CI/CD

The Vagrant environment can be used in CI pipelines:

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

## Security Notes

- Default credentials in test environments are for **testing only**
- Never use test configurations in production
- The Vagrant VM uses a private network (192.168.56.x) by default
- Port forwarding exposes services only on localhost (127.0.0.1)
