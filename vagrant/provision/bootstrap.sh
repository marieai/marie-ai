#!/bin/bash
# Marie-AI Vagrant VM Provisioning Script
# This script runs during 'vagrant up' to prepare the VM for Docker deployments

set -e

echo "=============================================="
echo "  Marie-AI VM Provisioning"
echo "=============================================="
echo ""

# ============================================================
# System Updates
# ============================================================
echo ">>> Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq

# ============================================================
# Install Prerequisites
# ============================================================
echo ">>> Installing prerequisites..."
apt-get install -y -qq \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    jq \
    htop \
    vim \
    git \
    unzip \
    wget \
    net-tools \
    dnsutils \
    iputils-ping

# ============================================================
# Install Docker CE
# ============================================================
echo ">>> Installing Docker CE..."

# Add Docker's official GPG key
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -qq

# Install Docker packages
apt-get install -y -qq \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# ============================================================
# Configure Docker
# ============================================================
echo ">>> Configuring Docker..."

# Add vagrant user to docker group
usermod -aG docker vagrant

# Configure Docker daemon
mkdir -p /etc/docker
cat > /etc/docker/daemon.json << 'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "live-restore": true,
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 65536,
      "Soft": 65536
    }
  }
}
EOF

# Enable and start Docker
systemctl enable docker
systemctl restart docker

# ============================================================
# System Tuning for Docker
# ============================================================
echo ">>> Applying system tuning..."

# Increase file descriptor limits
cat >> /etc/security/limits.conf << 'EOF'
* soft nofile 65536
* hard nofile 65536
root soft nofile 65536
root hard nofile 65536
EOF

# Kernel parameters for Docker
cat > /etc/sysctl.d/99-docker.conf << 'EOF'
# Docker performance tuning
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
net.core.netdev_max_backlog = 65535

# Memory settings
vm.max_map_count = 262144
vm.swappiness = 10
vm.overcommit_memory = 1

# File system
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 512
EOF

sysctl --system > /dev/null 2>&1

# ============================================================
# Create Marie Working Directory
# ============================================================
echo ">>> Setting up Marie-AI directories..."
mkdir -p /home/vagrant/marie/{Dockerfiles,config,data,logs}
chown -R vagrant:vagrant /home/vagrant/marie

# ============================================================
# Install Additional Tools
# ============================================================
echo ">>> Installing additional tools..."

# Install yq (YAML processor)
wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
chmod +x /usr/local/bin/yq

# Install lazydocker (Docker TUI)
curl -fsSL https://raw.githubusercontent.com/jesseduffield/lazydocker/master/scripts/install_update_linux.sh | bash
mv lazydocker /usr/local/bin/ 2>/dev/null || true

# ============================================================
# Pre-pull Common Images (Optional - speeds up first deployment)
# ============================================================
echo ">>> Pre-pulling common Docker images..."

# Pull common infrastructure images (matches what's in docker-compose files)
# These are pulled in background to speed up first deployment
echo ">>> Pre-pulling common infrastructure images..."
docker pull ghcr.io/ferretdb/postgres-documentdb:17-0.103.0 &
docker pull rabbitmq:3-management-alpine &
docker pull minio/minio:latest &
docker pull minio/mc:latest &
docker pull quay.io/coreos/etcd:v3.6.1 &
docker pull gitea/gitea:latest &
docker pull clickhouse/clickhouse-server:latest &

wait

# ============================================================
# Verification
# ============================================================
echo ""
echo ">>> Verifying installation..."
echo "Docker version: $(docker --version)"
echo "Docker Compose version: $(docker compose version)"
echo "Docker status: $(systemctl is-active docker)"

# Test Docker without sudo (as vagrant user)
su - vagrant -c "docker ps" > /dev/null 2>&1 && echo "Docker accessible by vagrant user: OK" || echo "Docker accessible by vagrant user: FAILED"

echo ""
echo "=============================================="
echo "  Provisioning Complete!"
echo "=============================================="
echo ""
echo "The VM is ready for Marie-AI deployment."
echo "Run 'vagrant ssh' to access the VM."
echo ""
