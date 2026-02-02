#!/bin/bash
# Standalone Docker CE Installation Script
# Can be used independently or called from bootstrap.sh

set -e

echo "Installing Docker CE..."

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VERSION=$VERSION_ID
else
    echo "Cannot detect OS. Exiting."
    exit 1
fi

case "$OS" in
    ubuntu|debian)
        # Remove old Docker versions
        apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

        # Install prerequisites
        apt-get update -qq
        apt-get install -y -qq \
            apt-transport-https \
            ca-certificates \
            curl \
            gnupg \
            lsb-release

        # Add Docker GPG key
        install -m 0755 -d /etc/apt/keyrings
        curl -fsSL "https://download.docker.com/linux/$OS/gpg" -o /etc/apt/keyrings/docker.asc
        chmod a+r /etc/apt/keyrings/docker.asc

        # Add Docker repository
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$OS \
          $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
          tee /etc/apt/sources.list.d/docker.list > /dev/null

        # Install Docker
        apt-get update -qq
        apt-get install -y -qq \
            docker-ce \
            docker-ce-cli \
            containerd.io \
            docker-buildx-plugin \
            docker-compose-plugin
        ;;

    fedora|centos|rhel|rocky|almalinux)
        # Remove old Docker versions
        dnf remove -y docker docker-client docker-client-latest docker-common \
            docker-latest docker-latest-logrotate docker-logrotate docker-engine 2>/dev/null || true

        # Install prerequisites
        dnf install -y dnf-plugins-core

        # Add Docker repository
        dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo

        # Install Docker
        dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        ;;

    *)
        echo "Unsupported OS: $OS"
        echo "Please install Docker manually: https://docs.docker.com/engine/install/"
        exit 1
        ;;
esac

# Enable and start Docker
systemctl enable docker
systemctl start docker

# Add current user to docker group (if not root)
if [ "$EUID" -ne 0 ] && [ -n "$SUDO_USER" ]; then
    usermod -aG docker "$SUDO_USER"
    echo "Added $SUDO_USER to docker group. Please log out and back in for this to take effect."
fi

# Verify installation
echo ""
echo "Docker installation complete!"
docker --version
docker compose version
