#!/bin/bash -xv
## #!/usr/bin/env bash

export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'

# check if docker is already installed
if [ -x "$(command -v docker)" ]; then
  echo "Docker is already installed"
#  exit 0
fi


# install docker
# https://docs.docker.com/engine/install/ubuntu/#set-up-the-repository

# Update the apt package index and install packages to allow apt to use a repository over HTTPS:
sudo apt-get update
sudo apt-get upgrade -y

sudo apt-get install -yq \
    ca-certificates \
    curl \
    gnupg

# Uninstall old versions of Docker
sudo apt-get purge -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo apt-get autoremove -y --purge

# 2 Add Docker’s official GPG key:
# Add Docker’s official GPG key
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker-archive-keyring.gpg

# Set up the stable repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null


## 3 Install Docker Engine
sudo apt-get update
sudo apt-get install -yq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Install the NVIDIA container toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Check if the NVIDIA container toolkit list file was created correctly
if grep -q "<!doctype" /etc/apt/sources.list.d/nvidia-container-toolkit.list; then
  echo "Error: The NVIDIA container toolkit list file contains HTML content. Please check the URL."
  exit 1
fi

# sudo apt-get update && sudo apt-get upgrade -y

sudo apt-get update
sudo apt-get install -yq nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker


# Check Docker API version
client_api_version=$(docker version --format '{{.Client.APIVersion}}')
server_api_version=$(docker version --format '{{.Server.APIVersion}}')

if [ "$client_api_version" != "$server_api_version" ]; then
  echo "Warning: Docker client API version ($client_api_version) is downgraded from server API version ($server_api_version)."
else
  echo "Docker client and server API versions match: $client_api_version"
fi

# Test
sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi

# cleanup
sudo apt autoremove -yq