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

# 2 Add Docker’s official GPG key:
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg

# Use the following command to set up the repository:
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

## 3 Install Docker Engine
sudo apt-get update
sudo apt-get install -yq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Install the NVIDIA container toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

# Check if nvidia-smi is installed
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "nvidia-smi found, installing NVIDIA Docker toolkit"

    # Remove existing NVIDIA repositories
    sudo rm /etc/apt/sources.list.d/nvidia-container-toolkit.list

	curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
	  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
	    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
	    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

		# Check if the NVIDIA container toolkit list file was created correctly
		if grep -q "<!doctype" /etc/apt/sources.list.d/nvidia-container-toolkit.list; then
		  echo "Error: The NVIDIA container toolkit list file contains HTML content. Please check the URL."
		  exit 1
		fi
        sudo apt-get update
        sudo apt-get install -yq nvidia-container-toolkit

        sudo nvidia-ctk runtime configure --runtime=docker

        sudo systemctl restart docker
        sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
else
    echo "nvidia-smi not found, skipping NVIDIA Docker toolkit installation"
fi

# cleanup
sudo apt autoremove -yq


