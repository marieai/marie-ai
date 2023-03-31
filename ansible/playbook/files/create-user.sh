#!/usr/bin/env bash

if [[ -z $1 ]]; then
    echo "No user specified"
    exit 1
fi

if id -u $1 >/dev/null 2>&1; then
    echo "User $1 already exists, skipping"
#    exit 0
fi

sudo groupadd -r app-svc -g 433
sudo useradd -u 431 --comment 'app-svc' --create-home app-svc  --shell /bin/bash -g app-svc

# add user to docker group
sudo usermod -aG docker app-svc
