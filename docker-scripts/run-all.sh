#!/usr/bin/env bash
source ./container.sh
exec 1> >(exec logger -s -t "${CONTAINER_NAME} [${0##*/}]") 2>&1
echo "Starting container : ${CONTAINER_NAME}"

if  [ $(id -u) = 0 ]; then
   echo "This script must not be run as root, run under 'docker user' account."
   exit 1
fi

# Mount local volume for configuration management
# Mount local volume for log management

# Running in interactive mode -i on port 8099  to map dest:src
# add --rm to remove a container after an exit, useful during troubleshooting


NAME=${CONTAINER_NAME}
# Determine GPUS to run this container on
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

printf '\e[1;32m%-6s\e[m\n' "GPU_COUNT : $GPU_COUNT"

for ((i = 0 ; i < ${GPU_COUNT} ; i++)); 
do
  printf '\e[1;32m%-6s\e[m\n' "Starting on GPU : $i"
  ./run.sh $i
done
