#!/usr/bin/env bash
source ./container.sh
# exec 1> >(exec logger -s -t "${CONTAINER_NAME} [${0##*/}]") 2>&1
echo "Starting interactive/dev container : ${CONTAINER_NAME}"

# Failfast on any errors
#set -eu -o pipefail

if  [ $(id -u) = 0 ]; then
   echo "This script must not be run as root, run under 'docker user' account."
  #  exit 1
fi

NAME=${CONTAINER_NAME}
# Determine GPUS to run this container on
DEVICES="all"
GPUS=$1
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
printf '\e[1;32m%-6s\e[m\n' "GPU_COUNT : $GPU_COUNT"

if [[ -n "$GPUS" ]]; then
  printf '\e[1;32m%-6s\e[m\n' "GPUS set : $GPUS"
  if [[ $GPUS -gt $GPU_COUNT || $GPUS -lt 0 ]]; then
    printf '\e[1;91m%-6s\e[0m \n' "Invalid GPU ID : $GPUS"
    exit 1
  fi
  # shellcheck disable=SC2089
  DEVICES='"device=$GPUS"'
  NAME="${CONTAINER_NAME}-$GPUS"
fi

echo "GPUS Selected > ${DEVICES}"
echo "NAME          > ${NAME}"

if [[ "${CONTAINER_ID}" ]]; then
  # container found.
  echo 'Container already exists'
  exit 1
fi

# container not found.
# -p 5100:5100  ${CONTAINER_NAME}:${CONTAINER_VERSION}

docker run -u 0 --user root -it --rm  --gpus "$DEVICES" --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864  \
--name "$NAME" \
-v `pwd`/../config:/etc/marie:rw \
-v /mnt/data/marie-ai/model_zoo:/opt/marie-icr/model_zoo:ro \
-v /opt/logs/marie-icr:/opt/marie-icr/logs:rw \
-v /opt/shares/medrxprovdata:/opt/marie-icr/share:rw  \
--env-file ./service.env  \
--network=host  \
-p 5100:5100  ${CONTAINER_REGISTRY}${CONTAINER_NAME}:${CONTAINER_VERSION}
