#!/usr/bin/env bash
source ./container.sh
# exec 1> >(exec logger -s -t "${CONTAINER_NAME} [${0##*/}]") 2>&1
echo "Starting interactive/dev container : ${CONTAINER_NAME}"

# Failfast on any errors
# set -eu -o pipefail

if  [ $(id -u) = 0 ]; then
   echo "This script must not be run as root, run under 'docker user' account."
   exit 1
fi

NAME=${CONTAINER_NAME}
# Determine GPUS to run this container on
DEVICES="all"
GPUS=$1
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
CONFIG="config"
PORT=6000

printf '\e[1;32m%-6s\e[m\n' "GPU_COUNT : $GPU_COUNT"

if [[ -n "$GPUS" ]]; then
  printf '\e[1;32m%-6s\e[m\n' "GPUS set : $GPUS"
  if [[ $GPUS -ge $GPU_COUNT || $GPUS -lt 0 ]]; then
    printf '\e[1;91m%-6s\e[0m \n' "Invalid GPU ID : $GPUS"
    exit 1
  fi
  # shellcheck disable=SC2089
#  DEVICES='"device=$GPUS"'
  DEVICES="device=$GPUS"
  NAME="${CONTAINER_NAME}-$GPUS"
  CONFIG="config-${GPUS}"
  PORT=$(($PORT + $GPUS))

  if [[ $GPUS == "all" ]]; then
    NAME="${CONTAINER_NAME}"
    CONFIG="config"
    PORT=6000
  fi
fi

CONTAINER_ID=$(docker inspect --format="{{.Id}}" ${NAME} 2> /dev/null)
CONFIG_DIR="/mnt/data/marie-ai/$CONFIG"

echo "GPUS Selected  > ${DEVICES}"
echo "CONTAINER_NAME > ${NAME}"
echo "CONTAINER_ID   > ${CONTAINER_ID}"
echo "CONFIG         > ${CONFIG}"
echo "PORT           > ${PORT}"
echo "CONFIG_DIR     > ${CONFIG_DIR}"

if [[ "${CONTAINER_ID}" ]]; then
  # container found.
  echo 'Container already exists'
  exit 1
fi

CONFIG_DIR="/mnt/data/marie-ai/$CONFIG"
if [ ! -d "${CONFIG_DIR}" ]; then
    echo "Config directory '$CONFIG_DIR' DOES NOT exists."
    # exit 1
fi

# exit 0
# container not found.
# -p 5100:5100  ${CONTAINER_NAME}:${CONTAINER_VERSION}
#--network=host  \
#-v `pwd`/../config:/etc/marie:rw \
# Currently using HOST for our networking but we will change to use dedicated bridge network


# shellcheck disable=SC2090
docker run -u 0 --user root -it --rm  --gpus $DEVICES --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864  \
-e MARIE_PORT=$PORT \
--env-file ./service.env \
--name $NAME \
-v $CONFIG_DIR:/etc/marie:rw \
-v /mnt/data/marie-ai/model_zoo:/opt/marie-icr/model_zoo:ro \
-v /opt/logs/marie-icr/$GPUS:/opt/marie-icr/logs:rw \
-v /opt/shares/medrxprovdata:/opt/marie-icr/share:rw \
--network=host \
-p $PORT:5000  ${CONTAINER_REGISTRY}${CONTAINER_NAME}:${CONTAINER_VERSION}


