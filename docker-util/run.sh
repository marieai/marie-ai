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

# exit 0

if [[ "${CONTAINER_ID}" ]]; then
  # container found.
	CID=$(docker ps -f status=running -f name=^/${NAME}$  --format '{{ .ID }}')
	if [ ! "${CID}" ]; then
	  echo "Container stopped"
	  docker start ${NAME}
	else
	  echo "Running > ${CID}"
	fi
	unset CID
else
  # container not found.
  echo 'Container not found, creating new one'
  #  -u 431

  # shellcheck disable=SC2090
docker run -d -u 0 --user root -it --gpus $DEVICES --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864  \
-e MARIE_PORT=$PORT \
--env-file ./service.env \
--name $NAME \
-v $CONFIG_DIR:/etc/marie:rw \
-v /mnt/data/marie-ai/model_zoo:/opt/marie-icr/model_zoo:ro \
-v /opt/logs/marie-icr/$GPUS:/opt/marie-icr/logs:rw \
-v /opt/shares/medrxprovdata:/opt/marie-icr/share:rw \
--network=host \
--restart unless-stopped \
-p $PORT:5000  ${CONTAINER_REGISTRY}${CONTAINER_NAME}:${CONTAINER_VERSION}

# docker run -d  -u 431 --name ${CONTAINER_NAME} -i -t  \
# -v `pwd`/config:/etc/marie:rw \
# -v `pwd`/model_zoo:/opt/marie-icr/model_zoo:r \
# -v /opt/logs/marie-icr:/opt/marie-icr/logs:rw  \
# --env-file ./service.env  \
# --network=host  \
# -p 5100:5100  ${CONTAINER_REGISTRY}${CONTAINER_NAME}:${CONTAINER_VERSION} 

fi
