#!/usr/bin/env bash
source ./container.sh id.gateway
exec 1> >(exec logger -s -t "[${0##*/}]") 2>&1
echo "Starting container : ${CONTAINER_NAME}"

if  [ $(id -u) = 0 ]; then
   echo "This script must not be run as root, run under 'docker user' account."
   exit 1
fi

# Set gateway-specific configuration
NAME=${CONTAINER_NAME}
CONFIG="config"
PORT=51000
GRPC_PORT=52000
CONFIG_DIR="/mnt/data/marie-ai/$CONFIG"

# Check if container already exists
CONTAINER_ID=$(docker inspect --format="{{.Id}}" ${NAME} 2> /dev/null)

echo "CONTAINER_NAME > ${NAME}"
echo "CONTAINER_ID   > ${CONTAINER_ID}"
echo "CONFIG         > ${CONFIG}"
echo "PORT           > ${PORT}"
echo "GRPC_PORT      > ${GRPC_PORT}"
echo "CONFIG_DIR     > ${CONFIG_DIR}"

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
  echo 'Container not found, creating new one'
  # shellcheck disable=SC2090
#  docker run -u 0 --user root -it  --entrypoint /bin/bash  --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864 \
#    -e MARIE_PORT=$PORT \
#    -e MARIE_DEBUG_QUERY_PLANNER=TRUE \
#    -e MARIE_DEFAULT_MOUNT=/mnt/data/marie-ai \
#    --env-file ./service.env \
#    --name $NAME \
#    -v $CONFIG_DIR:/etc/marie:rw \
#    -v /mnt/data/marie-ai/model_zoo:/mnt/data/marie-ai/model_zoo:ro \
#    -v /opt/logs/marie-ai:/opt/marie-ai/logs:rw \
#    -v /mnt/data/marie-ai/config/extra_py_modules:/mnt/data/marie-ai/config/extra_py_modules:ro \
#    --network=host \
#    --restart unless-stopped \
#    -p $PORT:51000 \
#    -p $GRPC_PORT:52000 \
#    ${CONTAINER_REGISTRY}${CONTAINER_NAME}:${CONTAINER_VERSION}
##    gateway --uses /mnt/data/marie-ai/config/service/gateway.yml --protocols HTTP GRPC --ports 51000 52000 --extra-search-paths /mnt/data/marie-ai/config/extra_py_modules

# -v /mnt/data/marie-ai/config/wheels:/mnt/data/marie-ai/config/wheels:rw \
  # shellcheck disable=SC2090
  docker run --rm -u 0 --user root -it --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864 \
    -e COLUMNS=180 \
    -e PYTHONUNBUFFERED=1 \
    -e GRPC_TRACE="${GRPC_TRACE:-ALL}" \
    -e GRPC_VERBOSITY="${GRPC_VERBOSITY:-DEBUG}" \
    -e MARIE_PORT=$PORT \
    -e MARIE_DEBUG_QUERY_PLANNER="${MARIE_DEBUG_QUERY_PLANNER:-FALSE}" \
    -e MARIE_DEFAULT_MOUNT=/mnt/data/marie-ai \
    -e MARIE_SCHEDULER_TRACE_ENABLED="${MARIE_SCHEDULER_TRACE_ENABLED:-true}" \
    -e MARIE_SCHEDULER_TRACE_PATH="${MARIE_SCHEDULER_TRACE_PATH:-/tmp/marie-scheduler-trace.jsonl}" \
    -e LLM_QUEUE_ENABLED="${LLM_QUEUE_ENABLED:-true}" \
    -e LLM_QUEUE_FABRIC_GROUP_ID="${LLM_QUEUE_FABRIC_GROUP_ID:-default}" \
    -e LLM_QUEUE_GATEWAY_ID="${LLM_QUEUE_GATEWAY_ID:-default}" \
    -e LLM_QUEUE_VALKEY_URL="${LLM_QUEUE_VALKEY_URL:-redis://localhost:6379/0}" \
    -e OPENAI_API_BASE="${OPENAI_API_BASE:-http://192.222.48.192}" \
    -e OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" \
    --env-file ./service.env \
    --name $NAME \
    -v $CONFIG_DIR:/etc/marie:rw \
    -v /mnt/data/marie-ai/:/mnt/data/marie-ai:rw \
    -v /mnt/data/marie-ai/model_zoo:/mnt/data/marie-ai/model_zoo:ro \
    -v /opt/logs/marie-ai:/opt/marie-ai/logs:rw \
    -v /mnt/data/marie-ai/config/extra_py_modules:/mnt/data/marie-ai/config/extra_py_modules:ro \
    --network=host \
    -p $PORT:51000 \
    -p $GRPC_PORT:52000 \
    ${CONTAINER_REGISTRY}${CONTAINER_NAME}:${CONTAINER_VERSION} \
    gateway --uses /etc/marie/service/marie-gateway-4.0.0.yml --protocols HTTP GRPC --ports 51000 52000 --extra-search-paths /mnt/data/marie-ai/config/extra_py_modules --metrics --metrics-exporter-host localhost --metrics-exporter-port 4317 --tracing --traces-exporter-host localhost --traces-exporter-port 4317
#docker run --rm  -it --entrypoint /bin/bash  marieai/marie-gateway:4.5rc1-cpu
# docker exec -it  marie-gateway  bash
fi
