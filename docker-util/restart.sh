#!/usr/bin/env bash

source ./container.sh
exec 1> >(exec logger -s -t "${CONTAINER_NAME} [${0##*/}]") 2>&1
echo "Restarting container : ${CONTAINER_NAME}"


if  [ $(id -u) = 0 ]; then
   echo "This script must not be run as root, run under 'app-svc' account."
   exit 1
fi

docker stop ${CONTAINER_NAME}
docker start ${CONTAINER_NAME}
docker ps -f name=${CONTAINER_NAME}
