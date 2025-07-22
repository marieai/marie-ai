#!/usr/bin/env bash

source ./container.sh
exec 1> >(exec logger -s -t "${CONTAINER_NAME} [${0##*/}]") 2>&1
echo "Stopping container[init] : ${CONTAINER_NAME}"

if  [ $(id -u) = 0 ]; then
   echo "This script must not be run as root, run under 'docker user' account."
   exit 1
fi

if [[ "${CONTAINER_ID}" ]]; then
   echo "Stopping container[stop] : ${CONTAINER_NAME}"
   docker container stop $(docker container ls -q --filter name=${CONTAINER_NAME})
   docker ps -f name=${CONTAINER_NAME}
else
   echo "Container not found : ${CONTAINER_NAME}"   
fi

