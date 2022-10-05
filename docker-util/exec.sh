#!/usr/bin/env bash

source ./container.sh
# exec 1> >(exec logger -s -t "${CONTAINER_NAME} [${0##*/}]") 2>&1
echo "Executing container : ${CONTAINER_NAME}"

if  [ $(id -u) = 0 ]; then
   echo "This script must not be run as root, run under 'docker user' account."
   exit 1
fi

docker exec -it ${CONTAINER_NAME} bash


