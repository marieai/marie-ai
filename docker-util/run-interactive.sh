#!/usr/bin/env bash
source ./container.sh
exec 1> >(exec logger -s -t "${CONTAINER_NAME} [${0##*/}]") 2>&1
echo "Starting interactive/dev container : ${CONTAINER_NAME}"

if  [ $(id -u) = 0 ]; then
   echo "This script must not be run as root, run under 'docker user' account."
   exit 1
fi

if [[ "${CONTAINER_ID}" ]]; then
  # container found.
  echo 'Container already exists'
  exit 1
fi

# container not found.

docker run --rm  -u 431 --name ${CONTAINER_NAME} -i -t  \
-v `pwd`/../config:/etc/marie:rw \
-v /opt/logs/marie-icr:/opt/marie-icr/logs:rw  \
--env-file ./service.env  \
--network=host  \
-p 5100:5100  ${CONTAINER_NAME}:${CONTAINER_VERSION} 