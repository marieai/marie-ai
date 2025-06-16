#!/usr/bin/env bash
source ./container.sh
# exec 1> >(exec logger -s -t "${CONTAINER_NAME} [${0##*/}]") 2>&1
echo "Starting interactive/dev container : ${CONTAINER_NAME}"

if  [ $(id -u) = 0 ]; then
   echo "This script must not be run as root, run under 'docker user' account."
  #  exit 1
fi

if [[ "${CONTAINER_ID}" ]]; then
  # container found.
  echo 'Container already exists'
  exit 1
fi

# container not found.
# -p 5100:5100  ${CONTAINER_NAME}:${CONTAINER_VERSION} 

docker run -u 0 --user root -it --rm  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  \
--name ${CONTAINER_NAME} \
-v `pwd`/../config:/etc/marie:rw \
-v `pwd`/../model_zoo:/opt/marie-icr/model_zoo:rw \
-v /opt/logs/marie-icr:/opt/marie-icr/logs:rw \
-v /opt/shares/medrxprovdata:/opt/marie-icr/share:rw  \
--env-file ./service.env  \
--network=host  \
-p 5100:5100  ${CONTAINER_REGISTRY}${CONTAINER_NAME}:${CONTAINER_VERSION} 