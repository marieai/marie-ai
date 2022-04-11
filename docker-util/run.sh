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

if [[ "${CONTAINER_ID}" ]]; then
  # container found.
	CID=$(docker ps -f status=running -f name=^/${CONTAINER_NAME}$  --format '{{ .ID }}')
	if [ ! "${CID}" ]; then
	  echo "Container stopped"
	  docker start ${CONTAINER_NAME}
	else
	  echo "Running > ${CID}"
	fi
	unset CID
else
  # container not found.
  echo 'Container not found, creating new one'
  # Run me if you want logs outside the container
  # docker run -d  -u 431 --name marie-icr  -i -t  -v /opt/containers/config/marie-icr:/opt/marie-icr/config -v /opt/logs:/opt/marie-icr/logs:rw  --env-file ./service.env  -p 8099:5000  marie-icr:2.0


  docker run -d  -u 431 --name ${CONTAINER_NAME} -i -t  \
  -v `pwd`/config:/etc/marie:rw \
  -v `pwd`/model_zoo:/opt/marie-icr/model_zoo:r \
  -v /opt/logs/marie-icr:/opt/marie-icr/logs:rw  \
  --env-file ./service.env  \
  --network=host  \
  -p 5100:5100  ${CONTAINER_REGISTRY}${CONTAINER_NAME}:${CONTAINER_VERSION} 

fi
