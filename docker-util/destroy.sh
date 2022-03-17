#!/usr/bin/env bash

source ./container.sh
exec 1> >(exec logger -s -t "${CONTAINER_NAME} [${0##*/}]") 2>&1
echo "Destroying container/images : ${CONTAINER_NAME}"

if  [ $(id -u) = 0 ]; then
   echo "This script must not be run as root, run under 'docker user' account."
   exit 1
fi


destroy(){
  echo 'Destroying data'
  # Stop all container
  docker stop $(docker ps -q  --filter "name=${CONTAINER_NAME}")

  # Delete all containers
  docker rm $(docker ps -a -q  --filter "name=${CONTAINER_NAME}")

  # Delete all images
  docker rmi $(docker images -q --filter=reference="$CONTAINER_REGISTRY${CONTAINER_NAME}:*")
}


while true; do
    read -p  "Do you wish to destroy all docker images/container for '${CONTAINER_NAME}? " yn
    case $yn in
        [Yy]* ) destroy; 
		exit 0
		;;
        [Nn]* ) echo 'NO';  
		exit 1
		;;
        * ) echo "Please answer yes or no.";;
    esac
done

