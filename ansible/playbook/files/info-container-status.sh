#!/usr/bin/env bash

if [[ -z $1 ]]; then
    echo "No container specified"
    exit 1
fi

CONTAINER_NAME=$1 

CID=$(docker ps -f status=running -f name=^/${CONTAINER_NAME}$  --format '{{ .ID }}  {{ .Image }}  {{ .Names }}  {{ .Status }}')
if [ ! "${CID}" ]; then
  echo "FAILED  : Stopped"
else
  echo "SUCCESS : Running > ${CID}"  
fi
unset CID