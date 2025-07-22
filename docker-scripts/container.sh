#!/usr/bin/env bash

# Collect container info from ID file
# Usage: source ./container.sh [id_file]
# If no id_file is provided, defaults to 'id'

ID_FILE=${1:-id}

CONTAINER_REF=`cat $ID_FILE`
CONTAINER_INFO=`cat $ID_FILE | awk '{print substr($0, match ($0, "\/[^\/]*$")+1)}'`
CONTAINER_REGISTRY=`cat $ID_FILE | awk '{print substr($0, 0, match ($0, "\/[^\/]*$"))}'`
CONTAINER_NAME=`echo $CONTAINER_INFO | awk '{split ($0, a,":"); print a[1]}'`
CONTAINER_VERSION=`echo $CONTAINER_INFO | awk '{split ($0, a,":"); print a[2]}'`
CONTAINER_ID=$(docker inspect --format="{{.Id}}" ${CONTAINER_NAME} 2> /dev/null)

echo $CONTAINER_REF
echo $CONTAINER_NAME
echo $CONTAINER_VERSION
echo $CONTAINER_REGISTRY
echo $CONTAINER_ID