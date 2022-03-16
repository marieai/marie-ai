#!/usr/bin/env bash

# Collect container info from ID file

CONTAINER_REF=`cat id`
CONTAINER_INFO=`cat id | awk '{print substr($0, match ($0, "\/[^\/]*$")+1)}'`
CONTAINER_REGISTRY=`cat id | awk '{print substr($0, 0, match ($0, "\/[^\/]*$"))}'`
CONTAINER_NAME=`echo $CONTAINER_INFO | awk '{split ($0, a,":"); print a[1]}'`
CONTAINER_VERSION=`echo $CONTAINER_INFO | awk '{split ($0, a,":"); print a[2]}'`
CONTAINER_ID=$(docker inspect --format="{{.Id}}" ${CONTAINER_NAME} 2> /dev/null)

echo $CONTAINER_REF
echo $CONTAINER_NAME
echo $CONTAINER_VERSION
echo $CONTAINER_REGISTRY
echo $CONTAINER_ID