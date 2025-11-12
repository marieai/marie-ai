#!/bin/bash
set -e

# MinIO alias
MC_ALIAS="localminio"
MC_CONTAINER="minio/mc"
MINIO_ENDPOINT="http://s3server:9000"
ROOT_USER="marieadmin"
ROOT_PASSWORD="marietopsecret"

# Create mc alias
docker run --rm --network marie_default \
  $MC_CONTAINER alias set $MC_ALIAS $MINIO_ENDPOINT $ROOT_USER $ROOT_PASSWORD

# Create user: Greg
docker run --rm --network marie_default \
  $MC_CONTAINER admin user add $MC_ALIAS accessKey1 verySecretKey1

# Assign readwrite policy to Greg
docker run --rm --network marie_default \
  $MC_CONTAINER admin policy set $MC_ALIAS readwrite user=accessKey1

# Create user: marie
docker run --rm --network marie_default \
  $MC_CONTAINER admin user add $MC_ALIAS MARIEACCESSKEY MARIESECRETACCESSKEY

# Assign readwrite policy to marie
docker run --rm --network marie_default \
  $MC_CONTAINER admin policy set $MC_ALIAS readwrite user=MARIEACCESSKEY

echo "✅ Users created successfully!"



# CONFIRMS USERS
# docker run --rm --network marie_default minio/mc admin user list localminio
