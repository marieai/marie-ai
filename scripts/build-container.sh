#!/usr/bin/env bash

set -e
CPU_COUNT=$(grep -c ^processor /proc/cpuinfo)
CPU_COUNT=$((CPU_COUNT-1))
MARIE_VERSION=$(sed -n '/^__version__/p' ./marie/__init__.py | cut -d \' -f2)

echo "Building Marie version: $MARIE_VERSION"
echo "CPU count: $CPU_COUNT"
#  --progress=plain

# https://forums.developer.nvidia.com/t/docker-build-starts-ignoring-default-runtime-setting-after-initially-working/111155/12
# we are expilitly setting DOCKER_BUILDKIT=0 to avoid the issue with the default-runtime=nvidia

NVIDIA_VISIBLE_DEVICES=all
DOCKER_BUILDKIT=0 docker build . --cpuset-cpus="0-$CPU_COUNT"  --build-arg PIP_TAG="[standard]" -f ./Dockerfiles/gpu-312.Dockerfile -t marieai/marie:$MARIE_VERSION-cuda #--no-cache
