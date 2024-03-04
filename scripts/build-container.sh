#!/usr/bin/env bash

set -e
CPU_COUNT=$(grep -c ^processor /proc/cpuinfo)
MARIE_VERSION=$(sed -n '/^__version__/p' ./marie/__init__.py | cut -d \' -f2)

echo "Building Marie version: $MARIE_VERSION"
echo "CPU count: $CPU_COUNT"

DOCKER_BUILDKIT=1 docker build . --cpuset-cpus="0-$CPU_COUNT"  --build-arg PIP_TAG="[standard]" -f ./Dockerfiles/gpu-310.Dockerfile -t marieai/marie:$MARIE_VERSION-cuda --no-cache
