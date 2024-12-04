#!/usr/bin/env bash

set -e
CPU_COUNT=$(grep -c ^processor /proc/cpuinfo)
CPU_COUNT=$((CPU_COUNT-1))
MARIE_VERSION=$(sed -n '/^__version__/p' ./marie/__init__.py | cut -d \' -f2)

echo "Building Marie version: $MARIE_VERSION"
echo "CPU count: $CPU_COUNT"
#  --progress=plain
DOCKER_BUILDKIT=1 docker build . --progress=plain --cpuset-cpus="0-$CPU_COUNT"  --build-arg PIP_TAG="[standard]" -f ./Dockerfiles/gpu-312.Dockerfile -t marieai/marie:$MARIE_VERSION-cuda # --no-cache
