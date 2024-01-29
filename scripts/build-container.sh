#!/usr/bin/env bash

set -e
DOCKER_BUILDKIT=1 docker build . --cpuset-cpus="0-32"  --build-arg PIP_TAG="[standard]" -f ./Dockerfiles/gpu-310.Dockerfile -t marieai/marie:3.0.26-cuda --no-cache