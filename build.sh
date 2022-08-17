#!/bin/sh
./hooks/post-commit

DOCKER_BUILDKIT=1 docker build . -f Dockerfile
#hash=$(docker images | awk '{print $3}' | awk 'NR==2')
hash=$(docker images --format='{{.ID}}' | head -1)
target="gregbugaj/marie-icr:2.4-cuda"
echo "Tagging : $hash as $target"
docker tag $hash $target