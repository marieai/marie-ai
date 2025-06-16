#!/usr/bin/env bash

if  [ $(id -u) = 0 ]; then
   echo "This script must not be run as root, run under 'docker user' account."
   exit 1
fi
 

docker exec -it marie-icr tail -f /opt/marie-icr/logs/server.log


