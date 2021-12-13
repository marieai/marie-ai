#!/usr/bin/env bash
exec 1> >(exec logger -s -t "ocr-service [${0##*/}]") 2>&1
echo "Restarting OCR Service container"

if  [ $(id -u) = 0 ]; then
   echo "This script must not be run as root, run under 'rms-svc' account."
   exit 1
fi

docker stop ocr-service
docker start ocr-service
docker ps -f name=ocr-service
