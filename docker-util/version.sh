#!/usr/bin/env bash
if  [ $(id -u) = 0 ]; then
   echo "This script must not be run as root, run under 'docker user' account."
   exit 1
fi

curl -v http://127.0.0.1:5000/api