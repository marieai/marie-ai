#!/bin/sh
set -eu

VALKEYCLI_AUTH="$(cat /run/valkey/password)"
export VALKEYCLI_AUTH
valkey-cli --user marie -h 127.0.0.1 ping | grep -qx PONG
