#!/usr/bin/env bash

PYTHONUNBUFFERED=1 MARIE_DEBUG=0 MARIE_DEBUG_PORT=5678 MARIE_DEFAULT_MOUNT=/mnt/data/marie-ai XXXXJINA_MP_START_METHOD=fork marie server --start --uses /mnt/data/marie-ai/config/service/marie.yml
