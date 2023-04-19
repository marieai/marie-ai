#!/bin/bash

nuctl deploy --project-name cvat \
  --path serverless/pytorch/bounding-boxes/nuclio \
  --platform local
