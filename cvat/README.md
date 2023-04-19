# Integrating MARIE-AI with CVAT for automatic annotation tasks 

## Supported Models / Tasks

* Bounding box detection for text
* Text recognition
* Named entity recognition


## Setup

For generic CVAT setup, please refer to the [CVAT documentation-Semi-automatic and Automatic Annotation](https://opencv.github.io/cvat/docs/administration/advanced/installation_automatic_annotation/).

Starting CVAT with some overrides and the serverless components:

```shell
docker compose -f ./docker-compose.yml -f ./docker-compose.override.yml  -f components/serverless/docker-compose.serverless.yml --project-directory . up --build
```

## List all registered serverless functions

```shell
nuctl get function --namespace nuclio
```
