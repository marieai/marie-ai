# Integrating MARIE-AI with CVAT for automatic annotation tasks 

## Supported Models / Tasks

* Bounding box detection for text
* Line segmentation
* Text recognition
* Named entity recognition


## Setup
For generic CVAT setup, please refer to the [CVAT documentation-Semi-automatic and Automatic Annotation](https://opencv.github.io/cvat/docs/administration/advanced/installation_automatic_annotation/).

User and admin setup : https://opencv.github.io/cvat/docs/administration/basics/admin-account/

```shell
  docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Starting CVAT with some overrides and the serverless components:
Latest version of nuclio is `1.11.22`

```shell
wget https://github.com/nuclio/nuclio/releases/download/1.11.22/nuctl-1.11.22-linux-amd64
```

New CVAT installations should create a `docker-compose.override.yml` file with the following content:

```yaml title='docker-compose.override.yml' file='docker-compose.override.yml'
version: '3.4'

services:
  cvat_server:
    environment:
      CVAT_SHARE_URL: 'Mounted from /mnt/share/cvat-data host directory'  
    volumes:
      - cvat_share:/home/django/share:ro
  cvat_worker_import:
    volumes:
      - cvat_share:/home/django/share:ro
  cvat_worker_export:
    volumes:
      - cvat_share:/home/django/share:ro
  cvat_worker_annotation:
    volumes:
      - cvat_share:/home/django/share:ro

volumes:
  cvat_share:
    driver_opts:
      type: none
      device: /mnt/share/cvat-data
      o: bind
```

Prepare a clean environment:

```shell
docker compose down --volumes --remove-orphans
```

```shell
export CVAT_HOST=cvat-003 && docker compose -f ./docker-compose.yml -f ./docker-compose.override.yml  -f components/serverless/docker-compose.serverless.yml --project-directory . up --build
```

Crate new `cvat` project

```shell
nuctl create project cvat
```


List all registered serverless functions

```shell
nuctl get function --namespace nuclio
```

## Common issues

```shell
Error - Project does not exist
    ...//nuclio/pkg/platform/abstract/platform.go:434

Call stack:
Project existence validation failed
    ...//nuclio/pkg/platform/abstract/platform.go:434
Failed to validate a function configuration
    /nuclio/pkg/platform/local/platform.go:1273
Failed to enrich and validate a function configuration

```

This means that the project is not created in nuclio. Create a new project with the same name as the CVAT project.

```shell
nuctl create project cvat
```

## References
https://github.com/opencv/cvat/blob/develop/site/content/en/docs/faq.md?plain=1
https://opencv.github.io/cvat/docs/manual/advanced/serverless-tutorial/
https://stephencowchau.medium.com/journey-using-cvat-semi-automatic-annotation-with-a-partially-trained-model-to-tag-additional-8057c76bcee2