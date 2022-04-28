# Marie-ICR

Integrate AI-powered OCR features into your applications

## Installation

create folder structure

```sh 
mkdir models config
```

Follow instructions from `pytorch` website

```sh
https://pytorch.org/get-started/locally/
```

Install required packages with `pip`

```sh
$ pip install -r ./requirements/requirements.txt
```

Build Docker Image

```sh
DOCKER_BUILDKIT=1 docker build . -t marie-icr:1.0
```

Starting in Development mode

```sh
python ./app.py
```

Starting in Production mode with `gunicorn`. Config
[gunicorn]settings (https://docs.gunicorn.org/en/stable/settings.html#settings)

```sh
gunicorn -c gunicorn.conf.py wsgi:app  --log-level=debug
```

Activate the environment as we used `PIP` to install `docker-compose` (python -m pip install docker-compose)

```sh
    source  ~/environments/pytorch/bin/activate
```

Build docker container

Remove dangling containers

```sh
docker rmi -f $(docker images -f "dangling=true" -q)
```

```sh
# --no-cache
DOCKER_BUILDKIT=1 docker build . -t marie-icr:1.0 --network=host  --no-cache
```

Start docker compose

```sh
DOCKER_BUILDKIT=1 docker-compose up

docker-compose down --volumes --remove-orphans && DOCKER_BUILDKIT=1 docker-compose up
```

Cleanup containers

```sh
    docker-compose down --volumes --remove-orphans
```

# Setup Redis

```sh
docker run --name marie_redis -d redis

docker run --rm --name marie_redis redis
```

```sh
docker exe -it marie_redis sh
```

## Codestyle / Formatting

```sh
black  ./utils/ --line-length 119
```
## Issues

There is a segmentation fault happening with `opencv-python==4.5.4.62` switching to `opencv-python==4.5.4.60` fixes the issue. 
[connectedComponentsWithStats produces a segfault ](https://github.com/opencv/opencv-python/issues/604)
```
pip install opencv-python==4.5.4.60
```

## References

[consul-catalog](https://doc.traefik.io/traefik/v2.2/providers/consul-catalog/)
[apispec](https://apispec.readthedocs.io/en/latest/install.html)
[gradio](https://gradio.app/)

[https://devonhubner.org/using_traefik_with_consul/](Consul / Traefik configuration)

## Box Detection

Implement secondary box detection method.
[TextFuseNet](TextFuseNethttps://github.com/ying09/TextFuseNet)


## Stream processing
[KSQL Stream processing example ](https://www.confluent.io/blog/sysmon-security-event-processing-real-time-ksql-helk/)
[KSQL](https://pypi.org/project/ksql/)
 

## Research 

[DocumentUnderstanding](https://github.com/bikash/DocumentUnderstanding)
[DocumentAI] (https://www.microsoft.com/en-us/research/project/document-ai/)




#Funsd Dataset
https://huggingface.co/datasets/nielsr/funsd/blob/main/funsd.py
