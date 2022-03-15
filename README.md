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
settings [https://docs.gunicorn.org/en/stable/settings.html#settings]

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

## References

https://doc.traefik.io/traefik/v2.2/providers/consul-catalog/
https://www.toptal.com/flask/flask-production-recipes
https://apispec.readthedocs.io/en/latest/install.html
https://github.com/gregbugaj/form-processor
https://gradio.app/

Consul / Traefik configuration [https://devonhubner.org/using_traefik_with_consul/]

## Box Detection

Implement secondary box detection method.
https://github.com/ying09/
https://github.com/ying09/TextFuseNet


## Stream processing
https://www.confluent.io/blog/sysmon-security-event-processing-real-time-ksql-helk/
https://pypi.org/project/ksql/
 

# Research 

https://www.microsoft.com/en-us/research/project/document-ai/
https://github.com/microsoft/unilm/tree/master/layoutlmv2