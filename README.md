[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Marie-AI

Integrate AI-powered document pipeline into your applications

## Documentation

See the [MarieAI docs](https://docs.marieai.co).

## Installation

You don't need this source code unless you want to modify the package. If you just
want to use the package, just run:

```sh
pip install --upgrade marieai
```

Install from source with:

```sh
pip install -e .
```

Build docker container:

```sh
DOCKER_BUILDKIT=1 docker build . --build-arg PIP_TAG="standard" -f ./Dockerfiles/gpu.Dockerfile  -t marieai/marie:3.0-cuda 
```

## Command-line interface

This library additionally provides an `marie` command-line utility which makes it easy to interact with the API 
from your terminal. Run `marie -h` for usage.

## Example code

Examples of how to use this library to accomplish various tasks can be found in the MarieAI documentation. 
It contains code examples for:

* Document cleanup
* Optical character recognition
* Textbox detection
* Named Entity Recognition
* Form detection
* And more


## Telemetry
https://telemetry.marieai.co/

TODO :MOVE TO DOCS

# S3 Cloud Storage
```shell
docker compose -f  docker-compose.s3.yml --project-directory . up  --build --remove-orphans
```

CrossFTP


## Configure AWS CLI Credentials.

```shell
vi ~/.aws/credentials
[marie] # this should be in the file
aws_access_key_id=your_access_key_id
aws_secret_access_key=your_secret_access_key
```

 

## Pull the Docker image.

```shell
docker pull zenko/cloudserver
```

## Create and start the container.


```sh
docker run --rm -it --name marie-s3-server -p 8000:8000 \
-e SCALITY_ACCESS_KEY_ID=MARIEACCESSKEY \
-e SCALITY_SECRET_ACCESS_KEY=MARIESECRETACCESSKEY \
-e S3DATA=multiple \
-e S3BACKEND=mem zenko/cloudserver
```

```
SCALITY_ACCESS_KEY_ID : Your AWS ACCESS KEY 
SCALITY_SECRET_ACCESS_KEY: Your AWS SECRET ACCESS KEY 
S3BACKEND: Currently using memory storage
```

## Verify Installation.

```shell
aws s3 mb s3://mybucket  --profile marie --endpoint-url http://localhost:8000 --region us-west-2
```

```shell
aws s3 ls --profile marie --endpoint-url http://localhost:8000
```

```shell
aws s3 cp some_file.txt s3://mybucket  --profile marie --endpoint-url http://localhost:8000
```


```shell
aws s3 --profile marie --endpoint-url=http://127.0.0.1:8000 ls --recursive s3://
```

# Production setup


Configuration for the S3 server will be stored in the following files:
https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628

