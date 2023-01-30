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