ARG CUDA_VERSION=11.6.0
ARG CUDNN_VERSION=8

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

# declare the image name
ARG JAXLIB_VERSION=0.3.0

# install python3
RUN apt update && apt install python3 -y

# install dependencies via uv
RUN uv pip install --system numpy scipy six wheel jaxlib==${JAXLIB_VERSION}+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html

RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git

ARG JINA_VERSION=

RUN uv pip install --system jina${JINA_VERSION:+==${JINA_VERSION}}

COPY requirements.txt requirements.txt
RUN uv pip install --system --compile-bytecode -r requirements.txt

COPY . /workdir/
WORKDIR /workdir

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
