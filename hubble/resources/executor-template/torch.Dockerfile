ARG TORCH_PACKAGE_VERSION=1.12.1
ARG CUDA_VERSION=11.3
ARG CUDNN_VERSION=8


FROM pytorch/pytorch:${TORCH_PACKAGE_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git

ARG JINA_VERSION=

RUN uv pip install --system jina${JINA_VERSION:+==${JINA_VERSION}}

COPY requirements.txt requirements.txt
RUN uv pip install --system --compile-bytecode -r requirements.txt

COPY . /workdir/
WORKDIR /workdir

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
