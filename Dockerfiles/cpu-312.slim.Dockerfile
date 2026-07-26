# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS build-image

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src

COPY pyproject.toml README.md MANIFEST.in setup.py ./
COPY requirements/uv/marie-gateway-cpu.lock.txt requirements/uv/
COPY marie/_version.py marie/_version.py
COPY wheels/etcd3-0.12.0-py2.py3-none-any.whl wheels/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python /usr/local/bin/python && \
    uv pip install \
        --python /opt/venv/bin/python \
        --require-hashes \
        --no-deps \
        --no-editable \
        --requirement requirements/uv/marie-gateway-cpu.lock.txt

COPY marie/ marie/
COPY marie_server/ marie_server/
COPY hubble/ hubble/
COPY packages/ packages/
COPY patches/ patches/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        --python /opt/venv/bin/python \
        --no-deps \
        --no-editable \
        . \
        packages/marie-cli \
        packages/marie-engine \
        packages/marie-instrumentation \
        packages/marie-wasm && \
    /opt/venv/bin/python patches/patch-omegaconf-py312.py --no-confirm && \
    test -x /opt/venv/bin/marie && \
    /opt/venv/bin/marie gateway --help >/dev/null && \
    /opt/venv/bin/python -c 'from marie.serve.runtimes.gateway.marie import MarieGateway; assert MarieGateway.__name__ == "MarieGateway"' && \
    /opt/venv/bin/python -c 'import sys; from marie.api.routes import create_mcp_router; assert callable(create_mcp_router); assert "marie.agent" not in sys.modules' && \
    /opt/venv/bin/python -c 'import importlib.util; blocked = ("albumentations", "cv2", "imagecodecs", "matplotlib", "networkx", "pandas", "pdf2image", "PIL", "pyarrow", "skimage", "torch", "torchvision", "wand"); assert not [name for name in blocked if importlib.util.find_spec(name)]'


FROM ${PYTHON_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /bin/uv

ARG TZ="Etc/UTC"
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
ARG BUILD_NUMBER=unknown
ARG MARIE_VERSION=unknown
ARG IMAGE_NAME=unknown

ENV TERM=xterm-256color \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=${TZ} \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update -o APT::Update::Error-Mode=any && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libjemalloc2 \
        tzdata && \
    rm -rf /var/lib/apt/lists/*

ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"

COPY --from=build-image /opt/venv /opt/venv

RUN /opt/venv/bin/python -c 'import sys; from marie.build_info import write_build_info; write_build_info(sys.argv[1], version=sys.argv[2], git_commit=sys.argv[3], build_time=sys.argv[4], build_number=sys.argv[5], image=sys.argv[6])' \
        /etc/marie-ai/build-info.json \
        "${MARIE_VERSION}" \
        "${VCS_REF}" \
        "${BUILD_DATE}" \
        "${BUILD_NUMBER}" \
        "${IMAGE_NAME}"

ENV MARIE_BUILD_INFO_PATH="/etc/marie-ai/build-info.json"

WORKDIR /marie

LABEL org.opencontainers.image.vendor="Marie AI" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="MarieAI Gateway" \
      org.opencontainers.image.description="Marie AI CPU gateway runtime" \
      org.opencontainers.image.authors="hello@marieai.co" \
      org.opencontainers.image.url="https://github.com/marieai/marie-ai" \
      org.opencontainers.image.documentation="https://docs.marieai.co" \
      org.opencontainers.image.source="https://github.com/marieai/marie-ai" \
      org.opencontainers.image.created=${BUILD_DATE} \
      org.opencontainers.image.version=${MARIE_VERSION} \
      org.opencontainers.image.revision=${VCS_REF}

ENTRYPOINT ["/opt/venv/bin/marie"]
