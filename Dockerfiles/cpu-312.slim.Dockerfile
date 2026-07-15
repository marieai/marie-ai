# !!! An ARG declared before a FROM is outside of a build stage, so it can’t be used in any instruction after a FROM
FROM ubuntu:24.04 AS build-image

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

ARG http_proxy
ARG https_proxy
ARG no_proxy="${no_proxy}"
ARG socks_proxy
ARG TZ="Etc/UTC"
ARG MARIE_CONFIGURATION="production"

# given by builder's env
ARG VCS_REF
ARG PY_VERSION=3.12
ARG BUILD_DATE
ARG MARIE_VERSION
ARG TARGETPLATFORM

# constant, wont invalidate cache
LABEL org.opencontainers.image.vendor="Marie AI" \
      org.opencontainers.image.licenses="Apache 2.0" \
      org.opencontainers.image.title="MarieAI" \
      org.opencontainers.image.description="Deploy production-ready AI agent systems for document processing, content analysis, and multimodal intelligence via containerized cloud services" \
      org.opencontainers.image.authors="hello@marieai.co" \
      org.opencontainers.image.url="https://github.com/marieai/marie-ai" \
      org.opencontainers.image.documentation="https://docs.marieai.co" \
      org.opencontainers.image.created=${BUILD_DATE} \
      org.opencontainers.image.source="https://github.com/marieai/marie-ai${VCS_REF}" \
      org.opencontainers.image.version=${MARIE_VERSION} \
      org.opencontainers.image.revision=${VCS_REF}


ENV DEBIAN_FRONTEND=noninteractive

# Tweak this list to reduce build time
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy


RUN apt-get update -o APT::Update::Error-Mode=any && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
        apt-get install wget -y \
        build-essential \
        curl \
        git \
        lshw \
        zlib1g \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-opencv \
        libopenblas-dev \
        libopenmpi-dev \
        openmpi-bin \
        openmpi-common \
        gfortran \
        libomp-dev \
        ninja-build \
        cmake \
        gcc \
        g++ \
        imagemagick \
        libmagickwand-dev \
        libtiff-dev \
        libjpeg-dev \
        libpng-dev \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove \
    && apt-get clean


# Ensure the correct symbolic links
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 \
&& ln -sf /usr/bin/python3.12 /usr/bin/python

# change on pyproject.toml or uv.lock will invalidate the dependency cache
COPY pyproject.toml uv.lock README.md /tmp/

# Copy directories
COPY packages/ /tmp/packages/
COPY patches/ /tmp/patches/
COPY wheels/ /tmp/wheels/

ENV VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Verify custom wheels expected by the CPU profile.
RUN set -eux; \
    test -f /tmp/wheels/etcd3-0.12.0-py2.py3-none-any.whl; \
    test -f /tmp/wheels/fastwer-0.1.3-cp312-cp312-linux_x86_64.whl; \
    sha256sum \
        /tmp/wheels/etcd3-0.12.0-py2.py3-none-any.whl \
        /tmp/wheels/fastwer-0.1.3-cp312-cp312-linux_x86_64.whl

# Install the CPU gateway profile from uv.lock. CPU gateway intentionally does
# not install torch.
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eux; \
    cd /tmp; \
    for i in 1 2 3; do \
        if uv sync --locked --no-dev --no-install-project --compile-bytecode --python /usr/bin/python3.12; then \
            break; \
        fi; \
        if [ "$i" = "3" ]; then \
            exit 1; \
        fi; \
        echo "Attempt $i failed, purging uv cache and retrying..."; \
        uv cache clean || true; \
        sleep 5; \
    done; \
    python3 /tmp/patches/patch-omegaconf-py312.py --no-confirm

# verify that virtual environment is used and python version is correct
RUN python3 --version \
    && which python3

# Verify the dependency set installed from the uv lock.
RUN uv pip check --python /opt/venv/bin/python


FROM ubuntu:24.04

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

ARG http_proxy
ARG https_proxy
ARG no_proxy="${no_proxy}"
ARG socks_proxy
ARG TZ="Etc/UTC"
ARG VCS_REF
ARG BUILD_DATE
ARG MARIE_VERSION

ENV TERM=xterm-256color \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    socks_proxy=${socks_proxy} \
    LANG='C.UTF-8' \
    LC_ALL='C.UTF-8' \
    TZ=${TZ} \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy

# the following label use ARG hence will invalid the cache
LABEL org.opencontainers.image.created=${BUILD_DATE} \
      org.opencontainers.image.source="https://github.com/marieai/marie-ai${VCS_REF}" \
      org.opencontainers.image.version=${MARIE_VERSION} \
      org.opencontainers.image.revision=${VCS_REF}


# Install necessary apt packages
RUN apt-get update -o APT::Update::Error-Mode=any && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -yq \
        ca-certificates \
        tzdata \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-opencv \
        git \
        git-lfs \
        ssh \
        curl \
        vim \
        imagemagick \
        libtiff-dev \
        libomp-dev \
        libjemalloc-dev \
        libgoogle-perftools-dev \
        libmagickwand-dev \
        poppler-utils && \
    ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    ln -s /usr/lib/x86_64-linux-gnu/libjemalloc.so /usr/lib/libjemalloc.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libtcmalloc.so /usr/lib/libtcmalloc.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libiomp5.so /usr/lib/libiomp5.so && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove \
    && apt-get clean

ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so"
ENV WORKDIR /marie

# Copy python virtual environment from build-image
COPY --from=build-image /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install and initialize MARIE-AI, copy all necessary files
# copy will almost always invalididate the cache
COPY ./im-policy.xml /etc/ImageMagick-6/policy.xml

# copy will almost always invalid the cache
COPY . /marie/

# Testing force copy
COPY ./marie/proto/docarray_v1/ /marie/proto/docarray_v1/
COPY ./marie/proto/docarray_v2/ /marie/proto/docarray_v2/

# this is important otherwise we will get python error that module is not found
# RUN export PYTHONPATH="/marie"
# ENV PYTHONPATH "${PYTHONPATH}:/marie"

# install marie from the locked uv project without resolving new dependencies
RUN cd /marie && \
    uv sync --locked --no-dev --no-editable --compile-bytecode --python /opt/venv/bin/python && \
    uv pip check --python /opt/venv/bin/python && \
    echo "MARIE-AI installed successfully"
    #rm -rf /tmp/* && rm -rf /marie

WORKDIR ${WORKDIR}
ENTRYPOINT ["marie"]
