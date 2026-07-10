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
ARG PYTHON_INSTALLER=pip
# Note: piwheels is for ARM/Raspberry Pi only, not needed for x86_64 builds
ARG PIP_EXTRA_INDEX_URL=""

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
ENV PIP_DEFAULT_TIMEOUT=100 \
    # Allow statements and log messages to immediately appear
    PYTHONUNBUFFERED=1 \
    # disable a pip version check to reduce run-time & log-spam
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # cache is useless in docker image, so disable to reduce image size
    PIP_NO_CACHE_DIR=1 \
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

# Install requirements
# change on extra-requirements.txt, setup.py will invalid the cache
COPY requirements.txt extra-requirements.txt setup.py /tmp/

# Copy directories
COPY patches/ /tmp/patches/
COPY wheels/ /tmp/wheels/
COPY requirements/ /tmp/requirements/

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && if [ "$PYTHON_INSTALLER" = "uv" ]; then \
        uv pip install --python /opt/venv/bin/python --compile-bytecode 'setuptools<81' 'pybind11[global]'; \
    else \
        python3 -m pip install 'setuptools<81' \
        && python3 -m pip install 'pybind11[global]'; \
    fi

# verify that virtual environment is used and python version is correct
RUN python3 --version \
    && which python3

# Install torch explicitly so the CPU profile is pinned to the same torch lane as the lock artifact.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    export PIP_NO_CACHE_DIR=0 && \
    for i in 1 2 3; do \
        if [ "$PYTHON_INSTALLER" = "uv" ]; then \
            uv pip install --python /opt/venv/bin/python --compile-bytecode --torch-backend cpu \
                torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu; \
        else \
            python3 -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu; \
        fi \
        && break \
        || { echo "Attempt $i failed, purging installer cache and retrying..."; python3 -m pip cache purge 2>/dev/null || true; uv cache clean 2>/dev/null || true; sleep 5; }; \
    done

# install custom wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$PYTHON_INSTALLER" = "uv" ]; then \
        uv pip install --python /opt/venv/bin/python --no-deps --compile-bytecode /tmp/wheels/etcd3-0.12.0-py2.py3-none-any.whl \
        && uv pip install --python /opt/venv/bin/python --no-deps --compile-bytecode /tmp/wheels/fastwer-0.1.3-cp312-cp312-linux_x86_64.whl; \
    else \
        python3 -m pip install /tmp/wheels/etcd3-0.12.0-py2.py3-none-any.whl \
        && python3 -m pip install /tmp/wheels/fastwer-0.1.3-cp312-cp312-linux_x86_64.whl; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$PYTHON_INSTALLER" = "uv" ]; then \
        uv pip install --python /opt/venv/bin/python --compile-bytecode omegaconf==2.3.0; \
    else \
        python3 -m pip install omegaconf==2.3.0; \
    fi \
    && python3 /tmp/patches/patch-omegaconf-py312.py --no-confirm

# Install marie packages (monorepo local packages)
COPY packages/ /tmp/packages/
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$PYTHON_INSTALLER" = "uv" ]; then \
        uv pip install --python /opt/venv/bin/python --compile-bytecode --torch-backend cpu --index-strategy unsafe-best-match -c /tmp/requirements/torch-2.12-cpu.txt /tmp/packages/marie-kernel \
        && uv pip install --python /opt/venv/bin/python --compile-bytecode --torch-backend cpu --index-strategy unsafe-best-match -c /tmp/requirements/torch-2.12-cpu.txt /tmp/packages/marie-wasm \
        && uv pip install --python /opt/venv/bin/python --compile-bytecode --torch-backend cpu --index-strategy unsafe-best-match -c /tmp/requirements/torch-2.12-cpu.txt /tmp/packages/marie-mem0 \
        && uv pip install --python /opt/venv/bin/python --compile-bytecode --torch-backend cpu --index-strategy unsafe-best-match -c /tmp/requirements/torch-2.12-cpu.txt /tmp/packages/marie-mcp; \
    else \
        python3 -m pip install /tmp/packages/marie-kernel \
        && python3 -m pip install /tmp/packages/marie-wasm \
        && python3 -m pip install /tmp/packages/marie-mem0 \
        && python3 -m pip install /tmp/packages/marie-mcp; \
    fi

# Install marie-ai package with dependencies
# Use --prefer-binary to speed up installation by using pre-built wheels when available
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /tmp/ \
    && if [ "$PYTHON_INSTALLER" = "uv" ]; then \
        uv pip install --python /opt/venv/bin/python --compile-bytecode --torch-backend cpu --index-strategy unsafe-best-match -c /tmp/requirements/torch-2.12-cpu.txt .; \
    else \
        python3 -m pip install --default-timeout=100 --compile --prefer-binary .; \
    fi


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
ARG PYTHON_INSTALLER=pip

ENV TERM=xterm-256color \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    socks_proxy=${socks_proxy} \
    LANG='C.UTF-8' \
    LC_ALL='C.UTF-8' \
    TZ=${TZ} \
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
        libreoffice-core \
        libreoffice-writer \
        libreoffice-calc \
        libreoffice-impress \
        poppler-utils \
        texlive-latex-base \
        djvulibre-bin && \
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

# Install PIP
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py


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

# install marie again but this time no deps
RUN cd /marie && \
    if [ "$PYTHON_INSTALLER" = "uv" ]; then \
        uv pip install --python /opt/venv/bin/python --no-deps --compile-bytecode .; \
    else \
        python3 -m pip install --no-deps --compile .; \
    fi && \
    echo "MARIE-AI installed successfully"
    #rm -rf /tmp/* && rm -rf /marie

WORKDIR ${WORKDIR}
ENTRYPOINT ["marie"]
