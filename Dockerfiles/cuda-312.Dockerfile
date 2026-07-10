# !!! An ARG declared before a FROM is outside of a build stage, so it can’t be used in any instruction after a FROM
ARG CUDA_VERSION=13.0.1

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS build-image

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
# https://developer.nvidia.com/cuda-gpus
ENV TORCH_CUDA_ARCH_LIST "7.5;8.0;8.6;8.9;9.0"

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy


RUN test -e /usr/local/cuda/bin/nvcc
RUN /usr/local/cuda/bin/nvcc --version

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
# change on pyproject.toml, extra-requirements.txt, or setup.py will invalidate the cache
COPY requirements.txt extra-requirements.txt setup.py pyproject.toml /tmp/
# Copy directories
COPY patches/ /tmp/patches/
COPY wheels/ /tmp/wheels/
COPY requirements/ /tmp/requirements/

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN uv pip install --python /opt/venv/bin/python --compile-bytecode 'setuptools<81' 'pybind11[global]'

# verify that virtual environment is used and python version is correct
RUN python3 --version \
    && which python3

RUN set -eux; \
    test -f /tmp/wheels/etcd3-0.12.0-py2.py3-none-any.whl; \
    test -f /tmp/wheels/fastwer-0.1.3-cp312-cp312-linux_x86_64.whl; \
    test "$(find /tmp/wheels -maxdepth 1 -name 'fairseq-*.whl' | wc -l)" -eq 1; \
    test "$(find /tmp/wheels -maxdepth 1 -name 'detectron2-*.whl' | wc -l)" -eq 1; \
    test "$(find /tmp/wheels -maxdepth 1 \( -name 'faiss*.whl' -o -name 'faiss_gpu_cu13-*.whl' \) | wc -l)" -eq 1; \
    sha256sum \
        /tmp/wheels/etcd3-0.12.0-py2.py3-none-any.whl \
        /tmp/wheels/fastwer-0.1.3-cp312-cp312-linux_x86_64.whl \
        /tmp/wheels/fairseq-*.whl \
        /tmp/wheels/detectron2-*.whl \
        /tmp/wheels/faiss*.whl

# Install the CUDA profile lock generated from pyproject dependency groups.
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eux; \
    cd /tmp; \
    for i in 1 2 3; do \
        if uv pip install --python /opt/venv/bin/python --compile-bytecode --torch-backend cu130 --index-strategy unsafe-best-match -r requirements/uv/marie-cuda-torch212-cu130.lock.txt; then \
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

# Install marie packages (monorepo local packages)
COPY packages/ /tmp/packages/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps --compile-bytecode /tmp/packages/marie-kernel \
    && uv pip install --python /opt/venv/bin/python --no-deps --compile-bytecode /tmp/packages/marie-wasm \
    && uv pip install --python /opt/venv/bin/python --no-deps --compile-bytecode /tmp/packages/marie-mem0 \
    && uv pip install --python /opt/venv/bin/python --no-deps --compile-bytecode /tmp/packages/marie-mcp

# Install marie-ai package from the staged source. Dependencies come from the uv lock above.
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /tmp/ \
    && uv pip install --python /opt/venv/bin/python --no-deps --compile-bytecode . \
    && python3 /tmp/patches/patch-detectron2-metadata.py --no-confirm \
    && uv pip check --python /opt/venv/bin/python


# No inference is being done currently 
#RUN git clone https://github.com/NVIDIA/apex && \
#    cd apex && git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82 && \
#    sed -i '/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/d' setup.py && \
#    python3 setup.py install --cpp_ext --cuda_ext

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04

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
    http_proxy=${http_proxy}   \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    socks_proxy=${socks_proxy} \
    LANG='C.UTF-8'  \
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
        graphviz \
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
    uv pip install --python /opt/venv/bin/python --no-deps --compile-bytecode . && \
    echo "MARIE-AI installed successfully"
    #rm -rf /tmp/* && rm -rf /marie

WORKDIR ${WORKDIR}
ENTRYPOINT ["marie"]
