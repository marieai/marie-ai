# syntax=docker/dockerfile:1
ARG CUDA_VERSION=13.0.1

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS build-image

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# Tweak this list to reduce build time
# https://developer.nvidia.com/cuda-gpus
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"

RUN test -e /usr/local/cuda/bin/nvcc && /usr/local/cuda/bin/nvcc --version

# Ubuntu 24.04 ships Python 3.12 natively; no external PPA needed.
RUN apt-get update -o APT::Update::Error-Mode=any && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        gcc \
        g++ \
        gfortran \
        git \
        imagemagick \
        libgl1 \
        libglib2.0-0 \
        libjpeg-dev \
        libmagickwand-dev \
        libomp-dev \
        libopenblas-dev \
        libopenmpi-dev \
        libpng-dev \
        libpq-dev \
        libtiff-dev \
        lshw \
        ninja-build \
        openmpi-bin \
        openmpi-common \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        unzip \
        wget \
        zlib1g && \
    rm -rf /var/lib/apt/lists/*

# Ensure the correct symbolic links
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# change on pyproject.toml or uv.lock will invalidate the dependency cache
COPY pyproject.toml uv.lock README.md /tmp/
# the project version is dynamic (attr: marie._version.__version__); uv sync needs it
# to build project metadata even with --no-install-project
COPY marie/_version.py /tmp/marie/_version.py
# Copy directories
COPY packages/ /tmp/packages/
COPY patches/ /tmp/patches/
COPY wheels/ /tmp/wheels/

ENV VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN set -eux; \
    test -f /tmp/wheels/etcd3-0.12.0-py2.py3-none-any.whl; \
    test -f /tmp/wheels/fastwer-0.1.3-cp312-cp312-linux_x86_64.whl; \
    test "$(find /tmp/wheels -maxdepth 1 -name 'fairseq-*.whl' | wc -l)" -eq 1; \
    test "$(find /tmp/wheels -maxdepth 1 -name 'detectron2-*.whl' | wc -l)" -eq 1; \
    test "$(find /tmp/wheels -maxdepth 1 \( -name 'faiss*.whl' -o -name 'faiss_gpu_cu13-*.whl' \) | wc -l)" -eq 1; \
    test "$(find /tmp/wheels -maxdepth 1 -name 'vllm-*.whl' | wc -l)" -eq 1; \
    sha256sum \
        /tmp/wheels/etcd3-0.12.0-py2.py3-none-any.whl \
        /tmp/wheels/fastwer-0.1.3-cp312-cp312-linux_x86_64.whl \
        /tmp/wheels/fairseq-*.whl \
        /tmp/wheels/detectron2-*.whl \
        /tmp/wheels/faiss*.whl \
        /tmp/wheels/vllm-*.whl

# Install the CUDA profile dependencies from uv.lock (project excluded so this
# layer stays cached across source-only changes).
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eux; \
    cd /tmp; \
    for i in 1 2 3; do \
        if uv sync --locked --no-dev --extra cu130 --no-install-project --compile-bytecode --python /usr/bin/python3.12; then \
            break; \
        fi; \
        if [ "$i" = "3" ]; then \
            exit 1; \
        fi; \
        echo "Attempt $i failed, retrying..."; \
        sleep 5; \
    done

# Full source; install the project itself into the venv, apply the installed-
# metadata patches, and verify the final dependency set. The runtime stage only
# copies the finished /opt/venv — it never runs uv.
COPY . /marie/
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /marie && \
    uv sync --locked --no-dev --group runtime --extra cu130 --no-editable --compile-bytecode --python /usr/bin/python3.12 && \
    test -x /opt/venv/bin/marie && \
    python3 -c 'import torch, vllm; assert torch.__version__.startswith("2.12.1"); print("vllm", vllm.__version__, "torch", torch.__version__)' && \
    python3 /tmp/patches/patch-omegaconf-py312.py --no-confirm && \
    python3 /tmp/patches/patch-detectron2-metadata.py --no-confirm && \
    uv pip check --python /opt/venv/bin/python && \
    python3 --version && which python3


FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /bin/uv

ARG TZ="Etc/UTC"
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
ARG BUILD_NUMBER=unknown
ARG MARIE_VERSION=unknown
ARG IMAGE_NAME=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    TERM=xterm-256color \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=${TZ} \
    PYTHONUNBUFFERED=1

# vLLM/Triton compile runtime launchers on first use.
# Ubuntu 24.04 ships Python 3.12 natively; no external PPA needed.
RUN apt-get update -o APT::Update::Error-Mode=any && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential \
        cuda-nvcc-13-0 \
        git \
        graphviz \
        imagemagick \
        libcurand-dev-13-0 \
        libgl1 \
        libglib2.0-0 \
        libjemalloc2 \
        libmagickwand-6.q16-7t64 \
        libopenblas0 \
        libpq5 \
        openssh-client \
        poppler-utils \
        python3.12 \
        python3.12-dev \
        tzdata && \
    test -x /usr/local/cuda/bin/nvcc && \
    test -f /usr/local/cuda/include/curand.h && \
    /usr/local/cuda/bin/nvcc --version && \
    ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

# jemalloc is the chosen allocator (runtime lib, not the -dev symlink).
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"

# Copy the finished python virtual environment from build-image
COPY --from=build-image /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN /opt/venv/bin/python -c 'import sys; from marie.build_info import write_build_info; write_build_info(sys.argv[1], version=sys.argv[2], git_commit=sys.argv[3], build_time=sys.argv[4], build_number=sys.argv[5], image=sys.argv[6])' \
        /etc/marie-ai/build-info.json \
        "${MARIE_VERSION}" \
        "${VCS_REF}" \
        "${BUILD_DATE}" \
        "${BUILD_NUMBER}" \
        "${IMAGE_NAME}"

ENV MARIE_BUILD_INFO_PATH="/etc/marie-ai/build-info.json"

COPY ./im-policy.xml /etc/ImageMagick-6/policy.xml

# Runtime assets (configs, templates, protos); the package itself is already
# installed non-editably inside /opt/venv.
COPY . /marie/

WORKDIR /marie

# Labels last: BUILD_DATE/VCS_REF change every build and would otherwise
# invalidate every layer after them.
LABEL org.opencontainers.image.vendor="Marie AI" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="MarieAI" \
      org.opencontainers.image.description="Deploy production-ready AI agent systems for document processing, content analysis, and multimodal intelligence via containerized cloud services" \
      org.opencontainers.image.authors="hello@marieai.co" \
      org.opencontainers.image.url="https://github.com/marieai/marie-ai" \
      org.opencontainers.image.documentation="https://docs.marieai.co" \
      org.opencontainers.image.source="https://github.com/marieai/marie-ai" \
      org.opencontainers.image.created=${BUILD_DATE} \
      org.opencontainers.image.version=${MARIE_VERSION} \
      org.opencontainers.image.revision=${VCS_REF}

ENTRYPOINT ["/opt/venv/bin/marie"]
