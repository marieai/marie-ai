# !!! An ARG declared before a FROM is outside of a build stage, so it can’t be used in any instruction after a FROM
#ARG CUDA_VERSION=11.6.1
ARG CUDA_VERSION=11.8.0
#ARG CUDA_VERSION=11.3.1

#FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04 as build-image
FROM ubuntu:22.04 as build-image

ARG http_proxy
ARG https_proxy
ARG no_proxy="${no_proxy}"
ARG socks_proxy
ARG TZ="Etc/UTC"
ARG MARIE_CONFIGURATION="production"

# given by builder's env
ARG VCS_REF
ARG PY_VERSION=3.10
ARG BUILD_DATE
ARG MARIE_VERSION
ARG TARGETPLATFORM
ARG PIP_EXTRA_INDEX_URL="https://www.piwheels.org/simple"

# constant, wont invalidate cache
LABEL org.opencontainers.image.vendor="Marie AI" \
      org.opencontainers.image.licenses="Apache 2.0" \
      org.opencontainers.image.title="MarieAI" \
      org.opencontainers.image.description="Build multimodal AI services via cloud native technologies" \
      org.opencontainers.image.authors="hello@marieai.co" \
      org.opencontainers.image.url="https://github.com/marieai/marie-ai" \
      org.opencontainers.image.documentation="https://docs.marieai.co"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.2;7.5;8.0;8.6;8.9;9.0" \
    PIP_DEFAULT_TIMEOUT=100

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    wget build-essential curl git lshw zlib1g pkg-config \
    python3.10 python3.10-dev python3-virtualenv python3.10-venv \
    python3-dev python3-pip python3-wheel python3-packaging \
    python3-opencv python3-setuptools libopenblas-dev libopenmpi-dev \
    openmpi-bin openmpi-common gfortran libomp-dev ninja-build cmake \
    gcc g++ imagemagick libmagickwand-dev libtiff5-dev libjpeg-dev \
    libpng-dev libpq-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y && apt-get clean

# Install requirements
# change on extra-requirements.txt, setup.py will invalid the cache
COPY requirements.txt extra-requirements.txt setup.py /tmp/

RUN python3.10 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -U pip==22.0.4 setuptools==53.0.0 wheel==0.36.2 && \
    pip install --no-cache-dir --upgrade setuptools && \
    pip install "pybind11[global]" && \
    pip install intel-openmp && \
    pip install --pre torch[dynamo] torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu --force && \
    pip install git+https://github.com/facebookresearch/fvcore && \
    pip install git+https://github.com/pytorch/fairseq.git && \
    pip install git+https://github.com/ying09/TextFuseNet.git && \
    pip install 'git+https://github.com/facebookresearch/detectron2.git' && \
    cd /tmp/ && \
    pip install --default-timeout=1000 --compile --extra-index-url ${PIP_EXTRA_INDEX_URL} .

FROM ubuntu:22.04

ARG http_proxy
ARG https_proxy
ARG no_proxy="${no_proxy}"
ARG socks_proxy
ARG TZ="Etc/UTC"

ENV TERM=xterm-256color \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    socks_proxy=${socks_proxy} \
    LANG='C.UTF-8' \
    LC_ALL='C.UTF-8' \
    TZ=${TZ}

# the following label use ARG hence will invalid the cache
LABEL org.opencontainers.image.created=${BUILD_DATE} \
      org.opencontainers.image.source="https://github.com/marieai/marie-ai${VCS_REF}" \
      org.opencontainers.image.version=${MARIE_VERSION} \
      org.opencontainers.image.revision=${VCS_REF}

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -y \
    ca-certificates tzdata python3-distutils python3-opencv git git-lfs \
    ssh curl vim imagemagick libtiff-dev libomp-dev libjemalloc-dev \
    libgoogle-perftools-dev libmagickwand-dev tesseract-ocr && \
    ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    ln -s /usr/lib/x86_64-linux-gnu/libjemalloc.so /usr/lib/libjemalloc.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libtcmalloc.so /usr/lib/libtcmalloc.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libiomp5.so /usr/lib/libiomp5.so && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y && apt-get clean

ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so" \
    WORKDIR /marie

# Copy python virtual environment from build-image
COPY --from=build-image /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copy all necessary files except the Python code
COPY ./im-policy.xml /etc/ImageMagick-6/policy.xml
COPY ./config /marie/config
COPY ./scripts /marie/scripts
COPY ./requirements.txt /marie/requirements.txt
COPY ./extra-requirements.txt /marie/extra-requirements.txt
COPY ./setup.py /marie/setup.py

# Install dependencies
RUN cd /marie && \
    pip install --no-deps --compile . && \
    rm -rf /tmp/*

# Copy only the Python code
COPY ./marie /marie/marie

WORKDIR ${WORKDIR}
ENTRYPOINT ["marie"]