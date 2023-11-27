# !!! An ARG declared before a FROM is outside of a build stage, so it can’t be used in any instruction after a FROM
#ARG CUDA_VERSION=11.6.1
ARG CUDA_VERSION=11.8.0
#ARG CUDA_VERSION=11.3.1

#FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04 as build-image
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 as build-image

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


ENV DEBIAN_FRONTEND=noninteractive

# constant, wont invalidate cache
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa


RUN apt-get update && apt-get upgrade -y && \
        apt install wget -y \
        build-essential \
        curl \
        git \
        lshw \
        zlib1g \
        pkg-config \
        python3.10 \
        python3.10-dev \
        python3-virtualenv \
        python3.10-venv \
        python3-dev \
        python3-pip \
        python3-wheel \
        python3-opencv \
        python3-venv \
        python3-setuptools \
        libopenblas-dev \
        libopenmpi-dev \
        openmpi-bin \
        openmpi-common \
        gfortran \
        libomp-dev \
        ninja-build \
        cmake \
        imagemagick \
        libmagickwand-dev \
        libtiff5-dev \
        libjpeg-dev \
        libpng-dev \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove \
    && apt-get clean

# Install requirements
# change on extra-requirements.txt, setup.py will invalid the cache
COPY requirements.txt extra-requirements.txt setup.py /tmp/

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python3 -m pip install --no-cache-dir -U pip==22.0.4 setuptools==53.0.0 wheel==0.36.2
RUN python3 -m pip install --no-cache-dir install --upgrade setuptools
RUN python3 -m pip install "pybind11[global]" # This prevents "ModuleNotFoundError: No module named 'pybind11'"

RUN python3 -m pip install intel-openmp
# USING NIGHTLY causes CUDA multiprocessing error
# https://forums.developer.nvidia.com/t/python-opencv-multiprocessing-doesnt-work-with-cuda/180195
#RUN python3 -m pip install --pre torch[dynamo] torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu118 --force
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  --force

# Order is important, need to install detectron2 last expected version is 0.6
RUN python3 -m pip install git+https://github.com/facebookresearch/fvcore && \
    python3 -m pip install git+https://github.com/pytorch/fairseq.git && \
    python3 -m pip install git+https://github.com/ying09/TextFuseNet.git && \
    python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN cd /tmp/ && \
    python3 -m pip install --default-timeout=1000  --compile --extra-index-url ${PIP_EXTRA_INDEX_URL} .

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04

ARG http_proxy
ARG https_proxy
ARG no_proxy="${no_proxy}"
ARG socks_proxy
ARG TZ="Etc/UTC"

ENV TERM=xterm-256color \
    http_proxy=${http_proxy}   \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    socks_proxy=${socks_proxy} \
    LANG='C.UTF-8'  \
    LC_ALL='C.UTF-8' \
    TZ=${TZ}

# the following label use ARG hence will invalid the cache
LABEL org.opencontainers.image.created=${BUILD_DATE} \
      org.opencontainers.image.source="https://github.com/marieai/marie-ai${VCS_REF}" \
      org.opencontainers.image.version=${MARIE_VERSION} \
      org.opencontainers.image.revision=${VCS_REF}


# Install necessary apt packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -yq \
        ca-certificates \
        tzdata \
        python3-distutils \
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
        libmagickwand-dev && \
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

## This is where we will map all of our configs
#RUN mkdir -p /etc/marie
#COPY ./config/marie.yml /etc/marie/marie.yml

# copy will almost always invalid the cache
COPY . /marie/

# this is important otherwise we will get python error that module is not found
# RUN export PYTHONPATH="/marie"
# ENV PYTHONPATH "${PYTHONPATH}:/marie"

# install marie again but this time no deps
RUN cd /marie && \
    pip install --no-deps --compile . && \
    rm -rf /tmp/* && rm -rf /marie

WORKDIR ${WORKDIR}
ENTRYPOINT ["marie"]
#ENTRYPOINT ["pip", "list"]