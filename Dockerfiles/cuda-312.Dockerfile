# !!! An ARG declared before a FROM is outside of a build stage, so it can’t be used in any instruction after a FROM
ARG CUDA_VERSION=12.4.1

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04 as build-image

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
ARG PIP_EXTRA_INDEX_URL="https://www.piwheels.org/simple"

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
ENV TORCH_CUDA_ARCH_LIST "7.0;7.2;7.5;8.0;8.6;8.9;9.0"

ENV PIP_DEFAULT_TIMEOUT=100 \
    # Allow statements and log messages to immediately appear
    PYTHONUNBUFFERED=1 \
    # disable a pip version check to reduce run-time & log-spam
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # cache is useless in docker image, so disable to reduce image size
    PIP_NO_CACHE_DIR=1


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
        libtiff5-dev \
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

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && python3 -m pip install --upgrade setuptools \
    && python3 -m pip install 'pybind11[global]'

# verify that virtual environment is used and python version is correct
RUN python3 --version \
    && which python3

# install custom wheels
RUN python3 -m pip install /tmp/wheels/etcd3-0.12.0-py2.py3-none-any.whl \
    && python3 -m pip install /tmp/wheels/fastwer-0.1.3-cp312-cp312-linux_x86_64.whl


RUN python3 -m pip install omegaconf==2.3.0 \
    && python3 /tmp/patches/patch-omegaconf-py312.py --no-confirm

# Order is important, need to install detectron2 last expected version is 0.6
# We also disable build isolation to avoid issues with error in detectron2 : No module named 'torch'
#
RUN python3 -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -m pip install git+https://github.com/facebookresearch/fvcore \
    && python3 -m pip install git+https://github.com/marieai/fairseq.git  \
    && python3 -m pip install --no-build-isolation  git+https://github.com/facebookresearch/detectron2.git -v

# Installing VLLM independently to avoid issues with torch version, down the road we will use as  --constraint constraints.txt
RUN python3 -m pip install psutil
RUN python3 -m pip install vllm==0.11.0
# Install flash-attn from pre-built wheel if available, otherwise build from source
RUN if [ -f /tmp/wheels/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl ]; then \
        echo "Installing flash-attn from pre-built wheel..." && \
        python3 -m pip install /tmp/wheels/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl; \
    else \
        echo "Pre-built wheel not found, building flash-attn from source..." && \
        python3 -m pip install flash-attn==2.8.3 --no-build-isolation; \
    fi

# Verify installation
RUN python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
RUN python3 -c "import flash_attn; print(f'flash_attn version: {flash_attn.__version__}')"

# ISSUE https://github.com/marieai/marie-ai/issues/136
RUN python3 -m pip install pillow==9.5.0

RUN cd /tmp/ \
    && python3 -m pip install --default-timeout=100 --compile --extra-index-url ${PIP_EXTRA_INDEX_URL} .


# No inference is being done currently 
#RUN git clone https://github.com/NVIDIA/apex && \
#    cd apex && git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82 && \
#    sed -i '/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/d' setup.py && \
#    python3 setup.py install --cpp_ext --cuda_ext

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04 

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
    python3 -m pip install --no-deps --compile . && \
    echo "MARIE-AI installed successfully"
    #rm -rf /tmp/* && rm -rf /marie

WORKDIR ${WORKDIR}
ENTRYPOINT ["marie"]
