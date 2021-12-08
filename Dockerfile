FROM ubuntu:20.04 as build-image

ARG PYTHON_VERSION=3.8

ARG http_proxy
ARG https_proxy
ARG no_proxy="${no_proxy}"
ARG socks_proxy
ARG MARIE_CONFIGURATION="production"


RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -yq \
        build-essential \
        curl \
        git \
        pkg-config \
        python3-dev \
        python3-pip \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*


# Install requirements
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python3 -m pip install --no-cache-dir -U pip==21.0.1 setuptools==53.0.0 wheel==0.36.2
#COPY ./requirements/ /tmp/requirements/
COPY requirements.txt /tmp/requirements/${MARIE_CONFIGURATION}.txt
#RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements/${MARIE_CONFIGURATION}.txt


FROM ubuntu:20.04

ARG http_proxy
ARG https_proxy
ARG no_proxy="${no_proxy}"
ARG socks_proxy
ARG TZ="Etc/UTC"

ENV TERM=xterm \
    http_proxy=${http_proxy}   \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    socks_proxy=${socks_proxy} \
    LANG='C.UTF-8'  \
    LC_ALL='C.UTF-8' \
    TZ=${TZ}


ARG USER="app-svc"
ARG MARIE_CONFIGURATION="production"
ENV MARIE_CONFIGURATION=${MARIE_CONFIGURATION}

# Install necessary apt packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -yq \
        ca-certificates \
        supervisor \
        tzdata \
        python3-distutils \
        git \
        git-lfs \
        ssh \
        curl && \
    ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*


# Add a non-root user
ENV USER=${USER}
ENV HOME /home/${USER}
RUN adduser --shell /bin/bash --disabled-password --gecos "" ${USER} && \
    if [ -z ${socks_proxy} ]; then \
        echo export "GIT_SSH_COMMAND=\"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30\"" >> ${HOME}/.bashrc; \
    else \
        echo export "GIT_SSH_COMMAND=\"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ProxyCommand='nc -X 5 -x ${socks_proxy} %h %p'\"" >> ${HOME}/.bashrc; \
    fi


# Copy python virtual environment from build-image
COPY --from=build-image /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install and initialize MARIE-ICR, copy all necessary files

RUN python3 --version    

# RUN all commands below as conteainer user 
USER ${USER}
WORKDIR ${HOME}
