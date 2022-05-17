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
        python3-opencv \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*


# Install requirements
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python3 -m pip install --no-cache-dir -U pip==22.0.4 setuptools==53.0.0 wheel==0.36.2
COPY requirements.txt /tmp/requirements/${MARIE_CONFIGURATION}.txt
RUN python3 -m pip install "pybind11[global]" # This prevents "ModuleNotFoundError: No module named 'pybind11'"
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
        python3-opencv \
        git \
        git-lfs \
        ssh \
        curl && \
    ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*


# Add a non-root user

ENV USER=${USER}
ENV GROUP=${USER}
ENV HOME /home/${USER}
ENV WORKDIR /opt/marie-icr

# Setup users
RUN groupadd -r app-svc -g 433
# RUN useradd -u 431 -r -g app-svc -d ${HOME} -s /sbin/nologin -c "app-svc user" app-svc

RUN useradd -u 431 -r -g ${GROUP} -m -d ${HOME} -s /sbin/nologin -c "${USER} user" ${USER} && \
    if [ -z ${socks_proxy} ]; then \
        echo export "GIT_SSH_COMMAND=\"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30\"" >> ${HOME}/.bashrc; \
    else \
        echo export "GIT_SSH_COMMAND=\"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ProxyCommand='nc -X 5 -x ${socks_proxy} %h %p'\"" >> ${HOME}/.bashrc; \
    fi

# Copy python virtual environment from build-image
COPY --from=build-image /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install and initialize MARIE-ICR, copy all necessary files

# RUN python3 --version

# Copy app resources
COPY --chown=${USER} ./src/info.py ${HOME}/
COPY --chown=${USER} ../ssh ${HOME}/.ssh
# COPY --chown=${USER} supervisord.conf ${HOME}/

COPY --chown=${USER} ./src/api/ /opt/marie-icr/api
COPY --chown=${USER} ./src/boxes/ /opt/marie-icr/boxes
COPY --chown=${USER} ./src/conf/ /opt/marie-icr/conf
COPY --chown=${USER} ./src/document/ /opt/marie-icr/document
COPY --chown=${USER} ./src/models/ /opt/marie-icr/models
COPY --chown=${USER} ./src/overlay/ /opt/marie-icr/overlay
COPY --chown=${USER} ./src/renderer/ /opt/marie-icr/renderer
COPY --chown=${USER} ./src/processors/ /opt/marie-icr/processors
COPY --chown=${USER} ./src/tasks/ /opt/marie-icr/tasks
COPY --chown=${USER} ./src/utils/ /opt/marie-icr/utils

COPY --chown=${USER} ./src/timer.py /opt/marie-icr/
COPY --chown=${USER} ./src/wsgi.py /opt/marie-icr/
COPY --chown=${USER} ./src/app.py /opt/marie-icr/
COPY --chown=${USER} ./src/logger.py /opt/marie-icr/

COPY --chown=${USER} ./src/register.py /opt/marie-icr/
COPY --chown=${USER} ./src/numpycontainer.py /opt/marie-icr/
COPY --chown=${USER} ./src/numpyencoder.py /opt/marie-icr/
COPY --chown=${USER} ../.build /opt/marie-icr/

# RUN python3 /opt/marie-icr/icr/info.py

RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# This is where we will map all of our configs
RUN mkdir -p /etc/marie
COPY --chown=${USER} ./config/marie.yml /etc/marie/marie.yml

# RUN all commands below as container user 
USER ${USER}
WORKDIR ${WORKDIR}

RUN mkdir ${HOME}/logs /tmp/supervisord 
RUN chown ${USER} ${HOME}/logs

EXPOSE 5000
ENTRYPOINT ["/usr/bin/supervisord"]
