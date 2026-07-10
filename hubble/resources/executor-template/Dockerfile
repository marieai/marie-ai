ARG JINA_VERSION=latest

FROM jinaai/jina:${JINA_VERSION}

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN uv pip install --system --compile-bytecode -r requirements.txt

# setup the workspace
COPY . /workdir/
WORKDIR /workdir

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
