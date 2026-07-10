# TODO use fixed jina version for deterministic execution
FROM jinaai/jina:test-pip

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

RUN uv pip install --system torch

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
