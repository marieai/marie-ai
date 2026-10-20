FROM golang:1.25-alpine AS build

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -o /out/marie-plugin-daemon ./cmd/server

FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv
COPY --from=build /out/marie-plugin-daemon /usr/local/bin/marie-plugin-daemon

RUN groupadd --gid 10001 marie && \
    useradd --uid 10001 --gid 10001 --create-home marie && \
    install -d -o 10001 -g 10001 /mnt/data/marie-ai && \
    python -c 'import ssl; assert ssl.create_default_context().get_ca_certs()'

ENV HOME=/home/marie \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MARIE_PLUGIN_STORAGE_ROOT=/mnt/data/marie-ai

USER 10001:10001
WORKDIR /mnt/data/marie-ai
EXPOSE 8099
ENTRYPOINT ["marie-plugin-daemon", "--addr", "0.0.0.0:8099"]
