FROM ghcr.io/ferretdb/postgres-documentdb:17-0.103.0

# Install maintenance tools without upgrading the base image's PostgreSQL server.
RUN set -eux; \
    repack_postgres_version="$(dpkg-query -W -f='${Version}' postgresql-17)"; \
    apt-get update; \
    apt-get install -y --no-install-recommends --no-remove \
        "postgresql-17=${repack_postgres_version}" \
        postgresql-17-repack=1.5.3-1.pgdg12+1; \
    rm -rf /var/lib/apt/lists/*
