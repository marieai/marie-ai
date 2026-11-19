FROM ghcr.io/documentdb/documentdb/documentdb-local:pg17-1.2.0@sha256:1d40546ccf53a0229f98a0848b7c3d6d74dc35a9fa535a1d1c8f70e3bed10d23

USER root

# Install maintenance tools without upgrading the base image's PostgreSQL server.
RUN set -eux; \
    repack_postgres_version="$(dpkg-query -W -f='${Version}' postgresql-17)"; \
    /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y; \
    apt-get update; \
    apt-get install -y --no-install-recommends --no-remove \
        "postgresql-17=${repack_postgres_version}" \
        postgresql-17-repack=1.5.3-1.pgdg13+1; \
    sed -i 's/echo "pg_cron,/echo "pg_stat_statements, pg_cron,/' \
        /home/documentdb/gateway/scripts/preload_libraries.sh; \
    grep -Fq 'echo "pg_stat_statements, pg_cron,' \
        /home/documentdb/gateway/scripts/preload_libraries.sh; \
    sed -i "s/^shared_preload_libraries = '/shared_preload_libraries = 'pg_stat_statements, /" \
        /data/postgresql.conf; \
    grep -Fq "shared_preload_libraries = 'pg_stat_statements," /data/postgresql.conf; \
    rm -rf /var/lib/apt/lists/*

COPY --chmod=755 entrypoint.sh /usr/local/bin/marie-documentdb-entrypoint
COPY --chmod=755 healthcheck.sh /usr/local/bin/marie-documentdb-healthcheck

ENV POSTGRES_USER=postgres POSTGRES_DB=postgres \
    DOCUMENTDB_USERNAME=marie POSTGRESQL_PORT=5432 DOCUMENTDB_PORT=27017 \
    ALLOW_EXTERNAL_CONNECTIONS=true DOCUMENTDB_USAGE_TELEMETRY=false

USER documentdb
EXPOSE 5432 27017
HEALTHCHECK --interval=10s --timeout=10s --start-period=120s --retries=3 \
    CMD ["/usr/local/bin/marie-documentdb-healthcheck"]
ENTRYPOINT ["/usr/local/bin/marie-documentdb-entrypoint"]
CMD []
