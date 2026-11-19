#!/bin/bash
set -Eeuo pipefail

: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"
export POSTGRES_USER="${POSTGRES_USER:-postgres}"
export POSTGRES_DB="${POSTGRES_DB:-postgres}"
export USERNAME="${DOCUMENTDB_USERNAME:-marie}"
export PASSWORD="${DOCUMENTDB_PASSWORD:-$POSTGRES_PASSWORD}"

if [[ "$POSTGRES_DB" != postgres ]]; then
    echo 'Marie storage requires POSTGRES_DB=postgres.' >&2
    exit 1
fi
if [[ "$POSTGRES_USER" == "$USERNAME" || "$POSTGRES_USER" == documentdb ]]; then
    echo 'POSTGRES_USER must differ from the document client and internal documentdb owner.' >&2
    exit 1
fi
if [[ "$USERNAME" == documentdb ]]; then
    echo 'DOCUMENTDB_USERNAME must differ from the internal documentdb owner.' >&2
    exit 1
fi
if [[ ! -r /docker-entrypoint-initdb.d/init.sql ]]; then
    echo 'Mount config/psql/init.sql at /docker-entrypoint-initdb.d/init.sql.' >&2
    exit 1
fi

marie_stop() {
    local status="$1"
    trap - EXIT
    if [[ -n "${marie_server_pid:-}" ]]; then
        kill -TERM "$marie_server_pid" 2>/dev/null || true
        wait "$marie_server_pid" || true
    fi
    exit "$status"
}
trap 'marie_stop "$?"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

rm -f /tmp/marie-postgres-ready
/home/documentdb/gateway/scripts/emulator_entrypoint.sh "$@" &
marie_server_pid=$!
while ! /usr/local/bin/documentdb-healthcheck >/dev/null 2>&1; do
    if ! kill -0 "$marie_server_pid" 2>/dev/null; then
        wait "$marie_server_pid"
        exit 1
    fi
    sleep 0.2
done

if [[ ! -f "${DATA_PATH:-/data}/.marie-sql-initialized" ]]; then
    psql --username documentdb --port "$POSTGRESQL_PORT" --dbname postgres \
        --set=ON_ERROR_STOP=1 <<'SQL'
\getenv app_user POSTGRES_USER
\getenv app_password POSTGRES_PASSWORD
SELECT format('CREATE ROLE %I LOGIN SUPERUSER PASSWORD %L', :'app_user', :'app_password')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'app_user')\gexec
SQL
    psql --username "$POSTGRES_USER" --port "$POSTGRESQL_PORT" --dbname postgres \
        --set=ON_ERROR_STOP=1 --file=/docker-entrypoint-initdb.d/init.sql
    touch "${DATA_PATH:-/data}/.marie-sql-initialized"
fi
mongosh --nodb --quiet --norc --eval '
const connection = new Mongo("mongodb://" + encodeURIComponent(process.env.USERNAME) +
    ":" + encodeURIComponent(process.env.PASSWORD) + "@127.0.0.1:" +
    process.env.DOCUMENTDB_PORT + "/?authMechanism=SCRAM-SHA-256&serverSelectionTimeoutMS=5000");
if (connection.getDB("admin").runCommand({ping: 1}).ok !== 1) {
    throw Error("Native DocumentDB authentication check failed");
}' >/dev/null
touch /tmp/marie-postgres-ready
wait "$marie_server_pid"
