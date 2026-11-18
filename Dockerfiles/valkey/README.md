# Require Valkey authentication

The Compose deployment requires one full-access `marie` account. The default user
is disabled. Every key, Pub/Sub channel and command remains available after login,
including Lua and administrative operations. Health checks use the same account.

Network publication stays on port 6379 on all host interfaces. TLS is deferred;
credentials and traffic are unencrypted. Authentication does not make the port private.

## Prepare credentials

From the Marie-AI repository root, generate a new bundle outside the configuration
tree mounted into application containers. Set `--host` to the address clients
already use to reach Valkey; `localhost` is appropriate for same-host clients.

```bash
export VALKEY_SECURITY_DIR=/mnt/data/marie-ai/valkey-security
python3 tools/setup-valkey-auth.py --output-dir "$VALKEY_SECURITY_DIR" --host localhost
```

Setup requires only Python's standard library. It prints file locations, never
credentials, and refuses an existing output directory. Directories are mode 0700
and files are mode 0600. A bundle-local ignore-all rule also prevents accidental
staging when a bundle is created inside a checkout. Keep it backed up securely:

- `server/users.acl`: disabled default user and a SHA-256 password hash for `marie`.
- `server/password`: password used by the authenticated health check.
- `clients/queue-url`: authenticated `redis://` URL for gateways and workers.

Only `server/` is mounted into Valkey, read-only. Startup copies it into a private
tmpfs owned by the unprivileged `valkey` account. Missing material or an ACL that
does not match the generated full-access account and password prevents startup.
The existing `/data` volume remains unchanged. Do not run setup again to add clients;
distribute the existing client connection file securely instead.

## Configure every gateway and worker

Store the client connection file outside Git with mode 0600 on each client host.
Load it into that host's existing deployment secret/environment configuration.
Use a reachable hostname in the URL on remote hosts, retaining the same credentials.
For a shell-launched deployment:

```bash
export LLM_QUEUE_URL="$(cat "$VALKEY_SECURITY_DIR/clients/queue-url")"
export LLM_QUEUE_VALKEY_URL="$LLM_QUEUE_URL"
```

The gateway, extract, G5 annotator and all-in-one gateway Compose definitions
already forward these settings. New credentials do not require an application
code change or rebuilding the gateway/worker images. Recreate client containers
to apply environment changes; `docker compose restart` retains their old environment.
Keep shell tracing disabled and do not print connection URLs.

## Back up and cut over

Schedule a coordinated restart. Pause submissions, drain active dispatches, then
stop queue clients with the existing deployment tooling. Capture the existing
Compose project and volume so the replacement uses the same dataset. Run the
following fail-fast block from the repository root with `VALKEY_SECURITY_DIR`
exported. It checks the planned volume before stopping anything, backs up the
stopped server, verifies the archive checksum, and checks the replacement mount:

```bash
bash -eu -o pipefail <<'SH'
VALKEY_COMPOSE_PROJECT="$(docker inspect marie-valkey --format '{{index .Config.Labels "com.docker.compose.project"}}')"
VALKEY_DATA_VOLUME="$(docker inspect marie-valkey --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}')"
VALKEY_IMAGE_ID="$(docker inspect marie-valkey --format '{{.Image}}')"
VALKEY_BACKUP_DIR="/mnt/data/marie-ai/backups/valkey-$(date -u +%Y%m%dT%H%M%SZ)"
test -n "$VALKEY_COMPOSE_PROJECT"
test -n "$VALKEY_DATA_VOLUME"
test -n "$VALKEY_IMAGE_ID"
test -s "$VALKEY_SECURITY_DIR/server/password"
test -s "$VALKEY_SECURITY_DIR/server/users.acl"
docker volume inspect "$VALKEY_DATA_VOLUME" >/dev/null
VALKEY_PLANNED_VOLUME="$(docker compose --env-file ./config/.env -p "$VALKEY_COMPOSE_PROJECT" \
  -f ./Dockerfiles/docker-compose.valkey.yml --project-directory . config --format json \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["volumes"]["valkey_data"]["name"])')"
test "$VALKEY_PLANNED_VOLUME" = "$VALKEY_DATA_VOLUME"
install -d -m 700 "$VALKEY_BACKUP_DIR"
docker stop marie-valkey
docker run --rm --network none --entrypoint sh \
  --mount "type=volume,src=$VALKEY_DATA_VOLUME,dst=/data,readonly" \
  --mount "type=bind,src=$VALKEY_BACKUP_DIR,dst=/backup" \
  "$VALKEY_IMAGE_ID" -ec '
    umask 077
    tar -C /data -czf /backup/valkey-data.tar.gz .
    cd /backup
    sha256sum valkey-data.tar.gz > valkey-data.sha256
    sha256sum -c valkey-data.sha256
  '
docker compose --env-file ./config/.env -p "$VALKEY_COMPOSE_PROJECT" \
  -f ./Dockerfiles/docker-compose.valkey.yml --project-directory . \
  up -d --force-recreate --wait valkey
test "$VALKEY_DATA_VOLUME" = "$(docker inspect marie-valkey --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}')"
docker exec marie-valkey sh /etc/valkey/healthcheck.sh
docker exec marie-valkey valkey-cli ping | grep -q NOAUTH
SH
```

The archive and checksum are owned by root. Restore through a privileged container
into a new volume, never over the original volume. Keep the original volume intact.

If any check fails, keep submissions paused and inspect the failure before
continuing. On each client host, load the new URL and recreate affected services
using that host's original project and Compose files. For example, after setting
`CLIENT_COMPOSE_PROJECT`, `CLIENT_COMPOSE_FILE` and `CLIENT_SERVICE`:

```bash
export LLM_QUEUE_URL="$(cat "$VALKEY_SECURITY_DIR/clients/queue-url")"
export LLM_QUEUE_VALKEY_URL="$LLM_QUEUE_URL"
docker compose --env-file ./config/.env -p "${CLIENT_COMPOSE_PROJECT:?set existing project}" \
  -f "${CLIENT_COMPOSE_FILE:?set existing Compose file}" --project-directory . \
  up -d --force-recreate "${CLIENT_SERVICE:?set gateway or worker service}"
```

After all client containers are recreated, submit one representative queued request
and confirm its reply. Do not use `down -v` on the shared deployment.

## Roll back or rotate credentials

If clients cannot authenticate, keep submissions paused and correct their URL.
To roll back the server, retain password authentication and the existing bundle;
switch to a compatible previous image while keeping this Compose configuration.
Reverting to the old unauthenticated configuration reopens anonymous access.

For rotation, generate a bundle in a new directory, prepare the new URL on all
clients, pause/drain work, point `VALKEY_SECURITY_DIR` at the new bundle, and
recreate Valkey and its clients together. Keep the previous bundle until verified.

## Verify locally

```bash
PYTHONPATH=packages/marie-engine/src python -m pytest -q \
  --confcutdir=tests/integration/deployment \
  tests/integration/deployment/test_valkey_authentication.py
```

The test creates a separate Compose project, binds an ephemeral loopback port,
uses its own data volume and generated credentials, and removes only its resources.
