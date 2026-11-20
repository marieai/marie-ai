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

Keep each component's credentials in its own directory under
`/mnt/data/marie-ai/security/`. Valkey uses `security/valkey/`; other components
can use sibling directories. The Valkey container mounts only its `server/` directory.

```bash
export VALKEY_SECURITY_DIR=/mnt/data/marie-ai/security/valkey
python3 tools/setup-valkey-auth.py --output-dir "$VALKEY_SECURITY_DIR" --host localhost
```

Setup requires only Python's standard library. It prints file locations, never
credentials, and refuses an existing output directory. Directories are mode 0700
and files are mode 0600. A bundle-local ignore-all rule also prevents accidental
staging when a bundle is created inside a checkout. Generated files:

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
For a shell-launched deployment, optionally load the generated URL:

```bash
export LLM_QUEUE_URL="$(cat "$VALKEY_SECURITY_DIR/clients/queue-url")"
export LLM_QUEUE_VALKEY_URL="$LLM_QUEUE_URL"
```

The gateway, extract, G5 annotator and all-in-one gateway Compose definitions
already forward these settings. New credentials do not require an application
code change or rebuilding the gateway/worker images. Recreate client containers
to apply environment changes; `docker compose restart` retains their old environment.
Keep shell tracing disabled and do not print connection URLs.

## Start Valkey

For a new setup, run from the Marie-AI repository root after preparing credentials:

```bash
export VALKEY_SECURITY_DIR=/mnt/data/marie-ai/security/valkey

docker network inspect marie_default >/dev/null 2>&1 \
  || docker network create marie_default

docker compose \
  --env-file ./config/.env \
  -p marie \
  -f ./Dockerfiles/docker-compose.valkey.yml \
  --project-directory . \
  up -d --force-recreate --wait valkey
```

## Verify authentication

```bash
docker exec marie-valkey sh /etc/valkey/healthcheck.sh \
  && echo "Authenticated access OK"

docker exec marie-valkey valkey-cli ping
```

The first command must print `Authenticated access OK`. The unauthenticated ping
must report `NOAUTH Authentication required.`

## Start gateways and workers

Start gateways and workers with the authenticated `LLM_QUEUE_URL` in their existing
deployment configuration. For existing Docker clients, use their original Compose
project and files with `up -d --force-recreate` to apply the new environment.
For host or IDE processes, restart them with the updated environment.

Submit one representative queued request and confirm its reply.

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
