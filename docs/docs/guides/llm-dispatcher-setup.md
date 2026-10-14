---
sidebar_position: 13
---

# LLM dispatcher setup

Run `tools/stress/llm_dispatch_deployment.py` to exercise the durable LLM dispatcher through a real gateway, scheduler, and external Document Annotator deployment. The gateway dispatches queued completion requests to AIMock; the Annotator processes synthetic pages and writes results to an isolated MinIO bucket.

The runner generates its own gateway configuration, credentials, fabric ID, and ports. Use the steps below to prepare the Python environment and Docker fixtures, start the deployment, inspect it, and stop it.

:::important Select a run mode

Use `--start-only` for startup, `--invoke-only` for ordinary annotation, or a qualification mode such as `--s03`. Running with only `--output` and `--fixture-manifest` starts the services and then exits with `Deployment coverage has not executed`. Readiness alone does not mean the failure scenarios passed.

:::

## Prepare the source environment

Run these commands from your `marie-ai` checkout:

```bash
cd ~/dev/marieai/marie-ai
.venv/bin/python --version
docker info --format '{{.ServerVersion}}'
docker compose version
```

Use Python 3.12 on Linux and a Docker Engine reachable by your user through the default local socket. The runner launches POSIX process groups, and the optional OOM case uses Linux cgroups and a systemd user session.

The interpreter must be available at `<checkout>/.venv/bin/python`. A `.venv` symlink to a centrally managed environment works. Activating a different virtual environment does not change the interpreter selected by this script.

If the source environment is missing dependencies, install the full source profile described in [Installation](../getting-started/installation.mdx). These commands use the dependency groups and compatibility patches from the repository setup helper:

```bash
scripts/fetch-wheels.sh fetch
uv sync --locked --extra cu130 --group dev --group legacy-gradio
.venv/bin/python patches/patch-omegaconf-py312.py --no-confirm
.venv/bin/python patches/patch-detectron2-metadata.py --no-confirm
```

This profile requires the local wheels declared in `pyproject.toml` under `wheels/`. The fetch command retrieves the wheel inventory; use your usual GitHub authentication if its release assets require access. The Annotator imports the ML stack, including FAISS and SentenceTransformers, even though this qualification sets `CUDA_VISIBLE_DEVICES=''` and runs without GPU inference. A minimal gateway environment is insufficient. The generated synthetic annotation does not require model downloads.

Check the main dependencies and entry point before creating containers:

```bash
.venv/bin/python - <<'PY'
import boto3
import httpx
import psutil
import psycopg
import torch
import transformers
import yaml
import marie.executor.extract

print('Deployment imports available')
PY
.venv/bin/python -m tools.stress.llm_dispatch_deployment --help
```

The controller restarts in a deliberately restricted environment. It sets its own `PYTHONPATH`, cache directories, generated configuration, and offline Hugging Face settings. Shell exports such as `OPENAI_API_KEY`, `LLM_QUEUE_URL`, `MARIE_DEFAULT_MOUNT`, and `DOCKER_HOST` are not inherited. Docker must be available in `/usr/local/bin`, `/usr/bin`, or `/bin`; a remote Docker context or an executable available only through your virtual environment will not carry over.

## Build and pull the required images

The runner creates PostgreSQL, etcd, and MinIO. It reuses Valkey and AIMock from the fixture stack. That stack also includes Redis for the separate Redis and Valkey qualification tools.

| Service | Image | Started by |
| --- | --- | --- |
| PostgreSQL with DocumentDB extensions | `marie-postgres-documentdb:17-0.103.0-repack-1.5.3` | Deployment runner |
| etcd | `quay.io/coreos/etcd:v3.7.0` | Deployment runner |
| MinIO | `minio/minio:latest` | Deployment runner |
| Valkey | `valkey/valkey:9.1.2` | Fixture Compose stack |
| Redis | `redis:7.4.2` | Fixture Compose stack |
| AIMock | `marie-aimock-llm-dispatch-task10:qualification` | Fixture Compose stack |

Build the two repository images and pull the others:

```bash
docker build \
  -f Dockerfiles/postgres/Dockerfile \
  -t marie-postgres-documentdb:17-0.103.0-repack-1.5.3 .

docker build \
  -f Dockerfiles/aimock/programmatic/Dockerfile \
  -t marie-aimock-llm-dispatch-task10:qualification \
  Dockerfiles/aimock/programmatic

docker pull quay.io/coreos/etcd:v3.7.0
docker pull minio/minio:latest
docker pull valkey/valkey:9.1.2
docker pull redis:7.4.2
```

Use this programmatic AIMock image: the qualification calls its admin `/fault-profile` endpoint as well as its OpenAI completion endpoint. An unrelated mock server with only `/v1/chat/completions` is insufficient. The deployment runner inspects infrastructure images before startup, so build or pull them first.

The fixture pins Valkey to `9.1.2` for reproducible qualification. Its manifest and live server version checks require that exact release. When updating the pin, update the fixture Compose file, `STORE_VERSIONS` in `tools/stress/llm_queued_dispatch_aimock_e2e.py`, and the manifest together. Start a fresh fixture project and regenerate its manifest when moving from an older version.

## Start the fixture stack

Keep the following variables in the same shell for the rest of the walkthrough. Use an absolute artifact directory and unused loopback ports:

```bash
mkdir -p "$HOME/tmp"
export QUAL_ROOT="$(mktemp -d "$HOME/tmp/llm-dispatch.XXXXXX")"
export QUAL_PROJECT="marie-llm-fixtures-$(date +%Y%m%d%H%M%S)"
export QUAL_COMPOSE_FILE="$PWD/Dockerfiles/docker-compose.llm-dispatch-qualification.yml"
export QUAL_MOCK_PORT=4030
export QUAL_ADMIN_PORT=4031
export QUAL_VALKEY_PORT=4032
export QUAL_REDIS_PORT=4033

ss -ltn '( sport = :4030 or sport = :4031 or sport = :4032 or sport = :4033 )'
```

If the command lists listeners, choose different values for the corresponding port variables. Keep all four ports distinct and keep the same values when restarting or inspecting these fixtures.

```bash
docker compose --env-file /dev/null \
  -p "$QUAL_PROJECT" -f "$QUAL_COMPOSE_FILE" up -d

docker compose --env-file /dev/null \
  -p "$QUAL_PROJECT" -f "$QUAL_COMPOSE_FILE" ps

docker compose --env-file /dev/null \
  -p "$QUAL_PROJECT" -f "$QUAL_COMPOSE_FILE" exec -T valkey valkey-cli ping

curl --fail --silent --show-error "http://127.0.0.1:$QUAL_MOCK_PORT/health"
curl --fail --silent --show-error "http://127.0.0.1:$QUAL_ADMIN_PORT/fault-profile"
```

Wait for Valkey to return `PONG` and for both HTTP checks to succeed before proceeding. No real provider API key is needed for this fixture.

## Generate the fixture manifest

The manifest records the current container IDs, image IDs, ownership labels, ports, and Compose project. Generate it from the running stack rather than copying IDs from another machine or an earlier run:

```bash
.venv/bin/python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

compose_file = str(Path(os.environ['QUAL_COMPOSE_FILE']).resolve())
project = os.environ['QUAL_PROJECT']
ports = {
    name: os.environ[name]
    for name in (
        'QUAL_REDIS_PORT', 'QUAL_VALKEY_PORT',
        'QUAL_MOCK_PORT', 'QUAL_ADMIN_PORT',
    )
}
assert len(set(ports.values())) == 4, 'Choose four distinct ports'
command = [
    'docker', 'compose', '--env-file', '/dev/null',
    '-p', project, '-f', compose_file,
]
manifest = {}
for service, version in [('redis', '7.4.2'), ('valkey', '9.1.2')]:
    identity = subprocess.check_output(
        command + ['ps', '-a', '-q', service], text=True
    ).strip()
    info = json.loads(subprocess.check_output(['docker', 'inspect', identity]))[0]
    labels = info['Config']['Labels']
    assert info['State']['Running'], f'{service} is not running'
    assert labels['marie.test.owner'] == 'llm-dispatch-task10'
    assert labels['com.docker.compose.project'] == project
    port = ports['QUAL_' + service.upper() + '_PORT']
    binding = [{'HostIp': '127.0.0.1', 'HostPort': port}]
    assert info['HostConfig']['PortBindings']['6379/tcp'] == binding
    manifest[service] = {
        'url': f'redis://127.0.0.1:{port}/0',
        'version': version,
        'container_id': info['Id'],
        'image_id': info['Image'],
        'compose_project': project,
        'compose_file': compose_file,
        'compose_ports': ports,
        'service': service,
        'owner_label': labels['marie.test.owner'],
        'bindings': {'6379/tcp': port},
    }
path = Path(os.environ['QUAL_ROOT']) / 'fixtures.json'
path.write_text(json.dumps(manifest, indent=2) + '\n')
print(path)
PY
```

Both entries use the `redis://` URL scheme. The deployment runner uses the Valkey entry and its AIMock ports. The Redis entry also makes the manifest compatible with the separate durable store qualification tools. If you recreate fixture containers, regenerate this manifest.

## Start and exercise the deployment

Use a new output directory for the initial command:

```bash
export DEPLOYMENT_RUN="$QUAL_ROOT/deployment"

.venv/bin/python -m tools.stress.llm_dispatch_deployment \
  --output "$DEPLOYMENT_RUN" \
  --fixture-manifest "$QUAL_ROOT/fixtures.json" \
  --start-only
```

The runner creates three containers with a unique `marie-deployment-*` Compose project, chooses free loopback ports, and starts the gateway and Annotator as source Python processes. It generates queued dispatch configuration with the `document-small` pool, the `aimock` endpoint, one concurrent dispatch, and synthetic annotation templates. PostgreSQL credentials, the gateway API key, and MinIO credentials are generated automatically.

Check readiness without printing credentials:

```bash
.venv/bin/python - "$DEPLOYMENT_RUN/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
print('status:', report['status'])
print('ports:', report.get('ports'))
print('failures:', report['failures'])
PY
```

Successful startup records `ready_not_qualified` with no failures. Submit the synthetic nine-page annotation and then inspect gateway operations:

```bash
.venv/bin/python -m tools.stress.llm_dispatch_deployment \
  --output "$DEPLOYMENT_RUN" \
  --fixture-manifest "$QUAL_ROOT/fixtures.json" \
  --resume --invoke-only

.venv/bin/python -m tools.stress.llm_dispatch_deployment \
  --output "$DEPLOYMENT_RUN" \
  --fixture-manifest "$QUAL_ROOT/fixtures.json" \
  --resume --inspect
```

Every subsequent command against this output directory needs `--resume`. Each invocation replaces `report.json` and archives the previous report under `reports/`; the original runtime identity and generated credentials remain in `private-runtime.json`.

## Run dispatcher recovery scenarios

Run each case with a separate, fresh output directory:

```bash
.venv/bin/python -m tools.stress.llm_dispatch_deployment \
  --output "$QUAL_ROOT/s03" \
  --fixture-manifest "$QUAL_ROOT/fixtures.json" \
  --s03

.venv/bin/python -m tools.stress.llm_dispatch_deployment \
  --output "$QUAL_ROOT/sigkill" \
  --fixture-manifest "$QUAL_ROOT/fixtures.json" \
  --death
```

| Mode | Behavior |
| --- | --- |
| `--prepare` | Create deployment infrastructure and runtime metadata without starting source processes. |
| `--start-only` | Start source processes and wait for dispatcher and executor readiness. |
| `--resume --invoke-only` | Submit synthetic annotation to an existing deployment. |
| `--resume --inspect` | Save authenticated gateway operations snapshots. |
| `--resume --restart --start-only` | Stop and restart source processes using the recorded infrastructure. |
| `--s03` | Exercise a provider outage and recovery during annotation. |
| `--death` | Kill the executor worker, check abandoned request cleanup, and explicitly launch a replacement Deployment. |
| `--oom` | Trigger and verify a kernel OOM kill inside an executor scope limited to 8 GiB. |

The OOM case requires cgroup v2 memory accounting and a working systemd user manager. Check `systemctl --user status` and the presence of `/sys/fs/cgroup/cgroup.controllers` first. Run this case only on a machine with memory available for an executor scope that deliberately fills an 8 GiB limit:

```bash
.venv/bin/python -m tools.stress.llm_dispatch_deployment \
  --output "$QUAL_ROOT/oom" \
  --fixture-manifest "$QUAL_ROOT/fixtures.json" \
  --oom
```

Each successful case retains a `case-report-*.json`. Verify the complete S03, SIGKILL, and OOM matrix before changing the source or stopping those deployments:

```bash
.venv/bin/python -m tools.stress.llm_dispatch_deployment_check \
  --reports \
    "$QUAL_ROOT/s03/case-report-S03.json" \
    "$QUAL_ROOT/sigkill/case-report-SIGKILL.json" \
    "$QUAL_ROOT/oom/case-report-OOM.json" \
  --output "$QUAL_ROOT/deployment-check.json"
```

This checker compares the retained startup manifests to the current checkout and requires all three cases. A startup or ordinary annotation run does not supply that matrix. The worker replacement in these scenarios is explicit; it does not demonstrate automatic executor supervision.

## Find logs and stop the deployments

Use the artifact directory to inspect results:

| Artifact | Contents |
| --- | --- |
| `report.json` and `reports/` | Latest result and earlier controller reports. |
| `commands/` | Subprocess commands, environment metadata, and process logs. |
| `gateway.yml`, `executor.yml`, `generated-config/` | Configuration generated for this run. |
| `probe.jsonl` | Process, dispatcher, admission, and output observations. |
| `operations-*.json` | Gateway operations snapshots from inspection. |
| `private-runtime.json` | Resource identities and generated credentials; keep this file private. |

Stop each output directory you created, including failed startup or scenario runs:

```bash
.venv/bin/python -m tools.stress.llm_dispatch_deployment \
  --output "$DEPLOYMENT_RUN" \
  --fixture-manifest "$QUAL_ROOT/fixtures.json" \
  --resume --stop-owned
```

Repeat with the `s03`, `sigkill`, or `oom` output path if you ran those cases. `--stop-owned` verifies recorded ownership and stops the source processes and deployment infrastructure containers. It retains containers, volumes, and artifacts, and it leaves the shared Valkey, Redis, and AIMock fixtures running. `--stop-sources` stops only the gateway and Annotator processes.

After all deployments using the fixtures have stopped, remove this fixture Compose project while retaining its data volumes:

```bash
docker compose --env-file /dev/null \
  -p "$QUAL_PROJECT" -f "$QUAL_COMPOSE_FILE" down
```

## Resolve setup failures

| Symptom | Action |
| --- | --- |
| Missing `.venv/bin/python` or an import fails | Repair the source environment in this checkout, including the ML profile and local wheels. |
| `command_image_postgres_exit_1` or another image inspection failure | Build or pull the exact image tag listed above. |
| Docker works in your shell but fails in the runner | Check the default Docker socket and restricted `PATH`; the runner does not inherit Docker context configuration. |
| `reused_valkey_identity` | Regenerate the manifest from the running fixture project; verify its image and ownership labels. |
| `FileExistsError` for the output directory | Use a new directory for a fresh deployment or `--resume` for an existing run. |
| `Gateway readiness timeout` | Inspect executor and gateway logs in `commands/`, the latest report, and Valkey/AIMock health. |
| `Address already in use` on a gateway port | Inspect all TCP connections, including outbound connections, and [reserve the gateway ports on Linux](#reserve-gateway-ports-on-linux). |
| `prewrite retry evidence missing` in the local stress suite | Use [direct AIMock networking](#use-direct-aimock-networking-for-the-local-stress-suite). A published Docker port can produce an ambiguous read failure rather than a connection refusal. |
| `Deployment coverage has not executed` | Select an explicit startup or qualification mode. Stop resources from the failed invocation before retrying. |
| OOM scope setup or identity fails | Check cgroup v2 and the systemd user session; use `--death` to test worker death without allocating the OOM scope. |
| Current-source matrix verification fails after edits | Rerun the cases against the current source; retained evidence is tied to the startup source inventory. |

## Configure a regular gateway separately

For an ordinary deployment, use `config/service/llm-dispatch.example.yml` as the dispatcher configuration reference. Enable the queue with `LLM_QUEUE_ENABLED=true` and set `LLM_QUEUE_URL` to the intended store. Queue enablement selects durable queued dispatch. A disabled queue uses direct batch execution. Enabled startup requires a complete registered policy; ordinary scheduler defaults or pool URLs do not provide endpoint registration.

Configure one consistent fabric ID in the gateway and producer, endpoint credentials through `credential_env`, lane bindings, and store/execution limits. Use explicit static policy or complete registered policy from scheduler persistence. Missing credentials, invalid policy, inconsistent fabric identity, or unavailable storage refuses enabled startup. Enable loopback only for a local fixture. Those settings are generated by the qualification runner above; exporting them in your shell will not override its isolated deployment configuration.

This is a fresh-install setup. The store initializes current fabric policy in
`llm:dispatch:{fabric:<id>}:`; producers attach to that policy. Startup does not
scan historical queue namespaces or require a legacy migration procedure.

### Use direct AIMock networking for the local stress suite

The local mixed-pool suite's `unreachable` phase qualifies an unsent retry:
the provider connection must fail before transmission and the dispatcher must
refund its scheduling charge. Stopping the inference listener inside a
bridge-networked container does not stop Docker's published-port proxy. That
proxy can accept the connection and produce `read_error` after transmission,
which retains an uncertain reservation instead of refunding the charge.

Programmatic AIMock uses host networking by default for a local gateway on Linux.
Run from the checkout root and use the same Compose project
name as your existing AIMock service (`marie` in this example):

```bash
docker compose --env-file /dev/null -p marie \
  -f Dockerfiles/docker-compose.mock-llm-programmatic.yml \
  up -d --build --force-recreate aimock-programmatic
```

This recreates the local mock service. The Compose file binds inference and admin
listeners directly to host loopback ports `4010` and `4011`, without bridge
membership or published ports. Set `AIMOCK_PORT` and `AIMOCK_ADMIN_PORT` if your
registered policy and suite use different ports. Set `AIMOCK_HOST` explicitly if
clients must reach the mock through another host interface; clients in isolated
bridge containers cannot use their own loopback address to reach this service.
See [Docker host networking](https://docs.docker.com/engine/network/drivers/host/).

Once the test fabric is clean, rerun the failed phase with a new artifact ID:

```bash
.venv/bin/python tools/stress/llm_dispatch_stress_suite.py \
  --runtime-mode queued-dispatch \
  --suite-id local-baseline-prewrite-recheck \
  --phases unreachable
```

Keep the refund assertion: successful document jobs alone do not qualify an
unsent retry. The suite also rejects ambiguous remote outcomes in this phase,
even if other calls were refunded. The timed outage must overlap actual dispatch;
if it ends before requests arrive, there is no failure evidence to qualify.

Changing networking or restarting a process does not settle reservations already
left by a failed run. Use affirmative provider evidence to reconcile those
attempts under the normal settlement procedure, or provision a separate clean
test fabric after confirming the old test executions have ended. Do not erase
held reservations or reinterpret read failures as unsent requests.

### Reserve gateway ports on Linux

The regular gateway uses HTTP port `51000` and gRPC port `52000`. These ports
fall within the usual Linux ephemeral range (`32768–60999`), so a browser or
another outbound client can occupy them before the gateway starts. Reserve the
configured gateway ports on the host where the gateway binds them.

Check the host's allocation range and existing reservations:

```bash
sysctl net.ipv4.ip_local_port_range net.ipv4.ip_local_reserved_ports
```

If the reserved-port list is empty, persist both gateway ports and apply the
setting immediately:

```bash
printf '%s\n' 'net.ipv4.ip_local_reserved_ports = 51000,52000' \
  | sudo tee /etc/sysctl.d/99-marie-gateway-ports.conf

sudo sysctl -p /etc/sysctl.d/99-marie-gateway-ports.conf
sysctl net.ipv4.ip_local_reserved_ports
```

If the host already reserves ports, include those ports or ranges alongside
`51000,52000` in the configuration value. Writing this setting replaces the
entire reservation list. Use the actual gateway ports if your deployment uses
different ones, and keep any existing sysctl configuration consistent with this
file. Reservations exclude ports from automatic outbound allocation while
allowing explicit server binds; see the [Linux kernel IP sysctl documentation](https://docs.kernel.org/networking/ip-sysctl.html#ip-variables).

Reservations do not close existing connections. If startup still reports
`Address already in use`, inspect all TCP states rather than only listeners:

```bash
ss -tanp '( sport = :51000 or sport = :52000 )'
```

An outbound connection can block a wildcard gateway bind even when no server is
listening. Close the conflicting connection or stop its owning application,
then retry once the port is free. A closed connection may remain in `TIME_WAIT`
until the kernel releases it.

See [LLM tracking and observability](./llm-tracking.md) for telemetry configuration in a regular deployment. This qualification disables OpenTelemetry export.
