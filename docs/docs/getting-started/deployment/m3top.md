---
sidebar_position: 4.5
title: m3top Operational Terminal
description: Install and use m3top to analyze Marie Runtime Fabric from a terminal.
---

# m3top Operational Terminal

`m3top` is the read-only operational terminal for Marie Runtime Fabric. It connects directly to a Marie gateway and gives operators a live view of executor capacity, scheduler pressure, jobs, DAGs, completion throughput, execution history, and dependency health.

M3 Forge is not required to run `m3top`. The binary uses the gateway's operational HTTP APIs and stores only its local target configuration and last selected gateway.

![m3top executor fleet showing logical executor groups, replica capacity, routed selections, and heartbeat age](/img/m3top/executor-fleet.png)

## When to use m3top

Use `m3top` during deployment verification and incident response to answer questions such as:

- Is ready work waiting because every matching executor slot is occupied?
- Does queued work have no matching registered capacity?
- Which executor replica is unhealthy, saturated, stale, or no longer selected?
- Where did a job or DAG stop progressing?
- What happened during each retained worker execution attempt?
- Are jobs completing, or is only scheduler selection activity increasing?
- Is PostgreSQL, etcd, discovery, or the gateway control plane degraded?

:::info Runtime boundary
`m3top` reads one concrete gateway at a time. Runtime Fabric names group operator targets locally, while the selected Marie gateway remains authoritative for scheduler, executor, job, DAG, and dependency state.
:::

## Requirements

- Linux on x86-64 or ARM64
- Direct network access to the Marie gateway operational URL, normally `http://gateway-host:51000`
- A gateway bearer token when operational API authentication is enabled
- A gateway revision that exposes the operational endpoints required by the selected screen
- `curl`, `tar`, `sha256sum`, and `install` for release installation

## Install m3top

Static Linux binaries and SHA-256 files are published in the [Marie-AI GitHub Releases](https://github.com/marieai/marie-ai/releases) repository.

### Install from a Marie-AI checkout

The repository installer detects x86-64 or ARM64, selects the newest `m3top-v*` release, verifies its checksum, and installs the binary to `~/.local/bin`:

```bash
./scripts/install-m3top.sh
```

Install a specific version or choose another user-writable directory:

```bash
M3TOP_VERSION=0.1.0 \
M3TOP_INSTALL_DIR="$HOME/bin" \
./scripts/install-m3top.sh
```

### Install directly from a release

The following example installs `m3top-v0.1.0`. Set `M3TOP_TARGET=aarch64-unknown-linux-musl` on ARM64.

```bash
M3TOP_VERSION=0.1.0
M3TOP_TARGET=x86_64-unknown-linux-musl

curl -fLO "https://github.com/marieai/marie-ai/releases/download/m3top-v${M3TOP_VERSION}/m3top-${M3TOP_VERSION}-${M3TOP_TARGET}.tar.gz"
curl -fLO "https://github.com/marieai/marie-ai/releases/download/m3top-v${M3TOP_VERSION}/m3top-${M3TOP_VERSION}-${M3TOP_TARGET}.tar.gz.sha256"
sha256sum -c "m3top-${M3TOP_VERSION}-${M3TOP_TARGET}.tar.gz.sha256"
tar -xzf "m3top-${M3TOP_VERSION}-${M3TOP_TARGET}.tar.gz"
install -Dm755 "m3top-${M3TOP_VERSION}-${M3TOP_TARGET}/m3top" "$HOME/.local/bin/m3top"
"$HOME/.local/bin/m3top" --version
```

Add `~/.local/bin` to `PATH` if your shell cannot find `m3top`.

## Connect a gateway

Start the terminal and press `Space` on the splash screen:

```bash
m3top
```

If no gateways are configured, the onboarding panel asks for:

1. A unique gateway name, such as `claims-prod-1`.
2. The direct gateway operational URL, normally `http://gateway-host:51000` or the corresponding HTTPS endpoint.
3. The Runtime Fabric that owns the gateway. A blank value uses the configured default fabric.
4. An optional bearer token. The field is masked while typing.

Select the Runtime Fabric and gateway, then press `Enter`. Press `a` from the target selector to register another gateway.

:::warning
Enter the direct Marie gateway URL, not the M3 Forge application URL. `m3top` must be able to reach the gateway operational APIs from the operator's machine.
:::

## Configuration

On Linux, `m3top` stores configuration at `~/.config/m3top/config.toml` with user-only permissions:

```toml
default_fabric = "default"

[[gateways]]
name = "local-gateway"
url = "http://localhost:51000"
fabric = "default"

[[gateways]]
name = "claims-prod-1"
url = "https://claims-gateway.example.net:51000"
fabric = "claims-prod"
token = "<claims-prod-1 bearer token>"
```

Each gateway can use a different token. Omit `token` when the gateway does not require authentication.

### Use an environment variable for the token

For environments that prohibit credentials in configuration files, set `token_env` to an environment variable name:

```toml
[[gateways]]
name = "claims-prod-1"
url = "https://claims-gateway.example.net:51000"
fabric = "claims-prod"
token_env = "M3TOP_CLAIMS_PROD_TOKEN"
```

```bash
export M3TOP_CLAIMS_PROD_TOKEN='<bearer token>'
m3top
```

Environment variable names must begin with a letter or underscore and contain only letters, digits, and underscores. Token values are not rendered in the target selector or included in connection errors.

### Choose the initial target

Use another configuration file or select the initial Runtime Fabric and gateway from the command line:

```bash
m3top --config /path/to/config.toml --fabric claims-prod --gateway claims-prod-1
```

## Analysis screens

| Key | Screen                | Operational question                                                                                 |
| --- | --------------------- | ---------------------------------------------------------------------------------------------------- |
| `1` | Targets               | Which Runtime Fabric and gateway am I analyzing?                                                     |
| `2` | Overview              | Is the gateway healthy, and where are work and capacity concentrated?                                |
| `3` | Executors             | Which executor group or replica is unhealthy, saturated, stale, or not receiving selections?         |
| `4` | Executor detail       | What capacity, topology, readiness, routing, and bounded error state explains this executor?         |
| `5` | Scheduler             | Is work ready, leased, blocked, waiting for capacity, or missing a matching executor?                |
| `6` | Jobs                  | Which jobs are queued too long, running too long, stale, retrying, failed, or terminally mismatched? |
| `7` | DAGs                  | How are child jobs progressing, and where does recorded execution time accumulate?                   |
| `8` | Completion throughput | How many plans and tasks reached terminal states, with what success rate and latency?                |
| `9` | Observe               | What do lifecycle events, flow pressure, attempts, and dependency health show?                       |

### Executor fleet and detail

Executor views combine logical groups and concrete replicas with:

- used, total, and free slots
- active jobs and routed gateway selections
- heartbeat and discovery readiness
- routing and circuit state
- deployment state and bounded recent errors
- scheduler-ready work that matches the executor

In Scheduler, `[WAIT]` means ready work has matching capacity but every slot is occupied. `[NONE]` means ready work has no registered matching capacity.

![m3top executor detail showing replica saturation, scheduler-ready work, and selected replica diagnostics](/img/m3top/executor-detail.png)

### Jobs, DAGs, and execution history

Jobs and DAGs are filtered, sorted, counted, and paged by PostgreSQL. `m3top` requests 25 rows per page instead of loading the full collection.

Use the attention filters to find queued-too-long, running-too-long, stale, retrying, failed, and terminal-mismatch records. From Job detail, press `d` to open the parent DAG. From Job or DAG detail, press `e` to inspect retained worker execution history.

![m3top DAG attention list showing failed and long-queued DAGs](/img/m3top/dag-attention.png)

Worker history is server-paged and includes safe operational fields such as worker status and message, executor, runtime, host, endpoint, attempt ID, and structured error location.

![m3top worker execution history showing state changes and selected-event details](/img/m3top/worker-history.png)

### Completion throughput

Completion throughput reports terminal outcomes over a bounded lookback. Press `w` to cycle through 1, 6, 24, 72, and 168 hours.

The report separates:

- scheduler system totals and plan success rates
- planner totals
- task totals, queues, endpoints, and execution latency
- the latest five UTC hourly buckets, with the current hour labeled `partial`

### Observe

Observe contains four read-only panels:

| Key | Panel              | Focus                                                                                      |
| --- | ------------------ | ------------------------------------------------------------------------------------------ |
| `e` | Recent Events      | Payload-free job, DAG, and execution-attempt lifecycle changes                             |
| `f` | Flow Pressure      | Arrival and terminal rates, queue pressure, active work, and durable lifecycle latency     |
| `a` | Execution Attempts | Server-paged scheduler attempt history, recovery, terminal state, and attention indicators |
| `h` | Dependency Health  | Safe PostgreSQL, etcd, discovery, and gateway control-plane health                         |

## Interpret metrics correctly

:::caution
`N/A` means the selected gateway did not provide an authoritative measurement. It does not mean zero, healthy, or idle.
:::

| Signal                | Meaning                                                                           |
| --------------------- | --------------------------------------------------------------------------------- |
| Routing rate          | Change in gateway executor-selection counters between polls                       |
| Selection throughput  | Jobs the scheduler selected for dispatch                                          |
| Completion throughput | DAGs and jobs that reached durable terminal outcomes during the selected lookback |

Routing and selection activity can increase without a corresponding increase in completed work. Request throughput, request latency, executor completion throughput, and some scheduler diagnostics remain `N/A` until the gateway exposes those measurements.

## Keyboard reference

| Key                     | Action                                                                      |
| ----------------------- | --------------------------------------------------------------------------- |
| `1` through `9`         | Open the corresponding analysis screen                                      |
| `Tab` / `Shift-Tab`     | Move between panes or Observe panels                                        |
| Arrow keys or `j` / `k` | Move the selection                                                          |
| `Enter`                 | Load a target or open the selected executor, job, or DAG                    |
| `Esc`                   | Return from detail to list, or from list to overview                        |
| `/`                     | Search the safe fields available on the current list                        |
| `n` / `p`               | Request the next or previous server page on paginated screens               |
| `r`                     | Refresh                                                                     |
| `p`                     | Pause polling on a live, non-paginated screen                               |
| `s`                     | Open source status unless the active screen assigns `s` to a visible filter |
| `?`                     | Open in-terminal help                                                       |
| `q`                     | Quit                                                                        |

The footer shows screen-specific filters and shortcuts.

## Operational API compatibility

`m3top` reads these gateway endpoint families:

| Capability                 | Endpoints                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------- |
| Gateway and executor state | `/api/capacity`, `/api/deployment-status`, `/api/debug`, `/api/deployments`, `/api/discovery/readiness` |
| Jobs and DAGs              | `/api/operations/jobs`, `/api/operations/dags`, `/api/operations/execution-history`                     |
| Completion throughput      | `/api/operations/throughput`                                                                            |
| Observe                    | `/api/operations/events`, `/api/operations/flow`, `/api/operations/attempts`, `/api/operations/health`  |

When a screen reports an unavailable endpoint, upgrade the selected gateway to a Marie-AI revision that implements the corresponding operational contract.

## Data and security boundaries

`m3top` performs read-only runtime requests. The operational APIs exclude job input data, output, document references, serialized DAGs, pickle data, submission references, projects, and policies. Worker history does not return raw task data, output, runtime environment JSON, or tracebacks.

Recent Events does not read the sensor payload log or raw application logs. Dependency Health excludes SQL query text, database connection details, etcd keys and values, and credentials.

Apply the same access controls used for other gateway operator tools:

- grant access only to the gateways an operator needs to inspect
- prefer HTTPS when the connection crosses an untrusted network
- keep the configuration file user-only, or use `token_env`
- rotate tokens according to the gateway credential policy

## Troubleshooting

| Symptom                           | What to check                                                                 |
| --------------------------------- | ----------------------------------------------------------------------------- |
| Connection refused or timed out   | Direct gateway host, operational port, network route, and TLS endpoint        |
| `401` or `403`                    | Selected gateway token or `token_env` value and authorization scope           |
| Screen reports an unavailable API | Gateway version and operational endpoint compatibility                        |
| Fields remain `N/A`               | Whether the gateway implements the authoritative diagnostic measurement       |
| Data becomes stale                | Press `r`, open source status with `s`, and inspect Dependency Health         |
| Tables feel compressed            | Widen the terminal; DAG and history inspector layouts activate at 160 columns |
| Shell cannot find `m3top`         | Run `~/.local/bin/m3top` or add `~/.local/bin` to `PATH`                      |

## Related documentation

- [Deployment overview](./index.md)
- [Observability](./observability.md)
- [Gateway](../../guides/gateway.md)
- [Job management](../job-management/index.md)
