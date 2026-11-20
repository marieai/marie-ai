# m3top

`m3top` is the standalone operational terminal for Marie Runtime Fabric. It connects directly to Marie gateway operational APIs; M3 Forge/Marie Studio is not required to configure or run it.

The standalone Rust crate lives at the top level of the Marie-AI repository in `m3top/`, alongside its gateway APIs. The compiled binary has no Studio API or frontend runtime dependency.

The implemented screens are the Runtime Fabric and gateway selector, operational overview, executor detail, scheduler detail, Jobs and Job detail, DAGs and DAG detail, worker execution history, completion throughput, and the read-only Observe panels.

## Build

Install a Rust toolchain with Cargo, then build from the Marie-AI repository root:

```bash
cargo build --manifest-path m3top/Cargo.toml --release --locked
```

The release binary is written to `m3top/target/release/m3top`. Verify it with:

```bash
./m3top/target/release/m3top --version
```

For a faster development build, omit `--release`; the binary will be written to `m3top/target/debug/m3top`.

To install `m3top` on the current user's Cargo path:

```bash
cargo install --path m3top --locked
m3top --version
```

## Install a release

Released Linux binaries are published under the public `marieai/marie-ai` GitHub project. The installer selects x86_64 or ARM64, downloads the newest `m3top-v*` release, verifies its SHA-256 checksum, and installs it to `~/.local/bin`:

```bash
curl -fsSL https://raw.githubusercontent.com/marieai/marie-ai/main/scripts/install-m3top.sh | sh
```

Install a specific version or choose another user-writable directory with:

```bash
curl -fsSL https://raw.githubusercontent.com/marieai/marie-ai/main/scripts/install-m3top.sh \
  | M3TOP_VERSION=0.2.0 M3TOP_INSTALL_DIR="$HOME/bin" sh
```

The installer does not require Marie Studio, Rust, Cargo, or GitHub authentication.

## Publish a release

The `Release m3top` workflow builds static Linux x86_64 and ARM64 binaries when an `m3top-v*` tag is pushed to `marieai/marie-ai`. It publishes the archives and checksum files in the same repository using GitHub's built-in repository token.

Follow [releasing.md](releasing.md) for versioning, validation, tagging, verification, and failed-run recovery. m3top keeps its own version and release tags, independently of the Marie-AI Python package.

## First run

Start the release binary:

```bash
./m3top/target/release/m3top
```

The splash renders the official M3 fox as responsive terminal ASCII art and blinks its blue eyes while waiting for Space. The packaged high-resolution ASCII master is derived from the [official M3 logo](https://www.m3forge.com/m3-logo.svg), so no Studio or image-rendering dependency is required. If no gateways are configured, M3top then opens the add-gateway panel. Enter:

- A unique gateway name.
- The direct gateway operational URL, normally `http://<host>:51000`.
- The Runtime Fabric that owns the gateway. A blank value assigns the gateway to `default_fabric`.
- An optional bearer token for that gateway. The field is masked while typing and the configuration file is written with user-only permissions.

Press `a` from the target selector to add another gateway later. Every gateway belongs to exactly one Runtime Fabric.

## Configuration

On Linux, M3top writes its standalone configuration to `~/.config/m3top/config.toml`:

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

If `fabric` is omitted or blank, the gateway is assigned to `default_fabric`. The selected fabric and gateway are saved separately under the platform state directory.

The onboarding panel accepts the selected gateway's token directly. Each gateway can have a different token. If the gateway does not require authentication, leave it blank.

For environments that prohibit credentials in configuration files, `token_env` remains available as a manual configuration option. Its value is the name of an environment variable:

```toml
token_env = "M3TOP_CTL_008_TOKEN"
```

```bash
export M3TOP_CTL_008_TOKEN='<bearer token>'
m3top
```

Names must use shell identifier syntax: a letter or underscore followed by letters, digits, or underscores. M3top rejects malformed names and does not echo their configured value in connection errors.

Existing configurations that placed a valid Marie `mas_` or `mau_` token in `token_env` are recognized automatically. Tokens are redacted from Rust debug output and never rendered in the target selector.

Use a different configuration or initial selection with:

```bash
m3top --config /path/to/config.toml --fabric claims-prod --gateway claims-prod-1
```

The overview polls the gateway's `/api/capacity`, `/api/deployment-status`, `/api/debug`, and `/api/operations/throughput` endpoints. It groups replicas by logical executor, keeps scheduler slot usage distinct from gateway-routed requests, and reports completed plans/hour, completed executor tasks/hour, and plan/task success percentages for the current 24-hour window.

Press `Enter` on an executor group to open executor detail. That screen also reads `/api/deployments` and `/api/discovery/readiness` directly from the selected gateway. It shows:

- Group capacity, active jobs, routed requests, free slots, scheduler-ready work, and the oldest ready age.
- Every known replica with its address, capacity use, routing state, circuit state, discovery probe, and heartbeat.
- Bounded scheduler-ready jobs from the scheduler's `stalest_ready` projection. Each compact row labels the first seven characters of the job ID; Selected detail shows the full ID and name.
- Selected-replica topology, routing counters, deployment activity, discovery state, and bounded errors.

The displayed routing rate is the change in gateway selection counters between M3top polls. It is not job completion throughput. Request throughput and request latency remain `N/A` until the gateway exposes authoritative values; M3top never turns missing metrics into synthetic zeroes.

Press `3` to open the executor fleet. It reads `/api/capacity`, `/api/deployment-status`, `/api/deployments`, and `/api/discovery/readiness` directly from the selected gateway and provides:

- Logical executor and replica modes, toggled with `x`.
- State, endpoint, node, and safe-text filters plus health, saturation, heartbeat, selection-count, and name sorting.
- Capacity, active work, routing selection counts, heartbeat, discovery, circuit, and deployment state.
- `N/A` for completion throughput and p95 latency until the gateway exposes authoritative executor-level measurements.

Press `Enter` to open S04 for the selected group or replica. Returning from S04 preserves the S03 filters, mode, and selection.

Press `5` after loading a gateway to open scheduler detail. It polls `/api/debug` and `/api/capacity` directly and shows:

- Scheduler runtime, DAG admission, event-queue depth, planner/frontier availability, batch size, lease TTL, dispatch-confirmation pressure, and cumulative scheduler anomaly counters.
- Frontier DAG, job, ready, leased, blocked, dependency, age, and SLA signals.
- Rolling selection count, selected jobs per second, selection latency, and the latest candidate/requested/selected result when the gateway exposes the current diagnostics contract.
- Ready work grouped by executor and correlated with matching used, total, and free capacity. `[WAIT]` means ready work exists while all matching slots are occupied; `[NONE]` means ready work has no registered capacity.
- Per-queue job and DAG state counts.

Selection throughput measures jobs selected for dispatch, not jobs completed. On a gateway version that predates the selection and dispatch diagnostics, those fields remain `N/A`; frontier, queue, DAG, and capacity data still render.

Press `6` to open Jobs. Press `7` to open DAGs. These screens call Marie-AI directly:

- `/api/operations/jobs` returns one server-filtered page of safe job metadata.
- `/api/operations/jobs/{job_id}` returns durable lifecycle states and at most ten safe attempt records.
- `/api/operations/dags` returns one server-filtered page with PostgreSQL child-job rollups.
- `/api/operations/dags/{dag_id}` returns safe DAG detail and one bounded child-job page.
- `/api/operations/execution-history` returns server-paged worker transitions and structured errors for every task in one DAG.
- `/api/operations/throughput` returns the scheduler system, planner, and task completion reports for a bounded lookback window.

M3top requests 25 rows per page and never loads the full job or DAG collection. PostgreSQL performs counts, filters, attention predicates, sorting, and DAG aggregation. Attention presets cover queued too long, running too long, stale update, retrying, failed, and terminal mismatch.

The operational endpoints exclude job data, output, document references, serialized DAGs, pickle data, submission refs, projects, and policies. Job detail shows only lifecycle stages that were durably recorded; it does not synthesize trace IDs, dispatch stages, incidents, or retryability.

Press `e` from Job detail or DAG detail to open worker execution history. From a Job, Marie resolves the parent DAG; from a DAG, it uses that DAG directly. The screen walks all retained `kv_store_worker_history` transitions for every child task, newest first, and shows worker status and message, executor, runtime, host, endpoint, attempt ID, plus structured error type, message, file, function, and line. It requests 50 rows at a time with `n` and `p`, so “complete retained history” means every available row is reachable through server paging rather than loaded into terminal memory. Raw `runtime_env_json`, task data, output, and tracebacks are never returned to m3top.

At 160 terminal columns or wider, DAGs and execution history use a list-and-inspector layout: the selected DAG or worker event remains visible in a right-side context pane while the list stays navigable on the left. Narrower terminals retain the stacked detail layout.

DAG detail defaults to a timegraph grouped by descending `job_level`. Each row plots the recorded start and terminal time inside the current bounded server page; active work extends to the current time. Press `t` to switch between the timegraph and the existing child-job table. Pagination remains server-side, so the graph never hydrates every child job.

Press `8` to open completion throughput. The screen shows:

- Window totals and success rates from `monitor_system_throughput`, plus the latest five UTC hourly buckets. The current hour is labeled `partial`.
- Bounded planner totals from `monitor_planner_throughput`.
- Bounded task/queue/endpoint totals and execution latency from `monitor_task_throughput`.

Press `w` to cycle the lookback through 1, 6, 24, 72, and 168 hours. These reports measure terminal DAG and job outcomes, not scheduler selections or retries. Marie-AI installs all three report functions during scheduler schema bootstrap; the selected gateway must run the corresponding Marie-AI revision before this screen is available.

Press `9` to open Observe. `Tab` and `Shift-Tab`, or `e`, `f`, `a`, and `h`, switch between the four P0 panels:

- Recent Events shows cursor-paged, payload-free job, DAG, and execution-attempt lifecycle changes. It never reads the sensor event payload log or raw application logs.
- Flow Pressure compares job arrival and terminal rates, current ready/active work, queue pressure, and durable ready-to-running and running-to-terminal latency. Lease and dispatch rates remain `N/A` until Marie exposes authoritative durable transitions.
- Execution Attempts shows the global `marie_scheduler.job_attempt` audit trail with database paging and safe owner, executor, gateway, terminal, recovery, and attention fields.
- Dependency Health combines safe PostgreSQL pool/activity aggregates, etcd connection and watch state, discovery readiness, and gateway control-plane state. It excludes query text, database connection information, etcd keys and values, and credentials.

The Observe endpoints are `/api/operations/events`, `/api/operations/flow`, `/api/operations/attempts`, and `/api/operations/health`. PostgreSQL owns the event, attempt, flow, filtering, ordering, and aggregation queries so they can be inspected and tuned without changing the TUI.

## Keys

- `Space` enters from the splash screen.
- `1` opens the target selector; `2` opens the loaded overview; `3` opens Executors; `4` opens the selected executor detail; `5` opens scheduler detail; `6` opens Jobs; `7` opens DAGs; `8` opens completion throughput; `9` opens Observe.
- `Tab` switches panes; arrows or `j`/`k` move.
- `Enter` loads the selected gateway or opens the selected executor group.
- `Enter` on a job or DAG opens its detail. Enter on a DAG child job opens Job detail.
- In Executors, `x` toggles groups and replicas; `f`, `e`, `n`, and `s` cycle state, endpoint, node, and sort; `/` searches executor names, addresses, and endpoints.
- `f`, `a`, `c`, and `s` cycle state, attention, queue, and sort on Jobs and DAGs. `/` searches safe operational fields. `n` and `p` request the next or previous server page.
- `d` opens the parent DAG from Job detail.
- `e` opens the complete retained, server-paged DAG worker history from Job detail or DAG detail.
- `t` toggles DAG detail between the timegraph and child-job table; `w` cycles the throughput lookback window.
- In Observe, `e`/`f`/`a`/`h` open Events/Flow/Attempts/Health; `v`, `c`, `g`, `x`, and `s` cycle the filters shown in the footer; `w` changes the event/flow window.
- `Esc` returns from a detail screen to its list, or from a list to the overview.
- `a` adds a gateway from the target selector.
- `p` pauses polling on non-paginated live screens; on paginated screens it requests the previous page. `r` refreshes.
- `s` opens source status; `?` opens help; `q` quits.
