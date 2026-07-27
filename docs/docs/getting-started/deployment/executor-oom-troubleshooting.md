---
sidebar_position: 9
---

# Production executor OOM troubleshooting

Use this runbook when a Marie executor disappears, GPU work stops, the gateway
reports an executor as unready, or Linux records an out-of-memory kill. The goal
is to identify the killed process, active job, input asset, retry behavior, and
resource that was exhausted without destroying incident evidence.

This runbook primarily uses Docker commands. Kubernetes equivalents are
included where the evidence source differs.

## What to establish

Answer these questions in order:

1. Did Linux kill a process because host RAM was exhausted?
2. Did a container exceed its memory limit?
3. Did the NVIDIA driver report a GPU fault or GPU-memory exhaustion?
4. Which container and executor replica owned the killed process?
5. Which job and asset were active immediately before the kill?
6. Did the scheduler retry the same job on another replica?
7. Was this one oversized request or memory accumulating across many requests?

Do not restart the container or delete scheduler records until the initial
evidence bundle has been captured. If a job is repeatedly killing workers,
stop or quarantine it through the approved scheduler or operations interface.
Do not repair the incident by editing scheduler tables directly.

## 1. Capture evidence

Choose a time window that begins before the first reported failure. Include an
explicit UTC offset.

```bash
incident_start='YYYY-MM-DDTHH:MM:SS-05:00'
incident_end='YYYY-MM-DDTHH:MM:SS-05:00'
incident_dir=$(mktemp -d /tmp/marie-oom.XXXXXX)

sudo journalctl --dmesg \
  --since "$incident_start" \
  --until "$incident_end" \
  -o short-iso-precise > "$incident_dir/kernel.log"

docker ps --no-trunc > "$incident_dir/docker-ps.txt"
docker stats --no-stream > "$incident_dir/docker-stats.txt"
ps -eo pid=,ppid=,user=,stat=,etime=,rss=,vsz=,cmd= --sort=-rss \
  | head -100 > "$incident_dir/processes-by-rss.txt"
nvidia-smi -q > "$incident_dir/nvidia-smi.txt"

printf 'Incident evidence: %s\n' "$incident_dir"
```

If `journalctl` does not retain the required interval, collect the current
kernel ring buffer as secondary evidence:

```bash
sudo dmesg -T > "$incident_dir/dmesg.txt"
```

`dmesg -T` normally renders local wall-clock time. Docker's `--timestamps`
output is normally RFC 3339 UTC with a trailing `Z`. Normalize both timelines
before correlating events.

Preserve the complete executor log before filtering it:

```bash
container_id='<full-or-unambiguous-container-id>'

docker logs --timestamps \
  --since "$incident_start" \
  --until "$incident_end" \
  "$container_id" > "$incident_dir/executor-raw.log" 2>&1
```

Do not save only `grep` or `rg` context as the primary artifact. Context filters
insert separators and omit intervening records that may contain the exact
failure stage.

For Kubernetes, capture current and previous container state before the pod is
replaced:

```bash
namespace='<namespace>'
pod='<executor-pod>'
container='<executor-container>'

kubectl get pod -n "$namespace" "$pod" -o yaml \
  > "$incident_dir/pod.yaml"
kubectl describe pod -n "$namespace" "$pod" \
  > "$incident_dir/pod-describe.txt"
kubectl top pod -n "$namespace" "$pod" --containers \
  > "$incident_dir/pod-top.txt"
kubectl logs -n "$namespace" "$pod" -c "$container" --timestamps \
  > "$incident_dir/executor-current.log" 2>&1
kubectl logs -n "$namespace" "$pod" -c "$container" \
  --previous --timestamps \
  > "$incident_dir/executor-previous.log" 2>&1
```

Inspect `.status.containerStatuses[].lastState.terminated.reason` for
`OOMKilled`. A pod-level `OOMKilled` does not by itself prove whether the pod
limit or the node was exhausted; obtain the node's kernel journal when the
container limit and pod events do not explain the kill.

## 2. Classify the failure

Search the kernel evidence:

```bash
rg -n -i \
  'out of memory|oom-killer|oom-kill|killed process|memory cgroup|NVRM|Xid' \
  "$incident_dir/kernel.log"
```

Interpret the results as follows:

| Evidence | Meaning |
| --- | --- |
| `Out of memory: Killed process ...` | Linux killed a process because RAM and available swap could not satisfy an allocation. |
| `global_oom` or `constraint=CONSTRAINT_NONE` | The host, not only one container limit, was out of memory. |
| `Memory cgroup out of memory` | A container or pod reached its configured memory limit. |
| `NVRM: Xid ...` | The NVIDIA driver reported a GPU fault. Record the Xid number and GPU UUID. |
| Application `CUDA out of memory` | GPU VRAM allocation failed, but the host kernel did not necessarily kill the process. |
| `r8169 ... XID ...` | Network-interface message; it is not an NVIDIA Xid. |

The process named before `invoked oom-killer` triggered the final allocation.
It is not necessarily the process consuming the memory. The authoritative
victim is the later `Killed process` record.

Extract context around every kill:

```bash
rg -n -B 80 -A 120 \
  'Out of memory: Killed process|Memory cgroup out of memory' \
  "$incident_dir/kernel.log"
```

In the kernel task table, compare `rss`, `rss_anon`, page-table usage, and swap
for all Python, Marie, and model-engine processes. The final kill line reports
RSS in KiB. Convert it to GiB with:

```bash
rss_kib='<anon-rss-value-from-kernel-log>'
awk -v value="$rss_kib" 'BEGIN {printf "%.2f GiB\n", value / 1024 / 1024}'
```

## 3. Identify the container and replica

Docker OOM records commonly include a cgroup such as:

```text
/system.slice/docker-<container-id>.scope
```

Resolve it without printing container environment variables:

```bash
container_id='<container-id-from-the-kernel-cgroup>'

docker ps --no-trunc --filter "id=$container_id" \
  --format 'table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}'

docker inspect --format \
  'name={{.Name}} image={{.Config.Image}} pid={{.State.Pid}} oom_killed={{.State.OOMKilled}} memory_limit={{.HostConfig.Memory}}' \
  "$container_id"
```

For a process that is still alive, map its host PID to its container namespace
PID:

```bash
host_pid='<host-pid>'
sudo sed -n '/^NSpid:/p' "/proc/$host_pid/status"
```

The last number in `NSpid` is the PID shown in Marie logs after `@`, for example
`rep-<replica>[]@<pid>`. Also capture both process views:

```bash
docker top "$container_id" -eo pid,ppid,user,stat,etime,rss,cmd
docker exec "$container_id" \
  ps -eo pid=,ppid=,user=,stat=,etime=,rss=,cmd= --sort=-rss
```

After a process has died, `/proc/<pid>` is gone. Use the container cgroup,
replica log PID, exact kill time, and the point where that replica stops logging
to establish the mapping. Record this as a timeline inference rather than a
direct PID mapping.

## 4. Correlate the application timeline

Create a filtered view only after preserving the raw log:

```bash
rg -n -C 3 \
  'requests TO MONITOR|Record job started|\[sem\] adopted ticket|Reading file from|Executing pipeline for document|Bursting frames|Skipping bursting|Processing llm pipeline/group|Running .*strategy|executor_completion_callback|Pipeline error' \
  "$incident_dir/executor-raw.log" \
  > "$incident_dir/executor-timeline.log"
```

For each killed replica PID, inspect its final records:

```bash
replica_pid='<container-namespace-pid>'
rg -n "@${replica_pid}\\b" "$incident_dir/executor-raw.log" | tail -100
```

Build a table with these timestamps:

| Event | Evidence |
| --- | --- |
| Request accepted | `requests TO MONITOR` |
| Durable work started | `Record job started` and semaphore adoption |
| Asset identified | `Reading file from ...` |
| Frame loading completed | `Executing pipeline for document` |
| OCR/indexing entered | `Bursting frames`, pipeline group, and indexer messages |
| Request completed | `executor_completion_callback completed` |
| Process killed | Kernel `Killed process` timestamp |

Strong evidence of a poison document consists of all of the following:

- the same job ID and asset appear on two fresh replicas;
- each replica follows a similar runtime and memory-growth pattern;
- neither attempt records a completion callback;
- each replica disappears at a corresponding kernel OOM kill;
- another healthy replica continues processing unrelated jobs.

By contrast, a process leak normally shows RSS increasing across many completed
jobs rather than one request reproducing the same failure on a fresh process.

## 5. Inspect the document without decoding pixels

An asset filename may contain a page count, but treat that only as a hint.
Confirm the actual frame count and estimated RGB size from the image metadata.

Use a copy of the asset or the existing temporary file. Avoid downloading a
large production asset onto a host that is still under memory pressure.

```bash
asset_path='/tmp/marie/<temporary-file>.tif'

docker exec -i "$container_id" python - "$asset_path" <<'PY'
import json
import sys

from PIL import Image

path = sys.argv[1]
with Image.open(path) as image:
    frame_count = image.n_frames
    decoded_bytes = 0
    largest_page = None

    for index in range(frame_count):
        image.seek(index)
        width, height = image.size
        page_bytes = width * height * 3
        decoded_bytes += page_bytes
        if largest_page is None or page_bytes > largest_page['rgb_bytes']:
            largest_page = {
                'index': index,
                'width': width,
                'height': height,
                'rgb_bytes': page_bytes,
            }

print(json.dumps({
    'path': path,
    'frames': frame_count,
    'estimated_rgb_bytes': decoded_bytes,
    'estimated_rgb_gib': round(decoded_bytes / 1024**3, 2),
    'largest_page': largest_page,
}, indent=2))
PY
```

This script seeks image metadata but does not call `load()`, `convert()`, or
create NumPy arrays. The estimate covers one RGB frame set. Actual peak RSS can
be two or three times higher when document objects, resized frames, OCR input,
or model preprocessing retain additional representations.

Marie applies these raster safeguards:

```text
MARIE_MAX_RASTER_PAGES=500
MARIE_MAX_RASTER_DECODED_BYTES=8589934592
```

The limits apply to selected pages. Selecting one page from a large TIFF loads
only that page. Set either value to `0` only when change control explicitly
allows the corresponding limit to be disabled.

## 6. Confirm scheduler history in PostgreSQL

Use the executor log job ID, not the document reference, as the correlation
key. Run the investigation in a read-only transaction. For additional job,
DAG, error-summary, and recovery queries, see
[scheduler SQL troubleshooting](./scheduler-sql-troubleshooting.md). Adapt the
container, database, and user placeholders to the deployment.

```bash
job_id='<job-uuid>'

docker exec -i '<database-container>' \
  psql -U '<database-user>' -d '<database-name>' \
  -v ON_ERROR_STOP=1 -v job_id="$job_id" <<'SQL'
BEGIN TRANSACTION READ ONLY;

SELECT
    j.id,
    j.dag_id,
    j.name AS queue_name,
    j.state,
    j.retry_count,
    j.retry_limit,
    j.run_attempt_id,
    j.started_on,
    j.completed_on,
    j.data->'metadata'->>'ref_id' AS ref_id,
    jsonb_path_query_array(j.data, '$.**.pages') AS requested_pages,
    j.output
FROM marie_scheduler.job j
WHERE j.id = :'job_id'::uuid;

SELECT
    ja.run_attempt_id,
    ja.executor,
    ja.attempt_state,
    ja.activated_at,
    ja.dispatch_started_at,
    ja.dispatch_confirmed_at,
    ja.dispatch_error,
    ja.terminal_at,
    ja.terminal_status,
    ja.terminal_source,
    ja.recovery_at,
    ja.recovery_state,
    ja.recovery_reason,
    ja.gateway_instance_id
FROM marie_scheduler.job_attempt ja
WHERE ja.job_id = :'job_id'::uuid
ORDER BY ja.activated_at;

SELECT
    jh.history_created_on,
    jh.state,
    jh.retry_count,
    jh.retry_limit,
    jh.run_attempt_id,
    jh.started_on,
    jh.completed_on,
    jh.output
FROM marie_scheduler.job_history jh
WHERE jh.id = :'job_id'::uuid
ORDER BY jh.history_created_on;

WITH worker_history AS (
    SELECT
        kh.change_time,
        kh.operation,
        kh.value,
        COALESCE(
            NULLIF(kh.value->>'runtime_env_json', '')::jsonb,
            '{}'::jsonb
        ) AS runtime_env
    FROM marie_scheduler.kv_store_worker_history kh
    WHERE kh.namespace = 'job'
      AND kh.key = 'marie_internal/job_info_' || :'job_id'
)
SELECT
    change_time,
    operation,
    value->>'status' AS status,
    value->>'message' AS message,
    value->>'run_attempt_id' AS run_attempt_id,
    runtime_env #>> '{attributes,host}' AS executor_host,
    runtime_env #>> '{attributes,executor}' AS executor,
    runtime_env #>> '{attributes,runtime_name}' AS runtime_name,
    runtime_env #>> '{attributes,executor_endpoint}' AS endpoint,
    runtime_env #>> '{error,type}' AS error_type,
    runtime_env #>> '{error,message}' AS error_message
FROM worker_history
ORDER BY change_time;

ROLLBACK;
SQL
```

Look for:

- multiple `job_attempt` rows for the same job;
- different `run_attempt_id` values;
- `RUN_LEASE_EXPIRED` recovery after a worker is killed;
- `RUNNING` worker records without a terminal worker record;
- a retry beginning shortly after the prior process disappears;
- requested pages being absent, empty, or unexpectedly broad.

A kernel `SIGKILL` does not let Python publish an exception or execute a
`finally` block. The absence of a worker exception is therefore expected in a
host OOM. PostgreSQL recovery and kernel evidence complete the timeline.

## 7. Check discovery after the process dies

An executor address can remain registered even though its gRPC listener has
stopped. Query the gateway's readiness view:

```bash
gateway_url='<gateway-base-url>'

curl -fsS \
  "$gateway_url/api/discovery/readiness?state=unready" \
  | jq '.result'
```

For every unready address, verify the listener on its host:

```bash
executor_port='<port>'
sudo ss -ltnp | rg ":${executor_port}\\b"
```

`registered: true` with `ready: false` means discovery contains the address but
the readiness probe cannot confirm a serving gRPC endpoint. Treat discovery
registration, process liveness, and endpoint readiness as separate signals.

## 8. Apply containment

Choose containment based on the evidence:

- Stop or quarantine the exact poison job before restarting workers.
- Set raster page and decoded-byte limits on the executor deployment.
- Reduce executor concurrency while determining a safe memory envelope.
- Add a container or pod memory limit so one executor cannot consume all host
  RAM. A container limit protects the host but does not fix oversized input.
- Drain an unhealthy host if multiple services are affected.
- Preserve the original asset for offline reproduction under controlled limits.

Example Kubernetes configuration:

```bash
kubectl set env deployment/<executor-deployment> \
  MARIE_MAX_RASTER_PAGES=500 \
  MARIE_MAX_RASTER_DECODED_BYTES=8589934592
```

Apply production mutations only through the deployment's normal change-control
process. Increasing swap or blindly increasing the container limit is not a
root-cause fix.

Current operational caveat: `DocumentTooLargeError` is marked
`retryable = False`, but scheduler-wide propagation of that flag is not yet
implemented. Until it is, an oversized document may consume its normal retry
budget. Quarantine the job when the first limit failure is observed.

## 9. Verify recovery

After containment and restart:

```bash
docker stats --no-stream "$container_id"
nvidia-smi
curl -fsS "$gateway_url/api/discovery/readiness" | jq '.result.summary'
curl -fsS "$gateway_url/api/capacity" | jq
```

Confirm all of the following:

- executor RSS stabilizes across completed jobs;
- no new kernel OOM or NVIDIA Xid records appear;
- the dead address is removed or remains quarantined;
- healthy executor capacity returns;
- the poison job is terminal or quarantined rather than cycling;
- unrelated jobs complete normally.

## Incident report checklist

Record facts separately from inferences:

```text
Host:
Incident window with timezone:
Container name and full ID:
Image tag and immutable digest:
Marie version and commit:
Kernel victim PID and RSS:
Container namespace PID and replica:
Job ID:
Run-attempt IDs:
Executor and endpoint:
Asset reference:
Selected page count:
Estimated decoded RGB bytes:
Last completed application stage:
Scheduler recovery reason:
NVIDIA Xid present: yes/no
Containment action:
Root cause:
Corrective action:
```

Capture immutable image identity without printing the container environment:

```bash
docker inspect --format 'image_id={{.Image}} image={{.Config.Image}}' "$container_id"
docker image inspect --format 'id={{.Id}} digests={{json .RepoDigests}}' \
  "$(docker inspect --format '{{.Config.Image}}' "$container_id")"
```

Do not include credentials, signed object URLs, document contents, or patient
data in the incident report.
