# 🩺 Marie-AI Log Healer Sidecar (Fluentd + Docker API)

## Overview

The **Log Healer Sidecar** monitors Docker logs for GPU or CUDA memory errors (e.g., *CUDA out of memory*, *cublas alloc failed*) and automatically restarts the affected container using the **Docker Engine API**.
It includes built-in **cooldown**, **rate limiting**, and **persistent restart logs** under `/tmp/marie-healer/`.

---

## 🧩 Architecture

```
+-----------------------+         +---------------------------+
| marieai-dev-server    |         | marieai-dev-log-healer    |
|  (main AI container)  |  logs→  |  Fluentd tail input       |
|                       |         |  grep GPU OOM patterns    |
|                       |         |  exec restart.sh          |
+-----------------------+         +-------------+-------------+
                                              |
                                              v
                                 +----------------------------+
                                 | Docker Engine API via sock |
                                 +----------------------------+
```

---

## 🧰 Components

| File                                                                         | Purpose                                                   |
| ---------------------------------------------------------------------------- | --------------------------------------------------------- |
| `/mnt/data/marie-ai/config/extract/observability/fluentd/fluent.conf`        | Fluentd pipeline (tail → grep → exec)                     |
| `/mnt/data/marie-ai/config/extract/observability/fluentd/trigger_restart.sh` | Reads matches from Fluentd and calls restart script       |
| `/mnt/data/marie-ai/config/extract/observability/fluentd/restart.sh`         | Restarts container via Docker API with cooldown & logging |
| `/tmp/marie-healer/`                                                         | Persists restart metadata, logs, and rate-limit files     |

---

## 🪶 Behavior

1. **Fluentd** tails Docker JSON logs for your target container (e.g., `marieai-dev-server`).
2. **Grep filter** matches GPU OOM patterns (`cuda out of memory`, etc.).
3. Matching lines trigger **`trigger_restart.sh`**, which calls `restart.sh`.
4. **`restart.sh`**:

   * Checks rate-limit and cooldown windows.
   * Calls Docker Engine API to restart the container.
   * Writes persistent logs in `/tmp/marie-healer/`.

---

## 🧾 Restart State in `/tmp/marie-healer/`

| File                                              | Description                |
| ------------------------------------------------- | -------------------------- |
| `last_restart.json`                               | Latest restart info (JSON) |
| `restarts.log`                                    | Append-only JSONL history  |
| `restart-<target>-<epoch>.marker`                 | Marker per event           |
| `<target>.last_restart` / `<target>.restart_hist` | Cooldown & rate-limit data |

Example:

```json
{
  "timestamp": 1760978852,
  "iso_time": "2025-10-20T21:07:32Z",
  "host": "asp-gpu002",
  "target": "marieai-dev-server",
  "container_id": "a5e01e0d66b8",
  "cooldown_sec": 300,
  "window_sec": 600,
  "window_count": 2
}
```

---

## ⚙️ Ansible Mounts

```yaml
volumes:
  - "/var/lib/docker/containers:/var/lib/docker/containers:ro"
  - "/mnt/data/marie-ai/config/extract:/etc/marie/config/extract:ro"
  - "/var/run/docker.sock:/var/run/docker.sock:rw"
  - "/tmp/marie-healer:/tmp/marie-healer:rw"
```

Pre-task to ensure `/tmp/marie-healer` exists:

```yaml
- name: Ensure /tmp/marie-healer directory exists
  ansible.builtin.file:
    path: /tmp/marie-healer
    state: directory
    mode: "1777"
```

---

## 🧪 Test Scenarios & Commands

### 1️⃣ Verify Fluentd Health Endpoint

Check that the Fluentd sidecar is running and healthy:

```bash
curl -s http://127.0.0.1:24220/api/plugins.json | jq '.plugins | length'
```

✅ You should see a list of Fluentd plugins or a non-empty count.

---

### 2️⃣ Verify Docker Engine Socket Access

Make sure Fluentd (and the restart script) can reach the Docker Engine:

```bash
docker exec -it marieai-dev-log-healer sh -lc \
  'curl --unix-socket /var/run/docker.sock http://localhost/_ping && echo'
```

✅ Expected output:

```
OK
```

---

### 3️⃣ Manually Trigger a Restart

Bypass cooldown to verify that `restart.sh` works:

```bash
docker exec -it marieai-dev-log-healer sh -lc \
  'TARGET_CONTAINER=marieai-dev-server COOLDOWN_SEC=0 /etc/marie/config/extract/observability/fluentd/restart.sh'
```

✅ Expected output:

```
[restart.sh] ... Restarting container 'marieai-dev-server' ...
[restart.sh] ... Restart complete.
```

Then confirm the log entries:

```bash
ls -l /tmp/marie-healer/
cat /tmp/marie-healer/restarts.log | tail -n 3
```

---

### 4️⃣ Test Rate Limiting

Run the same command repeatedly within a few seconds:

```bash
for i in $(seq 1 5); do
  docker exec -it marieai-dev-log-healer sh -lc \
    'TARGET_CONTAINER=marieai-dev-server /etc/marie/config/extract/observability/fluentd/restart.sh'
done
```

✅ After the configured number of allowed restarts (`MAX_PER_WINDOW`), you should see:

```
[restart.sh] Max restarts (4) in last 600s reached — skipping.
```

---

### 5️⃣ Simulate a GPU Error in Logs

Append a fake error message to the container log to trigger Fluentd automatically:

```bash
docker exec -it marieai-dev-server sh -c \
  'echo "{\"log\": \"RuntimeError: CUDA out of memory\", \"stream\": \"stderr\"}" >> /var/lib/docker/containers/$(hostname)/$(hostname)-json.log'
```

✅ Within a few seconds, Fluentd should detect it and trigger a restart.
You can watch Fluentd logs in real-time:

```bash
docker logs -f marieai-dev-log-healer
```

---

### 6️⃣ Inspect Persistent Restart State

Check what the healer recorded:

```bash
cat /tmp/marie-healer/last_restart.json | jq
```

✅ Shows metadata for the latest restart.

List markers:

```bash
ls -l /tmp/marie-healer/restart-*
```

---

### 7️⃣ Test Cooldown Logic

Run a restart, then immediately try again:

```bash
docker exec -it marieai-dev-log-healer sh -lc \
  'TARGET_CONTAINER=marieai-dev-server /etc/marie/config/extract/observability/fluentd/restart.sh'
```

✅ You should see:

```
[restart.sh] Cooldown active (45s < 300s) — skipping.
```

---

## ✅ Validation Checklist

| Check               | Command                                                          | Expected               |              |
| ------------------- | ---------------------------------------------------------------- | ---------------------- | ------------ |
| Fluentd running     | `docker ps                                                       | grep log-healer`       | Container up |
| Docker socket works | `curl --unix-socket /var/run/docker.sock http://localhost/_ping` | `OK`                   |              |
| Manual restart      | Run restart.sh manually                                          | "Restart complete"     |              |
| Rate limit enforced | Repeat restarts                                                  | “Max restarts reached” |              |
| Cooldown enforced   | Immediate retry                                                  | “Cooldown active”      |              |
| Logs persist        | `ls /tmp/marie-healer`                                           | Files present          |              |

---

## 🧾 Summary

* **Lightweight & safe:** automatic restart on GPU failures
* **Self-logging:** all actions tracked under `/tmp/marie-healer`
* **Infrastructure-agnostic:** works on any Docker host with Fluentd
* **Production ready:** cooldown, rate limits, persistence, and observability built-in


