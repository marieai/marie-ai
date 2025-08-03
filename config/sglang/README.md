# 📊 Benchmarking sglang with `wrk`

This guide shows how to benchmark your **Qwen/QwQ-32B deployment** using [`wrk`](https://github.com/wg/wrk), a modern
HTTP benchmarking tool.
We’ll simulate **multi‑user workloads** with both short and long prompts.

---

## 🔹 1. Install `wrk`

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install wrk -y
```

### Build from source (latest version)

```bash
sudo apt-get install build-essential libssl-dev git -y
git clone https://github.com/wg/wrk.git
cd wrk
make
sudo cp wrk /usr/local/bin
wrk --version
```

---

## 🔹 2. Lua Payload Scripts

Two Lua scripts are provided:

### `sglang_post.lua` (short prompt)

### `sglang_long_post.lua` (long prompt: \~2048 in / 1024 out)

---

## 🔹 3. Running Benchmarks

### Against single container (port `8000`)

```bash
# Short prompt, light load
wrk -t8 -c50 -d30s -s sglang_post.lua http://localhost:8000/v1/chat/completions

# Long prompt, heavier load
wrk -t12 -c100 -d60s -s sglang_long_post.lua http://localhost:8000/v1/chat/completions
```

### Against load‑balanced multi‑container (port `8080`)

```bash
wrk -t12 -c300 -d60s -s sglang_post.lua http://localhost:8080/v1/chat/completions
```

---

## 🔹 4. Test Profiles

We recommend three levels of testing:

* **Light load** → `-c50 -d30s` (simulate \~50 users)
* **Medium load** → `-c300 -d60s` (simulate \~300 users)
* **Heavy stress** → `-c1000 -d90s` (simulate \~1000 users)

Example (medium load, long prompts):

```bash
wrk -t16 -c300 -d60s -s sglang_long_post.lua http://localhost:8000/v1/chat/completions
```

---

## 🔹 5. What to Look At

`wrk` reports:

* **Requests/sec** → overall throughput
* **Latency** (avg, stdev, 95th/99th percentile) → multi‑user response times
* **Transfer/sec** → approximate bandwidth usage

Compare across:

* **Single 8‑GPU container (TP=8)** → lowest latency per request
* **Two 4‑GPU containers (TP=4 each, behind LB)** → higher sustained throughput with many users

---

## 🔹 6. Interpreting Results

* If **`wrk` req/sec << expected tokens/sec** from `sglang.bench_serving` → bottleneck is in API batching/scheduling,
  not the GPUs.
* Use `sglang_post.lua` for **short‑chat workloads** (low latency).
* Use `sglang_long_post.lua` for **stress testing long‑context generation**.

