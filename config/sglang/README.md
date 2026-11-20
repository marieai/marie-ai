# SGLang deployment profiles

For inventory-based deployment across GPU hosts, use the
[SGLang Ansible role and playbook](../../ansible/README.sglang.md).
The shell and Compose profiles below remain available for local testing.

## Qwen3.8-27B NVFP4 on one RTX 5090

Run `sglang-qwen3_8-27b-nvfp4-rtx5090.sh` on a Linux server with one
RTX 5090 (32 GB), Docker, NVIDIA Container Toolkit configured for Docker, and
a CUDA 13-compatible NVIDIA driver. See the
[SGLang installation guide](https://docs.sglang.io/docs/get-started/install).
Reserve approximately 30 GB of free disk space for the checkpoint, plus space
for the container image and download cache.

The profile uses the
[NVIDIA NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
and the SGLang `v0.5.19` image used by the
[upstream RTX 5090 recipes](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B#hw=rtx5090&variant=default&quant=nvfp4&nodes=single&spec=none&tier=low-latency&ssmDtype=float32).
Quantization is detected from the checkpoint's mixed ModelOpt metadata: NVFP4
MLPs and output head, FP8 attention projections, and BF16 vision/MTP tensors.
The profile explicitly selects FP8 KV cache and FlashInfer attention for SM120.
It retains the vision encoder and float32 Mamba state.

Start with these conservative limits: 32,768 tokens per request (input plus
output), a 32,768-token total KV pool, one active request, 2,048-token prefill
chunks, and 90% static GPU memory. Four Mamba cache slots match the single active
request with `extra_buffer_lazy`. Speculative decoding is off. These limits
adapt the upstream recipe; they require validation on the target server.

Launch from the `marie-ai` repository root:

```bash
bash config/sglang/sglang-qwen3_8-27b-nvfp4-rtx5090.sh
```

The first run downloads the Docker image and caches the model weights in the
persistent Hugging Face cache. Wait for the server to report that it is ready. The API is published on
`127.0.0.1:8000`; clients use `http://127.0.0.1:8000/v1` and model id
`Qwen3.8-27B`. Stop the foreground container with Ctrl+C.

Check readiness and send a text request from another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"Reply with hello."}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

For LAN access, set `BIND_IP=0.0.0.0` and export `SGLANG_API_KEY` in your shell
before launching. Authenticated clients send
`Authorization: Bearer <your-key>`. The script forwards an exported `HF_TOKEN`
when one is available.

Override deployment settings through environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GPU_ID` | `0` | Select one host GPU by index or UUID. |
| `PORT` | `8000` | Publish this host port; the container stays on port 8000. |
| `BIND_IP` | `127.0.0.1` | Select the host interface. |
| `HF_CACHE_DIR` | `$HOME/.cache/huggingface` | Mount an absolute host cache directory. |
| `CONTAINER_NAME` | `sglang-qwen3-8-27b-nvfp4` | Name the foreground container. |
| `SGLANG_IMAGE` | `lmsysorg/sglang:v0.5.19` | Override the pinned runtime image. |
| `MODEL_PATH` | `nvidia/Qwen3.8-27B-NVFP4` | Select another compatible NVFP4 checkpoint. |
| `SGLANG_API_KEY` | Unset | Enable API authentication. |

Additional arguments are passed to SGLang after the defaults, so they can
override memory limits. For example, reduce prefill activation memory:

```bash
GPU_ID=0 PORT=8001 bash config/sglang/sglang-qwen3_8-27b-nvfp4-rtx5090.sh \
  --chunked-prefill-size 1024
```

Check the boot log for the allocated token pool and `max_running_requests`.
Increasing context, concurrency, or adding speculative decoding requires sizing
the KV and Mamba pools together; use the upstream recipe's calculator. With the
current lazy cache strategy, budget four Mamba slots per active request and
increase `--cuda-graph-max-bs-decode` when increasing `--max-running-requests`.
If other processes occupy the GPU, free that memory before sizing the profile.
Treat a successful boot and smoke request as startup validation; measure
throughput and validate extraction quality separately.

## Qwen3.8-27B NVFP4 on four RTX 5090s

Use `sglang-qwen3_8-27b-nvfp4-rtx5090-x4.sh` for four 32 GB RTX 5090s on
one Linux host, with Docker Compose and the same software prerequisites as the
single-GPU profile. The Compose configuration starts four independent model
replicas, one per GPU, each with a stable endpoint. Assign one complete document
batch to a replica and send all of that batch's LLM requests to its endpoint.
The four replicas can process four batches concurrently.

| Service | Host GPU | OpenAI-compatible base URL |
| --- | --- | --- |
| `replica-0` | `0` | `http://127.0.0.1:8000/v1` |
| `replica-1` | `1` | `http://127.0.0.1:8001/v1` |
| `replica-2` | `2` | `http://127.0.0.1:8002/v1` |
| `replica-3` | `3` | `http://127.0.0.1:8003/v1` |

Each replica uses the single-GPU profile's limits: 32,768 context tokens, a
32,768-token total KV pool, one active LLM request, four lazy Mamba cache slots,
float32 state, and FP8 KV cache. The vision encoder is retained. A batch can
contain many documents and calls; the request token limit applies to each call.

**Batch ownership belongs to the batch dispatcher.** Keep one replica reserved
until its assigned batch finishes, then assign the next batch from the queue.
SGLang's `--max-running-requests 1` limits simultaneous LLM calls; it does not
reserve a replica for a batch. Calls within a batch run sequentially by default.
The profile supplies endpoints; it does not change Marie's dispatch code.

Start the four replicas in the background from the repository root:

```bash
bash config/sglang/sglang-qwen3_8-27b-nvfp4-rtx5090-x4.sh up -d
```

The wrapper forwards Docker Compose commands. Inspect status, follow logs, and
stop the deployment with:

```bash
bash config/sglang/sglang-qwen3_8-27b-nvfp4-rtx5090-x4.sh ps
bash config/sglang/sglang-qwen3_8-27b-nvfp4-rtx5090-x4.sh logs -f
bash config/sglang/sglang-qwen3_8-27b-nvfp4-rtx5090-x4.sh down
```

Run without arguments for foreground startup. Set `GPU_0` through `GPU_3` to
choose host GPU indices or UUIDs, and `PORT_0` through `PORT_3` to change the
published ports. Use four distinct GPUs and ports. `BIND_IP`, `HF_CACHE_DIR`,
`SGLANG_IMAGE`, `MODEL_PATH`, and `SGLANG_API_KEY` work as in the single-GPU
profile. `COMPOSE_PROJECT_NAME` overrides the default deployment name,
`sglang-qwen3-8-27b-nvfp4-x4`. Export overrides in your shell and keep them set
for subsequent Compose commands. Configuration comes from exported variables;
the wrapper disables automatic `.env` loading.

Use the same model id and smoke request as above, targeting each replica's port.
For LAN access, export `BIND_IP=0.0.0.0` and `SGLANG_API_KEY` before starting;
batch workers use the server address in place of `127.0.0.1`. Check each replica:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8001/health
curl --fail http://127.0.0.1:8002/health
curl --fail http://127.0.0.1:8003/health
```

Benchmark completed documents or batches per minute, batch completion time, and
extraction quality. Four replicas do not combine their memory for one request
or make one batch four times faster. This is a configuration starting point;
GPU startup, concurrent batch execution, and throughput need target-server
validation. Change per-replica inference limits in the shared `x-sglang` section
of `compose-qwen3_8-27b-nvfp4-rtx5090-x4.yaml` when tuning batch concurrency.

## Benchmark existing QwQ profiles with `wrk`

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

