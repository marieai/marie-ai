# Deploy SGLang batch replicas with Ansible

Use `playbook/sglang.yml` to deploy the Qwen3.8-27B NVFP4 profile across an
inventory of GPU hosts. The `sglang` role manages one Docker container per GPU,
with a stable endpoint for each replica. Its default layout matches the four
RTX 5090 Compose profile: GPUs 0–3, ports 8000–8003, and model `Qwen3.8-27B`.
Override the replica list for hosts with fewer or more GPUs.

The role uses `community.docker.docker_container` directly; Docker Compose and
the local launch scripts are not required on the managed hosts.

## Prepare the controller and hosts

Use an Ansible controller environment with Python 3.11+ and Ansible Core
2.19.9, the version used for local validation. Install the pinned collection:

```bash
cd ansible
ansible-galaxy collection install -r requirements.yml
```

Each managed host needs Linux, Python with `requests` available to Ansible's
interpreter, Docker, NVIDIA Container Toolkit configured for Docker, and a
CUDA 13-compatible NVIDIA driver. Provision those through your existing base
setup first. This role checks Docker and the selected GPUs but does not install
drivers, reset GPUs, or modify other workloads. Provide sufficient local disk
space for the image and model cache.

## Define the deployment inventory

Copy `inventories/sglang.example.yml` to your own inventory and replace the
example address, SSH user, and replica assignments. The example address is a
documentation placeholder. Add more hosts under `sglang_nodes` to expand the
deployment. Every host can override `sglang_replicas` independently:

```yaml
sglang_replicas:
  - name: batch-worker-0
    gpu_id: "0"
    port: 8000
  - name: batch-worker-1
    gpu_id: "1"
    port: 8001
```

Use GPU indices or GPU UUIDs. Names, GPU IDs, and published ports must be unique
within each host. These GPU assignments do not reserve devices against other
Docker deployments; operators must avoid assigning the same GPU to another
model service.

The default bind address is `127.0.0.1`. For remote batch workers, set
`sglang_bind_address` to the host's service IP or `0.0.0.0` and configure
`sglang_api_key` through your existing Ansible Vault or secret-management flow.
Use `sglang_hf_token` when model access requires a Hugging Face token. Container
tasks suppress their output, including diffs that could contain old credentials
when credentials are removed; do not place
credentials directly on the `ansible-playbook` command line.

## Validate and deploy

The following examples assume you are in `ansible/`. Substitute your inventory
and add your normal SSH, become, and Vault options. Syntax and variable
validation do not start containers:

```bash
ansible-playbook -i /path/to/sglang-hosts.yml playbook/sglang.yml --syntax-check
ansible-playbook -i /path/to/sglang-hosts.yml playbook/sglang.yml --tags sglang_validate
ansible-playbook -i /path/to/sglang-hosts.yml playbook/sglang.yml --check
```

Check mode performs read-only Docker and `nvidia-smi` preflight calls on the
selected hosts and asks the Docker module to predict changes. It needs access
to the hosts and Docker; it is not offline GPU qualification.

Drain active document batches on the selected host before applying changes
that recreate containers. Deploy one host first, then expand the rollout:

```bash
ansible-playbook -i /path/to/sglang-hosts.yml playbook/sglang.yml --limit gpu-01
ansible-playbook -i /path/to/sglang-hosts.yml playbook/sglang.yml
```

The playbook processes one host at a time and waits for each replica's HTTP
health endpoint before proceeding to the next replica. A failed host stops the
rollout. Re-running with the same settings does not force container recreation.
The first model download and startup can take time; the default readiness
timeout is 1,800 seconds per replica.

When migrating from the Compose profile, drain and stop the Compose deployment
first so it releases its GPUs and ports. Set `sglang_cache_dir` to the existing
host cache directory to reuse downloaded weights. The Ansible role manages its
own named containers and does not remove unrelated or old Compose containers.

## Customize the role

Put overrides in inventory/group/host variables or your normal variable files.
See `playbook/roles/sglang/defaults/main.yml` for the complete defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `sglang_image` | `lmsysorg/sglang:v0.5.19` | Pin a runtime tag or digest. |
| `sglang_model_path` | `nvidia/Qwen3.8-27B-NVFP4` | Select the compatible checkpoint. |
| `sglang_served_model_name` | `Qwen3.8-27B` | Set the model ID clients send. |
| `sglang_container_prefix` | `sglang-qwen3-8-27b-nvfp4` | Prefix each replica's container name. |
| `sglang_cache_dir` | `/var/cache/marie/sglang/huggingface` | Share downloaded weights between local replicas. |
| `sglang_bind_address` | `127.0.0.1` | Select the published interface. |
| `sglang_replicas` | Four entries | Assign names, GPU IDs, and ports per host. |
| `sglang_context_length` / `sglang_max_total_tokens` | `32768` / `32768` | Size each request and replica's shared KV pool. |
| `sglang_max_running_requests` | `1` | Limit active LLM requests per replica. |
| `sglang_max_mamba_cache_size` | `4` | Reserve at least four lazy Mamba slots per active request. |
| `sglang_mem_fraction_static` | `0.90` | Set static GPU memory allocation. |
| `sglang_chunked_prefill_size` | `2048` | Limit prefill activation memory. |
| `sglang_extra_args` | `[]` | Append additional SGLang arguments as a YAML list. |
| `sglang_wait_for_ready` / `sglang_ready_timeout` | `true` / `1800` | Control startup readiness checks. |
| `sglang_state` | `started` | Select `started`, `stopped`, or `absent`. |
| `sglang_serial` | `1` | Limit hosts deployed at once. |

Extra arguments override the defaults when SGLang accepts repeated options;
keep GPU topology and cache sizing consistent. The default command retains
FlashInfer attention, FP8 KV cache, float32 Mamba state, the vision encoder,
and checkpoint-detected mixed quantization. Do not force `--quantization fp8`
or `--quantization modelopt_fp4` on this mixed checkpoint.

To stop or remove only the configured replicas, drain their batches and run:

```bash
ansible-playbook -i /path/to/sglang-hosts.yml playbook/sglang.yml -e sglang_state=stopped
ansible-playbook -i /path/to/sglang-hosts.yml playbook/sglang.yml -e sglang_state=absent
```

Neither operation removes the shared model cache. Removed inventory entries
are not automatically deleted from hosts; explicitly retire their containers
before removing their replica definitions.

## Connect batch workers and monitor recovery

Clients use `http://<gpu-host>:<replica-port>/v1` with the configured model ID
and bearer API key. The batch dispatcher must reserve that endpoint for an
entire batch and release it when the batch finishes. This role does not
implement batch leases, checkpointing, retries, or node failover.

Containers use `unless-stopped` to recover from process exits and bounded local
log rotation. Their `/health` check reports HTTP readiness; it does not prove
GPU inference works, restart an unhealthy-but-running process, or recover a
wedged GPU. Ansible readiness checks happen during playbook execution, not as
a continuous recovery controller. Use your monitoring and operational recovery
procedures for those cases.

Validate a representative inference request on every endpoint, then exercise
concurrent batches and measure documents per minute and extraction quality.
Local Ansible checks alone do not qualify GPU startup, live deployment
idempotence, throughput, or recovery from hardware faults.
