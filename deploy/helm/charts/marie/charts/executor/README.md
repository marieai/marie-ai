# Render separate CPU and GPU executor pools

Give each enabled pool a unique lowercase DNS-label name. Keep CPU/GPU deployments of the same image or model in separate pools. `gpu`, `stateful` and `enabled` are booleans; omitted `gpu`/`stateful` retain false behavior. Disabled pools do not need complete compute settings.

The chart has one naming policy, with no version switches. Workloads and ordinary Services use `ex-<24 lowercase SHA-256 hex characters>`, hashing compact JSON `["marie.executor-workload", namespace, release, pool-name]`. Governing Services append `-headless`; StatefulSet `spec.serviceName` and workload `K8S_DEPLOYMENT_NAME`/`K8S_STATEFULSET_NAME` references use these same helpers. Logical pool IDs remain readable in `marie.ai/pool` labels and selectors, including numeric IDs as strings. Use those labels to identify pools.

Names remain short even with the 47-character hosted release convention: workloads are 27 characters, governing Services 36, and revision labels at most 38. Both CPU and GPU StatefulSets render under this policy. Duplicate generated identities reject. Stateful replica counts must be canonical whole values from 0 to 10,000; omitted replicas retain Kubernetes' default of one. Leading-zero strings reject to avoid YAML octal interpretation. Lifecycle phases that temporarily scale to zero still validate the selected count.

This policy is for the initial release. It changes workload and generated PVC identities compared with earlier development charts, even though the `data` claim template is unchanged. It supplies no migration or claim adoption for an installation made with those artifacts. Publish updated chart bytes as new immutable artifacts; do not replace old retained archives or apply the new chart to old state without an explicit migration.

CPU pools must omit the `nvidia.com/gpu` resource from both requests and limits. GPU pools require a positive whole `nvidia.com/gpu` limit, bounded to 1,000,000,000. An explicit GPU request must equal that limit; omitting it uses Kubernetes' limit-to-request default. This validation checks the existing NVIDIA resource seam, not arbitrary device resources or all Kubernetes CPU/memory quantity rules. [Kubernetes GPU scheduling rules](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/).

The chart preserves each pool's selected image, replicas, CPU/RAM resources, runtime class, node selector, affinity, tolerations and explicit environment entries. It does not inject `NVIDIA_VISIBLE_DEVICES=all` or `CUDA_VISIBLE_DEVICES=0`. The configured device plugin/runtime supplies GPU allocation; explicit operator visibility settings remain explicit configuration and need target qualification. [NVIDIA allocation strategies](https://github.com/NVIDIA/k8s-device-plugin#configuration-option-details).

Run local rendering regressions from the Marie AI repository:

```bash
python3 -m unittest discover -s deploy/helm/charts/marie/tests -p 'test_executor_*.py'
bash deploy/helm/charts/marie/tests/verify-gpu-executor-render.sh
```

These tests need Helm and PyYAML, render the real umbrella chart and never contact a cluster. Their pool labels, image references and runtime class are synthetic. They do not prove installed drivers/plugins, node capacity, placement policy, container GPU isolation, model compatibility or CPU/GPU execution. Update immutable publication versions and retained chart bytes together before using changed chart behavior in a release.

## Test setup with a mock executor

Apply `deploy/helm/charts/marie/values-sandbox-mock.yaml` as an explicit overlay to
select one `mock` pool. This replaces the entire executor pools list; use it for
an intentional setup environment, not an existing model deployment. Local and
hosted profiles must select this overlay explicitly; m3ctl's default installation
is unchanged. Retained publications must bind the mock pool to an approved,
immutable Marie executor image containing `IntegrationExecutorMock`.

The chart mounts `/etc/marie/config/service/mock/marie-mock-sandbox.yml` in both
Deployment and StatefulSet pods. It registers `mock_executor_a` with
`/document/process`. The Flow uses the chart's etcd and PostgreSQL environment
settings for discovery/registry. The mock does no asset I/O, model download,
plugin invocation or inference; no signing keys are needed. Its CPU request does
not qualify GPU scheduling or model execution.

Render the overlay from the Marie-AI root without contacting a cluster:

```bash
helm template sandbox-check deploy/helm/charts/marie \
  --namespace sandbox-check \
  -f deploy/helm/charts/marie/values-sandbox-mock.yaml
```

This preview inherits the base chart image defaults. Before deployment, select
approved image digests and the actual target's connections, Secret references
and storage. Publish new chart bytes and update retained publication locks; do
not overwrite an existing immutable archive.

Run the local setup regression with Helm and the development Python environment:

```bash
JINA_MP_START_METHOD=spawn .venv/bin/pytest -q \
  tests/integration/sandbox/test_mock_executor_flow.py
```

It boots the chart-rendered Flow on temporary local ports and exercises the real
gRPC request path. It disables discovery/registry explicitly because it owns no
etcd/PostgreSQL. It checks a successful result, a forced failure, and recovery.
It does not exercise the CLI server bootstrap, live registration, scheduler,
Kubernetes networking, image filesystem, persistent state or GPU.

Once an authorized disposable target runs the mock pool, select its pod and
forward the gRPC port in a separate terminal:

```bash
kubectl -n "$NAMESPACE" get pods -l 'app.kubernetes.io/component=executor,marie.ai/pool=mock'
kubectl -n "$NAMESPACE" port-forward "pod/$MOCK_POD" 52010:52010
```

Then run the result check from the Marie-AI root:

```bash
.venv/bin/python tools/sandbox/mock_executor_smoke.py --address 127.0.0.1:52010
# Negative control: must return exit 1; follow it with a normal probe.
.venv/bin/python tools/sandbox/mock_executor_smoke.py --address 127.0.0.1:52010 --force-fail
```

The probe prints JSON and exits zero only for the expected mock result with its
own echoed job ID. A successful RPC status or health check alone is insufficient:
the current executor completion path can report failures through the job
lifecycle while returning an empty RPC success response. This probe rejects that
response. The local port-forward uses plaintext gRPC; it is not a public endpoint
authentication test.

The mock overlay also registers `marie.query_planner.mock_query_plans` through
`server.queryPlanners`; normal chart defaults register none. `mock_simple` targets this executor
name, but enabling that planner and observing a scheduled job/terminal result is
a separate integration check. Full CPU/GPU execution and installed-state
Snapshot/restore remain release gates.

## Configure scheduled plugin execution

A plugin executor pool can use an explicitly provisioned execution daemon in
the same namespace. Set these fields on that pool, alongside its pinned image,
resources, replicas and placement:

```yaml
configFile: /etc/marie/plugin-executor.yml
pluginRuntime:
  serviceName: scheduled-plugins
  signingSecret: scheduled-plugin-signing
```

The approved executor image must contain the selected Flow and register
`plugin_daemon_executor` with `MariePluginDaemonExecutor`. The example path is
an image contract, not a file supplied by this chart. The path must be literal;
whitespace and `$` expansion are rejected. Configure that executor
with `start: false` for external runtime ownership. The chart injects
`MARIE_PLUGIN_DAEMON_URL=http://scheduled-plugins.<namespace>.svc:8099` and
signing ID/secret references to `key-id` and `secret` in the named Secret.
Both workload kinds support the binding. Reserved connection variables cannot
also appear in the pool's `env`; unselected pools receive no binding.

The provisioner must supply the Service, execution-capable daemon, persistent
runtime storage, matching signing material and required network access before
running jobs. It must install the selected package digest under the job's
organization/workspace identity. A binding does not create those resources or
prove package readiness. Missing Secret keys prevent the executor from starting;
an absent daemon or package is reported by the existing executor/runtime path.
Rotate credentials by restarting both clients and runtime.

Studio's preview daemon is a separate service with separate signing material.
Do not point scheduled pools at its Service. A remote daemon's subprocesses run
on its own host; assigning a GPU to the proxy executor does not allocate that
GPU to the remote plugin. OCR/model CPU/GPU execution remains independently
configured. This binding introduces no daemon replication or load-balancing
scheme; a replicated runtime needs its own package/state-routing design.
