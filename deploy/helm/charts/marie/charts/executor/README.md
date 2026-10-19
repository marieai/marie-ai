# Render separate CPU and GPU executor pools

Give each enabled pool a unique lowercase DNS-label name. The full rendered workload names must also remain unique after the chart's 63-character truncation. The default headless Service naming version (`v1`) appends `-headless` and cannot collide with another pool's Service. Under both versions, StatefulSet names must fit in 52 characters: Kubernetes appends a hyphen and up to ten revision-hash characters to a pod label limited to 63 characters. Invalid/colliding identities reject without silently renaming existing workloads or persistent claims. Numeric pool names remain strings in labels/selectors. Keep CPU/GPU deployments of the same image or model in separate pools. `gpu`, `stateful` and `enabled` are booleans; an omitted `gpu` or `stateful` uses the existing false behavior. Disabled pools do not need complete compute settings. [ControllerRevision naming](https://github.com/kubernetes/kubernetes/blob/v1.34.1/pkg/controller/history/controller_history.go), [StatefulSet revision labels](https://github.com/kubernetes/kubernetes/blob/v1.34.1/pkg/controller/statefulset/stateful_set_utils.go).

New approved publications may explicitly set `executor.headlessServiceNamingVersion: v2`. V2 names governing Services `ex-v2-<24 lowercase hex characters>-headless`, hashing compact JSON `["marie.executor-headless/v2", namespace, release, pool-name]` with SHA-256. The Service and StatefulSet `spec.serviceName` share this identity. Workload names, selectors and claim templates remain unchanged. Canonical examples live in `../../tests/executor-headless-identities.json`.

V2 also checks the pod hostname budget using the selected replica count (0–10,000, omitted defaults to one): `<StatefulSet-name>-<maximum-ordinal>` must fit 63 characters. Leading-zero replica strings reject to avoid YAML octal interpretation. Phase-zero replicas do not bypass the selected count's budget. The current hosted convention uses a 47-character release name, producing a 60-character StatefulSet for pool ID `cpu` or `gpu`; it remains unsupported even with short v2 Services. A separate full workload identity design is required. [StatefulSet pod identity](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/).

Keep v1 for existing revisions. Changing the governing Service on an installed StatefulSet requires a separate migration; this option does not implement or qualify one. Publish new chart bytes and explicit base values together for fresh environments. Studio does not inject v2 into existing publications.

CPU pools must omit the `nvidia.com/gpu` resource from both requests and limits. GPU pools require a positive whole `nvidia.com/gpu` limit, bounded to 1,000,000,000. An explicit GPU request must equal that limit; omitting it uses Kubernetes' limit-to-request default. This validation checks the existing NVIDIA resource seam, not arbitrary device resources or all Kubernetes CPU/memory quantity rules. [Kubernetes GPU scheduling rules](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/).

The chart preserves each pool's selected image, replicas, CPU/RAM resources, runtime class, node selector, affinity, tolerations and explicit environment entries. It does not inject `NVIDIA_VISIBLE_DEVICES=all` or `CUDA_VISIBLE_DEVICES=0`. The configured device plugin/runtime supplies GPU allocation; explicit operator visibility settings remain explicit configuration and need target qualification. [NVIDIA allocation strategies](https://github.com/NVIDIA/k8s-device-plugin#configuration-option-details).

Run local rendering regressions from the Marie AI repository:

```bash
python3 -m unittest discover -s deploy/helm/charts/marie/tests -p 'test_executor_*.py'
bash deploy/helm/charts/marie/tests/verify-gpu-executor-render.sh
```

These tests need Helm and PyYAML, render the real umbrella chart and never contact a cluster. Their pool labels, image references and runtime class are synthetic. They do not prove installed drivers/plugins, node capacity, placement policy, container GPU isolation, model compatibility or CPU/GPU execution. Update immutable publication versions and retained chart bytes together before using changed chart behavior in a release.
