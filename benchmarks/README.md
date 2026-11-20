# Marie Benchmarks

Benchmark-specific integrations live here.

Use this directory for benchmark planners, prompt and config assets, harness
glue, and small reproducibility notes that should live with Marie runtime code.

Do not use `packages/` for benchmark-only integrations. `packages/` is reserved
for reusable Marie extensions and product-adjacent modules.

Do not commit large benchmark outputs here. Official benchmark provider outputs
stay in the upstream benchmark repository's `runs/` directory; Marie runtime
artifacts stay in the configured object store.

## LongExtract

Use the [LongExtract setup and evaluation guide](longextract/README.md) for
long-document structured extraction. The
[five-document results](longextract/docs/results.md)
record the verified scores, run provenance, output artifacts, and remaining
integration work.
