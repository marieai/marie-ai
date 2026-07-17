# Marie Benchmarks

Benchmark-specific integrations live here.

Use this directory for benchmark planners, prompt and config assets, harness
glue, and small reproducibility notes that should live with Marie runtime code.

Do not use `packages/` for benchmark-only integrations. `packages/` is reserved
for reusable Marie extensions and product-adjacent modules.

Do not commit large benchmark outputs here. Official benchmark provider outputs
stay in the upstream benchmark repository's `runs/` directory; Marie runtime
artifacts stay in the configured object store.
