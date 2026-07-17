# marie-longextract

Benchmark-owned Marie integration for LongExtractBench runs.

Planner id: `longextract_bench`

The external entrypoint remains Marie Gateway `POST /api/v1/invoke`. The
benchmark provider passes the source PDF as `metadata.uri`, its media type as
`metadata.content_type`, and the schema/work/output object URIs under
`metadata.benchmark`.

The planner emits the existing format-aware document extraction node, the
shared `tables` annotator, a one-page schema aggregation-policy call, one
LongExtract extraction call per page, and the existing result-parser endpoint.
The external context provider supplies all schema units, the document-level
policy, and current-page table hints to each page prompt. Outputs therefore use
page names such as `00001.json`, not table-style names such as `00001_t0.json`.

The page prompt emits ordered records with an explicit continuation decision.
The registered `longextract-aggregated` parser walks those records in page order
and applies the schema-derived carry and sequence policy. It does not infer
policy from field names or reinterpret the table hints supplied to the LLM.

## Install

Install the integration as an editable package in the Marie development
environment:

```bash
uv pip install \
  --python ~/environments/marie-ai-pytorch-2-12/bin/python \
  --editable ~/dev/marieai/marie-ai/packages/marie-kernel \
  --editable ~/dev/marieai/marie-ai/benchmarks/longextract
```

Changes under `packages/marie-kernel/src` and `benchmarks/longextract/src` are
then available to that environment without reinstalling either package.

Install the adjacent benchmark harness in the same environment to use
`python -m longextract_bench.providers.marie`:

```bash
uv pip install \
  --python ~/environments/marie-ai-pytorch-2-12/bin/python \
  --editable ~/dev/marieai/longextract-bench
```

Download the public dataset into the benchmark repository's `.hf_cache`:

```bash
cd ~/dev/marieai/longextract-bench
lxbench-download
```

Run one document with the existing Gateway E2E configuration:

```bash
cd ~/dev/marieai/longextract-bench
python -m longextract_bench.providers.marie \
  --config ~/dev/marieai/marie-ai/tools/stress/gateway-e2e.config.json \
  --pdf ~/path/to/test/document.pdf \
  --schema ~/path/to/test/schema.json \
  --out runs/marie-smoke/output.json
```

## Evaluate the production Marie pipeline

Run the actual Marie table, page-extraction, parser, and LongExtractBench grader
path against page-aligned assets. Both table detection and page extraction send
the page image and the prepared MarkItDown text to the multimodal model.

The real LLM service must be running behind the Marie queue dispatcher. AIMock
returns a canned `Sample Document` payload and cannot produce a valid score; the
harness detects that payload and stops immediately.

```bash
cd ~/dev/marieai/longextract-bench

GENERATOR_DIR=~/.marie/generators/your-generator-id
DOCUMENT_SLUG=your-dataset-document-slug
DATASET_REVISION=your-dataset-revision
DATASET_DIR=.hf_cache/datasets--micro1-inc--longextract-bench-50/snapshots/$DATASET_REVISION/$DOCUMENT_SLUG

export OPENAI_API_KEY=queue-only
export OPENAI_API_BASE=http://127.0.0.1:1/v1
export LLM_QUEUE_ENABLED=true
export LLM_QUEUE_VALKEY_URL=redis://127.0.0.1:6379/0
export LLM_QUEUE_POOL_ID=document-small

python -m longextract_bench.marie_pipeline_eval \
  --asset-dir "$GENERATOR_DIR" \
  --metadata "$GENERATOR_DIR/$DOCUMENT_SLUG.meta.json" \
  --schema "$DATASET_DIR/schema.json" \
  --ground-truth "$DATASET_DIR/ground_truth.json" \
  --out-dir runs/prompt-eval/your-run-name \
  --config-root /mnt/data/marie-ai/config/extract
```

Use a fresh output directory after changing prompts or models. Add `--resume`
only to continue an interrupted run whose completed stages should be reused.
The final score is written to `evaluation.json`; the exact multimodal table
prompt for page 3 is `agent-output/tables/00003.png_prompt.txt`.

## Evaluate agentic boundary repair

Run the repair agent independently against a saved generator workspace before
adding it to a deployed query plan. The agent receives the previous, current,
and next page images, reads job-local extraction artifacts through scoped
filesystem tools, and returns a typed schema-boundary decision. The harness
applies only `unit_name` and `continuation.is_continuation` to a copied page
record, then reruns the existing `longextract-aggregated` parser and grader.

```bash
cd ~/dev/marieai/marie-ai

GENERATOR_DIR=~/.marie/generators/your-generator-id
DATASET_DIR=~/dev/marieai/longextract-bench/.hf_cache/your-dataset-snapshot/your-document
OUT_DIR=~/dev/marieai/longextract-bench/runs/agent-repair/your-run-name
REPAIR_API_BASE=http://your-openai-compatible-host

python benchmarks/longextract/tools/evaluate-agent-repair.py \
  --asset-dir "$GENERATOR_DIR" \
  --out-dir "$OUT_DIR" \
  --page 29 \
  --record-index 0 \
  --api-base "$REPAIR_API_BASE" \
  --request-timeout-seconds 300 \
  --model qwen_v3_30b_instruct \
  --schema "$DATASET_DIR/schema.json" \
  --ground-truth "$DATASET_DIR/ground_truth.json"
```

Use a new `OUT_DIR` for every attempt. The source generator workspace is never
modified. Inspect:

- `repair-evaluation.json` for the agent decision and grader metrics
- `agent-output/longextract-agent-repair/decision.json` for the accepted typed decision
- `agent-output/longextract-unit-extract/<page>.json` for the copied, repaired source record
- `parsed-result/longextract-result.json` for the parser-owned final result

The runtime-only command omits benchmark grading:

```bash
python -m marie_longextract.repair_eval \
  --asset-dir "$GENERATOR_DIR" \
  --out-dir "$OUT_DIR" \
  --page 29 \
  --record-index 0 \
  --api-base "$REPAIR_API_BASE" \
  --request-timeout-seconds 300
```

## Evaluate agentic leaf repair

Run the string-leaf repair against a saved production-pipeline run. This path
constructs `ReactAgent` directly; it does not route through the deployable agent
executor. The agent has a read-only tool rooted at the copied job workspace.
Each audit also receives the prepared MarkItDown page text plus the previous,
current, and next page images inline. Three fresh, independently validated
audits vote on every legal target, and only a target-level majority is applied.

The following command reproduces the focused ACS residual evaluation. The page
and field lists only scope the audit; they do not define expected replacements.

```bash
cd ~/dev/marieai/marie-ai

SOURCE_RUN=~/dev/marieai/longextract-bench/runs/prompt-eval/your-source-run
DATASET_DIR=~/dev/marieai/longextract-bench/.hf_cache/your-dataset-snapshot/your-document
OUT_DIR=~/dev/marieai/longextract-bench/runs/agent-repair/your-leaf-run
REPAIR_API_BASE=http://your-openai-compatible-host

python benchmarks/longextract/tools/evaluate-agent-leaves.py \
  --asset-dir "$SOURCE_RUN" \
  --out-dir "$OUT_DIR" \
  --pages 1,10,16,17,28 \
  --fields program_group,section_heading,percent_value,percent_margin_of_error \
  --api-base "$REPAIR_API_BASE" \
  --request-timeout-seconds 300 \
  --model qwen_v3_30b_instruct \
  --schema "$DATASET_DIR/schema.json" \
  --ground-truth "$DATASET_DIR/ground_truth.json"
```

Always use a new `OUT_DIR`; the source run is copied and never modified. The
ordered parser owns section-heading transitions. The agent handles existing
string leaves and cannot add rows, rewrite numeric values, or patch omitted
null fields. Ground truth is loaded only after repair to run the benchmark
grader and is never included in an agent prompt.

Inspect:

- `leaf-repair-evaluation.json` for the official score and run summary
- `agent-output/longextract-agent-repair/<page>.json` for every audit and the
  accepted target-level consensus
- `agent-output/longextract-unit-extract/<page>.json` for the copied, repaired
  page extraction
- `parsed-result/longextract-result.json` for the parser-owned final result

Import `marie_longextract.planners.longextract_bench` from the Gateway query
planner configuration. The annotator executor must load
`marie_longextract.context_providers`; the provider reads the branch contract
from generic `RunContext` invocation parameters and controls one extraction per
page. The parser executor must load `marie_longextract.parsers`. The executor
fragment in `config/executors.yml` shows both required module registrations.

Continuation aggregation is wired through Marie's existing parser executor.
The agentic repair can currently run independently against the same stable job
workspace. Gateway query-plan routing for that repair remains separate from the
evaluation harness; this package does not install a serverless dispatcher.

The active annotator assets and planner mapper must be copied or mounted at:

```text
config/extract/TID-longextract-bench/annotator/
config/extract/TID-longextract_bench/mapper.yml
```

## Artifact contract

- `agent-output/longextract-unit-extract/<page>.json`: page-scoped ordered records
- `agent-output/longextract-aggregation-policy/00001.json`: schema-derived parser policy
- `agent-output/longextract-aggregated/00001.json`: schema-shaped aggregate
- `agent-output/longextract-aggregated/trace.md`: continuation decision trace
- `parsed-result/longextract-result.json`: parser-stage schema-shaped result
- `work/stitched-result.json`: future verified merge artifact
- `work/verification-findings.json`: shape and coverage findings
- `work/repaired-result.json`: verifier-scoped repair output
- `output_uri`: final schema-shaped JSON consumed by LongExtractBench

Large run outputs remain outside this repository.

## Local checks

```bash
pytest benchmarks/longextract/tests -q
python -c "import marie_longextract.planners.longextract_bench"
```
