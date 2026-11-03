# marie-longextract

Benchmark-owned Marie integration for LongExtractBench runs.

Planner id: `longextract_bench`

The external entrypoint remains Marie Gateway `POST /api/v1/invoke`. The
benchmark provider passes the source PDF as `metadata.uri`, its media type as
`metadata.content_type`, and the schema/work/output object URIs under
`metadata.benchmark`.

The planner emits the existing format-aware document extraction node, the
shared `tables` annotator, a one-page schema aggregation-policy call, an optional
source-heading stage, an optional source-title stage, one LongExtract extraction call per page, and the existing
result-parser endpoint.
The external context provider supplies all schema units, the document-level
policy, bounded preceding-page source context, validated incoming headings and title attributes, and
current-page table hints to each page prompt. Outputs therefore use
page names such as `00001.json`, not table-style names such as `00001_t0.json`.

The page prompt emits ordered records with an explicit continuation decision.
The registered `longextract-aggregated` parser walks those records in page order
and applies the schema-derived carry and sequence policy. It does not infer
policy from field names or reinterpret the table hints supplied to the LLM.

Before aggregation, the parser compares a promoted refinement with its saved
initial pages using the recorded schema contract. If a refined page introduces
more row-schema violations, the parser selects that page's untouched initial
output and keeps the other valid refinements. It records the decision in
`debug/longextract-refinement-selection.json` and preserves both raw passes.
This check requires the saved refinement run and schema context; its report
states when those inputs are unavailable or do not match the promoted output.

## Verified five-document results

The verified evaluation of `qwen/qwen3.8-27b` matched **4,064/4,064
rows** across **127 pages**, with **99.3606% weighted leaf accuracy** and
**zero output-schema errors in all five documents**.

| Document | Pages | Leaf accuracy | Matched rows | Schema errors |
| --- | ---: | ---: | ---: | ---: |
| ACS | 30 | 96.9197% | 702 / 702 | 0 |
| Utility sales | 22 | 99.9908% | 1,560 / 1,560 | 0 |
| GDP | 24 | 99.9548% | 545 / 545 | 0 |
| Language census | 26 | 100.0000% | 765 / 765 | 0 |
| FDA | 25 | 99.3648% | 492 / 492 | 0 |
| **Combined** | **127** | **99.3606%** | **4,064 / 4,064** | **0** |

Row precision and recall are 100% for every document. FDA improved from
91.0522% leaf accuracy and 156 schema errors after source-cited title attributes
separated population, wear location, and sample count. All 25 FDA pages
completed extraction and refinement, followed by an isolated page 14 retry.

The other four saved predictions were independently regraded. Configurations,
selected passes, and reused stages vary across documents. The existing
`Validator 'tables' not found` runtime error remains an integration blocker;
value-level mismatches also remain despite valid output schemas.

See the [results report](docs/results.md) for metric definitions,
before/after results, exact run locations, verification commands, and full-output
artifact names. Large outputs remain in the adjacent benchmark harness checkout.

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

## Inspect the query plan

Build the LongExtract DAG through the registered planner and render it as a
PNG without starting Gateway or an executor:

```bash
cd ~/dev/marieai/marie-ai
python -m marie_longextract.planners.longextract_bench \
  --schema ~/path/to/test/schema.json \
  --output query_plan_graph.png
```

The command prints the registered planners, serialized plan, Mermaid graph,
and executor routes. It writes the visual graph to `query_plan_graph.png`.

## Evaluate the production Marie pipeline

Run the actual Marie table, page-extraction, parser, and LongExtractBench grader
path against page-aligned assets. Both table detection and page extraction send
the page image and the prepared MarkItDown text to the multimodal model.

Configure the test endpoint in `benchmarks/longextract/.env`. For a new checkout,
copy the example first, then edit `OPENAI_API_BASE` (including `/v1`) and
`OPENAI_API_KEY`:

```bash
cd ~/dev/marieai/marie-ai
cp benchmarks/longextract/.env.example benchmarks/longextract/.env
```

The local `.env` file is ignored by Git. `tools/run-with-env.py` loads it before
starting the benchmark command. File values override inherited environment
variables, and `MARIE_SKIP_LOAD_ENV=true` prevents Marie's default environment
loader from replacing them. Missing files or blank connection settings stop the
command before it starts. Values use literal dotenv syntax, without variable
interpolation. Use `--env-file /path/to/testing.env` before `--` to select another
file; the default path works from any working directory.

The example uses direct OpenAI-compatible requests with `LLM_QUEUE_ENABLED=false`.
For local inference, `LLM_HTTP_READ_TIMEOUT_S=1800` allows 30 minutes per request,
and `LLM_BATCH_TIMEOUT_S=7200` allows two hours per batch. Both values are seconds.
The local cohort is listed in `local-cohort.json`: five complete documents with
at most 30 pages each. The current selection contains 127 pages in total.
The model name remains in the annotator YAML configuration. AIMock returns a
canned `Sample Document` payload and cannot produce a valid score; the harness
detects that payload and stops immediately.

Table detection and refinement emit only `line_number` in `header_rows` and
`rows`. The existing parser supplies line text and location labels. This keeps
all source-line anchors without repeating per-line explanations; continuation
evidence stays in the table-level `continuation_rationale`.

Qwen3.8 thinking is configured per annotator. Table detection, page extraction,
and evidence-backed JSON repair explicitly set
`extra_body.chat_template_kwargs.enable_thinking=false`. The aggregation-policy
compiler enables thinking with `extra_body.reasoning_effort=low` to interpret
schema rules for carrying values and assigning sequences. Schemas with string
fields explicitly described as headings also use low-effort thinking in the
small `longextract-source-headings` call, capped at 8,192 output tokens. That
stage cites complete lines on the preceding page; the provider validates every
citation before supplying incoming headings to extraction. Current-page
headings end that incoming scope. Schemas without heading fields skip the stage.
Fields explicitly described as coming from table titles use the small
`longextract-source-titles` call with low-effort thinking and an 8,192-token cap.
It separates requested attributes for each printed title, including wrapped
titles. Joined source text blocks keep wrapped phrases readable while citations
retain the original line boundaries. The provider validates the current-page line range, complete source
quote, field types, and source support before supplying those attributes to
extraction and refinement. Schemas without title fields skip this stage.
The provider also corrects a contradictory incoming-continuation hint when a
single unambiguous printed table identifier changes; table counters and ambiguous
layouts remain subject to the table evidence. The ordered aggregation
parser remains deterministic. Page extraction keeps a 16,384-token output cap.
The server's total context must also fit the input prompt and image; a
32,768-token context does not provide 32,768 output tokens in addition to those
inputs. Each benchmark run freezes its own configuration so a change to thinking
mode does not alter an in-progress comparison.

```bash
cd ~/dev/marieai/marie-ai

GENERATOR_DIR=~/.marie/generators/your-generator-id
DOCUMENT_SLUG=your-dataset-document-slug
DATASET_REVISION=your-dataset-revision
BENCHMARK_REPO=~/dev/marieai/longextract-bench
DATASET_DIR=$BENCHMARK_REPO/.hf_cache/datasets--micro1-inc--longextract-bench-50/snapshots/$DATASET_REVISION/$DOCUMENT_SLUG

.venv/bin/python benchmarks/longextract/tools/run-with-env.py -- \
  .venv/bin/python -m longextract_bench.marie_pipeline_eval \
  --asset-dir "$GENERATOR_DIR" \
  --metadata "$GENERATOR_DIR/$DOCUMENT_SLUG.meta.json" \
  --schema "$DATASET_DIR/schema.json" \
  --ground-truth "$DATASET_DIR/ground_truth.json" \
  --out-dir "$BENCHMARK_REPO/runs/prompt-eval/your-run-name" \
  --config-root config/extract
```

Use a fresh output directory after changing prompts or models. Add `--resume`
only to continue an interrupted run whose completed stages should be reused.
The final score is written to `evaluation.json`; the exact multimodal table
prompt for page 3 is `agent-output/tables/00003.png_prompt.txt`.

## Evaluate agentic boundary repair

Run the repair agent independently against a saved generator workspace before
adding its node to the deployed query plan. This runs the same agent application
that the existing `AgentExecutor` hosts in production; it does not start another
Marie executor. The agent receives the previous, current, and next page images,
reads job-local extraction artifacts through scoped filesystem tools, and
returns a typed schema-boundary decision. The harness applies only `unit_name`
and `continuation.is_continuation` to a copied page record, then reruns the
existing `longextract-aggregated` parser and grader.

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
  --model qwen/qwen3.8-27b \
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
executor. It executes the same application code packaged for the existing
`AgentExecutor`. Each audit receives the prepared MarkItDown page text, table
evidence, schema, neighboring row context, and previous/current/next page images
inline. The current extracted target value is withheld so the agent selects its
source evidence independently. Three fresh, independently validated audits vote
on every legal target, and only a target-level majority is applied. Every audit
must return one review per target and cite an exact candidate from numbered
physical source lines. A review may cite an ordered multi-line span when the
schema and page structure establish one wrapped value. The validator preserves
token punctuation and verifies every fragment before the review can vote.

Use the same command for any downloaded LongExtract document by changing the
source run, dataset directory, pages, and optional field scope. The page and
field lists only scope the audit; they do not define expected replacements.

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
  --model qwen/qwen3.8-27b \
  --schema "$DATASET_DIR/schema.json" \
  --ground-truth "$DATASET_DIR/ground_truth.json"
```

Always use a new `OUT_DIR`; the source run is copied and never modified. The
ordered parser owns section-heading transitions. The agent handles existing
string leaves and cannot add rows, rewrite numeric values, or patch omitted
null fields. Each accepted non-null value must equal its cited source candidate;
the repair layer does not contain field-specific symbol or value normalizers.
Ground truth is loaded only after repair to run the benchmark grader and is
never included in an agent prompt.

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
fragment in `config/executors.yml` shows only those required module
registrations. It does not deploy another `AgentExecutor`. Production query-plan
agent nodes will route to the existing `AgentExecutor`, which hosts the packaged
LongExtract agent application through its deployment-owned plugin and route
allowlists.

Continuation aggregation is wired through Marie's existing parser executor.
The agentic repair can run independently against the same stable job workspace
for testing. Gateway execution will remain DAG-owned: each repair agent will be
an explicit query-plan node dispatched through the existing `AgentExecutor`;
the standalone harness does not own workflow order or install an executor.

The active annotator assets and planner mapper must be copied or mounted at:

```text
config/extract/TID-longextract-bench/annotator/
config/extract/TID-longextract_bench/mapper.yml
```

## Artifact contract

- `agent-output/longextract-unit-extract/<page>.json`: page-scoped ordered records
- `agent-output/longextract-aggregation-policy/00001.json`: schema-derived parser policy
- `agent-output/longextract-source-headings/<page>.json`: optional incoming headings with preceding-page source citations
- `agent-output/longextract-source-titles/<page>.json`: optional current-page title attributes with exact source citations
- `agent-output/longextract-aggregated/00001.json`: schema-shaped aggregate
- `agent-output/longextract-aggregated/trace.md`: continuation decision trace
- `parsed-result/longextract-result.json`: parser-stage schema-shaped result
- `debug/longextract-refinement-selection.json`: schema validation and page-pass selection audit
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
