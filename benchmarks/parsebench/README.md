# marie-parsebench

Standalone Marie integration for ParseBench. Dataset/runtime preparation,
versioned output contracts and native normalization/layout adapters are
implemented. The twelve-source Gateway preparation smoke run passed. Benchmark
page parsing integration produced five validated page artifacts through real
annotator/model calls. Source inspection found geometry and formatting errors;
these are unscored development observations. Document publication, native
provider transport and benchmark scoring remain later slices.

Keep this package under `benchmarks/parsebench/`, alongside `benchmarks/longextract/`.
Install benchmark dependencies explicitly in an isolated environment.

## Deploy the page annotator configuration

The benchmark wheel and configuration require the generic response-handler,
prompt-template and completion-attempt hooks in this integration tree. Both direct
and queued dispatch deliver attempt evidence before response validation. Record
the deployed Git revision and any local patch alongside each benchmark run.

Queued execution saves the prepared image, prompt, configuration, processing unit
and resolved per-page source identity before admission. The handler opens the
saved image temporarily under the shared preparation lock and verifies its hash.
Each physical provider attempt is saved before response normalization, including
refunded prewrite attempts and interrupted calls. Queued evidence is fenced by
claim and execution identity and retained in the existing durable request record.

Evidence plus the final result must fit the fabric's configured
`StoreLimits.result_allowance` (32 KiB by default; capture requires at least 256
bytes). Configure a larger allowance for long benchmark responses, accounting
for both raw attempts and the final completion. Overflow fails the result;
evidence is never silently truncated. A local evidence-copy failure leaves the
durable result unacknowledged under its normal bounded retention. Cancellation
retains its evidence budget while remote execution is unsettled and accepts late
completion evidence before settlement. Existing queue deadlines, retry policy,
trace context and remote reservation rules continue to apply.

The benchmark's recorded maximum of three transport attempts remains a validation
constraint. Use a benchmark fabric configured with that limit when qualifying
runs; the generic dispatcher continues to use its authoritative fabric policy.

Install the benchmark wheel in the host environment. Copy
`config/extract/TID-parsebench` into the active Marie `config/extract` directory,
then merge the executor fragment in `config/executors.yml` into the deployment
that serves `DocumentAnnotatorLLMExecutor`. Configure the host to import both the
root Marie package and every `packages/*/src` directory, including
`packages/marie-engine/src`, from that exact feature worktree. The executor
fragment imports `marie_parsebench.context_providers` so the existing component
registry can bind the package provider to `parsebench-page`.

Do not copy the benchmark configuration implicitly during package installation.
Use `load_page_annotator_config()` before a direct development invocation to
validate the fixed model settings and calculate the identity `config_sha256` from
the effective prompt template and generated candidate schema.

Handler protocol v5 accepts a bare JSON object or one complete outer
JSON-labeled or unlabeled Markdown code fence with no surrounding prose. It
removes only that envelope from the validation copy; immutable attempt evidence
keeps the exact raw response. Multiple or incomplete fences, other labels,
malformed JSON, and invalid candidate fields still fail without repair. Markdown
fences inside JSON string content are preserved as source data. Protocol v5 also
rejects source metadata that explicitly records `ocr_invoked=false` before its
compatibility boxes can be labeled as source OCR.

Run this import/signature preflight in the deployed host environment before any
page-parsing run. Set `PARSEBENCH_CONFIG` to the copied `config.yml`, not the
source-package copy. The assertions verify that the configured source tree and installed benchmark
package are the code actually imported by the interpreter.

```bash
set -e
export PARSEBENCH_MARIE_SOURCE=/absolute/path/to/marie-ai-parsebench
export PARSEBENCH_HOST_PYTHON=/absolute/path/to/host/bin/python
export PARSEBENCH_CONFIG=/absolute/path/to/active/config/extract/TID-parsebench/annotator/config.yml
git -C "$PARSEBENCH_MARIE_SOURCE" rev-parse HEAD
git -C "$PARSEBENCH_MARIE_SOURCE" diff --stat

parsebench_pythonpath="$PARSEBENCH_MARIE_SOURCE"
for package_source in "$PARSEBENCH_MARIE_SOURCE"/packages/*/src; do
  parsebench_pythonpath="$parsebench_pythonpath:$package_source"
done

PYTHONPATH="$parsebench_pythonpath" MARIE_SKIP_LOAD_ENV=true \
  "$PARSEBENCH_HOST_PYTHON" - <<'PY'
import inspect
import os
from pathlib import Path

from marie.engine.batch_processor import BatchProcessor
from marie.engine.openai_engine import OpenAIEngine
from marie.extract.annotators.llm_annotator import LLMAnnotator
from marie.extract.annotators.util import process_batch
from marie_parsebench.page_parsing import (
    RESPONSE_HANDLER,
    handle_page_response,
    load_page_annotator_config,
)

source = Path(os.environ['PARSEBENCH_MARIE_SOURCE']).resolve()
for symbol in (BatchProcessor, OpenAIEngine, LLMAnnotator, process_batch):
    module_path = Path(inspect.getfile(symbol)).resolve()
    assert source in module_path.parents, (symbol, module_path)

required_parameters = {
    process_batch: {'response_handler', 'prompt_template'},
    BatchProcessor.batch_generate: {'attempt_observer'},
    OpenAIEngine.batch_generate: {'attempt_observer'},
    handle_page_response: {'prompt_template', 'attempts'},
}
for callable_, required in required_parameters.items():
    missing = required - set(inspect.signature(callable_).parameters)
    assert not missing, (callable_, missing)

assert LLMAnnotator._load_response_handler(RESPONSE_HANDLER) is handle_page_response
loaded = load_page_annotator_config(os.environ['PARSEBENCH_CONFIG'])
assert loaded.annotator_conf['response_handler'] == RESPONSE_HANDLER
print(f'ParseBench core hook preflight passed: {loaded.config_sha256}')
PY
```

## Install the pinned harness

Use Python 3.12. Pydantic 2 validates the versioned runtime artifacts; preparation,
upstream evaluation, and development tools are optional dependency groups.
`requirements.lock` pins their combined environment, including the reviewed
ParseBench Git revision. It does not change Marie's root lock or Docker builds.

From this directory:

```bash
set -e
parsebench_checkout=/home/gbugaj/dev/marieai/ParseBench
test "$(git -C "$parsebench_checkout" rev-parse HEAD)" = 1dbb407c122d4b50b9896dc1f04c4c74ddd4fae3
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements.lock
uv pip install --python .venv/bin/python --no-deps --editable . --editable "$parsebench_checkout"
```

The final command installs the benchmark provider checkout in the isolated
environment. Record its Git status and any provider patch alongside a run.
An exact dependency sync restores the pinned Git installation; rerun the
editable-install command after syncing when developing the native provider.

To deliberately refresh the package-owned lock:

```bash
uv pip compile --python .venv/bin/python pyproject.toml \
  --extra prepare --extra harness --extra dev \
  --generate-hashes --output-file requirements.lock --quiet
```

## Prepare the dataset

The reviewed dataset revision is
`2805a1d940f95a203e0ae4b88be9934f7765b3fc`. `config/dataset-inventory.json`
contains its audited annotation hashes and counts, copied from the ParseBench
design's dataset inventory. Keep data and generated manifests outside Git.

```bash
set -e
parsebench_revision=2805a1d940f95a203e0ae4b88be9934f7765b3fc
parsebench_artifacts="$HOME/.cache/marie-benchmarks/parsebench/$parsebench_revision"
.venv/bin/python tools/download-dataset.py \
  --dataset-dir "$parsebench_artifacts/dataset" \
  --inventory config/dataset-inventory.json \
  --revision "$parsebench_revision"
.venv/bin/python tools/prepare-dataset.py \
  --dataset-dir "$parsebench_artifacts/dataset" \
  --output-dir "$parsebench_artifacts/manifests" \
  --inventory config/dataset-inventory.json \
  --revision "$parsebench_revision"
```

The downloader checks existing files before fetching missing files and verifies
all downloaded bytes against the pinned Hub Git-blob or LFS identities. A
corrupt existing file fails instead of being silently replaced. It uses public,
unauthenticated access and downloads the five annotation files, document
sources, and upstream evaluation configuration. Run this step before accepting
a newly supplied dataset directory as the official snapshot.

Preparation is offline. It verifies annotation hashes and counts, checks source
paths, hashes source bytes, and inspects real PDF/image page metadata. It emits:

- `input-manifest.json`: source identities and physical metadata, with separate
  evaluation memberships preserving original annotation IDs and page values.
- `smoke-manifest.json`: a deterministic selection of three sources per corpus
  (table, chart, text, layout). Text sources cover both text groups; layout
  includes a PDF, JPG, and PNG.

Expected inventory: 2,078 source paths, 2,553 group memberships, 169,011 annotation
rows, and 475 shared text sources. Physical page counts come from the files.
Neither manifest contains reference answers, rules, or layout labels. Membership
metadata is for evaluation bookkeeping and must not be forwarded to inference.

Repeat the preparation command to verify the same snapshot and manifest bytes.
Changed data, revision, inventory, or existing manifests fail verification; a
new protocol requires a new output directory. Offline preparation verifies the
local snapshot against the audited annotations and previously frozen manifest;
the download step establishes source-byte provenance against the remote revision.

## Verify locally

```bash
.venv/bin/python -m pytest -c pyproject.toml --confcutdir=. tests -q
.venv/bin/python -m ruff check src tools tests
.venv/bin/python -m black --check -S src tools tests
.venv/bin/python -m isort --check-only --profile black src tools tests
```

Use the package's pytest configuration and `--confcutdir` to keep preparation
tests independent of the root Marie runtime test setup.

## Integration boundary

The runtime wire models live in `src/marie_parsebench/models.py`.
`DocumentArtifact.model_validate_json(...)` validates serialized documents;
`normalize_bbox_xyxy(...)` converts rendered/cropped/rotated boxes into full
displayed-source normalized coordinates. Intrinsic PDF orientation is already
part of that source frame. Missing geometry stays absent.

The native converter is
`parse_bench.inference.providers.parse.marie_normalization.normalize_marie_artifact`.
It accepts the raw artifact plus the recorded example, pipeline, run, source hash,
configuration hash and expected page indices. Supply those expectations from the
request journal. Keep the artifact in `InferenceResult.raw_output` alongside the
returned ordinary `ParseOutput`, so later normalization can replay it without
another model call. The generated `document-v1.schema.json` must remain identical
in both repositories; native code also checks cross-field constraints.

Native evaluation registers provider `marie` for the exact pipeline
`marie_parsebench_v1` through the existing adapter and mapper package imports.
It preserves mixed page sizes in a fixed virtual coordinate frame and omits
unlocalized elements. Saved layout results must retain that frame and the flat
prediction representation. Hand-labeled fixtures verify evaluator behavior;
they are not measured Marie benchmark results. The existing document-wide AP
path does not distinguish source pages within a document; its per-page rule
path is verified separately.

Future planner, context provider, prompts, configuration, and result parser code
belong in this package. Native provider and output adapters belong in the
adjacent ParseBench checkout under `src/parse_bench/`. Add host dependencies only
when the runtime slices need them; local reusable Marie packages resolve through
`../../packages/<package>` from this directory.

The implemented tools prepare the dataset and inspect local runtime loading.
A real Gateway run processed all twelve frozen smoke sources (ten PDFs, one JPG
and one PNG) without retries. Readback verified 168 S3 artifacts, real page/OCR
geometry and distinct working directories, including identical source bytes
under different references. Benchmark parsing and scoring belong to later slices.

The 2026-09-09 evidence is outside Git under the pinned dataset cache:
`prep-20260909t200528z-1f36fbce/smoke-verification.json`, SHA-256
`3789be9be9e2d3c803fb4503a22082fdc57f0747365ee45cbba9fdf211eb3a98`.
This used the real `TextExtractionExecutor` with
`MARIE_LEGACY_WORKING_DIR=false` and the separately reviewed NumPy character-count
fix. Clean/burst exports are binarized derivatives; retain the original source
frames for color inputs. Cancellation remains unsupported, and successful job
status must still be followed by final artifact readback.

## Check runtime preparation

`config/preparation-profile.json` is a request template for the existing Gateway
`POST /api/v1/invoke` route. It selects the built-in `extract` planner and existing
`extract_executor://document/extract` endpoint. It is preparation-only; the full
ParseBench planner belongs to a later slice.

The template's `invoke_action.name="parsebench-preparation"` selects the benchmark
preparation queue. Current scheduler storage uses an unpartitioned `job` table;
queue names are logical routing metadata. The live check found a legacy database where
the queue functions had been updated but the physical unpartitioning migration
had not run. That mismatch rejected `parsebench-preparation` before execution.
Resolve it with the scheduler's maintenance migration; new queues should not
require physical partitions.

Send `Authorization: Bearer <api-key>` with the HTTP request and supply the same
key at `parameters.invoke_action.api_key` in memory. Both are required by the
current Gateway handlers. Populate `metadata.project_id`, `uri`, `ref_id`,
`queue_id`, and `content_type`.
Retain `metadata.benchmark.run_id`, `source_sha256`, and `config_sha256` for the
verification record. Use a unique lowercase `ref_id` for each run/source/config;
stage the exact verified source bytes at `uri`. Keep credentials out of saved
request artifacts. `pages`, when provided, is a comma-separated list of
zero-based source page indices, such as `"2,0"`; it preserves that order.

The required `metadata.run_params.parse_mode="ocr"` passes through the scheduler
and distributor into `TextExtractionExecutor._parse_mode()`. Planner
`op_params.parse_mode` alone does not select OCR. `return_ocr=true` retains the
preparation metadata for verification. Runtime-side tests exercise these actual
functions for PDF, JPG, and PNG; they do not submit a network job or run OCR.

Set `MARIE_LEGACY_WORKING_DIR=false` on the preparation executor to include
`ref_id` and `ref_type` in its local work-directory identity. The default legacy
layout only hashes rendered pixels and can reuse a directory across runs.
Record this setting from the deployed executor before accepting a live check.

Use an existing Marie host environment for the host tests:

```bash
parsebench_worktree=/home/gbugaj/dev/marieai/worktree/marie-ai-parsebench
parsebench_host_python=/home/gbugaj/environments/marie-ai-pytorch-2-12/bin/python
PYTHONPATH="$parsebench_worktree:$parsebench_worktree/benchmarks/parsebench/src" \
  "$parsebench_host_python" -m pytest -c pyproject.toml --confcutdir=. tests/host -q
```

The lightweight preparation environment skips host tests when Marie is absent.
Keep it independent of the host/GPU environment.

The local loader probe produces source-image evidence without running OCR:

```bash
parsebench_artifacts="$HOME/.cache/marie-benchmarks/parsebench/2805a1d940f95a203e0ae4b88be9934f7765b3fc"
PYTHONPATH="$parsebench_worktree:$parsebench_worktree/benchmarks/parsebench/src" \
  "$parsebench_host_python" tools/inspect-runtime-preparation.py \
  --dataset-dir "$parsebench_artifacts/dataset" \
  --input-manifest "$parsebench_artifacts/manifests/input-manifest.json" \
  --smoke-manifest "$parsebench_artifacts/manifests/smoke-manifest.json" \
  --output-dir "$parsebench_artifacts/local-runtime-preparation"
```

The probe invokes the existing source loader and frame-size cap, records page
mapping and resize information, and saves actual frame images. These establish
local source-loading behavior only. Use a new output directory for each run;
existing output and paths overlapping the dataset or manifests are rejected.
The accepted twelve-source Gateway smoke also verified executor configuration,
OCR geometry, pipeline transforms, artifact persistence and terminal status for
the recorded deployment. Reverify these when the runtime deployment changes.

Semantic extraction metadata with `extraction.route="plugin"` and
`ocr_invoked=false` contains synthetic compatibility boxes. Its text may be
useful, but its boxes and computed `imageSize` cannot establish visual grounding.
The versioned contract rejects synthetic compatibility provenance. Runtime
producers must supply independently verified rendered-frame/OCR or source-image
localization evidence.

Operational job status and preparation artifact readback are verified. The
inspected stop/delete routes only acknowledge requests, and storage errors can
be logged without failing the job. A scheduler completion or an S3-looking path
therefore cannot replace final benchmark artifact readback and validation.
