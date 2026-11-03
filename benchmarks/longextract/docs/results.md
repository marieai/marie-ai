# Five-document LongExtract results

Model: `qwen/qwen3.8-27b`. Status: diagnostic cross-document evaluation;
Gateway integration qualification remains pending.

## Results

The cohort contains five distinct PDFs, 127 pages in total, with at most 30
pages per document. The dataset is `micro1-inc/longextract-bench-50`, revision
`d4ca332f72f6d3649ecd11b69445a8f4ef5fcb02`. Document identifiers and page counts
are recorded in [local-cohort.json](../local-cohort.json).

| Document | Pages | Leaf accuracy | Matched / expected rows | Row precision / recall | Schema errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACS | 30 | 96.9197% | 702 / 702 | 100% / 100% | 0 |
| Utility sales | 22 | 99.9908% | 1,560 / 1,560 | 100% / 100% | 0 |
| GDP | 24 | 99.9548% | 545 / 545 | 100% / 100% | 0 |
| Language census | 26 | 100.0000% | 765 / 765 | 100% / 100% | 0 |
| FDA | 25 | 99.3648% | 492 / 492 | 100% / 100% | 0 |
| **Combined** | **127** | **99.3606%** | **4,064 / 4,064** | **100% / 100%** | **0** |

The combined score is **25,642 correct graded leaves out of 25,807**. The
unweighted mean of document leaf accuracies is **99.2460%**. All five outputs
pass their document JSON schemas.

Leaf accuracy uses matched-row leaves and document fields. Unmatched rows affect
row precision and recall and are excluded from the leaf denominator. All rows
matched in this snapshot. Output-schema validity checks structure, required
fields, and types; it does not establish that every value is correct.

## Improvements captured in this snapshot

- **GDP:** Correct source-table ownership restored 72 rows to the level-change
  unit. Matched rows increased from 473/545 to 545/545, row precision and recall
  rose from 86.79% to 100%, and schema errors fell from 145 to zero. Leaf accuracy
  increased from 99.9032% to 99.9548%; correcting ownership also changed the
  population of graded leaves.
- **Language census:** Source-table eligibility and governing-heading context
  removed 564 out-of-scope rows and recovered missing state labels. Precision
  increased from 57.56% to 100%, recall stayed at 100%, leaf accuracy increased
  from 96.7102% to 100%, and schema errors fell from 615 to zero. A page-level
  schema guard retained page 6's initial response when refinement introduced
  73 schema violations; the remaining valid refinements were retained.
- **FDA:** The earlier 91.0522% result was an initial-pass replay after response
  envelope normalization. Its 156 missing `wear_location` values remained.
  Source-cited title attributes now separate population, location, and sample
  count before row extraction. Required title fields, schema types, and exact
  current-page citations are validated. Joined source text blocks keep wrapped
  age units with their population phrase. The completed 25-page extraction and
  refinement, followed by a page 14 title/field retry, scored 99.3648% with zero
  schema errors.

Ground truth is used only for offline grading. Model inputs and context rules
use the supplied schemas and document evidence. Original raw responses and
both field passes remain preserved; pass selection and page retries have
recorded provenance.

## Run provenance

The aggregate combines saved runs with different configurations and selected
passes. It measures the same five documents across those recorded variants.

All paths below are relative to the adjacent harness checkout's
`runs/generalization/` directory.

| Document | Source run | Selection and reuse |
| --- | --- | --- |
| ACS | `cohort-five-qwen38-local4-20261026T194917Z` | Original frozen prompts; selected `pass_0`. |
| Utility sales | `token-budget-20261027T183010Z/utility-document-retry` | Nonthinking field extraction; selected `pass_1`; reused tables and aggregation policy. |
| GDP | `schema-guard-final-v2-20261029T172200Z` | Reaggregated completed initial/refined inference with the schema guard; selected `pass_1`; upstream stages reused. |
| Language census | `schema-guard-final-v2-20261029T172200Z` | Reaggregated completed initial/refined inference; page-wise schema guard selected the initial page 6 response; upstream stages reused. |
| FDA | `fda-title-final-20261029T220543Z` | Completed inference plus a page 14 title/field retry; reaggregated the other 24 pages unchanged. |

FDA's full 25-page initial and refinement passes are retained in
`fda-complete-recovery-20261029T214945Z`. That run recorded 50 field-model
responses. The final FDA run made three additional requests: one title request
and two field passes for page 14. Its `new_model_requests: 3` counts that retry,
not the full evaluation. Tables, aggregation policy, and source annotations have
explicit reuse records and hashes. The final manifest also records the origin
of every reused field-page response.

Frozen runtime/config snapshots and file hashes identify the evaluated code and
inputs. The final aggregate independently regrades the other four saved
predictions without new model requests for those documents.

## Recorded execution settings

The completed full FDA run used one document at a time and at most four
concurrent model requests. The final page retry ran afterward. Its recorded
timeouts were 1,800 seconds per HTTP request, 7,200 seconds per batch, and 14,400
seconds per document.

Table detection, field extraction/refinement, and JSON repair use nonthinking
requests. Schema-policy compilation and the small source-heading/title tasks
use low-effort thinking. Field extraction/refinement has a 16,384-token output
cap; source-heading and source-title requests have an 8,192-token cap. All 50
full-document field responses and all three page 14 retry responses finished
with `finish_reason: stop`.

The model endpoint is loaded through the dedicated benchmark environment file;
see the [configuration and run commands](../README.md#evaluate-the-production-marie-pipeline).
The mixed runs and reused stages do not support a controlled latency or
throughput comparison.

## Artifacts and verification

Large outputs remain outside this repository. The final artifact root on the
evaluation machine is:

```text
/home/gbugaj/dev/marieai/longextract-bench/runs/generalization/fda-title-final-20261029T220543Z
```

| Artifact under that root | Contents |
| --- | --- |
| `final-five-document-evaluation.json` | Per-document and aggregate metrics, input/output hashes, baseline comparison, run variants, and reuse metadata. |
| `final-five-document-evaluation.txt` | Compact five-document table. |
| `final-output-verification.json` | Independent all-five grading, schema checks, ZIP integrity, comparison metadata consistency, and preserved-source checks. |
| `verification.json` | Final FDA case verification, including frozen runtime/config hash checks and runtime validation errors. |
| `manifest.json` | Frozen files, upstream reuse hashes, and page-wise field-response provenance. |
| `completion.json` | Completion and verification status; final retry request count. |
| `outputs/<document-id>.json` | Each complete extracted document, including FDA's final result. |
| `all-five-document-outputs.json` | All five extracted documents in one JSON file. |
| `five-document-results.zip` | Five extracted JSON files and the detailed/compact evaluation reports. |
| `cases/fda/page-retry/00014/` | Exact title/field prompts, request settings, raw completions, and parsed responses for the final page retry. |

Pin the evaluation report and output bundle by SHA-256:

```text
ad4791e96d1b32deb19585eb03c14f7c40f86445d21e998b4ce8c526ba7369af  final-five-document-evaluation.json
19958d2193ad22f213dbedd2a19fd2172ddc44f8b8dcff9d969fa49f4396dc9a  five-document-results.zip
```

Repeat the final FDA case's grading and frozen-file verification without model
requests from the Marie-AI checkout:

```bash
LXBENCH_RESULT_ROOT=/home/gbugaj/dev/marieai/longextract-bench/runs/generalization/fda-title-final-20261029T220543Z
MARIE_SKIP_LOAD_ENV=true .venv/bin/python "$LXBENCH_RESULT_ROOT/verify-cohort.py"
sha256sum "$LXBENCH_RESULT_ROOT/final-five-document-evaluation.json" \
  "$LXBENCH_RESULT_ROOT/five-document-results.zip"
```

`verify-cohort.py` covers the final FDA case and its frozen files and refreshes
the derived `verification.json` report. The separate
`final-output-verification.json` records the all-five grading, schema validity,
ZIP integrity, metadata consistency, and preserved-response checks. Input and
prediction hashes are recorded in `final-five-document-evaluation.json`.

Code validation recorded with this result: **171 benchmark tests passed**
(124 warnings), **two external harness checks passed**, and Black and
`git diff --check` passed. Run the benchmark suite with:

```bash
MARIE_SKIP_LOAD_ENV=true .venv/bin/python -m pytest \
  -c benchmarks/longextract/pyproject.toml \
  --confcutdir=benchmarks/longextract benchmarks/longextract/tests -q
```

## Remaining work before integration

The cohort retains 165 graded value mismatches: 140 in ACS, 23 in FDA, one in
GDP, and one in utility sales. FDA's remaining mismatches are 22
`concurrence_marginal_totals.total_axis` values and one
`standards_referenced.standard_reference_code` value. Their JSON types are valid.

The FDA runtime validation summary still reports five parsers passed and one
failed, with `VALIDATOR_NOT_FOUND: Validator 'tables' not found`. Output-schema
checks passed separately. Resolve that validator registration/loading issue
before claiming runtime validation success.

Afterward, evaluate the five PDFs with one frozen current configuration and
complete pipeline validation before Gateway integration. Preserve any upstream
reuse explicitly and retain the raw model responses, both field passes, and
independent grading evidence.
