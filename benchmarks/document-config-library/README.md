# Marie document configuration library

This benchmark boundary accepts neutral, digest-locked candidates produced by
M3 Studio's `document-config-converter`. It does not read the source checkout or
execute the source runtime.

The first pilot is the generalized single-account bank statement. Importing a
candidate only proves package integrity and compatibility with Marie's current
query-plan contract. It does not make the candidate qualified or publishable.

## Import the pilot

Install the benchmark boundary as an editable package in the Marie development
environment:

```bash
uv pip install \
  --python /home/gbugaj/environments/marie-ai-pytorch-2-12/bin/python \
  --editable /home/gbugaj/dev/marieai/marie-ai/benchmarks/document-config-library \
  --no-deps
```

```bash
cd /home/gbugaj/dev/marieai/marie-ai
/home/gbugaj/environments/marie-ai-pytorch-2-12/bin/python \
  -m marie_document_config_library.import_candidate \
  --source /home/gbugaj/dev/marieai/marie-studio/generated/document-config-library/exports/document-config.financial-services-bank-statements-ai-generalized-single-account-statement \
  --destination benchmarks/document-config-library/imports
```

The importer verifies every declared artifact digest, rejects undeclared files,
symlinks, unsafe paths, remote executable references, invalid Marie routes, and
conflicting content under the same configuration ID. Re-importing the identical
package is idempotent.

## Preflight

```bash
/home/gbugaj/environments/marie-ai-pytorch-2-12/bin/python \
  -m marie_document_config_library.preflight \
  --candidate benchmarks/document-config-library/imports/document-config.financial-services-bank-statements-ai-generalized-single-account-statement \
  --config-root config/extract \
  --evidence-root benchmarks/document-config-library/qualification/document-config.financial-services-bank-statements-ai-generalized-single-account-statement
```

Preflight fails closed until the runtime layout configuration, reviewed expected
output, and real qualification result exist. Qualification evidence lives
outside the immutable imported package. AIMock output is not accepted as
qualification evidence.

## Focused checks

```bash
/home/gbugaj/environments/marie-ai-pytorch-2-12/bin/python -m pytest \
  benchmarks/document-config-library/tests -q
```
