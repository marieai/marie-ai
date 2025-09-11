## Validators

Validators are pluggable checks that verify inputs, intermediate artifacts, and final outputs at specific points in the annotation pipeline. They help you enforce schema conformance, file/artifact completeness, and domain constraints, and they surface actionable diagnostics early.

### What a validator does

- Accepts a ValidationContext containing:
  - Stage: where in the pipeline the check is running (e.g., pre-processing vs parse-output).
  - Input data and/or file system locations relevant to the check.
  - Arbitrary metadata you choose to pass in (e.g., agent_output_dir, folder sets).
- Returns a ValidationResult with:
  - valid: boolean pass/fail status.
  - errors and warnings: machine- and human-readable diagnostics (code, message, optional field).
  - metadata: structured details that help debugging or reporting.
  - execution_time: timing info for observability.

Validators only run for stages they declare support for.

### When validators run

Validators can be attached to different lifecycle stages, for example:
- PRE_PROCESSING: Validate that required folders or artifacts exist before heavy work runs.
- PARSE_OUTPUT: Validate the structure and semantics of the annotator’s outputs.
- POST_PROCESSING: Validate final normalized results (optional).

Only validators that declare support for a given stage will run during that stage.

### How validators are referenced

Validators are referenced by name in your annotator configuration. Example:

```yaml
annotators:
  claims:
    annotator_type: "llm"
    model_config:
      model_name: deepseek_r1_32
      prompt_path: "./claims.j2"
      system_prompt_text: >
        ### Task
        You are a helpful assistant that extracts claims from the given text.
      expect_output: "json"
    parser_name: default
    validators:
      - "claims"              # schema/structure for claims output
      - "document_structure"  # generic structural checks
```


Attach as many validators as you need; each runs independently and reports its own result.

### Adding validators to an annotator config

Attach validators to any annotator in the same way you attach parsers or model settings:

```yaml
annotators:
  tables:
    annotator_type: "llm"
    model_config:
      model_name: deepseek_r1_32
      prompt_path: "./tables.j2"
      expect_output: "json"
    parser_name: default
    validators:
      - "tables"                 # domain/schema validator
      - "folder_sets_audit"      # file set integrity
      - "document_structure"     # general structure
```

### Authoring a new validator

1) Define its goal and stage
- Decide which artifact or output you’re validating and at which stage it makes sense to run (e.g., PRE_PROCESSING to ensure inputs exist, PARSE_OUTPUT to validate JSON shape).

2) Decide the context metadata you need
- Common items: agent_output_dir, folder names, file extensions, expected file companions, schema hints.

3) Implement the validator
- Expose a unique name (used in config).
- Declare supported stages (it can support multiple).
- Read context metadata and perform checks.
- Return a result with:
  - valid flag.
  - zero or more errors/warnings (each with a code and human-readable message).
  - optional metadata for diagnostics (counts, paths, expectations).
  - the system records execution time automatically for observability.

4) Register the validator
- Ensure it can be referenced by name in annotator configs.

5) Test
- Provide a minimal test harness or run it in a dry-run pipeline to verify:
  - Correct handling of missing folders/files.
  - Comprehensible error/warning messages.
  - Accurate counts in metadata.

### Best practices

- Be explicit and stable:
  - Use consistent error codes (e.g., MISSING_ANNOTATIONS, FOLDER_MISSING, MISMATCHED_FILE_COUNTS).
  - Keep messages actionable (“missing 2 of 10 companions”).
- Keep checks cheap:
  - Validators often run before heavier stages—fail fast on missing inputs.
- Make diagnostics useful:
  - Include counts, file paths, and per-folder summaries in metadata to speed up debugging.
- Compose validators:
  - Prefer small, focused validators that can be combined in different annotators rather than large, one-off checks.

### Troubleshooting

- “Validator does not support stage …”:
  - Ensure the validator is configured for the stage it runs in. Add the desired stage to its supported set.
- “File/folder missing” but it exists:
  - Verify that agent_output_dir and other path inputs are pointing to the correct job/workspace. Check permissions and casing.
- “Mismatched file counts”:
  - Ensure your pipeline produces a consistent trio of files (.json, .png, and prompt sidecars) or the required companions for markdown/fragment workflows.
- Silent passes with unexpected outputs:
  - Add a stricter, domain-specific validator (e.g., claims, tables) to enforce schema and field requirements, not just file presence.

With validators in place, you can catch issues early, enforce invariants across annotators, and keep your extraction pipeline reliable and auditable.