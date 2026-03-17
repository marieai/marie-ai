---
sidebar_position: 2
---

# Configuration: directory structure, prompt resolution, and examples

This section describes how annotators are configured, where prompts live, and how the system locates and loads them at runtime.

### Config directory layout

- All prompt paths are resolved relative to the config base directory, which defaults to: `__config_dir__/extract/`
- Typical structure:

```plain text
extract/
  config/
    base-config.yml
    field-config.yml
  TID-<layout_id>/
    annotator/
      config.yml
      <prompt_a>.j2
      <prompt_b>.j2
      <prompt_c>.j2
```


- `<layout_id>` identifies the target document layout. Each layout keeps its own annotator config and prompt files under `TID-<layout_id>/annotator/`.

### Prompt path resolution

- `model_config.prompt_path` is treated as a path relative to the configuration base. For example:
  - `prompt_path: "./claims.j2"` is resolved within `TID-<layout_id>/annotator/claims.j2`.
- You can also supply an absolute prompt (e.g., via an alternate config root), but best practice is to keep prompts co-located with the annotator config for a specific layout under `TID-<layout_id>/annotator/`.
- If both `prompt_path` and `system_prompt_text` are present, the system uses `system_prompt_text` as the system role content and the Jinja2 prompt template as the user/content portion. This makes it easy to:
  - Keep a stable system prompt that describes the task.
  - Swap prompt templates per layout or per annotator.

### Annotator selection at runtime

- A job selects:
  - The annotator key (e.g., `claims`, `tables`, `key-values`) to specify which annotator to run.
  - The target `layout_id` to locate the right `TID-<layout_id>/annotator/config.yml` and its referenced prompt files.
- This keeps runtime requests small and declarative while enabling per-layout customization.

### Example YAML configuration

```yaml
# ALL prompt paths are relative to the config base directory defaulting to `__config_dir__/extract/`

# Structure of the config directory:
#   - extract/
#     - config/
#       - base-config.yml
#       - field-config.yml
#     - TID-<layout_id>/
#       - annotator/
#         - config.yml
#         - <prompt_a>.j2
#         - <prompt_b>.j2
#         - <prompt_c>.j2

annotators:
  # KEY-VALUE ANNOTATOR
  key-values:
    annotator_type: "llm"
    model_config:
      model_name: deepseek_r1_32
      prompt_path: "./key-value.j2"
      system_prompt_text: >
        ### Task
        You are a helpful assistant that extracts claims from the given text.
      top_p: 1.0
      frequency_penalty: 0
      presence_penalty: 0
      expect_output: "json"

    parser_name: default # parser to parse the key-value pairs
    validators: # Custom validators for key-values
      - "key-values"
      - "document_structure"


  # CLAIMS ANNOTATOR
  claims:
    annotator_type: "llm"
    model_config:
      model_name: deepseek_r1_32
      prompt_path: "./claims.j2"
      system_prompt_text: >
        ### Task
        You are a helpful assistant that extracts claims from the given text.
      top_p: 1.0
      frequency_penalty: 0
      presence_penalty: 0
      expect_output: "json"
    parser_name: default # parser to parse the claims
    validators: # Custom validators for claims
      - "claims"
      - "document_structure"

  # TABLE ANNOTATOR
  tables:
    annotator_type: "llm"
    model_config:
      model_name: deepseek_r1_32
      prompt_path: "./tables.j2"
      system_prompt_text: >
        ### Task
        You are a helpful assistant that extracts claims from the given text.
      top_p: 1.0
      frequency_penalty: 0
      presence_penalty: 0
      expect_output: "json"

    parser_name: default # parser to parse the tables
    validators: # Custom validators for tables
        - "tables"
        - "document_structure"

  # REMARK ANNOTATOR
  remarks:
    annotator_type: "llm"
    model_config:
      model_name: deepseek_r1_32
      prompt_path: "./remarks.j2"
      system_prompt_text: >
        ### Task
        You are a helpful assistant that extracts remarks from the given text.
      top_p: 1.0
      frequency_penalty: 0
      presence_penalty: 0
      expect_output: "json"

    parser_name: default # parser to parse the remarks
    validators: # Custom validators for remarks
      - "remarks"
      - "document_structure"

  # TABLE ExTRACTOR
  table-extract:
    annotator_type: "llm_table"
    parser: noop #other parser will be used to parse the table

    model_config:
      model_name: qwen_v2_5_vl
      prompt_path: "./table-extract.j2"
      multimodal: true
      system_prompt_text: >
        ### Task
        You are a helpful assistant that extracts tables from the given text.
      top_p: 1.0
      frequency_penalty: 0
      presence_penalty: 0
      expect_output: "markdown"

# Grounding keys should be defined in the child config
```


### Field-by-field guidance

- annotators: A map of annotator keys (e.g., `claims`, `tables`). Each key is how you select an annotator at runtime.
- annotator_type:
  - `"llm"`: A general-purpose LLM annotator for key-values, claims, remarks, and similar tasks.
  - `"llm_table"`: Specialized annotator for table understanding and extraction.
- model_config:
  - model_name: The model identifier to use.
  - prompt_path: Relative path (from the layout's annotator directory) to a Jinja2 prompt template.
  - system_prompt_text: Optional system-level instructions. Recommended for stable task framing across templates.
  - multimodal: Set true when the model supports and should consume images in addition to text.
  - top_p, frequency_penalty, presence_penalty: Sampling and output-shaping parameters (tune per task as needed).
  - expect_output: Hints how the downstream parser should treat model responses (e.g., `"json"` or `"markdown"`).
  - temperature: Controls randomness of model output. Lower values (e.g., `0.0`) produce more deterministic results.
  - extra_body: Additional API parameters passed through to the model provider.
  - min_pixels, max_pixels: Pixel dimension constraints for multimodal inputs.
  - mini_batch_size: Number of pages/units to process per batch.
  - refine_passes: Number of sequential refinement passes after the initial extraction (default `0`, see [Refinement Passes](#refinement-passes) below).
  - refine_prompt_path: Optional separate prompt template for refinement passes. If omitted, `prompt_path` is reused.
  - pass_temperatures: Optional list of per-pass temperature overrides. Length must be at least `refine_passes + 1`.
  - pass_models: Optional list of per-pass model overrides. Allows using a different model for each pass (e.g., cheap model for extraction, expensive model for refinement). Length must be at least `refine_passes + 1`.
  - refinement_validation: Controls how refinement pass outputs are validated before acceptance (see below).
- parser_name / parser:
  - Name of the parser to apply to model outputs. Use `noop` when the result is consumed by a different downstream parser (common for table pipelines).
- validators:
  - A list of validation profiles to run against extracted outputs. Useful for enforcing schema, structure, and domain-specific rules.
- Grounding keys:
  - Keep layout-specific grounding in child configs under `TID-<layout_id>`. This allows per-layout and per-annotator specialization without changing global configs.

### Refinement Passes

Refinement passes let the LLM Annotator run multiple sequential extraction rounds, feeding the previous accepted output back into the prompt so the model can correct errors and fill gaps. This applies only to `annotator_type: "llm"` — `"llm_table"` is not supported.

#### How it works

1. **Pass 0** (initial extraction) runs normally with the configured prompt.
2. For each subsequent refinement pass, the previous accepted output is injected into the prompt via the `PREVIOUS_EXTRACTION` variable.
3. Each refinement pass is validated against the previous accepted pass. A pass is accepted only if it does not regress (schema-valid JSON, no excessive element loss, no catastrophic size shrink).
4. The final output is the **last known good pass**, not blindly the last attempted pass.
5. The winning pass is atomically promoted into the live output directory. Intermediate passes are written to a hidden scratch area and never leak into parser-visible output.

#### Configuration

```yaml
annotators:
  claims:
    annotator_type: "llm"
    model_config:
      model_name: deepseek_r1_32
      prompt_path: "./claims.j2"
      expect_output: "json"
      temperature: 0.0

      # Refinement passes (default: 0 = disabled)
      refine_passes: 2
      refine_prompt_path: "./claims_refine.j2"   # optional, falls back to prompt_path
      pass_temperatures: [0.0, 0.2, 0.2]         # optional per-pass temperature
      pass_models: [deepseek_r1_32, gpt-4o, gpt-4o]  # optional per-pass model

      refinement_validation:
        require_same_units: true        # reject pass if processing units change
        max_segment_drop_ratio: 0.2     # reject pass if element count drops >20%
```

#### Key fields

- `refine_passes`: Number of refinement passes after pass 0. Set to `0` (default) to disable.
- `refine_prompt_path`: Path to a separate Jinja2 template for refinement passes. Must include a `PREVIOUS_EXTRACTION` placeholder. If omitted, `prompt_path` is reused.
- `pass_temperatures`: Per-pass temperature overrides. Index 0 applies to the initial extraction, index 1 to the first refinement pass, and so on. If not set, all passes use `temperature`.
- `pass_models`: Per-pass model overrides. Allows cost optimization by using a cheaper/faster model for initial extraction and a more capable model for refinement. Engine instances are cached — repeated model names share the same engine.
- `refinement_validation`: Controls how the system decides whether a refinement pass is accepted or rejected:
  - `require_same_units` (default `true`): Reject a refinement pass if it produces a different set of processing units (pages/units) than the previous accepted pass.
  - `max_segment_drop_ratio` (default `0.2`): Reject a refinement pass if the total element count drops by more than this ratio relative to the previous accepted pass.

#### Prompt integration

Prompts used for refinement should include a `PREVIOUS_EXTRACTION` placeholder:

```jinja2
{# claims_refine.j2 #}
{{ PREVIOUS_EXTRACTION }}

Extract the claims from the following document text.
...
```

On pass 0, `PREVIOUS_EXTRACTION` is stripped from the prompt so no placeholder text leaks into the initial extraction. On refinement passes, it is populated with the previous accepted JSON output wrapped in review instructions.

#### Output directory layout

When refinement is enabled, the output directory layout changes:

```text
agent-output/{annotator_name}/              # live, parser-visible (final only)
  frame_0001.json
  frame_0001.png
  _SUCCESS.yaml                             # promotion marker

agent-output/.{annotator_name}-refine/      # scratch (hidden from parser)
  runs/{run_id}/
    pass_0/
      frame_0001.json
      frame_0001.png
    pass_1/
      frame_0001.json
      frame_0001.png
  state.yaml
```

- The live directory contains only the promoted final artifacts plus a `_SUCCESS.yaml` marker.
- Intermediate pass outputs are written to the scratch directory and are invisible to downstream parsers.
- Reruns after interruption are safe: if `_SUCCESS.yaml` is missing, the output directory is treated as incomplete and rebuilt.

#### Backward compatibility

- `refine_passes: 0` (or omitted) preserves current single-pass behavior with no changes to output layout.
- No scratch directory is created when refinement is disabled.
- Downstream parsers continue to read only top-level `.json` files from the live output directory.

### Best practices

- Co-locate prompts with the annotator config for the corresponding layout to keep deployments self-contained.
- Reuse `system_prompt_text` for stable, task-wide guardrails; vary `prompt_path` for per-layout tuning.
- Use `expect_output` to standardize downstream parsing workflows:
  - `"json"` for key-values, claims, and remarks.
  - `"markdown"` for table extractions where Markdown tables or structured text are desired.
- For multimodal table extraction, set `multimodal: true` and ensure the prompt guides the model to consume visual table cues and return normalized structure.

### Troubleshooting

- Prompt not found:
  - Verify `prompt_path` is relative to the `TID-<layout_id>/annotator/` directory and that the file exists.
- Unexpected output shape:
  - Align `expect_output` with your parser and adjust prompt templates to produce consistent JSON/Markdown.
- Mixed layouts:
  - Ensure the correct `layout_id` is used so the proper `TID-<layout_id>/annotator/config.yml` and prompts are loaded.