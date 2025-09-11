# Configuration: directory structure, prompt resolution, and examples

This section describes how annotators are configured, where prompts live, and how the system locates and loads them at runtime.

### Config directory layout

- All prompt paths are resolved relative to the config base directory, which defaults to: `__config_dir__/extract/`
- Typical structure:

```plain text
g5/
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
#   - g5/
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
  - prompt_path: Relative path (from the layout’s annotator directory) to a Jinja2 prompt template.
  - system_prompt_text: Optional system-level instructions. Recommended for stable task framing across templates.
  - multimodal: Set true when the model supports and should consume images in addition to text.
  - top_p, frequency_penalty, presence_penalty: Sampling and output-shaping parameters (tune per task as needed).
  - expect_output: Hints how the downstream parser should treat model responses (e.g., `"json"` or `"markdown"`).
- parser_name / parser:
  - Name of the parser to apply to model outputs. Use `noop` when the result is consumed by a different downstream parser (common for table pipelines).
- validators:
  - A list of validation profiles to run against extracted outputs. Useful for enforcing schema, structure, and domain-specific rules.
- Grounding keys:
  - Keep layout-specific grounding in child configs under `TID-<layout_id>`. This allows per-layout and per-annotator specialization without changing global configs.

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