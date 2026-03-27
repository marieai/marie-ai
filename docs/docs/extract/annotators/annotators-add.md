---
sidebar_position: 3
---

# How to add a new annotator

This section walks you through designing, configuring, and validating a new annotator. It applies to all annotator types (LLM, LLM Table, rule-based, embedding/hybrid) and follows the configuration and prompt layout described earlier.

### 1) Choose the annotator type
- llm: General-purpose extraction (key-values, claims, remarks, summaries).
- llm_table: Table-specific extraction with optional multimodal vision.
- regex or rule-based: Deterministic pattern matching (IDs, dates, totals).
- embedding/hybrid: Robust matching under noisy OCR or variable terminology.

Pick the simplest type that satisfies your requirements. Favor llm for flexible semantics; llm_table for structured tabular data; regex for stable, patterned fields; hybrid for noisy/variable inputs.

### 2) Create or reuse prompts (if applicable)
- Place prompt templates in the layout’s annotator directory:
  - TID-layout_id/annotator/your-prompt.j2
- Keep a stable task-wide system_prompt_text and iterate on prompt templates per annotator/layout as needed.
- For multimodal use (tables, visual extraction), ensure the prompt instructs the model to rely on both text and visual cues and to return normalized structure.

### 3) Add the annotator config entry
- Open TID-layout_id/annotator/config.yml.
- Under annotators:, create a new key for your annotator (this is how you’ll call it at runtime).
- Provide annotator_type and model_config. For example:

```yaml
annotators:
  my-new-annotator:
    annotator_type: "llm"  # or "llm_table", "regex", etc.
    model_config:
      model_name: your_model_name
      prompt_path: "./my-new-annotator.j2"   # relative to the annotator directory
      system_prompt_text: >
        ### Task
        You are a helpful assistant that extracts <your-target> from the given text.
      top_p: 1.0
      frequency_penalty: 0
      presence_penalty: 0
      expect_output: "json"  # or "markdown" for table-like output
    parser_name: default     # or a custom parser name, if applicable
    validators:
      - "domain_validator"
      - "document_structure"
```


Notes:
- prompt_path is relative to the annotator directory for the selected layout.
- expect_output should align with your parser: usually "json" for fields; "markdown" for tables.

### 4) Parsers and validators (optional but recommended)
- Parsers convert model output into the system’s canonical structures. Use parser_name: default for standard JSON responses, or specify a custom parser if your output format is specialized.
- Validators enforce schema, domain, and structural rules (e.g., document_structure, claims, tables). Add relevant validators under validators:.

Tip: Start with default parser + a minimal validator. Add stricter validators once your prompt stabilizes.

### 5) Grounding keys and layout-specific settings
- Keep grounding keys and per-layout overrides in the child layout config (under TID-layout_id).
- This enables fine-tuning the same annotator across different document families without changing global configuration.

### 6) Naming and conventions
- Annotator key: short, kebab-case (e.g., claims, key-values, table-extract, remarks).
- Output format: return data that matches expect_output and is compatible with the parser.
- Idempotency: designs should allow re-runs to skip work when outputs already exist for the same job/context.

### 7) Test and validate
- Smoke test:
  - Use a small set of documents representative of the target layout.
  - Ensure the annotator produces structured output in its output directory and that the parser accepts it.
- Validation:
  - Enable validators and confirm they pass on known-good samples.
  - Add negative cases to ensure validators catch malformed output.
- Prompt iteration:
  - Adjust the Jinja2 prompt iteratively to stabilize field names, formats, and structure.

### 8) Enable refinement passes (optional)

For LLM annotators where extraction quality benefits from iterative correction, enable refinement passes. This runs the extraction multiple times, feeding each accepted result back into the prompt:

```yaml
annotators:
  my-new-annotator:
    annotator_type: "llm"
    model_config:
      model_name: your_model_name
      prompt_path: "./my-new-annotator.j2"
      expect_output: "json"

      refine_passes: 1                          # one refinement pass after initial extraction
      refine_prompt_path: "./my-new-annotator-refine.j2"  # optional separate refinement prompt
      pass_temperatures: [0.0, 0.2]             # optional per-pass temperature
      pass_models: [fast_model, accurate_model]  # optional: cheap model first, better model for refinement

      refinement_validation:
        require_same_units: true
        max_segment_drop_ratio: 0.2
```

Key considerations:
- `refine_passes: N` means `N + 1` total LLM calls per annotator run. Start with `1` and measure quality gain before adding more.
- Use `pass_models` to optimize cost: a cheap/fast model for initial extraction and a more capable model for refinement.
- The refinement prompt should include a `PREVIOUS_EXTRACTION` placeholder for prior-result injection.
- Refinement applies only to `annotator_type: "llm"`, not `"llm_table"`.
- Only enable refinement for layouts where measured quality improves — it increases both cost and latency.

See [Configuration: Refinement Passes](annotators-config.md#refinement-passes) for full details.

### 9) Performance and cost tips
- Keep prompts concise and deterministic; anchor field names and expected schema.
- Set top_p conservatively (e.g., 1.0 or lower) for more predictable outputs.
- For large documents, consider page/region batching if supported by your workflow.
- For tables, prefer llm_table with multimodal: true when visual structure matters.

### 10) Troubleshooting
- Prompt not found:
  - Verify prompt_path is correct and relative to TID-layout_id/annotator/.
- Output not parseable:
  - Align expect_output with your parser; update the prompt to produce strict JSON or Markdown.
- Inconsistent values:
  - Tighten prompts (explicit keys, formats), use validators, and consider a post-processor to normalize values (dates, amounts, enums).
- Cross-layout confusion:
  - Ensure the correct layout_id is selected at runtime so the correct config and prompts load.

### 11) Minimal templates to copy
- New LLM annotator skeleton:

```yaml
annotators:
  <your-key>:
    annotator_type: "llm"
    model_config:
      model_name: <your-model>
      prompt_path: "./your-key.j2"
      system_prompt_text: >
        ### Task
        You are a helpful assistant that extracts <describe target>.
      top_p: 1.0
      frequency_penalty: 0
      presence_penalty: 0
      expect_output: "json"
    parser_name: default
    validators:
      - "<your-domain>"
      - "document_structure"
```

ce a new annotator with minimal friction, keep configurations layout-scoped, and ensure outputs are predictable, parseable, and easy to validate.