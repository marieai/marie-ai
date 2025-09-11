# Layout Providers (Template Builders) — Developer Guide

This guide explains how to design, register, and maintain Layout Providers (aka Template Builders). It focuses on the conceptual model, configuration patterns, and lifecycle so you can add new layouts confidently and consistently.

## What is a Layout Provider?

A Layout Provider builds a Template that tells the extraction engine how to find and map fields in a specific document layout. It encapsulates:

- How to locate sections (layers) of a document.
- Which anchors or selectors define boundaries for each layer.
- How to map non-repeating fields and configure repeating structures (tables).
- How the layout is registered and discovered at runtime.

In practice, a Template is composed of one or more layers, each layer targeting a coherent part of the document (for example, header, claim details, remarks, tables).

## Core Responsibilities

- Validate the incoming configuration (e.g., the layout ID and layer keys exist).
- Build a Template object for a single layout ID.
- Create and configure one or more layers:
  - Define start/stop anchors or selector sets for each layer.
  - Map non-repeating fields to selectors.
  - Provide repeating-field/table configuration.
- Add layers to the Template in the correct order (when layers are interdependent).
- Return a fully formed Template to the engine.

## Concepts and Vocabulary

- Layout ID: A unique identifier for a layout that determines which Template Builder to use.
- Template: The assembled structure passed to the extraction engine.
- Layer: A “section” of the document governed by start/stop anchors (or selectors) and containing field mappings and/or tables.
- Anchors/Selectors: Patterns or rules that identify where a layer starts or stops and how fields are captured.
- Non-repeating fields: Single-value fields within a layer (e.g., Claim Number).
- Repeating fields / Tables: Structured, repeating data captured as one or more rows.

## Lifecycle

1. The engine identifies layout_id in the runtime config and locates the matching Layout Provider (registration).
2. The provider validates config and constructs a Template.
3. The provider defines layers with start/stop conditions (anchors/selectors), field mappings, and table configurations.
4. The provider returns the Template to the engine for parsing and extraction.

## Registration and Discovery

- Each Layout Provider is associated with a single layout ID.
- Providers are registered with the extraction framework under their layout ID.
- At runtime, the engine loads the provider for the current layout_id and calls the builder to obtain the Template.

Tip: Keep one provider per layout ID and keep the ID stable across releases. If a breaking change to the layout occurs, prefer introducing a new layout ID.

## Configuration-First Design

Layout Providers are driven by configuration to minimize branching logic in code. Typical config themes:

- selectors.start and selectors.stop per layer
- fields.non_repeating and fields.repeating definitions
- layer-specific mapping and optional overrides
- table schemas and header/column matchers

This separation allows you to evolve extraction patterns mostly through configuration.

## Layers and Anchors

- Start/stop anchors (or selector sets) define the scope for each layer.
- A layer processes content only within its boundaries.
- When layers have dependencies, add them to the Template in the dependency order. For example, a “remarks” layer might constrain or influence adjacent regions.

Examples of common layer patterns:

- Main/Claims layer: general header and claim-level information.
- Remarks layer: a bounded region based on remark headers/footers.
- Table layer(s): one or more repeating structures positioned by section anchors.

## Field Mapping

- Non-repeating fields:
  - Map each field to a selector set that identifies the value in the layer’s region.
  - Use the field key as the default name when not explicitly declared in field definitions.
  - Keep required/primary flags consistent with downstream validation rules.
- Repeating fields/Tables:
  - Provide a raw table configuration (column specs, header patterns, line-item detection).
  - Prefer config-driven approaches so table extraction logic doesn’t need code changes for each layout.

## Ordering and Dependencies

- The order you add layers to the Template can be significant when layers constrain or inform each other.
- Guideline:
  - Add ancillary or bounding layers first (e.g., remarks/notes, sectional bounds).
  - Add primary content layers (e.g., main/claims) next.
  - Add table layers last unless a preceding layer depends on table boundaries.

## Versioning and Compatibility

- Template version: Bump the version when you introduce non-trivial changes.
- Backwards compatibility: If the layout on documents changes meaningfully, create a new layout ID rather than forcing complex conditionals.
- Config evolution: Keep config changes backward compatible where feasible; document breaking changes clearly.

## Testing Checklist

- Configuration validation:
  - Missing layout_id is rejected with a clear error.
  - Unknown layout_id is rejected with a clear error.
  - Mandatory config sections exist (layers, selectors, fields).
- Layer anchoring:
  - Start/stop anchors correctly bound each layer for sample docs.
  - Overlapping or unbounded layers are handled or flagged.
- Field mapping:
  - Non-repeating fields found reliably across representative samples.
  - Required fields trigger validation failures when missing.
- Tables:
  - Column identification works with header variations.
  - Row segmentation handles multi-line cells and noise.
- End-to-end:
  - Extracted output matches expected field maps for multiple real samples.
  - Performance and stability under typical and stress inputs.

## Troubleshooting

- A layer isn’t detected:
  - Check that start/stop anchors are present, spelled correctly, and not overly strict.
  - Verify the layer order if dependencies exist.
- A field is empty or inconsistent:
  - Inspect the selector set specificity; broaden or refine as needed.
  - Confirm the field region truly lies within the layer bounds.
- Table columns are misaligned:
  - Revisit header detection rules and permissible variations.
  - Consider adding fallback column locators or widening tolerance.
- Config loads but extraction fails:
  - Ensure required fields or tables are correctly flagged or made optional.
  - Add explicit errors in config validation to fail fast.

## Recommended Patterns

- Prefer config-driven selector sets over code branching.
- Keep anchor terms concise and semantically strong.
- Group related layers and keep naming consistent across layouts.
- Document non-obvious assumptions in the config (e.g., “remarks section ends at ‘REMARK_END’ marker”).
- Treat tables as first-class: define robust header/column matching and row segmentation rules in config.

## Anti-Patterns

- Hardcoding field logic in the builder when a selector could express it.
- Overlapping layers that depend on implicit ordering rather than explicit boundaries.
- Silent fallbacks: Emit explicit errors when required elements aren’t found.
- Monolithic “catch-all” layers with ambiguous anchors.

## Example: Minimal YAML Skeleton

```yaml
layout_id: TID-XXXXX

selectors:
  # Defaults or layer-agnostic selectors (optional)
  # Usually layer selectors live inside each layer config.
  start: []
  stop: []

layers:
  layer-main:
    selectors:
      start: ["CLAIM NUMBER", "PATIENT ACCOUNT", "EMPLOYEE", "PROVIDER TAX ID"]
      stop: []  # Optional: could be implicit end of page/section
    non_repeating_fields:
      claim_number:
        # Field-specific selectors live here; structure depends on your selector schema
        annotation_selectors:
          - ["CLAIM NUMBER", ":right-of", ":same-line"]
      patient_account:
        annotation_selectors:
          - ["PATIENT ACCOUNT", ":right-of", ":same-line"]
    tables:
      - name: line_items
        header_selectors:
          - ["DOS", "CPT", "CHARGES", "PAID"]  # Example headers
        columns:
          - { key: service_date, patterns: ["DOS", "DATE"] }
          - { key: cpt_code, patterns: ["CPT", "PROC"] }
          - { key: charge_amount, patterns: ["CHARGES", "BILLED"] }
          - { key: paid_amount, patterns: ["PAID", "ALLOWED"] }

  layer-remarks:
    selectors:
      start: ["REMARK_START"]
      stop: ["REMARK_END"]
    non_repeating_fields:
      remarks_text:
        annotation_selectors:
          - ["*"]  # Everything inside the bounded region, per your selector semantics

fields:
  non_repeating:
    claim_number:
      name: Claim Number
      required: true
    patient_account:
      name: Patient Account
      required: false
    remarks_text:
      name: Remarks
      required: false

  repeating:
    line_items:
      # Table-level metadata and normalizers go here
      required: false
```


Notes:
- The exact schema for selectors and field definitions should match your project’s selector/annotation conventions.
- Keep layer names stable; they are used across configs, code integration, and tests.

## Quality Bar and Acceptance Criteria

A new Layout Provider is considered complete when:
- It is registered under a unique, stable layout ID.
- It validates configuration properly and fails fast on misconfiguration.
- It builds a Template with correctly ordered layers.
- It reliably extracts required fields and tables across representative samples.
- It includes tests for config validation, anchoring, field mapping, and tables.
- It is documented:
  - The intent of each layer.
  - Key anchors and non-obvious rules.
  - Known limitations and edge cases.

## Maintenance Tips

- Keep a changelog of layout-specific adjustments tied to sample documents.
- Capture heuristics in config comments; avoid institutional knowledge siloing.
- When anchor text changes in new document batches, evolve selectors conservatively and add tests using the new samples.
- If a change risks breaking existing datasets, introduce a new layout ID.

---
