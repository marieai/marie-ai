# Extract Engine — Entry Point

Use this page as your single-stop overview of how the Extract engine fits together and how to build on it. It connects the concepts, the runtime lifecycle, and the configuration-driven patterns so you can confidently add new layouts, annotators, processors, and validators.

---

## Mental Model

- You declare what to extract through configuration.
- The engine builds a layout-aware Template (layers/sections) and parses the document into structured regions.
- Annotators and region processors transform content into normalized structures (KV, tables).
- Validators enforce shape and domain rules.
- A registry connects everything via stable names so components can be swapped and extended without changing the core engine.

---

## Key Concepts (Glossary)

- Layout ID: Stable identifier for a document family (e.g., payer or form type) that selects a specific Template Builder (Layout Provider).
- Layout Provider (Template Builder): Builds a Template composed of layers (bounded regions with fields/tables). Configuration-first; minimal code branching.
- Layer: A bounded section of the document driven by start/stop selectors; contains non-repeating fields and/or tables.
- Structured Region: The standardized in-memory representation of the parsed document sections (sections, tables, rows, etc.).
- SectionRole and role_hint:
  - SectionRole: Layout placement (main/context).
  - role_hint: Semantic label (e.g., service_lines, claim_information) used to route processing and field mapping.
- Annotator: Pluggable extractor that enriches or extracts data (LLM, LLM-table, rule-based, embedding/hybrid).
- Region Processor: Transforms matched lines/spans into normalized structures (e.g., a service lines table).
- Validator: Pluggable checks run at specific stages to ensure inputs and outputs meet expectations.
- Component Registry: The central registry that discovers, registers, and resolves parsers, validators, template builders, and region processors by name.

---

## How It Flows (End-to-End)

1. Select layout_id
   - The engine uses layout_id to resolve the registered Layout Provider.
2. Build the Template
   - The Layout Provider constructs layers:
     - Start/stop selectors define bounds.
     - Non-repeating field mappings.
     - Table configurations (headers, columns, row segment rules).
3. Annotate (optional/when configured)
   - Annotators run based on job request and layout config to extract fields/tables or enrich content.
4. Detect and parse regions
   - Layer bounds create regions; the engine parses content within each region.
   - Sections are created with roles and role hints. Structured Region is built.
5. Route by role_hint
   - Sections tagged with role_hint are routed to the right processor or processing rule:
     - Example: service_lines → table processing.
     - claim_information → KV parsing.
     - remarks → custom remark logic.
6. Validate
   - Validators run at configured stages (pre-processing, parse-output, post-processing) to enforce structure and domain rules.
7. Return results
   - The engine outputs structured, validated data (KVs, tables, totals, remarks) with page/line provenance.

---

## What You Configure

- Layouts:
  - Layers with anchors/selectors and field/table definitions.
  - Layer order and dependencies when sections constrain one another.
- Annotators:
  - Which annotators to run for a layout; model/prompt configs and expected outputs.
- Role-driven processing:
  - Which sections get parsed as table/KV/custom based on role_hint.
- Validators:
  - Which validators run at which stages to enforce schema and integrity.

Tip: Favor configuration-first; aim for minimal code branching per layout.

---

## Core Building Blocks

- Layout Provider (per layout ID)
  - Builds and orders layers.
  - Encodes selectors (start/stop), field mappings, and table schemas.
  - Returns a Template for runtime use.
- Annotators
  - Types: llm, llm_table, regex/rule-based, retrieval/embedding hybrid.
  - Configured per layout with prompts/rules and validators.
- Structured Region
  - Sections with titles, roles, role_hint tags.
  - Blocks like TableSeries with segments/rows.
  - Provides APIs for accessing tables, sections, and spans.
- Region Processors
  - Named processors that turn matched spans into typed structures.
  - Attach provenance and role hints for downstream consumers.
- Validators
  - Pluggable checks for structure, schema, and domain rules.
  - Selectively attached per annotator or flow stage.
- Component Registry
  - Register and resolve all components by stable names.
  - Supports core and external modules; designed for extension.

---

## Role Hints: The Routing Bridge

- role_hint tags sections with business semantics (service_lines, claim_information, claim_totals, remarks).
- The engine or configured processing rules use role_hint to:
  - Choose parse method (table, kv, custom).
  - Apply field mappings scoped to that role.
  - Trigger specialized business logic or validation.

Best practices:
- Use descriptive, stable snake_case names.
- Keep naming consistent across layouts to maximize reuse.

---


## Minimal Developer Walkthrough

- Add or update a Layout Provider for a new document family:
  - Choose a new layout_id if documents change meaningfully to avoid brittle branching.
  - Define layers with clear start/stop anchors.
  - Map non-repeating fields and specify table definitions (headers, columns).
- Use role_hint to connect configuration to processing:
  - Tag sections like claim_information, service_lines, claim_totals, remarks.
  - Configure processing rules or processors that respond to those role hints.
- Add annotators (if needed):
  - Pick type: llm for flexible fields; llm_table for structured tables; regex for deterministic patterns; hybrid for robust matching.
  - Configure prompts/rules and expected output format; attach validators.
- Add validators:
  - Attach generic structure validators and domain-specific ones (e.g., tables, claims).
  - Fail fast on missing required fields or malformed structures.
- Test:
  - Validate anchoring across varied samples.
  - Check field/table extraction stability and performance.
  - Ensure validators produce actionable diagnostics.

---