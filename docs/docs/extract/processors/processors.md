---
sidebar_position: 1
---

# Processor Purpose and Role in the Extraction Pipeline

This document explains what “processors” are in the extraction pipeline, why they exist, and how they transform intermediate document representations into structured outputs.

## What is a Processor?

A processor is a focused component that:
- Receives a matched region of a document (a “section”) determined by upstream detection/annotation.
- Interprets the content within that region (e.g., lines and their metadata).
- Normalizes and structures that content into a consistent, typed format (e.g., tables with headers and rows).
- Produces a StructuredRegion object that downstream consumers can reliably use (parsers, exporters, validation/reporting, etc.).

In short, processors turn loosely structured, annotation-guided text into machine-friendly structures.

## Why Processors Exist

- Normalize heterogeneous document layouts into consistent data shapes.
- Encapsulate domain-specific logic for a specific region type (e.g., “remarks”, “tables”, “key-values”).
- Provide a clean boundary between raw OCR/annotations and structured, analyzable data.
- Support modular growth: new region types can be added without changing the core pipeline.

## Where Processors Fit in the Pipeline

1. Document is ingested and annotated (e.g., via layers that identify headers, data lines, or boundaries).
2. A region matcher identifies spans (page/line ranges) that correspond to a logical section.
3. A region processor is selected (by name/config) to handle that section type.
4. The processor:
   - Filters relevant lines within the spans.
   - Parses/normalizes content (e.g., split into fields, fix numbering).
   - Builds a structured output (tables/series/sections).
5. The system aggregates all StructuredRegion outputs for the final result.

## Outputs from a Processor

- StructuredRegion:
  - One or more sections, each with:
    - Title and role (e.g., “main”, “auxiliary”).
    - One or more table series (segments).
    - Tables with headers and rows.
  - Provenance tags (e.g., source processor, source layer).
  - Accurate page/line mapping for traceability.

This standardized structure makes it easy to:
- Render as tables in reports or UIs.
- Feed downstream parsers or validators.
- Compare and audit results consistently.

## Typical Processing Steps

- Collect lines within the matched spans.
- Filter only the lines relevant to the region’s semantics (based on annotations).
- Parse those lines into domain objects (e.g., rows with fields).
- Normalize/fix inconsistencies (e.g., sequence numbering, spacing).
- Build table metadata (title, page origin) and populate rows.
- Attach tags for provenance and role hints.
- Return a fully-formed StructuredRegion.

## Example Use Case (Conceptual)

For a “Remarks” region:
- Identify lines annotated as remark data within a bounded section.
- Parse each line into fields like sequence, remark code, and description.
- Normalize whitespace and backfill missing sequence numbers in order.
- Return a section titled “REMARKS” with a table having headers and rows.

This yields a predictable table that downstream steps can rely on without knowing document formatting specifics.

## Configuration and Roles

- Region processors can be configured with:
  - Title (used in output sections).
  - Role (e.g., “main”, “aux”); a role hint can be attached for consumers to adjust behavior or prioritization.
- These knobs let you tailor outputs while reusing the same logic across different providers or document types.

## When to Create a New Processor

Create a new processor when:
- You have a region with unique semantics (not covered by existing processors).
- The input lines/annotations require specialized parsing and normalization.
- You need a specific structured shape (headers, columns, role) for downstream tasks.
- Reusing or extending an existing processor would complicate logic or degrade clarity.

## Key Benefits

- Separation of concerns: keeps parsing logic isolated from detection/matching and high-level orchestration.
- Consistency: standardizes outputs as tables/sections with clear metadata.
- Extensibility: easy to register and configure new processors for new region types.
- Auditability: strong provenance and page/line traceability.

## Summary

Processors are the bridge between annotated text and structured data. They:
- Focus on a single region type’s semantics.
- Convert raw, annotated lines into structured tables and sections.
- Provide consistent, traceable outputs to power downstream validation, parsing, and analytics.