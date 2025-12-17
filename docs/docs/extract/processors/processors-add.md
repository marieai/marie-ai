---
sidebar_position: 2
---

# How to Create a New Region Processor (Hello World)

This page walks you through creating a minimal “Hello World” region processor that plugs into the extraction pipeline, registers itself, and produces a simple structured output.

## Overview

A region processor:
- Receives a matched section of a document (with spans/lines).
- Transforms lines into a standardized structure (e.g., a table).
- Returns a StructuredRegion with sections/blocks/rows.
- Is registered so the pipeline can discover and run it by name.

You’ll create a new, minimal processor that always emits a single table with one row: “Hello, world!”.

## Prerequisites

- Python 3.12 environment for the project.
- Basic understanding of how regions are matched and passed to processors.
- Access to add a new module under your processors package.

## 1) Create a Processor Module

Create a new Python module (for example, under your processors/regions package) and implement a function decorated with the region-processor registry decorator. The function should accept the standard processing arguments and return a StructuredRegion.

Example “hello_world_processor.py”:

```python
from typing import Dict, List

# Import the registry and core types from your extraction framework
from marie.extract.registry import register_region_processor
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import MatchSection, Span
from marie.extract.models.page_span import PageSpan
from marie.extract.results.span_util import create_aggregate_span
from marie.extract.results.role_util import normalize_role

from marie.extract.structures.cell_with_meta import CellWithMeta
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.extract.structures.structured_region import (
    RowRole,
    Section,
    StructuredRegion,
    TableBlock,
    TableRow,
    TableSeries,
    build_structured_region,
)
from marie.extract.structures.table import Table
from marie.extract.structures.table_metadata import TableMetadata


PROCESSOR_SOURCE = "hello_world"
TABLE_HEADERS = ["MESSAGE"]


@register_region_processor(PROCESSOR_SOURCE)
def process_hello_world(
    context: ExecutionContext,
    parent: MatchSection,
    section: MatchSection,
    region_parser_config: Dict,
    regions_config: List[Dict],
) -> StructuredRegion:
    # Aggregate span for the whole matched section
    spans: List[Span] = section.span
    aggregate_span: PageSpan = create_aggregate_span(spans)

    # Resolve title/role from config, with safe defaults
    region_cfg = regions_config[0] if regions_config else {}
    title = region_cfg.get("title", "HELLO WORLD")
    role_value = region_cfg.get("role", "main")
    role, role_hint = normalize_role(role_value)

    # Build a simple 1-column header
    header_cells = [CellWithMeta(lines=[LineWithMeta(line=h)]) for h in TABLE_HEADERS]
    header_row = TableRow(role=RowRole.HEADER, cells=header_cells)
    table_grid: List[List[CellWithMeta]] = [header_cells]

    # Add one data row with "Hello, world!"
    data_cells = [CellWithMeta(lines=[LineWithMeta(line="Hello, world!")])]
    data_row = TableRow(role=RowRole.BODY, cells=data_cells)
    table_grid.append(data_cells)

    # Table metadata (pick the first available line from the section if available)
    # If a section line is not guaranteed, you can set page_id to 0 and omit source line.
    page_id = 0
    if spans and spans[0].page_from is not None:
        page_id = int(spans[0].page_from)
    table_meta = TableMetadata(page_id=page_id, title=title)

    table = Table(cells=table_grid, metadata=table_meta)

    # Wrap in a single segment series
    segment = TableBlock(span=aggregate_span, table=table, rows=[header_row, data_row], segment_role="single")
    series = TableSeries(segments=[segment], span=aggregate_span, unified_header=TABLE_HEADERS)

    # Create a section and region
    hello_section = Section(title=title, role=role, blocks=[series], span=aggregate_span)
    if role_hint:
        hello_section.tags["role_hint"] = role_hint

    region = build_structured_region(
        region_id="hello_world_region",
        region_span=aggregate_span,
        sections=[hello_section],
    )

    # Tag for provenance
    region.tags["parsed"] = PROCESSOR_SOURCE
    region.tags["processor_generated"] = "true"

    return region
```


Notes:
- The decorator registers the processor by name (“hello_world”).
- The function constructs a minimal Table with a single header and one data row.
- Provenance tags help trace outputs back to this processor.

## 2) Make the Module Importable

Ensure your package’s __init__.py imports or otherwise exposes the new module so the registration decorator executes at import time. For example:

```python
from .hello_world_processor import process_hello_world  # noqa: F401
```

## 3) Reference It in Configuration

Wire your new processor into the pipeline by referencing its registered name in your region configuration. The exact structure may vary by your setup, but it typically includes the processor name and optional metadata like title/role.

Example configuration snippet:

```yaml
# yaml
regions:
  - name: hello_world_demo
    processor: hello_world
    title: "HELLO WORLD"
    role: "main"
    match:
      # Your matching strategy for when/how to invoke the processor
      # For a demo, match a simple page or section span
      type: "entire_document"  # or a specific rule in your system
```


Key field:
- processor: must match the name used in @register_region_processor.

## 4) Run and Verify

- Execute your pipeline with the configuration that includes the “hello_world” region.
- Inspect the resulting StructuredRegion:
  - A section titled “HELLO WORLD”
  - One table with header “MESSAGE”
  - One row containing “Hello, world!”
- Confirm that provenance tags indicate the processor ran.

## 5) Testing Tips

- Add a unit test that calls your processor with a synthetic section (one trivial span).
- Assert that:
  - The region contains exactly one section and one series/segment.
  - The header equals ["MESSAGE"].
  - The first data cell contains “Hello, world!”.
  - The region tags include “parsed: hello_world”.

Example test skeleton:

```python
def test_hello_world_processor_minimal():
    # Arrange: build a minimal ExecutionContext and MatchSection with one span
    # Act: call process_hello_world(...)
    # Assert: verify StructuredRegion shape and "Hello, world!" content
    ...
```


## 6) Extending Beyond “Hello World”

- Add more columns to TABLE_HEADERS for richer outputs.
- Populate rows from annotations/lines instead of a constant.
- Attach page/line provenance to each TableRow for traceability.
- Use configuration for titles, roles, or formatting.

## Summary

- Register a function with a unique processor name.
- Build and return a StructuredRegion with at least one table (header + row).
- Expose the module so registration happens at import time.
- Reference it in your configuration and run the pipeline to see your “Hello, world!” table in the output.

