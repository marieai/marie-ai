---
sidebar_position: 1
---

# Query Plan

The query plan is a tree structure that represents the steps that the query engine will take to execute a query. The
query plan is generated by the query optimizer, which is responsible for finding the most efficient way to extract the
data requested by the query.

## Query Plan Types

There are two main types of query plans: `Trivial` and `Non-Trivial`.
`Trivial` query plans represent the high-level steps that the extract engine will take to execute,
while `Non-Trivial` plans represent the low-level steps that the extract engine will take to execute.

## Query Plan Structure

The query plan is a tree structure that consists of nodes and edges. Each node represents a step in the query execution
process.

## Query Plan Generation

Plans can be generated via `Serial Plan Generation` or `Parallel Plan Generation` strategies.
`Trivial` plans are generated via `Serial Plan Generation` strategy, while `Non-Trivial` plans are generated via
`Parallel Plan Generation` strategy.

## Query Plan Execution

The query plan is executed by the query engine, which is responsible for executing the steps in the plan in the correct
order.

## Node Types

`Trivial` and `Non-Trivial` query plans consist of different types of nodes. Most of the trivial nodes can be executed
via a single pipeline step.

- Chunker
    - Splits the input data into chunks. (Pages,Frames)
- Annotator
    - Adds metadata(ROI, Embeddings) or labels to text that can be used in downstream tasks like matching, chunking, or
      extracting.
- Matcher
    - Splits the input data into chunks. (MatchSections, PageSpan)
- Extractor
    - Extracts the data from the input chunk. (Rows, Table, KeyValue, List, Section ...)
- Evaluator
    - Evaluates the extracted data. (Filter, Sort, Transform, ...)
- Aggregator
    - Gathers and combines data or content from multiple sources into a single, unified view or dataset.
- Collator
    - Organizes and arranges data or information in a systematic way, often focusing on sorting, aligning, or
      structuring items.
- Validator
    - Validates the extracted data. (Schema, Type, ...)
- Materializer
    - Materializes the extracted data into a final output. (S3, File, ...)

## Query Plan Engine

The query plan engine is responsible for executing the query plan. The engine is responsible for executing the steps in
the plan in the correct order and ensuring that the data is extracted correctly.