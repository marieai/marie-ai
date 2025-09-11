# Purpose and How They Fit Into the System

This document explains what “annotators” are in the extraction pipeline, why they exist, and how to choose and use them effectively.

## What is an Annotator?

An annotator is a pluggable component that enriches a document with machine-generated labels, structure, or extracted values. Think of an annotator as a focused “skill” that:
- Reads an unstructured or semi-structured document representation (text, pages, OCR, regions).
- Applies a strategy (rules, embeddings, LLMs, etc.) to identify or extract information.
- Emits structured results and artifacts that downstream steps can rely on.

Annotators expose a common interface and declare their capabilities, making them easy to orchestrate, swap, and compose.

## Why Annotators Exist

Annotators exist to:
- Standardize extraction logic across document types and models.
- Separate concerns between “how to find data” and “how to run workflows.”
- Enable hybrid strategies (rule-based + ML/LLM + retrieval) without changing the rest of the system.
- Improve reproducibility and auditability by writing outputs to predictable locations and keeping a consistent API.

## Core Concepts

- Capability declaration:
  - Annotators declare what they can do (e.g., EXTRACTOR, SEGMENTER). Orchestration can route documents to the right tool based on these capabilities.

- Configuration-driven:
  - Annotators are configured per layout/use-case. Configs define annotator identity, model options, prompts or rules, and expected outputs.

- Document contract:
  - Annotators operate on an internal “unstructured document” representation (text, OCR, page frames, regions, metadata).
  - They should validate inputs and fail early when preconditions are not met.

- Deterministic side effects:
  - Each annotator writes its artifacts to a dedicated output directory for the request/context. This allows idempotent runs, caching, and debugging.

- Synchronous and asynchronous operation:
  - Annotators may support both sync and async execution, which helps in scaling and distributed processing.

## The Main Annotator Types

- Rule-based (Regex Annotator)
  - Purpose: Fast, deterministic extraction when field patterns are stable (IDs, dates, totals).
  - Strengths: Simple, explainable, low cost, high precision when patterns are well-defined.
  - Use when: You have reliable string patterns and want speed and predictability.

- LLM Annotator
  - Purpose: Use a language model to interpret content and extract fields, labels, or regions—optionally multimodal (image + text).
  - Strengths: Robust to variation, can infer semantics, good for complex or loosely structured documents.
  - Use when: Layouts vary, semantics matter, or you need flexible reasoning beyond strict patterns.

- LLM Table Annotator
  - Purpose: Specialization focused on detecting, understanding, and extracting tabular content via LLM prompts.
  - Strengths: Handles fragmented tables, headers, and row/column semantics.
  - Use when: You need to reliably extract tabular data across varied document layouts.

- Embedding/Retrieval Hybrid Annotator
  - Purpose: Combine semantic embeddings, fuzzy matching, and memory of known labels to extract fields with high recall and robustness.
  - Strengths: Handles noisy OCR, misspellings, and partial matches; leverages retrieval to rank candidates.
  - Use when: Terminology and formatting vary widely or OCR is noisy, and you want resilient matching beyond rules.

## Where Annotators Fit in the Pipeline

- Orchestration (executors) invokes annotators based on operation parameters and layout configs.
- Documents are prepared (pages/frames, OCR, metadata) and passed to the chosen annotator.
- Annotators write their outputs (e.g., JSON results, intermediate artifacts) into a known output directory for the current job/context.
- Downstream components consume these structured results for final parsing, validation, and API responses.

## Inputs and Outputs (Conceptual)

- Inputs:
  - A document abstraction with text/OCR, page frames, and metadata.
  - Annotator configuration (name, model/rules, prompt, expected schema).
  - Optional runtime parameters (job identifiers, layout keys, etc.).

- Outputs:
  - Structured extraction results (e.g., field/value pairs, confidence).
  - Optional artifacts (image snippets, logs, intermediate files) useful for traceability and debugging.
  - Parsers can further normalize these outputs into the system’s canonical schema.

## Choosing the Right Annotator

- Choose Rule-based when:
  - Fields are well-known and consistently formatted.
  - You need predictable, transparent behavior and speed.

- Choose LLM when:
  - Documents vary in structure/wording.
  - You need semantic understanding or reasoning.
  - You’re okay with model latency and cost trade-offs.

- Choose LLM Table when:
  - Tabular structures are central to the task.
  - You want improved accuracy on table-specific semantics.

- Choose Embedding/Hybrid when:
  - Data is noisy or inconsistent.
  - You need a balance of recall and precision with fuzzy/semantic matching.

Often, combining annotators yields the best results: use fast regex for easy wins and fall back to LLM/Hybrid for hard cases.

## Extending the System with a New Annotator

- Implement the common annotator contract:
  - Declare capabilities.
  - Validate document input.
  - Implement annotate/aannotate and output parsing.
- Keep configuration external:
  - Don’t hardcode model names or prompts—load from config.
- Write reproducible outputs:
  - Use a dedicated output path per annotator and request context.
- Make it composable:
  - Favor structured outputs that downstream parsers can consume.

## Best Practices

- Idempotency:
  - Skip work if outputs already exist for the current context; allows retries without reprocessing everything.

- Observability:
  - Log configuration and important runtime parameters (model, prompt, output path).
  - Persist intermediate artifacts where helpful.

- Guardrails:
  - Validate inputs early and clearly.
  - Apply timeouts and error handling to avoid blocking pipelines.

- Performance:
  - Use the simplest annotator that achieves the goal.
  - Partition work across pages/regions to exploit parallelism.
  - Cache models/embeddings where possible.

- Quality:
  - For LLMs, control prompts, temperature/top-p, and expected schemas.
  - For hybrid approaches, tune thresholds and weights and maintain label memory.

## Summary

Annotators are the building blocks of the extraction system. They provide a consistent, configurable way to enrich documents with structured information—whether by simple rules, powerful language models, or hybrid semantic matching. By standardizing capabilities, inputs/outputs, and execution patterns, annotators make the pipeline flexible, scalable, and maintainable while enabling high-quality document understanding.