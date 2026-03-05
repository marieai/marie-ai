---
name: document-extraction
description: >
  Extract text, tables, and structured data from PDF files, images, and scanned documents.
  Use when working with document processing, OCR, or data extraction from files.
version: "1.0.0"
license: Apache-2.0
compatibility: Requires marie-ai document processing backend
allowed-tools: extract_document_ocr extract_document_data query_extraction_results
user-invokable: true
argument-hint: "[document path or extraction template]"
providers:
  - openai
  - claude
  - vllm
tags:
  - document-processing
  - ocr
  - extraction
  - pdf
metadata:
  author: marie-ai
  category: document-processing
---

# Document Extraction Skill

Extract structured data from documents using Marie-AI's document processing pipeline.

## When to Use

Use this skill when:
- User uploads or references a PDF, image, or scanned document
- User asks to extract text, tables, or specific fields from documents
- User mentions invoices, receipts, forms, contracts, or similar documents
- User wants to process multiple documents with a template
- User needs OCR (optical character recognition) on images

Do NOT use this skill when:
- User is asking general questions without document references
- User is working with structured data files (JSON, CSV, etc.)
- User needs to edit or create documents (this is extraction only)

## Instructions

1. **Identify the document**: Determine the document type from user input, file extension, or content analysis.

2. **Select extraction method**:
   - For full text extraction: Use `extract_document_ocr`
   - For structured field extraction: Use `extract_document_data` with appropriate template
   - For querying previous extractions: Use `query_extraction_results`

3. **Choose the right template** (for structured extraction):
   - `invoice`: Line items, totals, vendor info, dates
   - `receipt`: Items, amounts, merchant, date
   - `form`: Form fields and values
   - `contract`: Parties, dates, key clauses
   - `generic`: Auto-detect structure

4. **Process the document**:
   - Pass the document path/URL to the appropriate tool
   - Specify output format preferences if user requests specific format

5. **Present results**:
   - Format extracted data clearly
   - Highlight key fields the user asked about
   - Note any extraction confidence issues

## Examples

**User**: "Extract the line items from this invoice"
**Action**: Use `extract_document_data` with invoice template, focus on line_items field

**User**: "What does this document say?"
**Action**: Use `extract_document_ocr` for complete text extraction

**User**: "Process these receipts and give me a summary"
**Action**: Use `extract_document_data` with receipt template, then summarize totals

**User**: "Find the contract date and parties involved"
**Action**: Use `extract_document_data` with contract template, extract date and party fields

## Output Format

When presenting extraction results:

```
## Extracted Data

**Document Type**: Invoice
**Confidence**: 95%

### Key Fields
- **Invoice Number**: INV-2024-001
- **Date**: 2024-01-15
- **Total**: $1,234.56

### Line Items
| Item | Quantity | Price |
|------|----------|-------|
| Widget A | 10 | $50.00 |
| Widget B | 5 | $100.00 |
```

## Error Handling

- If document cannot be read, inform user and suggest alternatives (re-upload, different format)
- If extraction confidence is low, indicate uncertain fields
- If template doesn't match document type, suggest appropriate template
