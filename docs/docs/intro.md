---
sidebar_position: 1
---

# Introduction to Marie-AI

Marie-AI is an open-source document processing framework designed for extracting structured data from unstructured documents. Built on a microservice architecture, it provides scalable solutions for OCR, document classification, and intelligent data extraction.

## What is Marie-AI

Marie-AI helps you build document processing pipelines that can:

- **Extract text** from images and PDFs using OCR
- **Classify documents** by type and structure
- **Extract structured data** using rules, patterns, or AI models
- **Process at scale** with distributed microservices

## Key features

| Feature | Description |
| ------- | ----------- |
| **Microservice architecture** | Build scalable pipelines with independent, reusable components |
| **Multiple protocols** | Serve via gRPC, HTTP, or WebSocket |
| **GPU acceleration** | Optimized for GPU-based inference with health monitoring |
| **Cloud-native** | Deploy to Docker, Kubernetes, or cloud platforms |
| **Extensible** | Create custom Executors for your specific use cases |

## Core concepts

Marie-AI is built around these primary components:

- **[Executors](./guides/executor.md)**: Microservices that process documents
- **[Flows](./guides/flow.md)**: Orchestrate Executors into processing pipelines
- **Gateway**: Entry point that routes requests to Executors
- **Client**: Send requests to your document processing pipeline

## Quick example

Here's a minimal example of a Marie-AI document processing pipeline:

```python
from marie import Flow, Executor, requests
from docarray import DocList, BaseDoc

class TextExtractor(Executor):
    @requests(on='/extract')
    def extract(self, docs: DocList[BaseDoc], **kwargs) -> DocList[BaseDoc]:
        for doc in docs:
            # Extract text from document
            pass
        return docs

# Create and run the pipeline
f = Flow(port=54321).add(uses=TextExtractor)

with f:
    f.block()  # Serve until interrupted
```

## Installation

Install Marie-AI using pip:

```bash
pip install marie-ai
```

Or run with Docker:

```bash
docker run marieai/marie:latest
```

See the [Installation guide](./getting-started/installation.mdx) for detailed setup instructions.

## Use cases

Marie-AI is designed for document processing workflows such as:

- **Invoice processing**: Extract line items, totals, and vendor information
- **Form extraction**: Pull data from structured forms
- **Document classification**: Categorize documents by type
- **Receipt processing**: Extract merchant, date, and transaction details
- **Contract analysis**: Identify key terms and clauses

## Architecture overview

```text
┌─────────┐     ┌─────────┐     ┌─────────────────────────────┐
│         │     │         │     │            Flow             │
│ Client  │────▶│ Gateway │────▶│  ┌─────┐   ┌─────┐         │
│         │     │         │     │  │ OCR │──▶│Extract│        │
└─────────┘     └─────────┘     │  └─────┘   └─────┘         │
                                └─────────────────────────────┘
```

Learn more in the [Architecture overview](./guides/architecture-overview.md).

## Next steps

1. **[Installation](./getting-started/installation.mdx)**: Set up Marie-AI on your machine
2. **[Executors](./guides/executor.md)**: Learn to build document processing components
3. **[Flows](./guides/flow.md)**: Orchestrate components into pipelines
4. **[Deployment](./getting-started/deployment/index.md)**: Deploy to production

## Getting help

- **Documentation**: Browse these docs for guides and references
- **GitHub Issues**: Report bugs or request features at [marieai/marie-ai](https://github.com/marieai/marie-ai)
