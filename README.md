```text
 ██████   ██████   █████████   ███████████   █████ ██████████              █████████   █████      /\   /\   
░░██████ ██████   ███░░░░░███ ░░███░░░░░███ ░░███ ░░███░░░░░█             ███░░░░░███ ░░███      //\\_//\\     ____
 ░███░█████░███  ░███    ░███  ░███    ░███  ░███  ░███  █ ░             ░███    ░███  ░███      \_     _/    /   /
 ░███░░███ ░███  ░███████████  ░██████████   ░███  ░██████    ██████████ ░███████████  ░███       / * * \    /^^^]
 ░███ ░░░  ░███  ░███░░░░░███  ░███░░░░░███  ░███  ░███░░█   ░░░░░░░░░░  ░███░░░░░███  ░███       \_\O/_/    [   ] 
 ░███      ░███  ░███    ░███  ░███    ░███  ░███  ░███ ░   █            ░███    ░███  ░███        /   \_    [   /
 █████     █████ █████   █████ █████   █████ █████ ██████████            █████   █████ █████       \     \_  /  /
░░░░░     ░░░░░ ░░░░░   ░░░░░ ░░░░░   ░░░░░ ░░░░░ ░░░░░░░░░░            ░░░░░   ░░░░░ ░░░░░        [ [ /  \/ _/
```

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Marie-AI

Marie-AI is an **Agentic Document Intelligence Platform**. It orchestrates a network of autonomous, specialized AI agents—each responsible for tasks like OCR, classification, NER, and document transformation—delivering robust, scalable, and extensible document understanding for modern enterprise needs.

---

## 🚀 Quick Start

For the fastest way to try Marie-AI, see our [Quick Start Guide](./bootstrap.md).

---

## Orchestrating Intelligence

Marie-AI leverages an agentic design to deliver:
- **Autonomous Agents:** Modular executors act as agents, each with clear responsibilities.
- **Composable Workflows:** Agents are orchestrated and dynamically routed based on your goals.
- **Extensible & Modular:** Easily add, swap, or customize agents for new workflows.
- **Goal-Driven Automation:** Instruct Marie-AI with high-level objectives and let agents handle the execution.

> **Marie-AI brings the next generation of agentic AI to document processing and intelligence!**

---

## 🚀 Features

- **Agent-based Architecture:** Modular executors/agents for OCR, classification, NER, transformation, and more.
- **End-to-End Automation:** From ingestion to extraction and delivery, with minimal human intervention.
- **Job Scheduling & SLAs:** Advanced job orchestration with support for **soft and hard Service Level Agreements**.
- **DAG-Driven Execution:** Workflows and jobs are managed as Directed Acyclic Graphs (DAGs) for dependency-aware, parallel, and fault-tolerant execution.
- **Flexible Deployment:** Use as a Python package, Docker container, or standalone service.
- **CLI Utility:** `marie` command-line tool for streamlined local or remote processing.
- **Production Hardened:** Robust release/versioning, CI/CD, observability, and telemetry.

---

## 🏥 Real-World Usage: Document Extraction from Unstructured Data

Marie-AI excels at extracting structured information from unstructured or semi-structured documents across diverse industries, including:

- **Healthcare:** Extract patient data, clinical findings, and insurance details from forms, medical records, and lab reports.
- **Legal:** Parse contracts, agreements, and case documents for entity extraction, clause identification, and compliance checks.
- **Real Estate:** Digitize and extract fields from deeds, lease agreements, mortgage forms, and appraisals.
- **Finance:** Automate extraction from invoices, receipts, KYC documents, and bank statements.
- **Government/Education:** Process applications, certificates, and official records at scale.

Whether your data lives in scans, PDFs, or complex document batches, Marie-AI’s agentic pipeline automates the end-to-end process: OCR, classification, entity extraction, table/form detection, and custom business logic.

---

## 🔗 Model Support & LLM Integration

Marie-AI works seamlessly with a variety of language models:

- **Off-the-shelf LLMs:** Instantly connect to state-of-the-art models such as GPT, T5, BERT, and more for fast setup and broad generalization.
- **Fine-tuned/Custom LLMs:** Easily integrate your own or third-party fine-tuned models to enable domain-specific extraction, reasoning, and compliance—perfect for specialized use cases in healthcare, legal, real estate, and beyond.
- **Plug-and-play architecture:** Swap models or orchestrate multiple agents powered by different LLMs within a single workflow, ensuring flexibility as your needs and the LLM landscape evolve.

With Marie-AI, your document intelligence pipeline can adapt to the latest advances in large language models—whether you need generic understanding or highly specialized, compliant extraction.

---

## ⏱️ Job Scheduling, SLAs, and DAGs

Marie-AI includes a production-grade job scheduling subsystem for managing and orchestrating document processing workflows at scale.

- **Job Scheduler:** Schedule, enqueue, and monitor jobs with support for immediate or delayed execution.
- **SLA Management:** Define **soft** and **hard** SLAs per job or workflow, ensuring deadline-driven execution and automatic escalation or retries.
- **DAG Workflows:** Jobs are executed as **Directed Acyclic Graphs (DAGs)**, allowing complex dependency management, parallel step execution, and robust error handling. Each node in the DAG can represent an agent or processing step.
- **Policies & Retry:** Fine-grained control over job policies, priority, backoff strategies, and retries.
- **Monitoring:** Real-time state tracking for all jobs and DAGs, with SQL and API interfaces for querying job status, logs, and history.

> *Example:*
> - Submit a batch extraction job as a DAG. The scheduler will coordinate all dependent steps, track their state, and enforce SLA requirements.
> - Use SQL or API to monitor progress and handle maintenance (pause, stop, restart, or purge jobs).

---

## 📚 Documentation

See [**docs.marieai.co**](https://docs.marieai.co) for full guides and advanced usage.

---

## 💾 Installation

**Stable release via PyPI:**
```sh
pip install --upgrade marieai
```

**From source:**
```sh
pip install -e .
```

**Docker:**
```sh
DOCKER_BUILDKIT=1 docker build . --build-arg PIP_TAG="standard" -f ./Dockerfiles/gpu.Dockerfile -t marieai/marie:3.0-cuda
```
Or pull an official image:
```sh
docker pull marieai/marie:latest
```

---

## 🛠️ Quick Start

Run with default entrypoint:
```sh
docker run --rm -it marieai/marie:3.0.19-cuda
```

Run server with custom entrypoint:
```sh
docker run --rm -it --entrypoint /bin/bash marieai/marie:3.0.30-cuda
marie server --start --uses sample.yml
```

---

## 🖥️ Command-Line Interface

Interact with the API from your terminal:
```sh
marie -h
```

---

## 🔥 Example Use Cases

- Autonomous document cleanup and OCR
- Agent-driven document classification and splitting
- Named Entity Recognition (NER) as an agent
- Form and table detection
- Custom agent pipeline composition

See [MarieAI docs](https://docs.marieai.co) for full code samples and advanced scenarios.

---

## 🔄 Release & Versioning

Marie is released via PyPI and Docker Hub. See [RELEASE.md](RELEASE.md) for details on versioning, release cycles, and manual release workflows.

---

## 📝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards, naming conventions, and how to get started. Please open issues or pull requests for bugs, feature requests, or documentation improvements.

---

## 📰 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a summary of recent changes and new features.

---

## 📄 License

Marie-AI is [Apache 2.0 licensed](LICENSE).

---

## 🙏 Credits

This project uses and builds upon many open source components. See [NOTICE](NOTICE) for details.

---


