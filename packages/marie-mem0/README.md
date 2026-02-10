# marie-mem0

Mem0 memory integration for Marie AI agents.

## Overview

This package provides Mem0 SDK integration for persistent agent memory, allowing agents to store and retrieve memories across conversations using pgvector storage.

## Installation

```bash
pip install marie-mem0
```

Or install from the monorepo:

```bash
cd packages/marie-mem0
pip install -e .
```

## Usage

### Basic Usage

```python
from marie_mem0 import Mem0Config, Mem0Memory

# Configure mem0
config = Mem0Config(
    enabled=True,
    vector_store=Mem0VectorStoreConfig(
        config={
            "host": "localhost",
            "port": 5432,
            "dbname": "mem0",
            "user": "postgres",
            "password": "123456",
        }
    ),
)

# Create memory client
memory = Mem0Memory(config)

# Add a memory
memory.add(
    messages=[{"role": "user", "content": "My name is John"}],
    user_id="user-123",
)

# Search memories
results = memory.search(
    query="What is my name?",
    user_id="user-123",
)
```

### With Context Provider

```python
from marie_mem0 import Mem0Provider

# Create provider for agent integration
provider = Mem0Provider(
    user_id="user-123",
    agent_id="my-agent",
)

# Use with agents
async with provider:
    # Provider handles memory storage/retrieval automatically
    context = await provider.get_context(messages)
```

## Configuration

The package uses environment variables for LLM/embedding routing:

- `OPENAI_BASE_URL`: LiteLLM proxy URL (e.g., `http://localhost:4000/v1`)
- `OPENAI_API_KEY`: API key for LiteLLM

## Database Setup

Requires PostgreSQL with pgvector extension:

```sql
CREATE DATABASE mem0;
\c mem0
CREATE EXTENSION IF NOT EXISTS vector;
```

The Mem0 SDK handles table creation automatically on first use.
