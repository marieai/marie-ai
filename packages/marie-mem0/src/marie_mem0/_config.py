"""Configuration models for Mem0 memory integration.

This module provides Pydantic configuration classes for configuring
the Mem0 SDK with pgvector, LLM, and embedder settings.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field


class Mem0VectorStoreConfig(BaseModel):
    """pgvector configuration for Mem0.

    Configures the vector store backend for storing memory embeddings.
    Uses the existing PostgreSQL instance with pgvector extension.
    """

    provider: Literal["pgvector"] = "pgvector"
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "host": "marie-psql-server",
            "port": 5432,
            "user": "postgres",
            "password": "123456",
            "dbname": "mem0",
            "collection_name": "memories",
            "embedding_model_dims": 1536,
            "hnsw": True,
        }
    )


class Mem0LLMConfig(BaseModel):
    """LLM configuration for Mem0.

    Configures the LLM backend used by Mem0 for memory extraction
    and summarization. Routes through LiteLLM via OPENAI_BASE_URL.
    """

    provider: Literal["openai", "litellm"] = "openai"
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 2000,
        }
    )


class Mem0EmbedderConfig(BaseModel):
    """Embedder configuration for Mem0.

    Configures the embedding model for vectorizing memories.
    Routes through LiteLLM via OPENAI_BASE_URL.
    """

    provider: Literal["openai"] = "openai"
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "text-embedding-3-small",
        }
    )


class Mem0Config(BaseModel):
    """Configuration for Mem0 memory integration.

    Main configuration class that combines vector store, LLM, and embedder
    settings for the Mem0 SDK. Disabled by default.

    Example:
        ```python
        from marie_mem0 import Mem0Config, Mem0VectorStoreConfig

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
        ```
    """

    enabled: bool = Field(
        default=False,
        description="Whether Mem0 memory integration is enabled",
    )
    vector_store: Mem0VectorStoreConfig = Field(
        default_factory=Mem0VectorStoreConfig,
        description="pgvector configuration for memory storage",
    )
    llm: Mem0LLMConfig = Field(
        default_factory=Mem0LLMConfig,
        description="LLM configuration for memory extraction",
    )
    embedder: Mem0EmbedderConfig = Field(
        default_factory=Mem0EmbedderConfig,
        description="Embedder configuration for memory vectorization",
    )
