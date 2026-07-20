"""Configuration-driven embedding construction."""

from __future__ import annotations

from typing import Any, Dict

from marie.embeddings.base import EmbeddingsBase
from marie.embeddings.gemini import GeminiEmbeddings
from marie.embeddings.jina import JinaEmbeddings
from marie.embeddings.qwen import QwenVLEmbeddings


def _normalize_type(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "")


def setup_embeddings(config: Dict[str, Any]) -> EmbeddingsBase:
    """Create an embedding implementation from a pipeline-style config."""
    model_name = config.get("model_name_or_path") or config.get("model_name")
    if not model_name:
        raise ValueError("Embedding config requires model_name_or_path")

    configured_type = config.get("type")
    if configured_type:
        embedding_type = _normalize_type(str(configured_type))
    else:
        normalized_model = str(model_name).lower()
        if "qwen" in normalized_model:
            embedding_type = "qwen"
        elif "jina" in normalized_model:
            embedding_type = "jina"
        elif "gemini" in normalized_model:
            embedding_type = "gemini"
        else:
            raise ValueError(
                "Embedding config requires type for an unrecognized model. "
                "Supported types: qwen, jina, gemini"
            )

    dimension = int(config.get("dimension", config.get("embedding_dim", 1024)))
    batch_size = int(config.get("batch_size", 4))
    use_gpu = bool(config.get("use_gpu", True))

    if embedding_type in {"qwen", "qwenvl", "qwenvlembeddings"}:
        return QwenVLEmbeddings(
            model_name_or_path=model_name,
            task=str(config.get("task", "retrieval")),
            truncate_dim=dimension,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )

    if embedding_type in {"jina", "jinaembeddings"}:
        return JinaEmbeddings(
            model_name_or_path=model_name,
            task=str(config.get("task", "retrieval")),
            truncate_dim=dimension,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )

    if embedding_type in {"gemini", "geminiembeddings"}:
        gemini_model = str(model_name)
        if gemini_model.startswith("google/"):
            gemini_model = gemini_model.removeprefix("google/")
        return GeminiEmbeddings(
            model_name=gemini_model,
            output_dimensionality=dimension,
            task_type=config.get("task_type", "RETRIEVAL_DOCUMENT"),
            batch_size=batch_size,
            api_key=config.get("api_key"),
        )

    raise ValueError(
        f"Invalid embedding type: {configured_type!r}. "
        "Supported types: qwen, jina, gemini"
    )
