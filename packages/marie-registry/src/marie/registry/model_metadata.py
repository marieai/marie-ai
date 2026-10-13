from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from marie.registry.model_config import ModelMetadata


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Model metadata must be an object: {path}")
    return value


def read_model_metadata(
    resolved_path: str | Path,
    revision: str | None = None,
    *,
    metadata_files: dict[str, Path] | None = None,
) -> ModelMetadata:
    """Parse metadata files already resolved by a registry handler."""
    path = Path(resolved_path)
    if path.is_dir():
        config_path = path / "config.json"
        if not config_path.is_file():
            config_path = path / "marie.json"
    else:
        config_path = path
    config = _read_json(config_path)
    sections: dict[str, dict[str, Any]] = {
        "processor": {},
        "tokenizer": {},
        "generation": {},
    }
    for filename, section in (
        ("preprocessor_config.json", "processor"),
        ("processor_config.json", "processor"),
        ("tokenizer_config.json", "tokenizer"),
        ("generation_config.json", "generation"),
    ):
        auxiliary = (
            metadata_files.get(filename)
            if metadata_files is not None
            else config_path.parent / filename
        )
        if auxiliary is not None and auxiliary.is_file():
            sections[section].update(_read_json(auxiliary))
    return ModelMetadata(config=config, revision=revision, **sections)


def infer_multimodal(metadata: dict[str, Any]) -> bool | None:
    """Infer image support from metadata or upstream Transformers task mappings."""
    if isinstance(metadata.get("vision_config"), dict) or isinstance(
        metadata.get("img_processor"), dict
    ):
        return True
    model_type = metadata.get("model_type")
    if not isinstance(model_type, str):
        return None
    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
        )
    except ImportError as exc:
        raise ImportError(
            "Install marie-registry[huggingface] to infer capability from model_type, "
            "or configure an explicit multimodal override"
        ) from exc
    if model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
        return True
    if model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return False
    return None
