from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    config: dict[str, Any]
    processor: dict[str, Any] = Field(default_factory=dict)
    tokenizer: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    revision: str | None = None


class PromptSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    system: str | None = None
    template: str | None = None


class ProviderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = True
    loading: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    processor: dict[str, Any] = Field(default_factory=dict)
    image: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    prompt: PromptSettings = Field(default_factory=PromptSettings)
    cache: bool = False

    @model_validator(mode="after")
    def _validate_limits(self) -> Self:
        reserved = {"model", "revision", "tokenizer_revision", "mm_processor_kwargs"}
        for section_name in ("loading", "runtime"):
            misplaced = reserved.intersection(getattr(self, section_name))
            if misplaced:
                raise ValueError(
                    f"{sorted(misplaced)} must be configured through model identity or processor settings"
                )
        requires = self.loading.get("requires")
        if requires is not None and (
            not isinstance(requires, str) or not requires.strip()
        ):
            raise ValueError("loading.requires must be a nonempty module name or null")
        fallback = self.loading.get("fallback")
        if fallback is not None:
            if not isinstance(fallback, dict):
                raise ValueError("loading.fallback must be a mapping or null")
            if reserved.intersection(fallback) or {"requires", "fallback"}.intersection(
                fallback
            ):
                raise ValueError(
                    "loading.fallback must contain provider loading settings only"
                )
        if "max_new_tokens" in self.generation:
            raise ValueError(
                "Use generation.max_tokens; max_new_tokens is upstream metadata only"
            )
        structured = self.generation.get("structured_outputs")
        if structured is not None and not isinstance(structured, dict):
            raise ValueError("generation.structured_outputs must be a mapping or null")
        for key in ("max_model_len", "tensor_parallel_size", "max_num_seqs"):
            if key in self.runtime:
                value = self.runtime[key]
                if type(value) is not int or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
        if "gpu_memory_utilization" in self.runtime:
            value = self.runtime["gpu_memory_utilization"]
            if type(value) not in (int, float) or not 0 < value <= 1:
                raise ValueError("gpu_memory_utilization must be in (0, 1]")
        for key in ("min_pixels", "max_pixels"):
            value = self.image.get(key)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"image.{key} must be a positive integer or null")
        if (
            self.image.get("min_pixels") is not None
            and self.image.get("max_pixels") is not None
            and self.image["min_pixels"] > self.image["max_pixels"]
        ):
            raise ValueError("image.min_pixels must not exceed image.max_pixels")
        for key in ("max_tokens", "n"):
            if key in self.generation:
                value = self.generation[key]
                if type(value) is not int or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
        return self


class ModelConfigOverrides(ProviderSettings):
    """Explicit constructor settings; only supplied fields override configuration."""

    multimodal: bool | None = None


class ResolvedModelConfig(ProviderSettings):
    name: str
    provider: str
    revision: str | None = None
    multimodal: bool | None
    source_path: str
    profile: str | None = None
    metadata: ModelMetadata | None = None
