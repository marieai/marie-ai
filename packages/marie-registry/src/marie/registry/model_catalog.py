from __future__ import annotations

import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Self

import yaml
from marie.registry.model_config import (
    ModelConfigOverrides,
    ModelMetadata,
    ProviderSettings,
    ResolvedModelConfig,
)
from marie.registry.model_metadata import infer_multimodal
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

_ModelName = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
MetadataResolver = Callable[[str, str | None], ModelMetadata]
_config_path: Path | None = None


def _identity(name: str) -> str:
    return name.removeprefix("hf://").removeprefix("transformer://")


class _UniqueLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise yaml.constructor.ConstructorError(
                    "while reading model configuration",
                    node.start_mark,
                    f"duplicate key: {key}",
                    key_node.start_mark,
                )
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: _ModelName
    revision: _ModelName | None = None
    multimodal: bool | None = None
    profile: _ModelName | None = None
    providers: dict[_ModelName, ProviderSettings] = Field(default_factory=dict)


class _Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    defaults: dict[_ModelName, ProviderSettings] = Field(default_factory=dict)
    profiles: dict[_ModelName, dict[_ModelName, ProviderSettings]] = Field(
        default_factory=dict
    )
    models: dict[_ModelName, ModelConfig]

    @model_validator(mode="after")
    def _validate_aliases(self) -> Self:
        capabilities: dict[str, bool] = {}
        revisions: dict[str, str | None] = {}
        identities = {_identity(model.name) for model in self.models.values()}
        for alias, model in self.models.items():
            alias_identity = _identity(alias)
            if alias_identity in identities and _identity(model.name) != alias_identity:
                raise ValueError(f"Alias '{alias}' shadows a full model name")
        for model in self.models.values():
            identity = _identity(model.name)
            if model.profile is not None and model.profile not in self.profiles:
                raise ValueError(f"Unknown model profile '{model.profile}'")
            if identity in revisions and revisions[identity] != model.revision:
                raise ValueError(f"Conflicting revisions for '{identity}'")
            revisions[identity] = model.revision
            if model.multimodal is not None:
                previous = capabilities.get(identity)
                if previous is not None and previous != model.multimodal:
                    raise ValueError(
                        f"Conflicting multimodal capabilities for '{identity}'"
                    )
                capabilities[identity] = model.multimodal
        return self


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class ModelCatalog:
    """Resolve complete model settings from one external catalog."""

    def __init__(
        self, path: Path, metadata_resolver: MetadataResolver | None = None
    ) -> None:
        from marie.registry.model_registry import ModelRegistry

        self.path = path
        try:
            contents = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Model catalog not found: {path}") from exc
        try:
            self._config = _Catalog.model_validate(
                yaml.load(contents, Loader=_UniqueLoader)
            )
        except (yaml.YAMLError, ValidationError) as exc:
            raise ValueError(f"Invalid model catalog {path}: {exc}") from exc
        self.models = self._config.models
        self.name_map = {alias: model.name for alias, model in self.models.items()}
        self.reverse_name_map = {name: alias for alias, name in self.name_map.items()}
        self._metadata_resolver = metadata_resolver or ModelRegistry.metadata
        self._metadata: dict[tuple[str, str | None], ModelMetadata] = {}
        self._capabilities = {
            _identity(model.name): model.multimodal
            for model in self.models.values()
            if model.multimodal is not None
        }

    def _definition(self, name: str) -> ModelConfig:
        if name in self.models:
            return self.models[name]
        matches = [
            model
            for model in self.models.values()
            if _identity(model.name) == _identity(name)
        ]
        if matches:
            first = matches[0]
            if any(
                model.profile != first.profile or model.providers != first.providers
                for model in matches[1:]
            ):
                raise ValueError(
                    f"Model '{name}' has multiple configured profiles in {self.path}; use an alias"
                )
            return first
        return ModelConfig(name=name)

    def _model_metadata(self, model: ModelConfig) -> ModelMetadata:
        key = (_identity(model.name), model.revision)
        if key not in self._metadata:
            try:
                self._metadata[key] = self._metadata_resolver(
                    model.name, model.revision
                )
            except (OSError, ValueError, ImportError) as exc:
                exc.add_note(f"Resolving model '{model.name}' from catalog {self.path}")
                raise
        return self._metadata[key]

    def is_multimodal(self, name: str) -> bool:
        """Prefer an explicit capability override, otherwise use model metadata."""
        model = self._definition(name)
        identity = _identity(model.name)
        if identity not in self._capabilities:
            capability = infer_multimodal(self._model_metadata(model).config)
            if capability is None:
                raise ValueError(
                    f"Cannot determine multimodal capability for '{model.name}'. "
                    f"Set 'multimodal' explicitly in {self.path}"
                )
            self._capabilities[identity] = capability
        return self._capabilities[identity]

    def resolve_config(
        self,
        name: str,
        provider: str,
        *,
        overrides: ModelConfigOverrides | None = None,
    ) -> ResolvedModelConfig:
        """Merge provider defaults, one profile, model settings, and explicit overrides."""
        from marie.registry.model_registry import ModelRegistry

        model = self._definition(name)
        merged: dict[str, Any] = {}
        layers = [self._config.defaults.get(provider)]
        if model.profile is not None:
            layers.append(self._config.profiles[model.profile].get(provider))
        layers.append(model.providers.get(provider))
        layers.append(overrides)
        for layer in layers:
            if layer is not None:
                values = layer.model_dump(exclude_unset=True)
                values.pop("multimodal", None)
                if "processor" in values:
                    merged["image"] = _merge(
                        merged.get("image", {}), values["processor"]
                    )
                merged = _merge(merged, values)
        metadata = self._model_metadata(model) if merged.get("enabled", True) else None
        if overrides is not None and overrides.multimodal is not None:
            multimodal = overrides.multimodal
        elif merged.get("enabled", True):
            multimodal = self.is_multimodal(name)
        else:
            multimodal = self._capabilities.get(_identity(model.name))
        if metadata is not None:
            generation = {
                key: value
                for key, value in metadata.generation.items()
                if key
                in {"temperature", "top_p", "top_k", "max_new_tokens", "max_tokens"}
                and value is not None
            }
            if "max_new_tokens" in generation:
                generation["max_tokens"] = generation.pop("max_new_tokens")
            merged = _merge({"generation": generation}, merged)
        revision = (metadata.revision if metadata else None) or model.revision
        try:
            return ResolvedModelConfig(
                name=ModelRegistry.resolve_reference(model.name, revision),
                provider=provider,
                revision=revision,
                multimodal=multimodal,
                source_path=str(self.path),
                profile=model.profile,
                metadata=metadata.model_copy(deep=True)
                if metadata is not None
                else None,
                **merged,
            )
        except ValidationError as exc:
            raise ValueError(
                f"Invalid model settings for {name} in {self.path}: {exc}"
            ) from exc

    @property
    def multimodal_models(self) -> list[str]:
        """Return the configured model names supporting image input."""
        return list(
            dict.fromkeys(
                name for name in self.name_map.values() if self.is_multimodal(name)
            )
        )


def configure_model_catalog(path: str | Path) -> None:
    """Set the host catalog path without reading it or resolving models."""
    global _config_path
    _config_path = Path(path).expanduser()
    _load_catalog.cache_clear()


def invalidate_model_catalog() -> None:
    """Discard resolved settings after registry path changes."""
    _load_catalog.cache_clear()


@lru_cache(maxsize=1)
def _load_catalog(path: Path) -> ModelCatalog:
    return ModelCatalog(path)


def load_model_catalog() -> ModelCatalog:
    """Load the selected external catalog on first model use."""
    catalog_path = os.getenv("MARIE_MODEL_CATALOG")
    if catalog_path:
        path = Path(catalog_path).expanduser()
    elif _config_path is not None:
        path = _config_path
    elif mount_path := os.getenv("MARIE_DEFAULT_MOUNT"):
        path = Path(mount_path).expanduser() / "config" / "models.yaml"
    else:
        raise ValueError(
            "Set MARIE_MODEL_CATALOG to the external models.yaml file or "
            "MARIE_DEFAULT_MOUNT to the Marie data directory before model lookup"
        )
    return _load_catalog(path)
