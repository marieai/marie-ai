from typing import Literal

from pydantic import Field

from marie.extension.settings import ExtensionModel


class RuntimeResources(ExtensionModel):
    memory_bytes: int | None = Field(default=None, alias="memoryBytes")
    timeout_seconds: int | None = Field(default=None, alias="timeoutSeconds")


class RuntimeSpec(ExtensionModel):
    type: str = "metadata_only"
    language: str | None = None
    version: str | None = None
    entrypoint: str | None = None
    isolation: str | None = None
    resources: RuntimeResources = Field(default_factory=RuntimeResources)


class RuntimePolicy(ExtensionModel):
    timeout_ms: int = Field(default=30_000, alias="timeoutMs", gt=0)
    max_concurrent: int = Field(default=1, alias="maxConcurrent", gt=0)
    max_memory_bytes: int = Field(default=536_870_912, alias="maxMemoryBytes", gt=0)
    network_policy: Literal["none", "manifest_declared", "internal_only"] = Field(
        default="none", alias="networkPolicy"
    )
