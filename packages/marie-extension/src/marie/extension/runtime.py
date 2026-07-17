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
