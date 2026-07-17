from pydantic import Field

from marie.extension.settings import ExtensionModel

TRUSTED_WILDCARD_LEVELS = {"builtin", "system"}


class NetworkPermissions(ExtensionModel):
    enabled: bool = False
    allowed_hosts: list[str] = Field(default_factory=list, alias="allowedHosts")
    allow_private_networks: bool = Field(default=False, alias="allowPrivateNetworks")

    def is_host_allowed(self, host: str, trust_level: str = "community") -> bool:
        if not self.enabled:
            return False
        if "*" in self.allowed_hosts:
            return trust_level in TRUSTED_WILDCARD_LEVELS
        return any(
            host == allowed or host.endswith(f".{allowed}")
            for allowed in self.allowed_hosts
        )


class SecretPermissions(ExtensionModel):
    enabled: bool = False
    allowed_names: list[str] = Field(default_factory=list, alias="allowed")

    def is_secret_allowed(self, name: str, trust_level: str = "community") -> bool:
        if not self.enabled:
            return False
        if "*" in self.allowed_names:
            return trust_level in TRUSTED_WILDCARD_LEVELS
        return name in self.allowed_names


class StoragePermissions(ExtensionModel):
    enabled: bool = False
    scopes: list[str] = Field(default_factory=list)
    max_bytes: int | None = Field(default=None, alias="maxBytes")


class RuntimeResourceLimits(ExtensionModel):
    timeout_ms: int = Field(default=30_000, alias="timeoutMs")
    max_concurrent: int = Field(default=1, alias="maxConcurrent")
    max_memory_bytes: int | None = Field(default=None, alias="maxMemoryBytes")
    max_processes: int = Field(default=1, alias="maxProcesses")


class ModelAccessPermissions(ExtensionModel):
    enabled: bool = False
    allowed_providers: list[str] = Field(default_factory=list, alias="allowedProviders")
    allowed_models: list[str] = Field(default_factory=list, alias="allowedModels")


class EndpointPermissions(ExtensionModel):
    enabled: bool = False
    allowed_methods: list[str] = Field(default_factory=list, alias="allowedMethods")
    public_exposure: bool = Field(default=False, alias="publicExposure")


class ExtensionPermissions(ExtensionModel):
    network: NetworkPermissions = Field(default_factory=NetworkPermissions)
    secrets: SecretPermissions = Field(default_factory=SecretPermissions)
    storage: StoragePermissions = Field(default_factory=StoragePermissions)
    runtime: RuntimeResourceLimits = Field(default_factory=RuntimeResourceLimits)
    model: ModelAccessPermissions = Field(default_factory=ModelAccessPermissions)
    endpoints: EndpointPermissions = Field(default_factory=EndpointPermissions)
