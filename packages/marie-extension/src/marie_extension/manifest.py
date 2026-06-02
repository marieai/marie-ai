from pydantic import Field, model_validator

from marie_extension.datasource import DatasourceDefinition
from marie_extension.endpoint import EndpointDefinition
from marie_extension.permissions import ExtensionPermissions
from marie_extension.provider import ProviderDefinition
from marie_extension.runtime import RuntimeSpec
from marie_extension.settings import ExtensionModel
from marie_extension.tool import ToolDefinition
from marie_extension.trigger import TriggerDefinition
from marie_extension.trust import TrustSpec


class PackageMetadata(ExtensionModel):
    id: str
    name: str
    version: str
    display_name: str | dict[str, str] | None = Field(default=None, alias="displayName")
    author: str | None = None
    description: str | dict[str, str] | None = None
    icon: str | None = None
    tags: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    license: str | None = None
    readme: str | None = None
    privacy: str | None = None
    i18n: dict[str, dict[str, str]] = Field(default_factory=dict)


class CompatibilitySpec(ExtensionModel):
    marie_ai: str | None = Field(default=None, alias="marieAi")
    marie_studio: str | None = Field(default=None, alias="marieStudio")
    schema_version: str = Field(default="v1alpha1", alias="schema")


class ScopeSpec(ExtensionModel):
    install_target: str = Field(default="organization", alias="installTarget")
    workspace_binding: dict[str, object] = Field(
        default_factory=dict, alias="workspaceBinding"
    )
    tenant_compatibility: dict[str, object] = Field(
        default_factory=dict, alias="tenantCompatibility"
    )


class ExtensionPackage(ExtensionModel):
    api_version: str = Field(alias="apiVersion")
    kind: str
    metadata: PackageMetadata
    compatibility: CompatibilitySpec = Field(default_factory=CompatibilitySpec)
    trust: TrustSpec = Field(default_factory=TrustSpec)
    scope: ScopeSpec = Field(default_factory=ScopeSpec)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    permissions: ExtensionPermissions = Field(default_factory=ExtensionPermissions)
    providers: list[ProviderDefinition] = Field(default_factory=list)
    tools: list[ToolDefinition] = Field(default_factory=list)
    datasources: list[DatasourceDefinition] = Field(default_factory=list)
    triggers: list[TriggerDefinition] = Field(default_factory=list)
    endpoints: list[EndpointDefinition] = Field(default_factory=list)
    ui: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_kind(self) -> "ExtensionPackage":
        if self.kind != "ExtensionPackage":
            raise ValueError("kind must be ExtensionPackage")
        return self
