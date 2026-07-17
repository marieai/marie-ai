from pydantic import Field

from marie.extension.settings import ExtensionModel, ParameterDefinition


class ModelPricing(ExtensionModel):
    input: str | None = None
    output: str | None = None
    unit: str | None = None
    currency: str | None = None


class ModelDefinition(ExtensionModel):
    ref: str
    model_id: str = Field(alias="modelId")
    model_type: str = Field(alias="modelType")
    display_name: str | None = Field(default=None, alias="displayName")
    features: list[str] = Field(default_factory=list)
    properties: dict[str, object] = Field(default_factory=dict)
    parameters: list[ParameterDefinition] = Field(default_factory=list)
    pricing: ModelPricing | None = None
