from pydantic import Field

from marie_extension.settings import ExtensionModel


class TrustSignature(ExtensionModel):
    required: bool = False
    key_id: str | None = Field(default=None, alias="keyId")


class TrustSpec(ExtensionModel):
    source: str = "local"
    publisher: str = "community"
    level: str = "community"
    signature: TrustSignature = Field(default_factory=TrustSignature)
    checksums: dict[str, str | None] = Field(default_factory=dict)
