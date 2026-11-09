from pathlib import Path

import pytest
import yaml
from marie.extension.manifest import ExtensionPackage

FIXTURES = Path(__file__).parent / "fixtures"


def test_valid_manifest_parses() -> None:
    data = yaml.safe_load(
        (FIXTURES / "minimal-tool" / "marie-extension.yaml").read_text()
    )

    manifest = ExtensionPackage.model_validate(data)

    assert manifest.kind == "ExtensionPackage"
    assert manifest.providers[0].tools[0].invocation_schema.required == ["text"]


def test_invalid_kind_rejected() -> None:
    data = yaml.safe_load(
        (FIXTURES / "minimal-tool" / "marie-extension.yaml").read_text()
    )
    data["kind"] = "Plugin"

    with pytest.raises(ValueError, match="kind must be ExtensionPackage"):
        ExtensionPackage.model_validate(data)


def test_datasource_provider_must_be_declared_with_the_correct_type() -> None:
    data = yaml.safe_load(
        (FIXTURES / 'minimal-tool' / 'marie-extension.yaml').read_text()
    )
    data['datasources'] = [
        {
            'ref': 'datasources/wiki',
            'name': 'Wiki',
            'providerRef': data['providers'][0]['ref'],
        }
    ]
    with pytest.raises(ValueError, match='datasource_provider'):
        ExtensionPackage.model_validate(data)
    data['providers'][0]['type'] = 'datasource_provider'
    assert (
        ExtensionPackage.model_validate(data).datasources[0].provider_ref
        == data['providers'][0]['ref']
    )


def test_agent_strategy_provider_has_typed_agent_contract() -> None:
    data = yaml.safe_load(
        (FIXTURES / "minimal-agent" / "marie-extension.yaml").read_text()
    )

    manifest = ExtensionPackage.model_validate(data)
    provider = manifest.providers[0]
    agent = provider.agents[0]

    assert provider.type == "agent_strategy_provider"
    assert agent.ref == "agents/repair"
    assert agent.invocation_schema.required == ["page_file"]
    assert agent.output.schema_["type"] == "object"
    assert agent.credentials[0].key == "llm_api_key"
    assert agent.model_capabilities == ["vision", "tool_calling"]
    assert agent.runtime_policy is not None
    assert agent.runtime_policy.network_policy == "internal_only"
    assert manifest.runtime_policy.max_concurrent == 2
