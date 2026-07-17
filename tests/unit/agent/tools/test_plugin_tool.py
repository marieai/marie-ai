from __future__ import annotations

import pytest

from marie.agent.tools.registry import resolve_tools
from marie.plugins.agent_tool import (
    PluginTool,
    PluginToolSpec,
    is_plugin_tool_spec,
)

# A Studio `extension_tool` dropped-tool record, as persisted on an AGENT node
# (camelCase keys, exactly what reactFlowToQueryPlan serializes).
DROPPED_TOOL = {
    "id": "tool-uuid",
    "name": "Jina Reader",
    "slug": "jina_reader",
    "toolType": "function",
    "source": "extension_tool",
    "runtimeExecution": "disabled",
    "installId": "inst-1",
    "providerId": "prov-1",
    "packageId": "pkg-1",
    "toolRef": "jina_reader",
    "description": "Read a web page",
}


def test_spec_ingests_studio_camelcase_dropped_tool():
    spec = PluginToolSpec.model_validate(DROPPED_TOOL)
    assert spec.tool_name == "jina_reader"
    assert spec.tool_ref == "jina_reader"
    assert spec.install_id == "inst-1"
    assert spec.provider_id == "prov-1"
    assert spec.package_id == "pkg-1"
    assert spec.source == "extension_tool"


def test_tool_name_falls_back_to_slug_then_tool_ref():
    spec = PluginToolSpec.model_validate(
        {"source": "extension_tool", "toolRef": "do_thing", "installId": "i"}
    )
    assert spec.tool_name == "do_thing"


@pytest.mark.parametrize(
    "spec",
    [
        {"source": "extension_tool", "slug": "x"},  # missing tool_ref
        {"source": "extension_tool", "toolRef": "x"},  # missing install identity
    ],
)
def test_incomplete_spec_is_rejected(spec):
    with pytest.raises(Exception):
        PluginToolSpec.model_validate(spec)


def test_is_plugin_tool_spec_discriminates():
    assert is_plugin_tool_spec(DROPPED_TOOL) is True
    # native/base tool (no source/toolRef/installId)
    assert (
        is_plugin_tool_spec(
            {"id": "x", "name": "calculator", "slug": "calculator", "toolType": "function"}
        )
        is False
    )
    # MCP spec
    assert (
        is_plugin_tool_spec(
            {"type": "mcp", "remote_tool_name": "search", "server_url": "http://x"}
        )
        is False
    )


def test_plugin_tool_metadata():
    tool = PluginTool(PluginToolSpec.model_validate(DROPPED_TOOL))
    assert tool.name == "jina_reader"
    assert tool.description == "Read a web page"
    params = tool.metadata.get_parameters_dict()
    assert params["type"] == "object"


def test_call_without_client_raises():
    tool = PluginTool(PluginToolSpec.model_validate(DROPPED_TOOL))
    with pytest.raises(RuntimeError, match="no daemon client"):
        tool.call(url="https://example.com")


def test_resolve_tools_routes_dropped_dict_to_plugin_tool():
    resolved = resolve_tools([DROPPED_TOOL])
    assert "jina_reader" in resolved
    assert isinstance(resolved["jina_reader"], PluginTool)


def test_build_payload_is_dify_invoke_tool_shape():
    spec = PluginToolSpec.model_validate({**DROPPED_TOOL, "providerRef": "jina"})
    tool = PluginTool(spec)
    payload = tool.build_payload({"url": "https://example.com"})
    assert payload == {
        "type": "tool",
        "action": "invoke_tool",
        "provider": "jina",
        "tool": "jina_reader",
        "credentials": {},
        "tool_parameters": {"url": "https://example.com"},
    }
