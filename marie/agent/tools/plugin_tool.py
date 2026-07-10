"""Plugin tool adapter — invoke an installed plugin tool as an agent tool.

Mirrors `marie.agent.tools.mcp_tool.MCPRemoteTool` / `marie.mcp.runtime.MCPToolSpec`.
An installed extension (plugin) tool dropped onto an agent is resolved by
`resolve_tools()` into a `PluginTool`, which (from Slice 3a) invokes the marie
plugin daemon (`POST /v1/dispatch/invoke`).

The spec carries only the stable IDENTITY of the tool. The daemon wire identity
(`packageRef`/`provider`/`tool`), the parameter schema, credentials, and the
org/workspace context are injected at call time — never persisted here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput
from marie.secret_store import CredentialRequirement

if TYPE_CHECKING:
    from marie.agent.tools.plugin_daemon_client import PluginDaemonClient


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class PluginToolSpec(BaseModel):
    """Identity of an installed-plugin tool, as dropped onto an agent in Studio.

    Ingests the Studio `extension_tool` dropped-tool dict (camelCase keys) thanks
    to ``alias_generator=_to_camel`` + ``populate_by_name=True``.
    """

    model_config = ConfigDict(populate_by_name=True, alias_generator=_to_camel)

    type: str = Field(default="extension_tool")
    source: str | None = None

    # Invocable name + identity
    tool_name: str | None = None
    name: str | None = None
    slug: str | None = None
    tool_ref: str | None = None
    install_id: str | None = None
    provider_id: str | None = None
    provider_ref: str | None = None
    package_id: str | None = None
    package_ref: str | None = None
    # Daemon trust-policy claims (required by /v1/dispatch/invoke). Hydrated from
    # the extension catalog; for the local hello-world they are passed as constants.
    package_digest: str | None = None
    package_trust_level: str | None = None

    description: str | None = None
    # Hydrated fresh at call time (Slice 3a); not persisted into saved plans.
    input_schema: dict[str, Any] | None = None
    # Credential requirements (name + bound secret_ref); resolved in marie-ai by
    # the CredentialResolver at call time. Empty/absent -> no credentials.
    credential_requirements: list[CredentialRequirement] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize(self) -> "PluginToolSpec":
        tool_name = self.tool_name or self.slug or self.tool_ref or self.name
        if not tool_name:
            raise ValueError(
                "plugin tool spec must include a tool name (slug / tool_ref / name)"
            )
        self.tool_name = tool_name

        if not self.tool_ref:
            raise ValueError("plugin tool spec must include 'tool_ref'")

        if not (self.install_id or self.package_ref or self.provider_ref):
            raise ValueError(
                "plugin tool spec must include an install identity "
                "(install_id / package_ref / provider_ref)"
            )
        return self


def is_plugin_tool_spec(spec: dict[str, Any]) -> bool:
    """True when a tool-config dict denotes an installed plugin (extension) tool."""
    if spec.get("source") == "extension_tool" or spec.get("type") == "extension_tool":
        return True
    return any(
        key in spec for key in ("tool_ref", "toolRef", "install_id", "installId")
    )


class PluginTool(AgentTool):
    """AgentTool backed by an installed plugin tool (invoked via the plugin daemon).

    Mirrors `MCPRemoteTool`. Resolution is complete here; `call()` is wired in
    Slice 3a (build the marie envelope, dispatch to ``/v1/dispatch/invoke``).
    """

    def __init__(
        self, spec: PluginToolSpec, client: "PluginDaemonClient | None" = None
    ):
        self._spec = spec
        self._client = client

    @property
    def spec(self) -> PluginToolSpec:
        return self._spec

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self._spec.tool_name or "plugin_tool",
            description=self._spec.description or self._default_description(),
            parameters=self._spec.input_schema
            or {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
        )

    def bind_client(self, client: "PluginDaemonClient") -> "PluginTool":
        """Attach the daemon client used to invoke this tool (set by the runtime)."""
        self._client = client
        return self

    def build_payload(self, tool_parameters: dict[str, Any]) -> dict[str, Any]:
        """Build the dify `invoke_tool` request the daemon forwards to the plugin.

        The dify_plugin SDK routes by ``type``+``action`` and parses the request as
        a ``ToolInvokeRequest`` (``provider``/``tool``/``credentials``/
        ``tool_parameters`` at the top level). ``user_id`` (also required) is added
        by the daemon client. `credentials` is empty here — credential resolution is
        Slice 3b (the daemon stays stateless re: secrets).
        """
        return {
            "type": "tool",
            "action": "invoke_tool",
            "provider": self._spec.provider_ref or self._spec.tool_ref,
            "tool": self._spec.tool_ref,
            "credentials": {},
            "tool_parameters": tool_parameters,
        }

    def call(self, **kwargs: Any) -> ToolOutput:
        if self._client is None:
            raise RuntimeError(
                "PluginTool has no daemon client bound; the execution environment "
                "must call bind_client() (or construct PluginTool(spec, client)) "
                f"before invoking tool={self._spec.tool_name!r}"
            )
        return self._client.invoke(self._spec, self.build_payload(kwargs))

    def _default_description(self) -> str:
        return f"Invoke the installed plugin tool {self._spec.tool_name}."
