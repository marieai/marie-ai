"""Hello-world: call an installed plugin tool from marie-ai on the local machine.

Canonical reference for the agent-tool -> plugin-daemon path. A tool dropped onto
an agent in Studio (an ``extension_tool`` record) becomes a callable `PluginTool`
at runtime, invoked via the marie plugin daemon over a *signed* envelope.

WHAT IT DOES
  1. Builds the plugin-tool spec in the Studio `extension_tool` shape.
  2. resolve_tools() turns it into a `PluginTool` (same path an agent uses).
  3. Connects to the daemon and HMAC-signs the runtime envelope (the daemon
     verifies it — same scheme Studio uses).
  4. Calls the tool -> prints the ToolOutput (marie-ai -> /v1/dispatch/invoke ->
     daemon -> plugin stdio -> SSE -> ToolOutput).

DAEMON CONNECTION (use the EXISTING compiled binary or a running daemon)
  - Remote (a daemon already running):
        export MARIE_PLUGIN_DAEMON_URL=http://127.0.0.1:8099
  - Or spawn the existing compiled binary (no rebuild):
        export MARIE_PLUGIN_DAEMON_BIN=packages/marie-plugin-daemon/dist/marie-plugin-daemon

SIGNING (the daemon verifies HMAC; use the SAME key the daemon was started with)
        export MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID=marie-api-test
        export MARIE_PLUGIN_DAEMON_SIGNING_SECRET=test-runtime-secret

Then fill in KNOWN_INSTALL / DROPPED_TOOL below for a plugin installed in that
daemon and run:
        python examples/agents/agent_plugin_tool_hello_world.py
"""

from __future__ import annotations

import os

from marie.agent.tools.plugin_daemon_client import PluginDaemonClient
from marie.agent.tools.plugin_tool import PluginTool
from marie.agent.tools.registry import resolve_tools

# --- Tenant + signing (set to match the daemon you're talking to) ------------
ORG_ID = os.getenv("MARIE_ORG_ID", "11111111-1111-1111-1111-111111111111")
WORKSPACE_ID = os.getenv("MARIE_WORKSPACE_ID", "22222222-2222-2222-2222-222222222222")
USER_ID = os.getenv("MARIE_USER_ID", "44444444-4444-4444-4444-444444444444")
SIGNING_KEY_ID = os.getenv("MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID")
SIGNING_SECRET = os.getenv("MARIE_PLUGIN_DAEMON_SIGNING_SECRET")

# Identity of the installed plugin tool (the daemon requires packageRef + digest).
KNOWN_INSTALL = {
    "package_ref": os.getenv("MARIE_PLUGIN_PACKAGE_REF", "ext.m3forge.reader"),
    "package_digest": os.getenv("MARIE_PLUGIN_PACKAGE_DIGEST", "sha256:replace-me"),
}

# The tool as Studio drops it onto an agent node (an `extension_tool` record).
DROPPED_TOOL = {
    "id": "tool-uuid",
    "name": "Web Reader",
    "slug": "web_reader",
    "toolType": "function",
    "source": "extension_tool",
    "installId": os.getenv(
        "MARIE_PLUGIN_INSTALL_ID", "33333333-3333-3333-3333-333333333333"
    ),
    "providerId": os.getenv(
        "MARIE_PLUGIN_PROVIDER_ID", "55555555-5555-5555-5555-555555555555"
    ),
    # providerRef is the provider NAME the plugin declares (e.g. "jina"), used as
    # the dify invoke_tool `provider`.
    "providerRef": os.getenv("MARIE_PLUGIN_PROVIDER_REF", "web_reader"),
    "packageId": os.getenv(
        "MARIE_PLUGIN_PACKAGE_ID", "77777777-7777-7777-7777-777777777777"
    ),
    "toolRef": os.getenv("MARIE_PLUGIN_TOOL_REF", "web_reader"),
    "packageRef": KNOWN_INSTALL["package_ref"],
    "packageDigest": KNOWN_INSTALL["package_digest"],
    "description": "Read a web page",
    # Credential requirements resolved in marie-ai (env-backed here). jina's api_key
    # is OPTIONAL: resolved + sent if JINA_API_KEY is set in the env, skipped if not.
    "credentialRequirements": [
        {"name": "api_key", "secretRef": "env:JINA_API_KEY", "required": False},
    ],
}

# The arguments passed to the tool (forwarded as the plugin invocation payload).
TOOL_ARGS = {"url": "https://example.com"}


def main() -> None:
    # 1-2. Resolve the dropped tool into a PluginTool — exactly how an agent does
    #      it via resolve_tools() (AgentExecutor(tools=[DROPPED_TOOL]) would too).
    tools = resolve_tools([DROPPED_TOOL])
    tool = next(t for t in tools.values() if isinstance(t, PluginTool))
    print(f"resolved PluginTool: name={tool.name!r} ref={tool.spec.tool_ref!r}")

    # The daemon verifies a signed envelope — fail fast with a clear message.
    if not SIGNING_KEY_ID or not SIGNING_SECRET:
        raise SystemExit(
            "Set MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID and "
            "MARIE_PLUGIN_DAEMON_SIGNING_SECRET (the key the daemon was started with)."
        )

    # 3. Connect to the running daemon (default localhost:8099). Set
    #    MARIE_PLUGIN_DAEMON_SPAWN=1 to spawn-and-own the binary instead.
    spawn = os.getenv("MARIE_PLUGIN_DAEMON_SPAWN", "").lower() in {"1", "true", "yes"}
    base_url = (
        None if spawn else os.getenv("MARIE_PLUGIN_DAEMON_URL", "http://127.0.0.1:8099")
    )
    client = PluginDaemonClient(
        organization_id=ORG_ID,
        workspace_id=WORKSPACE_ID,
        user_id=USER_ID,
        signing_key_id=SIGNING_KEY_ID,
        signing_secret=SIGNING_SECRET,
        base_url=base_url,
        spawn_local=spawn,
        env=dict(os.environ),
    )
    # Optional: capture the call with DebugCaptureMiddleware (the agent debug
    # middleware). In a REAL agent run you'd just do
    #   YourAgent(tools=[DROPPED_TOOL, ...], middlewares=[DebugCaptureMiddleware(output_dir=...)])
    # and the plugin tool is captured automatically (it's a normal AgentTool).
    # Here there is no LLM, so we wire it manually — mirroring how
    # BaseAgent._call_tool emits the tool.* events DebugCaptureMiddleware listens for.
    debug_dir = os.getenv("MARIE_AGENT_DEBUG_DIR")
    emitter = None
    capture = None
    if debug_dir:
        from marie.agent.emitter import Emitter, emit_sync
        from marie.agent.middleware.debug_capture import DebugCaptureMiddleware

        emitter = Emitter()
        capture = DebugCaptureMiddleware(output_dir=debug_dir)
        capture.bind(emitter)
        emit_sync(
            emitter,
            "tool.start",
            {"tool_name": tool.name, "arguments": TOOL_ARGS},
            source=tool.name,
        )

    try:
        tool.bind_client(client)
        # 4. Invoke via safe_call() — the OBSERVABLE path: it opens the OTel
        #    `tool:<name>` span, emits start/success/finish events, logs + times
        #    the call (visible in your tracing pipeline), then runs PluginTool.call
        #    -> signed envelope -> /v1/dispatch/invoke -> daemon -> plugin -> SSE.
        output = tool.safe_call(TOOL_ARGS)
        print(f"ToolOutput (is_error={output.is_error}):")
        print(output.content)

        if emitter is not None:
            from marie.agent.emitter import emit_sync

            emit_sync(
                emitter,
                "tool.error" if output.is_error else "tool.success",
                {"tool_name": tool.name, "result": output.content[:240]},
                source=tool.name,
            )
            emit_sync(
                emitter,
                "tool.finish",
                {"tool_name": tool.name, "success": not output.is_error},
                source=tool.name,
            )
            print(f"debug artifacts written to: {capture.debug_dir}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
