# Marie Agent

`marie-agent` is the reusable agent runtime distributed under Marie's PEP 420
namespace. The distribution name is `marie-agent`; its Python API is
`marie.agent`.

The package does not contain `src/marie/__init__.py`. It can therefore share
the `marie` namespace with `marie-ai`, `marie-instrumentation`, and future
Marie integration distributions without one distribution owning the namespace
root.

```python
from marie.agent.agents import ReactAgent
from marie.agent.llm import OpenAICompatibleWrapper
from marie.agent.messages import ContentItem, Message
from marie.agent.tools.filesystem import FileReadTool
```

Install optional provider support only when needed:

```bash
pip install "marie-agent[openai]"
pip install "marie-agent[a2a]"
pip install "marie-agent[autogen]"
pip install "marie-agent[haystack]"
```

Marie server adapters such as `AgentExecutor`, EmbeddedPlugin tools, MCP tools,
and database tools remain in the `marie-ai` distribution. The optional
`marie.engine` bridge is supplied by the separate `marie-engine[agent]`
distribution. Both depend on this provider-independent package; this package
does not depend on either engine or server.
