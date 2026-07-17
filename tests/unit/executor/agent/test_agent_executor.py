from __future__ import annotations

import importlib
import json
from unittest.mock import AsyncMock, patch

import pytest
from docarray import DocList
from docarray.documents import TextDoc
from jsonschema import ValidationError

from marie.agent.backends.base import (
    AgentResult,
    AgentStatus,
)
from marie.agent.backends.openai_backend import (
    OpenAIAgentBackend,
    OpenAIBackendConfig,
)
from marie.agent.message import Message
from marie.executor.agent import AgentExecutor


def executor_with_result(result: AgentResult) -> tuple[AgentExecutor, AsyncMock]:
    executor = AgentExecutor(enable_conversation_store=True)
    backend = AsyncMock()
    backend.run.return_value = result
    executor._backend = backend
    executor._initialized = True
    return executor, backend


def test_agent_executor_only_uses_executor_namespace() -> None:
    import marie.agent

    assert not hasattr(marie.agent, "AgentExecutor")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("marie.agent.executor")
    assert AgentExecutor.__module__ == "marie.executor.agent.agent_executor"


@pytest.mark.asyncio
async def test_execute_is_stateless_and_validates_structured_output() -> None:
    executor, backend = executor_with_result(
        AgentResult(output='```json\n{"decision":"repair",}\n```', iterations=2)
    )
    await executor._conversation_store.add_message("run-1", Message.user("ignored"))

    result = await executor.execute_endpoint(
        DocList[TextDoc]([TextDoc(text="Review the finding")]),
        parameters={
            "execution_id": "run-1",
            "output_schema": {
                "type": "object",
                "properties": {"decision": {"const": "repair"}},
                "required": ["decision"],
                "additionalProperties": False,
            },
        },
    )

    assert json.loads(result[0].text) == {"decision": "repair"}
    messages = backend.run.call_args.kwargs["messages"]
    assert all("ignored" not in message.text_content for message in messages)
    assert messages[0].text_content == "Review the finding"


@pytest.mark.asyncio
async def test_execute_preserves_multimodal_content_items() -> None:
    executor, backend = executor_with_result(AgentResult(output="done"))

    await executor.execute_endpoint(
        DocList[TextDoc]([TextDoc(text="Review the current page")]),
        parameters={"content_items": [{"image": "/tmp/page-29.png"}]},
    )

    message = backend.run.call_args.kwargs["messages"][0]
    assert message.content[0].text == "Review the current page"
    assert message.content[1].image == "/tmp/page-29.png"


@pytest.mark.asyncio
async def test_execute_retries_invalid_structured_output() -> None:
    executor, backend = executor_with_result(AgentResult(output="[]"))
    backend.run.side_effect = [
        AgentResult(output="[]"),
        AgentResult(output='{"decision":"repair"}'),
    ]

    result = await executor.execute_endpoint(
        DocList[TextDoc]([TextDoc(text="Review")]),
        parameters={
            "output_schema": {
                "type": "object",
                "properties": {"decision": {"const": "repair"}},
                "required": ["decision"],
            }
        },
    )

    assert json.loads(result[0].text) == {"decision": "repair"}
    assert backend.run.await_count == 2
    retry_messages = backend.run.call_args.kwargs["messages"]
    assert "did not satisfy" in retry_messages[-1].text_content


@pytest.mark.asyncio
async def test_execute_rejects_invalid_structured_output() -> None:
    executor, _backend = executor_with_result(AgentResult(output='{"decision":"skip"}'))

    with pytest.raises(ValidationError):
        await executor.execute_endpoint(
            DocList[TextDoc]([TextDoc(text="Review")]),
            parameters={
                "output_schema": {
                    "type": "object",
                    "properties": {"decision": {"const": "repair"}},
                    "required": ["decision"],
                }
            },
        )


@pytest.mark.asyncio
async def test_execute_raises_on_backend_failure() -> None:
    executor, _backend = executor_with_result(
        AgentResult(
            output="",
            status=AgentStatus.FAILED,
            error="model unavailable",
        )
    )

    with pytest.raises(RuntimeError, match="model unavailable"):
        await executor.execute_endpoint(
            DocList[TextDoc]([TextDoc(text="Review")]),
        )


def test_openai_backend_keeps_model_configuration() -> None:
    executor = AgentExecutor(
        backend="openai",
        backend_config={
            "api_key": "test-key",
            "max_iterations": 4,
            "temperature": 0.0,
            "max_tokens": 4096,
            "timeout_seconds": 45,
            "model": "qwen_v3_30b_instruct",
            "base_url": "http://llm.test/v1",
        },
        system_message="Repair extraction findings.",
    )

    backend = executor._create_backend()

    assert isinstance(backend, OpenAIAgentBackend)
    assert backend.config.max_iterations == 4
    assert backend.config.temperature == 0.0
    assert backend.config.max_tokens == 4096
    assert backend.config.timeout_seconds == 45
    assert backend.config.model == "qwen_v3_30b_instruct"
    assert backend.config.base_url == "http://llm.test/v1"
    assert backend.config.system_message == "Repair extraction findings."

    with patch(
        "marie.agent.backends.openai_backend.OpenAICompatibleWrapper"
    ) as wrapper:
        backend._get_llm()

    wrapper.assert_called_once_with(
        api_key="test-key",
        model="qwen_v3_30b_instruct",
        base_url="http://llm.test/v1",
        tool_call_format="auto",
        timeout=45,
        max_retries=0,
    )


@pytest.mark.asyncio
async def test_openai_backend_reports_llm_transport_failure() -> None:
    class FailingLLM:
        def chat(self, **_kwargs):
            raise TimeoutError("model request timed out")

    backend = OpenAIAgentBackend(
        config=OpenAIBackendConfig(model="test-model"),
        llm=FailingLLM(),
    )

    result = await backend.run([Message.user("Review")])

    assert result.status == AgentStatus.FAILED
    assert result.error == "model request timed out"
