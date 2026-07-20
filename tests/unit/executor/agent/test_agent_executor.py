from __future__ import annotations

import importlib
import json
from unittest.mock import AsyncMock, patch

import pytest
from docarray import DocList
from docarray.documents import TextDoc
from jsonschema import ValidationError
from pydantic import ValidationError as PydanticValidationError

from marie.agent.backends.base import (
    AgentResult,
    AgentStatus,
)
from marie.agent.backends.openai_backend import (
    OpenAIAgentBackend,
    OpenAIBackendConfig,
)
from marie.agent.message import Message
from marie.api import AssetKeyDoc
from marie.executor.agent import AgentExecutor
from marie.plugins.embedded import PluginInvocationResult


def executor_with_result(result: AgentResult) -> tuple[AgentExecutor, AsyncMock]:
    executor = AgentExecutor(enable_conversation_store=True)
    backend = AsyncMock()
    backend.run.return_value = result
    executor._backend = backend
    executor._initialized = True
    return executor, backend


def plugin_executor(
    *,
    route: dict | None = None,
) -> tuple[AgentExecutor, AsyncMock]:
    executor = AgentExecutor(
        enable_conversation_store=False,
        plugins=[
            {
                'package': 'marie/fixture-agent',
                'path': '/not-read-during-routing.zip',
                'actions': ['run'],
            }
        ],
        agent_routes={
            'fixture.echo': route
            or {
                'package': 'marie/fixture-agent',
                'action': 'run',
            }
        },
    )
    invocation = AsyncMock(
        return_value=PluginInvocationResult(
            result={'proposal': {'decision': 'repair'}},
            frames=(
                {'type': 'stream', 'data': {'proposal': {'decision': 'repair'}}},
                {'type': 'end', 'data': {}},
            ),
            request_id='task-1',
            trace_id='trace-1',
        )
    )
    executor.embedded_plugins.invoke_async = invocation
    return executor, invocation


def plugin_parameters(**request_overrides) -> dict:
    request = {
        'agent_ref': 'fixture.echo',
        'input': {'finding': 'missing value'},
        'artifacts': {'schema_uri': 's3://bucket/schema.json'},
        'idempotency_key': 'effect-1',
    }
    request.update(request_overrides)
    return {
        'payload': {'op_params': request},
        'job_id': 'task-1',
        'dag_id': 'dag-1',
        'node_task_id': 'task-1',
        'run_attempt_id': 'attempt-2',
    }


def test_agent_executor_only_uses_executor_namespace() -> None:
    import marie.agent

    assert not hasattr(marie.agent, "AgentExecutor")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("marie.agent.executor")
    assert AgentExecutor.__module__ == "marie.executor.agent.agent_executor"
    assert AgentExecutor.agent_run_endpoint.__requests__ == {'on': '/agent/run'}


@pytest.mark.asyncio
async def test_agent_run_routes_scheduler_request_to_embedded_plugin() -> None:
    executor, invocation = plugin_executor()

    docs = await executor.agent_run_endpoint(
        DocList[TextDoc](),
        parameters=plugin_parameters(),
    )

    response = json.loads(docs[0].text)
    assert response == {
        'agent_ref': 'fixture.echo',
        'result': {'proposal': {'decision': 'repair'}},
        'frames': [
            {'type': 'stream', 'data': {'proposal': {'decision': 'repair'}}},
            {'type': 'end', 'data': {}},
        ],
        'request_id': 'task-1',
        'trace_id': 'trace-1',
    }
    assert executor._backend is None

    package, action, payload = invocation.await_args.args
    assert package == 'marie/fixture-agent'
    assert action == 'run'
    assert payload == {
        'agent_ref': 'fixture.echo',
        'input': {'finding': 'missing value'},
        'artifacts': {'schema_uri': 's3://bucket/schema.json'},
        'idempotency_key': 'effect-1',
    }
    assert invocation.await_args.kwargs == {
        'execution_metadata': {
            'dag_id': 'dag-1',
            'task_id': 'task-1',
            'attempt': 'attempt-2',
            'job_id': 'task-1',
        },
        'request_id': 'task-1',
    }


@pytest.mark.asyncio
async def test_agent_run_rejects_unknown_route_before_plugin_invocation() -> None:
    executor, invocation = plugin_executor()

    with pytest.raises(ValueError, match='Agent route is not configured'):
        await executor.agent_run_endpoint(
            DocList[TextDoc](),
            parameters=plugin_parameters(agent_ref='fixture.unknown'),
        )

    invocation.assert_not_awaited()


@pytest.mark.parametrize(
    'request_overrides, match',
    [
        ({'package_path': '/tmp/agent.zip'}, 'extra_forbidden'),
        ({'input': {'credentials': {'api_key': 'request-secret'}}}, 'credentials'),
        ({'artifacts': {'source': '/tmp/source.json'}}, 'host filesystem'),
        ({'artifacts': {'source': '../source.json'}}, 'host filesystem'),
        ({'input': {'packagePath': '/tmp/agent.zip'}}, 'packagepath'),
    ],
)
@pytest.mark.asyncio
async def test_agent_run_rejects_request_owned_host_configuration(
    request_overrides: dict,
    match: str,
) -> None:
    executor, invocation = plugin_executor()

    with pytest.raises(PydanticValidationError, match=match):
        await executor.agent_run_endpoint(
            DocList[TextDoc](),
            parameters=plugin_parameters(**request_overrides),
        )

    invocation.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_run_enforces_route_model_profiles() -> None:
    executor, invocation = plugin_executor(
        route={
            'package': 'marie/fixture-agent',
            'action': 'run',
            'model_profile': {
                'name': 'repair-model',
                'model': 'fixture-model',
                'base_url': 'http://model.test/v1',
            },
        }
    )

    with pytest.raises(ValueError, match='Model profile is not allowed'):
        await executor.agent_run_endpoint(
            DocList[TextDoc](),
            parameters=plugin_parameters(model_profile='request-model'),
        )
    invocation.assert_not_awaited()

    await executor.agent_run_endpoint(
        DocList[TextDoc](),
        parameters=plugin_parameters(),
    )
    assert invocation.await_args.args[2]['model_profile'] == {
        'name': 'repair-model',
        'model': 'fixture-model',
        'base_url': 'http://model.test/v1',
        'request_timeout_seconds': 300.0,
    }


@pytest.mark.asyncio
async def test_agent_run_resolves_current_job_workspace(
    tmp_path,
) -> None:
    executor, invocation = plugin_executor(
        route={
            'package': 'marie/fixture-agent',
            'action': 'run',
            'requires_workspace': True,
        },
    )
    docs_from_asset = patch(
        'marie.utils.docs.docs_from_asset',
        return_value=(['asset-doc'], '/tmp/source.tif'),
    )
    frames_from_docs = patch(
        'marie.utils.docs.frames_from_docs',
        return_value=['frame'],
    )
    prepare_asset_directory = patch(
        'marie.utils.asset_util.prepare_asset_directory',
        return_value=(str(tmp_path), str(tmp_path / 'frames'), 'metadata.json'),
    )
    parameters = plugin_parameters()
    parameters.update({'ref_id': 'document-1', 'ref_type': 'extract'})

    with (
        docs_from_asset as load,
        frames_from_docs as frames,
        prepare_asset_directory as prepare,
    ):
        await executor.agent_run_endpoint(
            DocList[AssetKeyDoc](
                [AssetKeyDoc(asset_key='s3://bucket/source.tif', pages=[0])]
            ),
            parameters=parameters,
        )

    assert invocation.await_args.args[2]['workspace'] == {
        'root': str(tmp_path.resolve()),
        'access': 'read_only',
    }
    load.assert_called_once_with(
        's3://bucket/source.tif',
        [0],
        return_file_path=True,
    )
    frames.assert_called_once_with(['asset-doc'])
    assert prepare.call_args.kwargs['ref_id'] == 'document-1'
    assert prepare.call_args.kwargs['ref_type'] == 'extract'
    assert prepare.call_args.kwargs['restore_dirs'] == [
        'agent-output',
        'parsed-result',
        'work',
    ]


@pytest.mark.asyncio
async def test_agent_run_requires_scheduler_task_identity_match() -> None:
    executor, invocation = plugin_executor()
    parameters = plugin_parameters()
    parameters['node_task_id'] = 'different-task'

    with pytest.raises(ValueError, match='node_task_id must match job_id'):
        await executor.agent_run_endpoint(
            DocList[AssetKeyDoc](),
            parameters=parameters,
        )

    invocation.assert_not_awaited()


def test_agent_route_must_reference_allowlisted_plugin_action() -> None:
    with pytest.raises(ValueError, match='action is not configured'):
        AgentExecutor(
            enable_conversation_store=False,
            plugins=[
                {
                    'package': 'marie/fixture-agent',
                    'path': '/not-read-during-routing.zip',
                    'actions': ['run'],
                }
            ],
            agent_routes={
                'fixture.echo': {
                    'package': 'marie/fixture-agent',
                    'action': 'request-selected-action',
                }
            },
        )


def test_agent_executor_closes_owned_embedded_plugins() -> None:
    executor, _ = plugin_executor()
    with patch.object(executor.embedded_plugins, 'close') as close:
        executor.close()
    close.assert_called_once_with()


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
