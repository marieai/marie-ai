import logging
import os
from collections.abc import Iterator
from unittest.mock import Mock

import httpx
import pytest
from marie.engine import get_engine
from marie.engine import openai_engine as engine_module
from marie.engine.batch_processor import BatchProcessor
from marie.engine.completion_contract import CompletionCallParams
from marie.engine.llm_queue.config import LlmQueueProducerConfig
from marie.engine.llm_queue.store import StoreUnavailable
from marie.engine.openai_engine import OpenAIEngine
from openai import AsyncOpenAI, AuthenticationError

from marie.excepts import BatchExecutionError
from marie.extract.annotators import util

QUEUE_URL = 'redis://127.0.0.1:1/0'
PROVIDER_URL = 'http://127.0.0.1:1/v1'


@pytest.fixture(autouse=True)
def isolated_setup(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    util.close_all_engines()
    for name in (
        'OPENAI_API_KEY',
        'OPENAI_API_BASE',
        'OPENAI_BASE_URL',
        'LLM_QUEUE_ENABLED',
        'LLM_QUEUE_URL',
        'LLM_QUEUE_VALKEY_URL',
        'LLM_QUEUE_FABRIC_GROUP_ID',
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    util.close_all_engines()


@pytest.fixture
def no_provider(monkeypatch: pytest.MonkeyPatch) -> Mock:
    builder = Mock(
        side_effect=AssertionError('queued mode must not build a provider client')
    )
    monkeypatch.setattr(engine_module, 'build_async_openai_client', builder)
    return builder


def _config(
    enabled: bool = True, queue_url: str | None = QUEUE_URL
) -> LlmQueueProducerConfig:
    return LlmQueueProducerConfig.from_env(
        enabled=enabled, queue_url=queue_url, fabric_group_id='fabric'
    )


def _call() -> CompletionCallParams:
    return CompletionCallParams(model='model', messages=[])


@pytest.mark.parametrize('entry', ['bare', 'factory'])
def test_queued_engine_never_builds_provider_client(
    monkeypatch: pytest.MonkeyPatch, no_provider: Mock, entry: str
) -> None:
    monkeypatch.setenv('LLM_QUEUE_ENABLED', 'true')
    monkeypatch.setenv('LLM_QUEUE_URL', QUEUE_URL)
    monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', 'fabric')
    runner = Mock(
        side_effect=AssertionError('queued close must not create a direct runner')
    )
    producer = Mock(
        side_effect=AssertionError('construction must not create a producer')
    )
    monkeypatch.setattr(BatchProcessor, '_get_direct_runner', runner)
    monkeypatch.setattr(BatchProcessor, '_get_queued_executor', producer)

    engine = (
        OpenAIEngine('model')
        if entry == 'bare'
        else util.route_llm_engine('model', False)
    )
    try:
        assert engine.uses_queued_dispatch
        assert engine.client is None
        assert engine.batch_processor.client is None
        assert engine.complete_many([]) == []
    finally:
        engine.close()
        engine.close()

    no_provider.assert_not_called()
    runner.assert_not_called()
    producer.assert_not_called()


@pytest.mark.parametrize('entry', ['bare', 'factory', 'processor'])
def test_configuration_is_resolved_once_at_the_owning_boundary(
    monkeypatch: pytest.MonkeyPatch, no_provider: Mock, entry: str
) -> None:
    monkeypatch.setenv('LLM_QUEUE_ENABLED', 'true')
    monkeypatch.setenv('LLM_QUEUE_URL', QUEUE_URL)
    monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', 'fabric')
    original = LlmQueueProducerConfig.from_env
    snapshots = []

    def resolve(**kwargs: object) -> LlmQueueProducerConfig:
        snapshot = original(**kwargs)
        snapshots.append(snapshot)
        return snapshot

    resolver = Mock(side_effect=resolve)
    monkeypatch.setattr(LlmQueueProducerConfig, 'from_env', resolver)
    if entry == 'processor':
        owned = BatchProcessor(None, 'model', logging.getLogger(__name__))
        processor = owned
    else:
        owned = (
            OpenAIEngine('model')
            if entry == 'bare'
            else util.route_llm_engine('model', False)
        )
        processor = owned.batch_processor
    try:
        resolver.assert_called_once()
        assert processor._queue_config is snapshots[0]
        assert processor.client is None
    finally:
        owned.close()
    no_provider.assert_not_called()


@pytest.mark.parametrize('enabled', [False, True])
def test_supplied_snapshot_selects_mode_without_resolving_again(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    snapshot = _config(enabled)
    monkeypatch.setenv('LLM_QUEUE_ENABLED', 'false' if enabled else 'true')
    resolver = Mock(
        side_effect=AssertionError('supplied snapshot must not be re-resolved')
    )
    monkeypatch.setattr(LlmQueueProducerConfig, 'from_env', resolver)
    builder = Mock(wraps=engine_module.build_async_openai_client)
    monkeypatch.setattr(engine_module, 'build_async_openai_client', builder)
    engine = OpenAIEngine(
        'model',
        api_key='synthetic-test-key',
        base_url=PROVIDER_URL,
        queue_config=snapshot,
    )
    try:
        assert engine.uses_queued_dispatch is enabled
        assert engine.batch_processor._queue_config is snapshot
        if enabled:
            assert engine.client is None
            builder.assert_not_called()
        else:
            builder.assert_called_once_with(
                api_key='synthetic-test-key', base_url=PROVIDER_URL
            )
            assert engine.client.max_retries == 0
    finally:
        engine.close()
    resolver.assert_not_called()


@pytest.mark.parametrize('enabled', [False, True])
def test_explicit_queue_mode_overrides_environment(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    monkeypatch.setenv('LLM_QUEUE_ENABLED', 'false' if enabled else 'true')
    builder = Mock(wraps=engine_module.build_async_openai_client)
    monkeypatch.setattr(engine_module, 'build_async_openai_client', builder)
    engine = OpenAIEngine(
        'model',
        api_key='synthetic-test-key',
        base_url=PROVIDER_URL,
        queue_enabled=enabled,
        queue_url=QUEUE_URL,
        queue_fabric_group_id='fabric',
    )
    try:
        assert engine.uses_queued_dispatch is enabled
        assert builder.call_count == (0 if enabled else 1)
    finally:
        engine.close()


@pytest.mark.parametrize('target', ['engine', 'processor'])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('queue_enabled', False),
        ('queue_enabled', True),
        ('queue_url', ''),
        ('queue_valkey_url', ''),
        ('queue_fabric_group_id', ''),
    ],
)
def test_snapshot_and_explicit_queue_options_conflict_before_resources(
    no_provider: Mock, target: str, enabled: bool, name: str, value: object
) -> None:
    kwargs = {'queue_config': _config(enabled), name: value}
    with pytest.raises(ValueError, match='^queue_configuration_conflict$'):
        if target == 'engine':
            OpenAIEngine('model', **kwargs)
        else:
            BatchProcessor(None, 'model', logging.getLogger(__name__), **kwargs)
    no_provider.assert_not_called()


def test_forbidden_pool_check_precedes_snapshot_conflict(no_provider: Mock) -> None:
    with pytest.raises(ValueError, match='^caller_pool_forbidden$'):
        OpenAIEngine(
            'model', pool_id='pool', queue_config=_config(), queue_enabled=False
        )
    no_provider.assert_not_called()


def test_direct_engine_keeps_missing_key_validation(no_provider: Mock) -> None:
    with pytest.raises(ValueError, match='^api_key is required for direct execution$'):
        OpenAIEngine('model', queue_enabled=False)
    no_provider.assert_not_called()


@pytest.mark.parametrize(
    ('key', 'base'), [(None, None), ('test', None), (None, PROVIDER_URL)]
)
def test_direct_factory_keeps_key_and_base_validation(
    monkeypatch: pytest.MonkeyPatch,
    no_provider: Mock,
    key: str | None,
    base: str | None,
) -> None:
    if key is not None:
        monkeypatch.setenv('OPENAI_API_KEY', key)
    if base is not None:
        monkeypatch.setenv('OPENAI_API_BASE', base)
    with pytest.raises(
        ValueError, match='^OPENAI_API_KEY and OPENAI_API_BASE must be set$'
    ):
        util.route_llm_engine('model', False)
    no_provider.assert_not_called()


@pytest.mark.parametrize('enabled', [False, True])
def test_processor_rejects_non_provider_clients_in_both_modes(enabled: bool) -> None:
    with pytest.raises(
        ValueError, match='Client must be an instance of OpenAI API client'
    ):
        BatchProcessor(
            object(), 'model', logging.getLogger(__name__), queue_enabled=enabled
        )


@pytest.mark.parametrize('snapshot', [False, True])
def test_direct_processor_rejects_absent_provider_client(snapshot: bool) -> None:
    options = {'queue_config': _config(False)} if snapshot else {'queue_enabled': False}
    with pytest.raises(
        ValueError, match='Client must be an instance of OpenAI API client'
    ):
        BatchProcessor(None, 'model', logging.getLogger(__name__), **options)


def test_queued_close_does_not_create_direct_runner(
    monkeypatch: pytest.MonkeyPatch, no_provider: Mock
) -> None:
    runner = Mock(
        side_effect=AssertionError('queued close must not create a direct runner')
    )
    monkeypatch.setattr(BatchProcessor, '_get_direct_runner', runner)
    first = OpenAIEngine('first', queue_config=_config())
    second = OpenAIEngine('second', queue_config=_config())
    producer = Mock()
    first.batch_processor._queued_executor = producer
    try:
        first.close()
        first.close()
        producer.close.assert_called_once()
        assert second.client is None
        assert second.batch_processor._queued_executor is None
        assert second.complete_many([]) == []
    finally:
        second.close()
    runner.assert_not_called()
    no_provider.assert_not_called()


def test_missing_queue_url_fails_on_first_nonempty_submission(
    no_provider: Mock,
) -> None:
    engine = OpenAIEngine('model', queue_enabled=True, queue_fabric_group_id='fabric')
    try:
        assert engine.complete_many([]) == []
        with pytest.raises(
            ValueError, match='Queued dispatch requires a configured queue URL'
        ):
            engine.complete_many([_call()])
        assert engine.batch_processor._queued_executor is None
        assert engine.batch_processor._direct_runner is None
    finally:
        engine.close()
    no_provider.assert_not_called()


def test_queue_startup_failure_never_falls_back_to_provider(
    monkeypatch: pytest.MonkeyPatch, no_provider: Mock
) -> None:
    from marie.engine.llm_queue import producer

    failure = StoreUnavailable('synthetic unavailable store')
    factory = Mock(side_effect=failure)
    monkeypatch.setattr(producer, 'QueuedDispatchProducer', factory)
    engine = OpenAIEngine('model', queue_config=_config())
    try:
        with pytest.raises(StoreUnavailable) as raised:
            engine.complete_many([_call()])
        assert raised.value is failure
        factory.assert_called_once()
        assert engine.batch_processor._direct_runner is None
    finally:
        engine.close()
    no_provider.assert_not_called()


def test_existing_positional_constructor_parameters_keep_their_meaning(
    no_provider: Mock,
) -> None:
    engine = OpenAIEngine(
        'model', 'system', False, False, None, None, 2, 3, queue_config=_config()
    )
    processor = BatchProcessor(
        None,
        'model',
        logging.getLogger(__name__),
        None,
        2,
        3,
        None,
        True,
        QUEUE_URL,
        None,
        'fabric',
    )
    try:
        assert engine.system_prompt == 'system'
        assert engine.max_concurrency == processor.max_concurrency == 2
        assert engine.batch_timeout == processor.batch_timeout == 3
        assert processor._queue_config.queue_url == QUEUE_URL
    finally:
        engine.close()
        processor.close()
    no_provider.assert_not_called()


@pytest.mark.parametrize('queued', [False, True])
def test_engine_consumes_and_retains_explicit_key_without_environment_lookup(
    monkeypatch: pytest.MonkeyPatch, queued: bool
) -> None:
    snapshot = _config(queued)
    monkeypatch.setenv('OPENAI_API_KEY', 'ambient-key-must-not-be-used')
    original_getenv = os.getenv

    def getenv(name: str, default: str | None = None) -> str | None:
        if name == 'OPENAI_API_KEY':
            raise AssertionError('engine must consume its explicit key')
        return original_getenv(name, default)

    monkeypatch.setattr(os, 'getenv', getenv)
    builder = Mock(wraps=engine_module.build_async_openai_client)
    monkeypatch.setattr(engine_module, 'build_async_openai_client', builder)
    engine = OpenAIEngine(
        'model', api_key='explicit-key', base_url=PROVIDER_URL, queue_config=snapshot
    )
    try:
        assert engine.api_key == 'explicit-key'
        if queued:
            assert engine.client is None
            builder.assert_not_called()
        else:
            builder.assert_called_once_with(
                api_key='explicit-key', base_url=PROVIDER_URL
            )
            assert engine.client.max_retries == 0
    finally:
        engine.close()


@pytest.mark.parametrize('api_key', [None, ''])
def test_direct_engine_rejects_missing_explicit_key_despite_ambient_key(
    monkeypatch: pytest.MonkeyPatch, no_provider: Mock, api_key: str | None
) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'ambient-key-must-not-be-used')
    with pytest.raises(ValueError, match='^api_key is required for direct execution$'):
        OpenAIEngine('model', api_key=api_key, queue_enabled=False)
    no_provider.assert_not_called()


@pytest.mark.parametrize('entry', ['route', 'generic'])
@pytest.mark.parametrize('queued', [False, True])
def test_factory_resolves_optional_key_once_for_both_modes(
    monkeypatch: pytest.MonkeyPatch, entry: str, queued: bool
) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'factory-key')
    monkeypatch.setenv('OPENAI_API_BASE', PROVIDER_URL)
    monkeypatch.setenv('LLM_QUEUE_ENABLED', 'true' if queued else 'false')
    monkeypatch.setenv('LLM_QUEUE_URL', QUEUE_URL)
    monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', 'fabric')
    getter = Mock(wraps=os.environ.get)
    monkeypatch.setattr(os.environ, 'get', getter)
    engine = (
        util.route_llm_engine('model', False)
        if entry == 'route'
        else get_engine('model', provider='openai', base_url=PROVIDER_URL)
    )
    try:
        assert engine.api_key == 'factory-key'
        assert (engine.client is None) is queued
        key_reads = [
            call for call in getter.call_args_list if call.args[0] == 'OPENAI_API_KEY'
        ]
        assert len(key_reads) == 1
    finally:
        engine.close()


@pytest.mark.parametrize('api_key', [None, 'explicit-key'])
def test_generic_factory_preserves_explicit_key_without_environment_fallback(
    monkeypatch: pytest.MonkeyPatch, no_provider: Mock, api_key: str | None
) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'ambient-key-must-not-be-used')
    getter = Mock(wraps=os.environ.get)
    monkeypatch.setattr(os.environ, 'get', getter)
    engine = get_engine(
        'model', provider='openai', api_key=api_key, queue_config=_config()
    )
    try:
        assert engine.api_key == api_key
        assert engine.client is None
        assert not any(
            call.args[0] == 'OPENAI_API_KEY' for call in getter.call_args_list
        )
    finally:
        engine.close()
    no_provider.assert_not_called()


@pytest.mark.parametrize('queued', [False, True])
def test_route_cache_separates_resolved_key_and_direct_endpoint(
    monkeypatch: pytest.MonkeyPatch, queued: bool
) -> None:
    monkeypatch.setenv('LLM_QUEUE_ENABLED', 'true' if queued else 'false')
    monkeypatch.setenv('LLM_QUEUE_URL', QUEUE_URL)
    monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', 'fabric')
    monkeypatch.setenv('OPENAI_API_BASE', PROVIDER_URL)
    monkeypatch.setenv('OPENAI_API_KEY', 'first-key')
    first = util.route_llm_engine('model', False)
    assert util.route_llm_engine('model', False) is first
    monkeypatch.setenv('OPENAI_API_KEY', 'second-key')
    second = util.route_llm_engine('model', False)
    assert second is not first
    assert first.api_key == 'first-key'
    assert second.api_key == 'second-key'
    assert 'first-key' not in repr(tuple(util._engine_cache))
    assert 'second-key' not in repr(tuple(util._engine_cache))
    monkeypatch.setenv('OPENAI_API_KEY', 'first-key')
    assert util.route_llm_engine('model', False) is first
    monkeypatch.setenv('OPENAI_API_BASE', 'http://127.0.0.1:2/v1')
    endpoint_changed = util.route_llm_engine('model', False)
    assert (endpoint_changed is first) is queued


@pytest.mark.parametrize('status', [200, 401])
def test_direct_completion_uses_client_key_without_ambient_credentials(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if status == 401:
            return httpx.Response(
                401,
                json={
                    'error': {'message': 'invalid key', 'type': 'invalid_request_error'}
                },
            )
        return httpx.Response(
            200,
            json={
                'id': 'direct-test',
                'object': 'chat.completion',
                'created': 0,
                'model': 'model',
                'choices': [
                    {
                        'index': 0,
                        'message': {'role': 'assistant', 'content': 'direct result'},
                        'finish_reason': 'stop',
                    }
                ],
            },
        )

    client = AsyncOpenAI(
        api_key='explicit-key',
        base_url=PROVIDER_URL,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        max_retries=0,
    )
    builder = Mock(return_value=client)
    monkeypatch.setattr(engine_module, 'build_async_openai_client', builder)
    engine = OpenAIEngine(
        'model', api_key='explicit-key', base_url=PROVIDER_URL, queue_enabled=False
    )
    try:
        assert 'OPENAI_API_KEY' not in os.environ
        if status == 200:
            assert engine.complete_many([_call()]) == ['direct result']
        else:
            with pytest.raises(BatchExecutionError) as raised:
                engine.complete_many([_call()])
            assert isinstance(raised.value.primary_error, AuthenticationError)
        builder.assert_called_once_with(api_key='explicit-key', base_url=PROVIDER_URL)
        assert len(requests) == (1 if status == 200 else 3)
        assert all(
            request.headers['authorization'] == 'Bearer explicit-key'
            for request in requests
        )
    finally:
        engine.close()
