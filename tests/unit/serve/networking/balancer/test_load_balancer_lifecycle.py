import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import pytest
from grpc.aio import AioRpcError
from marie.engine.circuit_breaker import CircuitBreakerConfig

from marie.excepts import EstablishGrpcConnectionError, InternalNetworkError
from marie.serve.networking import GrpcConnectionPool
from marie.serve.networking.balancer.least_connection_balancer import (
    LeastConnectionsLoadBalancer,
)
from marie.serve.networking.balancer.load_balancer import LoadBalancer
from marie.serve.networking.balancer.round_robin_balancer import RoundRobinLoadBalancer
from marie.serve.networking.replica_list import _ReplicaList


@pytest.fixture(params=[RoundRobinLoadBalancer, LeastConnectionsLoadBalancer])
def balancer(request):
    lb = request.param('executor', Mock())
    lb.update_connections([SimpleNamespace(address='a'), SimpleNamespace(address='b')])
    return lb


class Channel:
    def __init__(self):
        self.closing = asyncio.Event()
        self.gate = None
        self.closed = False

    async def close(self, grace):
        self.closing.set()
        if self.gate is not None:
            await self.gate.wait()
        self.closed = True


@pytest.fixture
def replicas(monkeypatch):
    channels = []

    def create(address, **kwargs):
        channel = Channel()
        channels.append(channel)
        connection = SimpleNamespace(
            address=address, deployment_name=kwargs['deployment_name'], channel=channel
        )
        return connection, channel

    monkeypatch.setattr(
        'marie.serve.networking.replica_list.create_async_channel_stub', create
    )
    replicas = _ReplicaList(
        None, Mock(), 'review', load_balancer_type='least_connection'
    )
    replicas.add_connection('a', 'executor')
    replicas.add_connection('b', 'executor')
    replicas.created_channels = channels
    return replicas


async def test_selection_excludes_tried_addresses(balancer):
    selected = await balancer.get_next_connection(exclude_addresses={'a'})
    assert selected.address == 'b'
    with pytest.raises(EstablishGrpcConnectionError):
        await balancer.get_next_connection(exclude_addresses={'a', 'b'})


async def test_acquisition_counts_and_release_is_exactly_once(balancer):
    first = await balancer.acquire_connection(exclude_addresses={'b'})
    second = await balancer.acquire_connection(exclude_addresses={'b'})
    assert balancer.get_active_count('a') == 2
    balancer.release_connection(first)
    balancer.release_connection(first)
    assert balancer.get_active_count('a') == 1
    balancer.release_connection(second)
    assert balancer.get_active_count('a') == 0


async def test_outstanding_usage_survives_remove_and_readd(balancer):
    old = await balancer.acquire_connection(exclude_addresses={'b'})
    b = balancer._connections[1]
    balancer.update_connections([b])
    assert balancer.get_active_counter() == {'b': 0}
    balancer.update_connections([SimpleNamespace(address='a'), b])
    new = await balancer.acquire_connection(exclude_addresses={'b'})
    assert balancer.get_active_count('a') == 2
    balancer.release_connection(old)
    assert balancer.get_active_count('a') == 1
    balancer.release_connection(old)
    assert balancer.get_active_count('a') == 1
    balancer.release_connection(new)


async def test_release_does_not_recreate_removed_address(balancer):
    lease = await balancer.acquire_connection(exclude_addresses={'b'})
    balancer.update_connections([SimpleNamespace(address='b')])
    balancer.release_connection(lease)
    assert balancer.get_active_counter() == {'b': 0}


async def test_legacy_release_does_not_recreate_removed_address(balancer):
    balancer.incr_usage('a')
    balancer.update_connections([SimpleNamespace(address='b')])
    balancer.decr_usage('a')
    assert balancer.get_active_counter() == {'b': 0}


async def test_close_rejects_selection_and_acquisition(balancer):
    lease = await balancer.acquire_connection()
    balancer.close()
    balancer.release_connection(lease)
    assert balancer.connection_count() == 0
    assert balancer.get_active_counter() == {}
    with pytest.raises(EstablishGrpcConnectionError):
        await balancer.get_next_connection()
    with pytest.raises(EstablishGrpcConnectionError):
        await balancer.acquire_connection()


@pytest.mark.parametrize('operation', ['remove', 'reset'])
async def test_membership_published_before_channel_close(replicas, operation):
    old = replicas.get_all_connections()[0]
    channel = old.channel
    channel.gate = asyncio.Event()
    if operation == 'remove':
        task = asyncio.create_task(replicas.remove_connection('a'))
    else:
        task = asyncio.create_task(replicas.reset_connection('a', 'executor'))
    await channel.closing.wait()
    try:
        selected = await replicas.get_next_connection()
        assert selected is not old
        assert selected.address == ('b' if operation == 'remove' else 'a')
    finally:
        channel.gate.set()
        await task


async def test_one_owned_channel_per_connection(replicas):
    assert len(replicas.created_channels) == 2
    await replicas.close()
    assert all(channel.closed for channel in replicas.created_channels)


async def test_replica_close_publishes_before_await_and_prevents_add(replicas):
    channel = replicas.get_all_connections()[0].channel
    channel.gate = asyncio.Event()
    task = asyncio.create_task(replicas.close())
    await channel.closing.wait()
    try:
        with pytest.raises(EstablishGrpcConnectionError):
            await replicas.get_next_connection()
        with pytest.raises(EstablishGrpcConnectionError):
            replicas.add_connection('c', 'executor')
    finally:
        channel.gate.set()
        await task
    await replicas.close()


@pytest.mark.parametrize('kind', ['batch', 'stream'])
async def test_retry_uses_busy_untried_replica_without_spinning(replicas, kind):
    a, b = replicas.get_all_connections()
    replicas.incr_usage('b')
    error = AioRpcError(grpc.StatusCode.UNAVAILABLE, (), (), 'not the leader', 'test')

    async def fail(**kwargs):
        raise error

    async def success(**kwargs):
        assert replicas.get_load_balancer().get_active_count('b') == 2
        return 'response', 'metadata'

    async def fail_stream(**kwargs):
        raise error
        yield

    async def success_stream(**kwargs):
        assert replicas.get_load_balancer().get_active_count('b') == 2
        yield 'response', 'metadata'

    a.send_requests, b.send_requests = fail, success
    a.send_single_doc_request, b.send_single_doc_request = fail_stream, success_stream
    original = replicas.get_next_connection
    selections = 0

    async def bounded_selection(**kwargs):
        nonlocal selections
        selections += 1
        if selections > 8:
            raise RuntimeError(
                'retry selection spun instead of excluding tried address'
            )
        return await original(**kwargs)

    replicas.get_next_connection = bounded_selection
    pool = object.__new__(GrpcConnectionPool)
    pool._logger = Mock()
    pool.compression = None
    request = SimpleNamespace(request_id='test')
    if kind == 'batch':
        result = await pool._send_requests([request], replicas, retries=1)
        assert result == ('response', 'metadata')
    else:
        result = [
            item
            async for item in pool._send_single_doc_request(
                request, replicas, retries=1
            )
        ]
        assert result == [('response', 'metadata')]
    assert replicas.get_load_balancer().get_active_counter() == {'a': 0, 'b': 1}


@pytest.mark.parametrize(
    'policy', [RoundRobinLoadBalancer, LeastConnectionsLoadBalancer]
)
async def test_open_circuit_rejected_by_default(policy):
    lb = policy(
        'executor',
        Mock(),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=3600
        ),
    )
    lb.update_connections([SimpleNamespace(address='a')])
    lb.record_failure('a')
    with pytest.raises(EstablishGrpcConnectionError):
        await lb.get_next_connection()


@pytest.mark.parametrize(
    'policy', [RoundRobinLoadBalancer, LeastConnectionsLoadBalancer]
)
async def test_half_open_admits_one_probe_and_releases_on_completion(policy):
    lb = policy(
        'executor',
        Mock(),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0, half_open_max_calls=1
        ),
    )
    lb.update_connections([SimpleNamespace(address='a')])
    lb.record_failure('a')
    first = await lb.acquire_connection()
    with pytest.raises(EstablishGrpcConnectionError):
        await lb.acquire_connection()
    lb.release_connection(first)
    next_probe = await lb.acquire_connection()
    lb.release_connection(next_probe)
    assert lb.get_active_count('a') == 0


@pytest.mark.parametrize(
    'status',
    [
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.PERMISSION_DENIED,
        grpc.StatusCode.CANCELLED,
    ],
)
async def test_client_errors_do_not_open_circuit(replicas, status):
    from marie.engine.circuit_breaker import CircuitBreaker

    lb = replicas.get_load_balancer()
    lb._circuit_breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=1), Mock()
    )
    pool = object.__new__(GrpcConnectionPool)
    pool._logger = Mock()
    error = AioRpcError(status, (), (), 'request error', 'test')
    await pool._handle_aiorpcerror(
        error,
        current_address='a',
        current_deployment='executor',
        connection_list=replicas,
    )
    assert lb.is_connection_available('a')


async def test_disabled_debug_does_not_format_connection_state():
    class Address(str):
        def __repr__(self):
            raise AssertionError('hot path formatted the entire state')

    logger = Mock(debug_enabled=False)
    lb = LeastConnectionsLoadBalancer('executor', logger)
    address = Address('a')
    lb.update_connections([SimpleNamespace(address=address)])
    connection = await lb.get_next_connection()
    lb.incr_usage(connection.address)
    lb.decr_usage(connection.address)
    assert lb.get_active_count(address) == 0


def test_unsupported_policy_is_rejected():
    with pytest.raises(NotImplementedError):
        LoadBalancer.create_load_balancer('consistent_hashing', 'executor', Mock())


async def test_fail_open_is_explicit_and_never_bypasses_probe_limit():
    config = CircuitBreakerConfig(
        failure_threshold=1, recovery_timeout=3600, fail_open=True
    )
    lb = LeastConnectionsLoadBalancer('executor', Mock(), circuit_breaker_config=config)
    lb.update_connections([SimpleNamespace(address='a')])
    lb.record_failure('a')
    fallback = await lb.acquire_connection()
    lb.release_connection(fallback)
    config.recovery_timeout = 0
    probe = await lb.acquire_connection()
    with pytest.raises(EstablishGrpcConnectionError):
        await lb.acquire_connection()
    lb.release_connection(probe)


async def test_old_probe_release_does_not_release_new_probe():
    config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0)
    lb = LeastConnectionsLoadBalancer('executor', Mock(), circuit_breaker_config=config)
    lb.update_connections([SimpleNamespace(address='a')])
    lb.record_failure('a')
    old = await lb.acquire_connection()
    lb.record_failure('a', lease=old)
    new = await lb.acquire_connection()
    lb.release_connection(old)
    with pytest.raises(EstablishGrpcConnectionError):
        await lb.acquire_connection()
    lb.release_connection(new)


async def test_old_success_does_not_close_readded_address_circuit():
    config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=3600)
    lb = LeastConnectionsLoadBalancer('executor', Mock(), circuit_breaker_config=config)
    lb.update_connections([SimpleNamespace(address='a')])
    old = await lb.acquire_connection()
    lb.update_connections([])
    lb.update_connections([SimpleNamespace(address='a')])
    lb.record_failure('a')
    lb.record_success('a', lease=old)
    assert not lb.is_connection_available('a')
    assert lb.get_circuit_breaker_stats()['a']['total_successes'] == 0
    lb.release_connection(old)


async def test_acquisition_rolls_back_if_interceptor_fails(balancer):
    interceptor = Mock()
    interceptor.on_connection_acquired.side_effect = ValueError('trace failed')
    balancer.tracing_interceptors = [interceptor]
    with pytest.raises(ValueError):
        await balancer.acquire_connection()
    assert balancer.get_active_counter() == {'a': 0, 'b': 0}


async def test_concurrent_acquisitions_balance_outstanding_load(balancer):
    leases = await asyncio.gather(*(balancer.acquire_connection() for _ in range(20)))
    assert balancer.get_active_counter() == {'a': 10, 'b': 10}
    for lease in leases:
        balancer.release_connection(lease)
    assert balancer.get_active_counter() == {'a': 0, 'b': 0}


@pytest.mark.parametrize('kind', ['batch', 'stream', 'discovery'])
async def test_cancellation_releases_usage_and_probe(replicas, kind):
    from marie.engine.circuit_breaker import CircuitBreaker

    await replicas.remove_connection('b')
    lb = replicas.get_load_balancer()
    lb._circuit_breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0), Mock()
    )
    lb.record_failure('a')
    connection = replicas.get_all_connections()[0]
    started = asyncio.Event()

    async def wait(**kwargs):
        started.set()
        await asyncio.Event().wait()

    async def wait_stream(**kwargs):
        await wait(**kwargs)
        yield

    connection.send_requests = wait
    connection.send_single_doc_request = wait_stream
    connection.send_discover_endpoint = wait
    pool = object.__new__(GrpcConnectionPool)
    pool._logger = Mock()
    pool.compression = None
    request = SimpleNamespace(request_id='test')
    if kind == 'batch':
        task = pool._send_requests([request], replicas, retries=0)
    elif kind == 'discovery':
        task = asyncio.create_task(pool._send_discover_endpoint(replicas, retries=0))
    else:

        async def consume():
            async for _ in pool._send_single_doc_request(request, replicas, retries=0):
                pass

        task = asyncio.create_task(consume())
    await started.wait()
    assert lb.get_active_count('a') == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lb.get_active_count('a') == 0
    probe = await lb.acquire_connection()
    lb.release_connection(probe)


async def test_retry_reuses_single_replica_with_bounded_attempts(replicas):
    await replicas.remove_connection('b')
    connection = replicas.get_all_connections()[0]
    attempts = 0

    async def send(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise AioRpcError(
                grpc.StatusCode.UNAVAILABLE, (), (), 'not the leader', 'test'
            )
        return 'response', None

    connection.send_requests = send
    pool = object.__new__(GrpcConnectionPool)
    pool._logger = Mock()
    pool.compression = None
    result = await pool._send_requests(
        [SimpleNamespace(request_id='test')], replicas, retries=2
    )
    assert result == ('response', None)
    assert attempts == 3
    assert replicas.get_load_balancer().get_active_count('a') == 0


async def test_closing_stream_early_releases_usage(replicas):
    connection = replicas.get_all_connections()[0]

    async def stream(**kwargs):
        yield 'response', None
        await asyncio.Event().wait()

    connection.send_single_doc_request = stream
    pool = object.__new__(GrpcConnectionPool)
    pool.compression = None
    iterator = pool._send_single_doc_request(
        SimpleNamespace(request_id='test'), replicas, retries=0
    )
    assert await anext(iterator) == ('response', None)
    assert replicas.get_load_balancer().get_active_count('a') == 1
    await iterator.aclose()
    assert replicas.get_load_balancer().get_active_count('a') == 0


async def test_cancelled_removal_still_closes_retired_channel(replicas):
    channel = replicas.get_all_connections()[0].channel
    channel.gate = asyncio.Event()
    task = asyncio.create_task(replicas.remove_connection('a'))
    await channel.closing.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not replicas.has_connection('a')
    channel.gate.set()
    await asyncio.sleep(0)
    assert channel.closed


async def test_release_interceptor_failure_does_not_replace_rpc_result(balancer):
    lease = await balancer.acquire_connection()
    interceptor = Mock()
    interceptor.on_connection_released.side_effect = ValueError('tracing unavailable')
    balancer.tracing_interceptors = [interceptor]
    balancer.release_connection(lease)
    assert balancer.get_active_counter() == {'a': 0, 'b': 0}


@pytest.mark.parametrize('kind', ['batch', 'stream', 'discovery'])
@pytest.mark.parametrize('initially_open', [True, False])
@pytest.mark.parametrize(
    'status', [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]
)
async def test_circuit_rejection_preserves_classified_network_error(
    replicas, kind, initially_open, status
):
    from marie.engine.circuit_breaker import CircuitBreaker

    await replicas.remove_connection('b')
    lb = replicas.get_load_balancer()
    lb._circuit_breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=1, recovery_timeout=3600), Mock()
    )
    if initially_open:
        lb.record_failure('a')
    root_error = AioRpcError(
        status, (), (('backend', 'executor'),), 'backend down', 'test'
    )

    async def fail(**kwargs):
        raise root_error

    async def fail_stream(**kwargs):
        raise root_error
        yield

    connection = replicas.get_all_connections()[0]
    connection.send_requests = fail
    connection.send_single_doc_request = fail_stream
    connection.send_discover_endpoint = fail
    pool = object.__new__(GrpcConnectionPool)
    pool._logger = Mock()
    pool.compression = None
    request = SimpleNamespace(request_id='test')
    if kind == 'batch':
        error = await pool._send_requests([request], replicas, retries=1)
    elif kind == 'stream':
        responses = [
            item
            async for item in pool._send_single_doc_request(
                request, replicas, retries=1
            )
        ]
        error = responses[0][0]
    else:
        with pytest.raises(InternalNetworkError) as caught:
            await pool._send_discover_endpoint(replicas, retries=1)
        error = caught.value
    assert isinstance(error, InternalNetworkError)
    assert error.code() == (grpc.StatusCode.UNAVAILABLE if initially_open else status)
    assert lb.get_active_count('a') == 0
    if not initially_open:
        assert error.og_exception is root_error
        assert error.trailing_metadata() == (('backend', 'executor'),)
        assert error.dest_addr == {'a'}
