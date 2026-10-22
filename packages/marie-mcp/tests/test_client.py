import json

import httpx
import pytest
from marie.mcp.server.clients.marie_client import MarieClient, MarieClientError

JOB_ID = '019ce927-9621-7000-8000-000000000001'


@pytest.fixture
def client():
    return MarieClient(base_url='http://gateway', api_key='test-key')


async def install_transport(client, handler):
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={'Authorization': f'Bearer {client.api_key}'},
    )


@pytest.mark.asyncio
async def test_submit_matches_gateway_envelope(client):
    def gateway(request):
        action = json.loads(request.content)['parameters']['invoke_action']
        assert action['metadata']['ref_id'] == 'Invoice.PDF'
        assert action['api_key'] == 'test-key'
        assert request.headers['authorization'] == 'Bearer test-key'
        return httpx.Response(200, json={'parameters': {'job_id': JOB_ID}})

    await install_transport(client, gateway)
    async with client:
        assert await client.submit_job('extract', {'ref_id': 'Invoice.PDF'}) == {
            'job_id': JOB_ID
        }


@pytest.mark.asyncio
async def test_lost_submission_response_is_not_retried(client):
    requests = []

    def gateway(request):
        requests.append(request)
        raise httpx.ReadTimeout('accepted but reply lost', request=request)

    await install_transport(client, gateway)
    async with client:
        with pytest.raises(MarieClientError):
            await client.submit_job('extract', {})
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_status_uses_operational_detail(client):
    def gateway(request):
        assert request.url.path == f'/api/operations/dags/{JOB_ID}'
        return httpx.Response(
            200, json={'status': 'OK', 'result': {'state': 'completed'}}
        )

    await install_transport(client, gateway)
    async with client:
        assert (await client.get_job_status(JOB_ID))['result']['state'] == 'completed'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'body',
    [
        {'status': 'error', 'message': 'invalid state'},
        [],
        {'parameters': {'status': 'error', 'job_id': JOB_ID}},
    ],
)
async def test_application_errors_are_not_success(client, body):
    await install_transport(client, lambda _: httpx.Response(200, json=body))
    async with client:
        with pytest.raises(MarieClientError):
            await client.submit_job('extract', {})


@pytest.mark.asyncio
async def test_reads_send_auth_and_debug_scope():
    requests = []
    async with MarieClient(base_url='http://gateway', api_key='test-key') as client:
        # Keep the real configured headers while replacing only the transport.
        client._client._transport = httpx.MockTransport(
            lambda request: (
                requests.append(request) or httpx.Response(200, json={'status': 'OK'})
            )
        )
        await client.get_deployments()
        await client.get_capacity()
        await client.list_jobs('active')
        await client.health_check()
        await client.get_debug_info(fabric_group_id='fabric-a')
    assert all(r.headers.get('authorization') == 'Bearer test-key' for r in requests)
    assert requests[-1].url.params['fabric_group_id'] == 'fabric-a'


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['stop_job', 'delete_job'])
async def test_unsupported_mutations_do_not_call_gateway(client, method):
    def unexpected(_):
        pytest.fail('unsupported mutation sent a request')

    await install_transport(client, unexpected)
    async with client:
        with pytest.raises(MarieClientError, match='not supported'):
            await getattr(client, method)(JOB_ID)


@pytest.mark.asyncio
async def test_auth_failure_is_not_retried(client):
    requests = []
    await install_transport(
        client,
        lambda r: (
            requests.append(r) or httpx.Response(401, json={'detail': 'unauthorized'})
        ),
    )
    async with client:
        with pytest.raises(MarieClientError, match='401'):
            await client.list_jobs()
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_transient_read_failure_is_retried(client):
    requests = []

    def gateway(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, json={'detail': 'unavailable'})
        return httpx.Response(200, json={'status': 'OK', 'result': []})

    await install_transport(client, gateway)
    async with client:
        assert (await client.list_jobs())['result'] == []
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_status_falls_back_for_child_job_uuid(client):
    paths = []

    def gateway(request):
        paths.append(request.url.path)
        if '/dags/' in request.url.path:
            return httpx.Response(404, json={'detail': 'DAG not found'})
        return httpx.Response(
            200, json={'status': 'OK', 'result': {'state': 'completed'}}
        )

    await install_transport(client, gateway)
    async with client:
        assert (await client.get_job_status(JOB_ID))['result']['state'] == 'completed'
    assert paths == [f'/api/operations/dags/{JOB_ID}', f'/api/operations/jobs/{JOB_ID}']


@pytest.mark.asyncio
async def test_status_does_not_mask_authorization_failure(client):
    paths = []
    await install_transport(
        client,
        lambda r: (
            paths.append(r.url.path)
            or httpx.Response(403, json={'detail': 'forbidden'})
        ),
    )
    async with client:
        with pytest.raises(MarieClientError, match='403'):
            await client.get_job_status(JOB_ID)
    assert paths == [f'/api/operations/dags/{JOB_ID}']
