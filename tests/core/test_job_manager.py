import asyncio
import sys

import pytest

from marie_server.job.common import JobStatus
from marie_server.job.job_manager import JobManager
from marie_server.storage.in_memory import InMemoryKV
from marie_server.storage.psql import PostgreSQLKV
from tests.core.test_utils import async_wait_for_condition_async_predicate, async_delay

import multiprocessing
import time
from dataclasses import dataclass

import pytest

from marie import Document, DocumentArray, Executor, requests, Deployment
from marie.excepts import ExecutorError
from marie.serve.runtimes.asyncio import AsyncNewLoopRuntime
from marie.serve.runtimes.servers import BaseServer
from marie.serve.runtimes.worker.request_handling import WorkerRequestHandler
from marie.serve.runtimes.gateway.streamer import GatewayStreamer
from marie.types.request import Request
from marie.types.request.data import DataRequest
from tests.helper import _generate_pod_args


async def check_job_succeeded(job_manager, job_id):
    data = await job_manager.get_job_info(job_id)
    status = data.status
    if status == JobStatus.FAILED:
        raise RuntimeError(f"Job failed! {data.message}")
    assert status in {JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUCCEEDED}
    return status == JobStatus.SUCCEEDED


async def check_job_failed(job_manager, job_id):
    status = await job_manager.get_job_status(job_id)
    assert status in {JobStatus.PENDING, JobStatus.RUNNING, JobStatus.FAILED}
    return status == JobStatus.FAILED


async def check_job_stopped(job_manager, job_id):
    status = await job_manager.get_job_status(job_id)
    assert status in {JobStatus.PENDING, JobStatus.RUNNING, JobStatus.STOPPED}
    return status == JobStatus.STOPPED


async def check_job_running(job_manager, job_id):
    status = await job_manager.get_job_status(job_id)
    assert status in {JobStatus.PENDING, JobStatus.RUNNING}
    return status == JobStatus.RUNNING


@pytest.mark.asyncio
@pytest.fixture
async def job_manager(tmp_path):
    storage = InMemoryKV()

    storage_config = {"hostname": "127.0.0.1", "port": 5432, "username": "postgres", "password": "123456",
                      "database": "postgres",
                      "default_table": "kv_store_a", "max_pool_size": 5,
                      "max_connections": 5}

    storage = PostgreSQLKV(config=storage_config, reset=True)
    yield JobManager(storage)


@pytest.mark.asyncio
async def test_list_jobs_empty(job_manager: JobManager):
    assert await job_manager.list_jobs() == dict()


@pytest.mark.asyncio
async def test_list_jobs(job_manager: JobManager):
    await job_manager.submit_job(entrypoint="echo hi", submission_id="1")

    runtime_env = {"env_vars": {"TEST": "123"}}
    metadata = {"foo": "bar"}
    await job_manager.submit_job(
        entrypoint="echo hello",
        submission_id="2",
        runtime_env=runtime_env,
        metadata=metadata,
    )

    _ = asyncio.create_task(async_delay(update_job_status(job_manager, "1", JobStatus.SUCCEEDED), 1))
    _ = asyncio.create_task(async_delay(update_job_status(job_manager, "2", JobStatus.SUCCEEDED), 1))

    await async_wait_for_condition_async_predicate(
        check_job_succeeded, job_manager=job_manager, job_id="1"
    )
    await async_wait_for_condition_async_predicate(
        check_job_succeeded, job_manager=job_manager, job_id="2"
    )
    jobs_info = await job_manager.list_jobs()
    assert "1" in jobs_info
    assert jobs_info["1"].status == JobStatus.SUCCEEDED

    assert "2" in jobs_info
    assert jobs_info["2"].status == JobStatus.SUCCEEDED
    assert jobs_info["2"].message is not None
    assert jobs_info["2"].end_time >= jobs_info["2"].start_time
    assert jobs_info["2"].runtime_env == runtime_env
    assert jobs_info["2"].metadata == metadata


async def update_job_status(job_manager, job_id, job_status):
    await job_manager.job_info_client().put_status(job_id, job_status)


@pytest.mark.asyncio
async def test_pass_job_id(job_manager):
    submission_id = "my_custom_id"

    returned_id = await job_manager.submit_job(
        entrypoint="echo hello", submission_id=submission_id
    )
    assert returned_id == submission_id

    _ = asyncio.create_task(async_delay(update_job_status(job_manager, submission_id, JobStatus.SUCCEEDED), 1))

    await async_wait_for_condition_async_predicate(
        check_job_succeeded, job_manager=job_manager, job_id=submission_id
    )

    # Check that the same job_id is rejected.
    with pytest.raises(ValueError):
        await job_manager.submit_job(
            entrypoint="echo hello", submission_id=submission_id
        )


@pytest.mark.asyncio
async def test_simultaneous_submit_job(job_manager):
    """Test that we can submit multiple jobs at once."""
    job_ids = await asyncio.gather(
        job_manager.submit_job(entrypoint="echo hello"),
        job_manager.submit_job(entrypoint="echo hello"),
        job_manager.submit_job(entrypoint="echo hello"),
    )

    for job_id in job_ids:
        _ = asyncio.create_task(async_delay(update_job_status(job_manager, job_id, JobStatus.SUCCEEDED), 1))

        await async_wait_for_condition_async_predicate(
            check_job_succeeded, job_manager=job_manager, job_id=job_id
        )


@pytest.mark.asyncio
async def test_simultaneous_with_same_id(job_manager):
    """Test that we can submit multiple jobs at once with the same id.

    The second job should raise a friendly error.
    """
    with pytest.raises(ValueError) as excinfo:
        await asyncio.gather(
            job_manager.submit_job(entrypoint="echo hello", submission_id="1"),
            job_manager.submit_job(entrypoint="echo hello", submission_id="1"),
        )
    assert "Job with submission_id 1 already exists" in str(excinfo.value)

    # Check that the (first) job can still succeed.
    _ = asyncio.create_task(async_delay(update_job_status(job_manager, "1", JobStatus.SUCCEEDED), 1))

    await async_wait_for_condition_async_predicate(
        check_job_succeeded, job_manager=job_manager, job_id="1"
    )


class StreamerTestExecutor(Executor):
    @requests
    def foo(self, docs, parameters, **kwargs):
        text_to_add = parameters.get('text_to_add', 'default ')
        for doc in docs:
            doc.text += text_to_add


def _create_worker_runtime(port, uses, name=''):
    args = _generate_pod_args()
    args.port = [port]
    args.name = name
    args.uses = uses
    with AsyncNewLoopRuntime(args, req_handler_cls=WorkerRequestHandler) as runtime:
        runtime.run_forever()


def _setup(pod0_port, pod1_port):
    pod0_process = multiprocessing.Process(
        target=_create_worker_runtime, args=(pod0_port, 'StreamerTestExecutor')
    )
    pod0_process.start()

    pod1_process = multiprocessing.Process(
        target=_create_worker_runtime, args=(pod1_port, 'StreamerTestExecutor')
    )
    pod1_process.start()

    assert BaseServer.wait_for_ready_or_shutdown(
        timeout=5.0,
        ctrl_address=f'0.0.0.0:{pod0_port}',
        ready_or_shutdown_event=multiprocessing.Event(),
    )
    assert BaseServer.wait_for_ready_or_shutdown(
        timeout=5.0,
        ctrl_address=f'0.0.0.0:{pod1_port}',
        ready_or_shutdown_event=multiprocessing.Event(),
    )
    return pod0_process, pod1_process


@pytest.mark.parametrize(
    'parameters, target_executor, expected_text',
    [  # (None, None, 'default default '),
        ({'pod0__text_to_add': 'param_pod0 '}, None, 'param_pod0 default '),
        (None, 'pod1', 'default '),
        ({'pod0__text_to_add': 'param_pod0 '}, 'pod0', 'param_pod0 '),
    ],
)
@pytest.mark.parametrize('results_in_order', [False, True])
@pytest.mark.asyncio
async def test_gateway_job_manager(
        port_generator, parameters, target_executor, expected_text, results_in_order
):
    pod0_port = port_generator()
    pod1_port = port_generator()
    pod0_process, pod1_process = _setup(pod0_port, pod1_port)
    graph_description = {
        "start-gateway": ["pod0"],
        "pod0": ["pod1"],
        "pod1": ["end-gateway"],
    }
    pod_addresses = {"pod0": [f"0.0.0.0:{pod0_port}"], "pod1": [f"0.0.0.0:{pod1_port}"]}
    # send requests to the gateway
    gateway_streamer = GatewayStreamer(
        graph_representation=graph_description, executor_addresses=pod_addresses
    )

    try:
        input_da = DocumentArray.empty(60)
        resp = DocumentArray.empty(0)
        num_resp = 0
        async for r in gateway_streamer.stream_docs(
                docs=input_da,
                request_size=10,
                parameters=parameters,
                target_executor=target_executor,
                results_in_order=results_in_order,
        ):
            num_resp += 1
            resp.extend(r)

        assert num_resp == 6
        assert len(resp) == 60
        for doc in resp:
            assert doc.text == expected_text

        request = DataRequest()
        request.data.docs = DocumentArray.empty(60)
        unary_response = await gateway_streamer.process_single_data(request=request)
        assert len(unary_response.docs) == 60

    except Exception:
        assert False
    finally:  # clean up runtimes
        pod0_process.terminate()
        pod1_process.terminate()
        pod0_process.join()
        pod1_process.join()
        await gateway_streamer.close()


@pytest.mark.asyncio
async def test_deployment_job_manager():
    class PIDExecutor(Executor):

        @requests
        def foo(self, docs, **kwargs):
            import os
            for doc in docs:
                # print(f"{os.getpid()} : {doc.id}")
                doc.tags['pid'] = os.getpid()

    # dep = Deployment(uses=PIDExecutor, include_gateway=True, shards=1, replicas=3)
    dep = Deployment(uses=PIDExecutor, include_gateway=False, replicas=4)

    # dep.to_docker_compose_yaml('/tmp/marieai/docker-compose.yml')

    # create a graph from the deployment
    graph_description = {}
    pod_addresses = {}

    gateway_streamer = GatewayStreamer(
        graph_representation=graph_description, executor_addresses=pod_addresses
    )

    request = DataRequest()
    request.data.docs = DocumentArray.empty(5)
    unary_response = await gateway_streamer.process_single_data(request=request)

    print(unary_response)
    # assert len(unary_response.docs) == 60
    with dep:
        docs = dep.post(on='/', inputs=DocumentArray.empty(20), request_size=1)

        dep.block()

    returned_pids = set([doc.tags['pid'] for doc in docs])
    print(returned_pids)
    assert len(returned_pids) == 1


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
