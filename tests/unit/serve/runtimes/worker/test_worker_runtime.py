import asyncio
import multiprocessing
import socket
import time
from multiprocessing import Process
from threading import Event

import grpc
import pytest
import requests as req

from docarray import Document
from marie import DocumentArray, Executor, requests
from marie.clients.request import request_generator
from marie.parsers import set_pod_parser
from marie.proto import jina_pb2, jina_pb2_grpc
from marie.serve.networking import GrpcConnectionPool
from marie.serve.runtimes.asyncio import AsyncNewLoopRuntime
from marie.serve.runtimes.worker import WorkerRuntime
from marie.serve.runtimes.worker.request_handling import WorkerRequestHandler


@pytest.mark.slow
@pytest.mark.timeout(5)
def test_worker_runtime():
    args = set_pod_parser().parse_args([])

    cancel_event = multiprocessing.Event()

    def start_runtime(args, cancel_event):
        with WorkerRuntime(args, cancel_event=cancel_event) as runtime:
            runtime.run_forever()

    runtime_thread = Process(
        target=start_runtime,
        args=(args, cancel_event),
        daemon=True,
    )
    runtime_thread.start()

    assert AsyncNewLoopRuntime.wait_for_ready_or_shutdown(
        timeout=5.0,
        ctrl_address=f'{args.host}:{args.port}',
        ready_or_shutdown_event=Event(),
    )

    target = f'{args.host}:{args.port}'
    with grpc.insecure_channel(
        target,
        options=GrpcConnectionPool.get_default_grpc_options(),
    ) as channel:
        stub = jina_pb2_grpc.JinaSingleDataRequestRPCStub(channel)
        response, call = stub.process_single_data.with_call(_create_test_data_message())

    cancel_event.set()
    runtime_thread.join()

    assert response

    assert not AsyncNewLoopRuntime.is_ready(f'{args.host}:{args.port}')
