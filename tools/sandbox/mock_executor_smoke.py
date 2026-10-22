#!/usr/bin/env python3
"""Check a forwarded mock executor endpoint with an actual data request."""

import argparse
import json
import math
from typing import Any

import grpc
from docarray import DocList

from marie.api.docs import AssetKeyDoc
from marie.proto import jina_pb2, jina_pb2_grpc
from marie.types_core.request.data import DataRequest


def probe(address: str, timeout: float, force_fail: bool = False) -> dict[str, Any]:
    """Return a setup result; transport success alone does not count as execution."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    request = DataRequest()
    request.header.exec_endpoint = '/document/process'
    request.header.target_executor = 'mock_executor_a'
    request.document_array_cls = DocList[AssetKeyDoc]
    request.data.docs = DocList[AssetKeyDoc]([AssetKeyDoc(asset_key='mock://setup')])
    job_id = request.header.request_id
    request.parameters = {
        'job_id': job_id,
        'setup_probe': 'sandbox-mock',
        'force_fail': force_fail,
    }
    with grpc.insecure_channel(address) as channel:
        response = jina_pb2_grpc.JinaSingleDataRequestRPCStub(
            channel
        ).process_single_data(request, timeout=timeout)
    if response.header.status.code != jina_pb2.StatusProto.SUCCESS:
        return {'ok': False, 'job_id': job_id, 'error': 'executor_response_failed'}
    results = response.parameters.get('__results__')
    if isinstance(results, dict) and len(results) == 1:
        result = next(iter(results.values()))
        if isinstance(result, dict):
            executor = result.get('executor')
            parameters = result.get('parameters')
            if (
                isinstance(executor, str)
                and (
                    executor == 'mock_executor_a'
                    or executor.startswith('mock_executor_a/')
                )
                and result.get('processed_docs') == 1
                and isinstance(parameters, dict)
                and parameters.get('job_id') == job_id
                and parameters.get('setup_probe') == 'sandbox-mock'
                and parameters.get('force_fail') is force_fail
            ):
                return {
                    'ok': True,
                    'job_id': job_id,
                    'executor': executor,
                    'processed_docs': 1,
                }
    return {'ok': False, 'job_id': job_id, 'error': 'mock_result_missing_or_invalid'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--address',
        required=True,
        help='host:port, normally a local kubectl port-forward',
    )
    parser.add_argument('--timeout', type=float, default=10)
    parser.add_argument(
        '--force-fail',
        action='store_true',
        help='Exercise the negative check; expect exit 1',
    )
    args = parser.parse_args()
    try:
        result = probe(args.address, args.timeout, args.force_fail)
    except grpc.RpcError as error:
        result = {'ok': False, 'error': f'rpc_{error.code().name.lower()}'}
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
