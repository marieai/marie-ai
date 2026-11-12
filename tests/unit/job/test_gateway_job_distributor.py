from __future__ import annotations

from marie.job.common import JobInfo, JobStatus
from marie.job.gateway_job_distributor import GatewayJobDistributor


async def test_build_payload_preserves_typed_upstream_and_content_reference():
    reference = {'kind': 'extraction', 'ref_id': 'upstream', 'ref_type': 'document'}
    upstream = {'output': {'records': [{'moves': ['e4']}]}}
    info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint='vector_store_executor://embed_and_store',
        metadata={
            'metadata': {
                'uri': 's3://fixture/one.pdf',
                'input_ref': reference,
                'upstream': upstream,
            },
        },
    )
    parameters, _ = await GatewayJobDistributor()._build_payload('job', info)
    assert parameters['input_ref'] == reference
    assert parameters['upstream'] == upstream


async def test_build_payload_preserves_registered_item_identity() -> None:
    identity = dict(
        index_id='index',
        item_id='opaque:one',
        kb_indexing_dag_id='attempt',
        owner_type='record',
        owner_id='one',
    )
    info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint='vector_store_executor://embed_and_store',
        metadata={
            'metadata': {'uri': 's3://fixture/one.pdf', **identity},
        },
    )
    parameters, _ = await GatewayJobDistributor()._build_payload('job', info)
    assert {key: parameters.get(key) for key in identity} == identity


async def test_build_payload_passes_mock_failure_controls_to_executor_parameters() -> (
    None
):
    distributor = GatewayJobDistributor()
    job_info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint="mock_executor:///document/process",
        metadata={
            "metadata": {
                "uri": "s3://marie/extract/sample.tif",
                "ref_id": "sample",
                "ref_type": "extract",
                "failure_rate": 0.5,
                "failure_mode": "timeout",
                "force_fail": True,
                "randomize_time": True,
            }
        },
    )

    parameters, asset_doc = await distributor._build_payload("job-a", job_info)

    assert asset_doc.asset_key == "s3://marie/extract/sample.tif"
    assert parameters["job_id"] == "job-a"
    assert parameters["failure_rate"] == 0.5
    assert parameters["failure_mode"] == "timeout"
    assert parameters["force_fail"] is True
    assert parameters["randomize_time"] is True


async def test_build_payload_preserves_datasource_for_retry() -> None:
    distributor = GatewayJobDistributor()
    payload = {
        "uri": "s3://marie/extract/sample.tif",
        "ref_id": "sample",
        "ref_type": "extract",
    }
    job_info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint="mock_executor:///document/process",
        metadata={"metadata": payload},
    )

    first_parameters, first_asset = await distributor._build_payload("job-a", job_info)
    second_parameters, second_asset = await distributor._build_payload(
        "job-a", job_info
    )

    assert payload["uri"] == "s3://marie/extract/sample.tif"
    assert first_asset.asset_key == second_asset.asset_key == payload["uri"]
    assert "uri" not in first_parameters["payload"]
    assert "uri" not in second_parameters["payload"]


async def test_build_payload_forwards_submission_context_on_each_attempt() -> None:
    distributor = GatewayJobDistributor()
    carrier = {
        'traceparent': '00-00000000000000000000000000000001-0000000000000002-00',
        'tracestate': 'vendor=value',
    }
    job_info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint='mock_executor:///document/process',
        metadata={
            'metadata': {'uri': 's3://marie/extract/sample.tif'},
            'submission_trace_context': carrier,
            'dag_id': 'dag-1',
        },
    )
    for _ in range(2):
        parameters, _ = await distributor._build_payload('job-a', job_info)
        assert parameters['submission_trace_context'] == carrier
        assert parameters['job_id'] == 'job-a'
        assert parameters['dag_id'] == 'dag-1'
        assert 'submission_trace_context' not in parameters['payload']


def test_resolve_endpoint_accepts_mock_annotator_llm_route() -> None:
    distributor = GatewayJobDistributor(
        deployment_nodes={
            "annotator_llm": [
                {"endpoint": "/annotator/llm"},
            ],
        }
    )

    assert distributor._resolve_endpoint(
        "job-a",
        "annotator_llm://annotator/llm",
    ) == ("annotator_llm", "/annotator/llm")


async def test_build_payload_passes_run_attempt_identity() -> None:
    distributor = GatewayJobDistributor()
    job_info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint="guardrail_executor://evaluate",
        metadata={
            "metadata": {
                "uri": "s3://marie/extract/sample.json",
                "ref_id": "sample",
                "ref_type": "extract",
            },
            "dag_id": "00000000-0000-0000-0000-000000000201",
            "node_task_id": "00000000-0000-0000-0000-000000000101",
            "run_owner": "scheduler-1",
            "run_attempt_id": "00000000-0000-0000-0000-000000000301",
        },
    )

    parameters, _ = await distributor._build_payload("job-a", job_info)

    assert parameters["run_owner"] == "scheduler-1"
    assert parameters["run_attempt_id"] == job_info.metadata["run_attempt_id"]


def test_resolve_endpoint_routes_connector_to_plugin_daemon_execute() -> None:
    # W2: a CONNECTOR node serializes to the FIXED endpoint
    # plugin_daemon_executor://execute (identity rides in params, not the path).
    distributor = GatewayJobDistributor(
        deployment_nodes={
            "plugin_daemon_executor": [
                {"endpoint": "/execute"},
            ],
        }
    )

    assert distributor._resolve_endpoint(
        "job-a",
        "plugin_daemon_executor://execute",
    ) == ("plugin_daemon_executor", "/execute")


def test_resolve_endpoint_routes_agent_application() -> None:
    distributor = GatewayJobDistributor(
        deployment_nodes={
            "agent_executor": [
                {"endpoint": "/agent/run"},
            ],
        }
    )

    assert distributor._resolve_endpoint(
        "job-a",
        "agent_executor://agent/run",
    ) == ("agent_executor", "/agent/run")


async def test_repository_refresh_dispatches_saved_source_without_file_payload():
    metadata = {
        'planner': 'kb_repository_refresh',
        'tenant_id': 'tenant',
        'index_id': 'index',
        'source_id': 'source',
        'organization_id': 'org',
        'workspace_id': 'workspace',
    }
    info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint='repository_executor://refresh_repository',
        metadata={
            'metadata': metadata,
            'dag_id': 'dag',
            'node_task_id': 'job',
            'run_owner': 'scheduler',
            'run_attempt_id': 'attempt',
        },
    )
    parameters, asset = await GatewayJobDistributor()._build_payload('job', info)
    assert asset is None
    assert parameters['payload'] == metadata
    assert parameters['source_id'] == 'source'
    assert parameters['dag_id'] == 'dag'
    assert (
        parameters['run_owner'] == 'scheduler'
        and parameters['run_attempt_id'] == 'attempt'
    )


async def test_parameter_only_jobs_do_not_require_a_file_payload():
    metadata = {'operation': 'calculate', 'run_params': {'values': [1, 2, 3]}}
    info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint='arbitrary_executor://run',
        metadata={'metadata': metadata},
    )
    parameters, asset = await GatewayJobDistributor()._build_payload('job', info)
    assert asset is None
    assert parameters['payload'] == metadata
    assert parameters['run_params'] == metadata['run_params']


async def test_declared_file_input_is_still_parsed_for_any_executor(tmp_path):
    import pytest

    info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint='arbitrary_executor://run',
        metadata={'metadata': {'srcFile': str(tmp_path / 'missing.pdf')}},
    )
    with pytest.raises(Exception, match='File not found'):
        await GatewayJobDistributor()._build_payload('job', info)


async def test_executor_parameters_are_opaque_and_do_not_need_registration():
    import copy

    metadata = {
        'application_id': 'app',
        'settings': {'format': 'pgn', 'moves': ['e4', 'e5']},
        'enabled': False,
        'limit': 0,
        'custom_items': [],
    }
    original = copy.deepcopy(metadata)
    info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint='arbitrary_executor://run',
        metadata={'metadata': metadata},
    )
    parameters, asset = await GatewayJobDistributor()._build_payload('job', info)
    assert asset is None
    assert {key: parameters.get(key) for key in metadata} == metadata
    assert parameters['payload'] == original and metadata == original


async def test_runtime_attempt_identity_overrides_application_parameters():
    runtime = {
        'dag_id': 'dag',
        'node_task_id': 'node',
        'run_owner': 'scheduler',
        'run_attempt_id': 'attempt',
    }
    metadata = {
        'job_id': 'forged',
        'payload': 'forged',
        **{key: 'forged' for key in runtime},
    }
    info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint='arbitrary_executor://run',
        metadata={'metadata': metadata, **runtime},
    )
    parameters, _ = await GatewayJobDistributor()._build_payload('job', info)
    assert parameters['job_id'] == 'job'
    assert {key: parameters[key] for key in runtime} == runtime
    assert parameters['payload'] == metadata
    info.metadata = {'metadata': metadata}
    parameters, _ = await GatewayJobDistributor()._build_payload('job', info)
    assert all(key not in parameters for key in runtime)


async def test_binary_asset_is_materialized_without_forwarding_inline_bytes():
    import base64

    metadata = {
        'srcBase64': base64.b64encode(b'%PDF-1.4\nfixture').decode(),
        'custom_option': 7,
    }
    info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint='arbitrary_executor://run',
        metadata={'metadata': metadata},
    )
    parameters, asset = await GatewayJobDistributor()._build_payload('job', info)
    assert parameters['custom_option'] == 7
    assert parameters['uri'] == asset.asset_key
    assert 'srcBase64' not in parameters and 'srcBase64' not in parameters['payload']
    assert 'srcBase64' in metadata


async def test_application_parameters_cannot_preseed_runtime_response_storage():
    for value in ['invalid', {'forged-worker': {'status': 'success'}}]:
        info = JobInfo(
            status=JobStatus.PENDING,
            entrypoint='arbitrary_executor://run',
            metadata={'metadata': {'__results__': value, 'setting': 'opaque'}},
        )
        parameters, _ = await GatewayJobDistributor()._build_payload('job', info)
        assert '__results__' not in parameters
        assert parameters['setting'] == 'opaque'
