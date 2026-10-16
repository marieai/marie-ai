from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip('marie', reason='Requires the existing Marie host environment')

from marie.executor.text.text_extraction_executor import TextExtractionExecutor
from marie.extraction import FormatRouter
from marie.job.common import JobInfo, JobStatus
from marie.job.gateway_job_distributor import GatewayJobDistributor
from marie.job.job_manager import generate_job_id
from marie.query_planner.base import QueryPlanRegistry
from marie.query_planner.ocr_planner import query_planner_extract
from marie.scheduler.models import JobSubmissionModel, WorkInfo, WorkState
from marie.scheduler.planner_util import query_plan_work_items

PROFILE = Path(__file__).resolve().parents[2] / 'config/preparation-profile.json'


@pytest.mark.parametrize(
    'extension,content_type',
    [
        ('pdf', 'application/pdf'),
        ('jpg', 'image/jpeg'),
        ('png', 'image/png'),
    ],
)
def test_preparation_profile_reaches_actual_executor_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extension: str,
    content_type: str,
) -> None:
    packet = json.loads(PROFILE.read_text())
    message = packet['parameters']['invoke_action']
    metadata = message['metadata']
    identity = {
        'run_id': 'preparation-001',
        'source_sha256': 'a' * 64,
        'config_sha256': 'b' * 64,
    }
    uri = f's3://marie/parsebench-preparation/input.{extension}'
    metadata.update(
        {
            'project_id': 'parsebench',
            'uri': uri,
            'ref_id': 'preparation-001-' + 'a' * 64,
            'content_type': content_type,
            'queue_id': 'parsebench-preparation',
            'pages': '0',
            'benchmark': identity,
        }
    )
    original = copy.deepcopy(message)
    JobSubmissionModel(**message)
    monkeypatch.chdir(tmp_path)
    if 'extract' not in QueryPlanRegistry.list_planners():
        QueryPlanRegistry.register('extract', query_planner_extract)
    now = datetime.now(timezone.utc)
    work = WorkInfo(
        id=generate_job_id(),
        name=message['name'],
        data=message,
        state=WorkState.CREATED,
        retry_limit=0,
        retry_delay=0,
        retry_backoff=False,
        start_after=now,
        expire_in_seconds=0,
        keep_until=now + timedelta(days=1),
    )
    plan, nodes = query_plan_work_items(work)
    extraction_nodes = [
        node
        for node in nodes
        if node.data['metadata']['on'] == 'extract_executor://document/extract'
    ]
    assert len(extraction_nodes) == 1
    node = extraction_nodes[0]
    job_data = copy.deepcopy(node.data)
    job_data.update({'dag_id': work.id, 'node_task_id': node.id})
    job = JobInfo(
        status=JobStatus.PENDING,
        entrypoint=node.data['metadata']['on'],
        metadata=job_data,
    )
    parameters, asset = asyncio.run(
        GatewayJobDistributor()._build_payload(node.id, job)
    )

    assert TextExtractionExecutor._parse_mode(parameters) == 'ocr'
    assert FormatRouter().route(extension, 'ocr', ocr_supported=True) == 'ocr'
    assert asset.asset_key == uri
    assert asset.pages == [0]
    assert parameters['payload']['benchmark'] == identity
    assert parameters['payload']['return_ocr'] is True
    assert parameters['payload']['format'] == 'xywh'
    assert parameters['ref_id'] == metadata['ref_id']
    assert parameters['ref_type'] == 'parsebench-preparation'
    assert parameters['dag_id'] == work.id
    assert parameters['node_task_id'] == node.id
    assert parameters['queue_id'] == 'parsebench-preparation'
    assert message == original


def test_parse_mode_in_op_params_does_not_select_ocr() -> None:
    assert (
        TextExtractionExecutor._parse_mode(
            {'payload': {'op_params': {'parse_mode': 'ocr'}}}
        )
        is None
    )


def test_source_page_order_survives_payload_conversion() -> None:
    job = JobInfo(
        status=JobStatus.PENDING,
        entrypoint='extract_executor://document/extract',
        metadata={
            'metadata': {
                'uri': 's3://marie/parsebench-preparation/multipage.pdf',
                'ref_id': 'multipage',
                'ref_type': 'parsebench-preparation',
                'pages': '2,0',
                'run_params': {'parse_mode': 'ocr'},
            }
        },
    )
    parameters, asset = asyncio.run(
        GatewayJobDistributor()._build_payload('prep-pages', job)
    )
    assert asset.pages == [2, 0]
    assert parameters['payload']['pages'] == '2,0'
    assert TextExtractionExecutor._parse_mode(parameters) == 'ocr'
