import asyncio
import io
import json
import os
import socket
from typing import Any

import yaml
from docarray import DocList

from marie.api.docs import AssetKeyDoc
from marie.executor.marie_executor import MarieExecutor
from marie.runtime import requests
from marie.storage import StorageManager
from marie.utils.asset_util import s3_asset_path
from marie.utils.docs import docs_from_asset, frames_from_docs


def _write_json(uri: str, value: dict[str, Any]) -> None:
    if not StorageManager.write(
        io.BytesIO(json.dumps(value).encode()), uri, overwrite=True
    ):
        raise IOError(f'Failed to upload mock handoff artifact: {uri}')


def _asset_root(parameters: dict[str, Any]) -> str:
    if parameters['ref_type'] != 'mock_llm_handoff':
        raise ValueError(
            'Use ref_type=mock_llm_handoff and a fresh ref_id for this test'
        )
    return s3_asset_path(parameters['ref_id'], parameters['ref_type'])


class LlmHandoffExecutorMock(MarieExecutor):
    """Prepare synthetic OCR and verify persisted output from the real LLM annotator."""

    @requests(on='/mock/llm/prepare', request_schema=DocList[AssetKeyDoc])
    async def prepare(
        self, docs: DocList[AssetKeyDoc], parameters: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        root = _asset_root(parameters)
        uri = f"{root}/{parameters['ref_id']}.meta.json"
        if await asyncio.to_thread(StorageManager.exists, uri):
            previous = json.loads(await asyncio.to_thread(StorageManager.read, uri))
            if previous.get('mock_handoff', {}).get('dag_id') != parameters['dag_id']:
                raise ValueError('Use a fresh ref_id for each handoff DAG')
        if len(docs) != 1 or docs[0].pages:
            raise ValueError('The handoff test requires one document with all pages')
        source = await asyncio.to_thread(docs_from_asset, docs[0].asset_key, None)
        frames = frames_from_docs(source)
        if not frames:
            raise ValueError('The handoff test document has no pages')
        metadata = {
            'pages': len(frames),
            'ocr': [
                {
                    'meta': {
                        'page': page,
                        'imageSize': {
                            'width': frame.shape[1],
                            'height': frame.shape[0],
                        },
                        'lines': [],
                        'lines_bboxes': [],
                    },
                    'lines': [],
                    'words': [],
                }
                for page, frame in enumerate(frames)
            ],
            'mock_handoff': {'dag_id': parameters['dag_id']},
        }
        await asyncio.to_thread(_write_json, uri, metadata)
        return {'status': 'success', 'metadata_uri': uri, 'pages': len(frames)}

    @requests(on='/mock/llm/read', request_schema=DocList[AssetKeyDoc])
    async def read(
        self, docs: DocList[AssetKeyDoc], parameters: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        root = _asset_root(parameters)
        metadata = json.loads(
            await asyncio.to_thread(
                StorageManager.read, f"{root}/{parameters['ref_id']}.meta.json"
            )
        )
        if metadata['mock_handoff']['dag_id'] != parameters['dag_id']:
            raise ValueError('Handoff metadata belongs to a different DAG')
        output_root = f'{root}/agent-output/mock-llm'
        marker = yaml.safe_load(
            await asyncio.to_thread(StorageManager.read, f'{output_root}/_SUCCESS.yaml')
        )
        expected = [f'{page:05d}.json' for page in range(1, metadata['pages'] + 1)]
        if not expected or not set(expected).issubset(marker['files']):
            raise ValueError('Annotation success marker is missing expected pages')
        annotations = []
        for name in expected:
            value = json.loads(
                await asyncio.to_thread(StorageManager.read, f'{output_root}/{name}')
            )
            if not isinstance(value, (dict, list)):
                raise ValueError(f'Annotation {name} is not structured JSON')
            annotations.append(value)
        receipt = {
            'dag_id': parameters['dag_id'],
            'job_id': parameters['job_id'],
            'reader_host': socket.gethostname(),
            'reader_pid': os.getpid(),
            'pages_read': len(annotations),
            'annotations': annotations,
        }
        uri = f'{root}/handoff-receipt.json'
        await asyncio.to_thread(_write_json, uri, receipt)
        return {'status': 'success', 'receipt_uri': uri, 'pages_read': len(annotations)}
