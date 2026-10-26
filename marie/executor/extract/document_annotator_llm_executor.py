from __future__ import annotations

import asyncio
import copy
import io
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Union

import torch
from docarray import DocList
from jsonschema import SchemaError
from jsonschema.validators import validator_for
from marie.engine.llm_queue.config import llm_queue_enabled
from omegaconf import OmegaConf

from marie.api.docs import DOC_KEY_PAGE_NUMBER, AssetKeyDoc
from marie.executor.extract import document_annotator_executor as annotation_runtime
from marie.executor.extract.document_annotator_executor import (
    DocumentAnnotatorExecutor,
)
from marie.executor.extract.util import layout_config
from marie.extract.annotators.llm_annotator import LLMAnnotator
from marie.extract.annotators.refinement_provider import ProcessingKey
from marie.extract.readers.meta_reader.meta_reader import MetaReader
from marie.extract.registry.loader import initialize_components_from_config
from marie.logging_core.logger import MarieLogger
from marie.logging_core.mdc import MDC
from marie.logging_core.predefined import default_logger as logger
from marie.models.utils import torch_gc
from marie.prompt import PromptTemplate
from marie.runtime import requests
from marie.storage import StorageManager
from marie.utils.asset_util import prepare_asset_directory, s3_asset_path
from marie.utils.docs import docs_from_asset, frames_from_docs
from marie.utils.json import load_json_file


def _load_schema_definition(
    root_config_dir: str, layout: str, key: str
) -> tuple[dict[str, Any], PromptTemplate, dict[str, Any]]:
    if (
        not isinstance(layout, str)
        or not re.fullmatch(r'[\w.-]+', layout)
        or layout in {'.', '..'}
    ):
        raise ValueError('Schema extraction requires a valid configured layout')
    if (
        not isinstance(key, str)
        or not re.fullmatch(r'[\w.-]+', key)
        or key in {'.', '..'}
    ):
        raise ValueError('Schema extraction requires a configured annotator key')
    cfg = layout_config(root_config_dir, layout)
    if key not in cfg.annotators:
        raise ValueError(f'Invalid annotator key: {key}')
    conf = OmegaConf.to_container(cfg.annotators[key], resolve=True)
    conf['name'] = key
    if not conf.get('enabled', True):
        raise ValueError(f"Schema extraction annotator '{key}' is disabled")
    model_config = conf.get('model_config', {})
    if model_config.get('expect_output') != 'json':
        raise ValueError('Schema extraction requires model_config.expect_output: json')
    prompt_dir = Path(root_config_dir) / f'TID-{layout}' / 'annotator'
    prompt = PromptTemplate.from_file_with_fallback(
        model_config.get('prompt_path'),
        prompt_dir=str(prompt_dir),
        layout_id=layout,
        name=f'{key}-prompt',
    )
    resolved_prompt = Path(prompt.file_path)
    base_dir = Path(root_config_dir) / 'base'
    relative_prompt = resolved_prompt.relative_to(
        prompt_dir if resolved_prompt.is_relative_to(prompt_dir) else base_dir
    )
    companion = relative_prompt.with_suffix('.output.json')
    candidates = [prompt_dir / companion, base_dir / companion]
    schema_path = next((path for path in candidates if path.is_file()), None)
    if schema_path is None:
        raise ValueError(f'Missing saved output schema; expected {candidates[0]}')
    try:
        schema_config = load_json_file(str(schema_path))
        schema = schema_config.get('schema', schema_config)
        if not isinstance(schema, dict) or schema.get('type') != 'object':
            raise ValueError('output schema must describe an object')
        validator_for(schema).check_schema(schema)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        SchemaError,
        ValueError,
        AttributeError,
    ) as exc:
        raise ValueError(f'Invalid saved output schema {schema_path}: {exc}') from exc
    prompt = PromptTemplate(
        prompt.source,
        file_path=prompt.file_path,
        name=prompt.name,
        defaults={'EXTRACTOR_SCHEMA': json.dumps(schema, ensure_ascii=False)},
    )
    return conf, prompt, schema


def _read_schema_page_results(
    output_dir: str, source_pages: list[int], *, require_all_pages: bool = False
) -> list[dict[str, Any]]:
    results = []
    paths = list(Path(output_dir).glob('*.json'))
    if not paths:
        raise ValueError('Schema extraction produced no JSON output')
    keyed_paths = [(ProcessingKey.from_filename(path.name), path) for path in paths]
    keyed_paths.sort(key=lambda item: (item[0].page_number, item[0].unit_index or 0))
    for processing_key, path in keyed_paths:
        page_index = processing_key.page_number - 1
        if not 0 <= page_index < len(source_pages):
            raise ValueError(f'Invalid extraction output page in {path.name}')
        try:
            data = load_json_file(str(path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f'Invalid extraction JSON output {path.name}: {exc}'
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(
                f'Extraction output {path.name} must be a schema-shaped object'
            )
        results.append({'page': source_pages[page_index], 'data': data})
    if require_all_pages:
        missing_pages = sorted(
            set(source_pages) - {result['page'] for result in results}
        )
        if missing_pages:
            raise ValueError(
                f'Missing extraction output for source pages: {missing_pages}'
            )
    return results


class DocumentAnnotatorLLMExecutor(DocumentAnnotatorExecutor):
    """Executor for document annotation"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, Any] = None,
        llm_tracking: dict[str, Any] = None,
        parsers: dict[str, Any] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ) -> None:

        kwargs['storage'] = storage
        kwargs['llm_tracking'] = llm_tracking
        super().__init__(**kwargs)
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger
        queue_enabled = llm_queue_enabled()
        submission_mode = "queued-dispatch" if queue_enabled else "direct-batch"
        self.logger.info(
            f"LLM request submission mode: {submission_mode} "
            f"(LLM_QUEUE_ENABLED={str(queue_enabled).lower()})"
        )

        if parsers:
            result = initialize_components_from_config(parsers)
            logger.info(f"Loaded modules: {result['loaded']}")
            logger.info(f"Failed modules: {result['failed']}")
            logger.info(
                f"Total context providers: {result.get('total_context_providers', 0)}"
            )

        logger.info(f"Started executor : {self.__class__.__name__}")

    def deployment_status_details(self) -> dict[str, Any]:
        enabled = llm_queue_enabled()
        return {
            "llm_dispatch": {
                "enabled": enabled,
                "mode": "queued-dispatch" if enabled else "direct-batch",
            }
        }

    @requests(on='/annotator/schema')
    async def annotator_schema(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ) -> dict[str, Any]:
        """Extract using the mounted prompt/schema profile and persist this DAG's result."""
        from marie.extract.extractor.schema_extraction_result import (
            build_schema_extraction_result,
        )

        setup_error = self._setup_request(docs, parameters)
        if setup_error is not None:
            raise ValueError(str(setup_error))
        try:
            dag_id = parameters.get('dag_id')
            if not isinstance(dag_id, str) or not re.fullmatch(r'[\w-]+', dag_id):
                raise ValueError('Schema extraction requires a valid dag_id')
            ref_id, ref_type = parameters.get('ref_id'), parameters.get('ref_type')
            if (
                ref_type != 'extractor_document'
                or not isinstance(ref_id, str)
                or not ref_id
                or Path(ref_id).name != ref_id
                or '\\' in ref_id
                or '\x00' in ref_id
            ):
                raise ValueError(
                    'Schema extraction requires extractor_document identity'
                )
            payload = parameters.get('payload') or {}
            op_params = payload.get('op_params') or {}
            if set(op_params) - {'layout', 'key'}:
                raise ValueError(
                    'Schema extraction accepts only saved layout/key operator parameters'
                )
            layout, key = op_params.get('layout'), op_params.get('key')
            conf, prompt, schema = _load_schema_definition(
                self.root_config_dir, layout, key
            )
            source_doc = docs[0]
            source_docs, local_path = docs_from_asset(
                source_doc.asset_key, source_doc.pages, return_file_path=True
            )
            frames = frames_from_docs(source_docs)
            source_pages = [doc.tags[DOC_KEY_PAGE_NUMBER] for doc in source_docs]
            root_asset_dir, _, metadata_file = prepare_asset_directory(
                frames=frames,
                local_path=local_path,
                ref_id=ref_id,
                ref_type=ref_type,
                logger=self.logger,
            )
            metadata = load_json_file(metadata_file)
            document = MetaReader.from_data(
                frames=frames,
                ocr_meta=metadata['ocr'],
                unstructured_meta={
                    'ref_id': ref_id,
                    'ref_type': ref_type,
                    'job_id': parameters['job_id'],
                    'requested_pages': source_doc.pages,
                    'source_metadata': metadata,
                },
            )
            run_context = None
            if annotation_runtime.MARIE_KERNEL_AVAILABLE:
                run_context = annotation_runtime.RunContext(
                    annotation_runtime.TaskInstanceRef(
                        tenant_id=str(
                            parameters.get('project_id')
                            or payload.get('project_id')
                            or 'default'
                        ),
                        dag_name='schema_extract',
                        dag_id=dag_id,
                        task_id=key,
                        try_number=1,
                    ),
                    annotation_runtime.FileSystemStateBackend(base_path=root_asset_dir),
                    parameters=op_params,
                )
            with TemporaryDirectory(prefix=f'marie-schema-{dag_id}-') as run_dir:
                output_dir = Path(run_dir) / key
                annotator = LLMAnnotator(
                    working_dir=root_asset_dir,
                    output_dir=str(output_dir),
                    prompt_dir=str(
                        Path(self.root_config_dir) / f'TID-{layout}' / 'annotator'
                    ),
                    prompt_template=prompt,
                    annotator_conf=conf,
                    layout_conf={'layout_id': layout},
                    run_context=run_context,
                    job_id=parameters['job_id'],
                    dag_id=dag_id,
                    node_task_id=parameters.get('node_task_id'),
                    ref_id=ref_id,
                    ref_type=ref_type,
                    requested_pages=source_doc.pages,
                )
                await annotator.aannotate(
                    document, frames if annotator.requires_frames else []
                )
                page_results = _read_schema_page_results(
                    str(output_dir),
                    source_pages,
                    require_all_pages=annotator.processing_mode == 'per-page',
                )
                absolute_ocr = copy.deepcopy(metadata['ocr'])
                for page, ocr_page in zip(source_pages, absolute_ocr):
                    ocr_page['meta']['page'] = page
                result = await asyncio.to_thread(
                    build_schema_extraction_result,
                    schema,
                    page_results,
                    absolute_ocr,
                    dag_id,
                    annotator.model_name,
                )
                result_bytes = json.dumps(result, ensure_ascii=False).encode('utf-8')
                result_uri = (
                    f'{s3_asset_path(ref_id, ref_type)}/extraction/{dag_id}.json'
                )
                written = await asyncio.to_thread(
                    StorageManager.write,
                    io.BytesIO(result_bytes),
                    result_uri,
                    overwrite=True,
                )
                if written is not True:
                    raise RuntimeError(
                        f'Failed to persist extraction result to {result_uri}'
                    )
            return {
                'status': 'success',
                'runtime_info': self.runtime_info,
                'dag_id': dag_id,
                'result_uri': result_uri,
                'error': None,
            }
        finally:
            torch_gc()
            MDC.remove('request_id')

    @requests(on="/annotator/llm")
    async def annotator_llm(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        """
        Document annotator executor
        Much of this is hardcoded in here and need to be moved into proper pipeline.

        EXAMPLE USAGE

            As Executor

            .. code-block:: python

                exec = AnnotatorExecutor()

        :param parameters:
        :param docs: Documents to process
        :param kwargs:
        :return:
        """
        try:
            return await self._process_annotation_request(
                docs, parameters, LLMAnnotator, *args, **kwargs
            )

        except Exception as e:
            # Since we are not throwing an exception the job will be marked as successful
            # we log the error and return an error message, later we can improve this to mark the job as failed
            # by introducing an improved error handling mechanism
            self.logger.error(f"Error during annotation: {e}")
            raise e
            return {'status': 'error', 'message': str(e)}
