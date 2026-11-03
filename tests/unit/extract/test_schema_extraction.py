import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from marie.executor.extract import document_annotator_llm_executor as module
from marie.executor.extract.document_annotator_llm_executor import (
    DocumentAnnotatorLLMExecutor,
)
from marie.extract.annotators import llm_annotator as annotator_module
from marie.extract.annotators.llm_annotator import LLMAnnotator
from marie.extract.annotators.util import _write_single_result


@pytest.fixture
def definition(tmp_path: Path) -> Path:
    root = tmp_path / 'extract'
    (root / 'base').mkdir(parents=True)
    (root / 'base/base-config.yml').write_text('annotators: {}\ngrounding: {}\n')
    (root / 'base/field-config.yml').write_text('grounding: {}\n')
    profile = root / 'TID-claims/annotator'
    profile.mkdir(parents=True)
    (profile / 'config.yml').write_text(
        'annotators:\n'
        '  extract:\n'
        '    mode: per-page\n'
        '    model_config:\n'
        '      model_name: configured-model\n'
        '      prompt_path: ./claim.j2\n'
        '      expect_output: json\n'
        '      temperature: 0.3\n'
    )
    (profile / 'claim.j2').write_text(
        'Custom instructions {{ EXTRACTOR_SCHEMA }} {{ OCR_TEXT }}'
    )
    (profile / 'claim.output.json').write_text(
        json.dumps(
            {
                'version': '1.0',
                'strict': True,
                'schema': {
                    'type': 'object',
                    'properties': {'payer': {'type': 'string'}},
                },
            }
        )
    )
    return root


def test_saved_definition_loads_custom_prompt_and_reloads_current_files(
    definition: Path,
) -> None:
    first_conf, first_prompt, first_schema = module._load_schema_definition(
        str(definition), 'claims', 'extract'
    )
    assert first_conf['name'] == 'extract'
    assert first_conf['model_config']['temperature'] == 0.3
    assert first_prompt.render({'OCR_TEXT': 'Acme'}).startswith('Custom instructions ')
    assert '"payer"' in first_prompt.render({'OCR_TEXT': 'Acme'})
    assert first_schema['properties'] == {'payer': {'type': 'string'}}

    profile = definition / 'TID-claims/annotator'
    (profile / 'claim.j2').write_text('Updated {{ EXTRACTOR_SCHEMA }}')
    (profile / 'claim.output.json').write_text(
        json.dumps(
            {
                'type': 'object',
                'properties': {'total': {'type': 'number'}},
            }
        )
    )
    _, next_prompt, next_schema = module._load_schema_definition(
        str(definition), 'claims', 'extract'
    )
    assert next_prompt.render().startswith('Updated ')
    assert '"total"' in next_prompt.render()
    assert next_schema['properties'] == {'total': {'type': 'number'}}


def test_base_prompt_uses_layout_schema_before_inherited_schema(
    definition: Path,
) -> None:
    profile = definition / 'TID-claims/annotator'
    (profile / 'claim.j2').unlink()
    (definition / 'base/claim.j2').write_text('Base {{ EXTRACTOR_SCHEMA }}')
    (definition / 'base/claim.output.json').write_text(
        json.dumps(
            {
                'type': 'object',
                'properties': {'base': {'type': 'string'}},
            }
        )
    )
    _, prompt, schema = module._load_schema_definition(
        str(definition), 'claims', 'extract'
    )
    assert prompt.file_path == str(definition / 'base/claim.j2')
    assert 'payer' in schema['properties']
    (profile / 'claim.output.json').unlink()
    _, prompt, schema = module._load_schema_definition(
        str(definition), 'claims', 'extract'
    )
    assert 'base' in schema['properties']
    assert '"base"' in prompt.render()


@pytest.mark.parametrize(
    'schema', [None, '{bad', '{"version":"1.0","schema":{"type":42}}']
)
def test_saved_definition_requires_valid_schema(
    definition: Path, schema: str | None
) -> None:
    path = definition / 'TID-claims/annotator/claim.output.json'
    if schema is None:
        path.unlink()
    else:
        path.write_text(schema)
    with pytest.raises(ValueError, match='schema'):
        module._load_schema_definition(str(definition), 'claims', 'extract')


@pytest.mark.parametrize(
    'layout,key', [('../claims', 'extract'), ('claims', 'missing')]
)
def test_saved_definition_rejects_invalid_profile(
    definition: Path, layout: str, key: str
) -> None:
    with pytest.raises(ValueError):
        module._load_schema_definition(str(definition), layout, key)


def test_page_results_read_actual_annotator_files_and_map_absolute_pages(
    tmp_path: Path,
) -> None:
    _write_single_result(
        None,
        'prompt',
        '/frames/00001.png',
        'task-1',
        '{"payer":"A"}',
        str(tmp_path),
        'json',
    )
    _write_single_result(
        None,
        'prompt',
        '/frames/00002.png',
        'task-2',
        '{"payer":"B"}',
        str(tmp_path),
        'json',
        '_t0',
    )
    _write_single_result(
        None,
        'prompt',
        '/frames/00002.png',
        'task-3',
        '{"payer":"C"}',
        str(tmp_path),
        'json',
        '_t1',
    )
    assert module._read_schema_page_results(str(tmp_path), [2, 5]) == [
        {'page': 2, 'data': {'payer': 'A'}},
        {'page': 5, 'data': {'payer': 'B'}},
        {'page': 5, 'data': {'payer': 'C'}},
    ]


def test_page_results_reject_empty_malformed_and_out_of_range_outputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match='output'):
        module._read_schema_page_results(str(tmp_path), [0])
    path = tmp_path / '00001.json'
    path.write_text('[]')
    with pytest.raises(ValueError, match='object'):
        module._read_schema_page_results(str(tmp_path), [0])
    path.write_text('{bad')
    with pytest.raises(ValueError, match='JSON'):
        module._read_schema_page_results(str(tmp_path), [0])
    path.unlink()
    (tmp_path / '00002.json').write_text('{}')
    with pytest.raises(ValueError, match='page'):
        module._read_schema_page_results(str(tmp_path), [0])


def test_annotator_output_directory_is_independent_of_configured_name(
    monkeypatch, definition: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        annotator_module, 'route_llm_engine', lambda *_args: MagicMock()
    )
    conf = {
        'name': 'extract',
        'model_config': {'model_name': 'configured-model', 'prompt_path': './claim.j2'},
    }
    first = LLMAnnotator(
        str(tmp_path),
        conf,
        {'layout_id': 'claims'},
        prompt_dir=str(definition / 'TID-claims/annotator'),
        output_dir=str(tmp_path / 'agent-output/schema/dag-1/extract'),
    )
    second = LLMAnnotator(
        str(tmp_path),
        conf,
        {'layout_id': 'claims'},
        prompt_dir=str(definition / 'TID-claims/annotator'),
        output_dir=str(tmp_path / 'agent-output/schema/dag-2/extract'),
    )
    (Path(first.output_dir) / '00001.json').write_text('{}')
    assert first.name == second.name == 'extract'
    assert first.output_dir != second.output_dir
    assert not list(Path(second.output_dir).iterdir())
    assert first._scratch_root() != second._scratch_root()
    (Path(first._scratch_root()) / 'runs/active').mkdir(parents=True)
    second._cleanup_orphaned_runs()
    assert (Path(first._scratch_root()) / 'runs/active').exists()


@pytest.fixture
def runtime(monkeypatch, definition: Path, tmp_path: Path):
    executor = object.__new__(DocumentAnnotatorLLMExecutor)
    executor.logger = MagicMock()
    executor.root_config_dir = str(definition)
    executor.runtime_info = {'worker': 'test'}
    executor._setup_request = lambda *_args: None
    frames = [np.zeros((32, 64, 3), dtype=np.uint8)] * 2
    source_docs = [SimpleNamespace(tags={'page_number': p}) for p in [2, 5]]
    monkeypatch.setattr(
        module, 'docs_from_asset', lambda *_args, **_kwargs: (source_docs, '/doc.tif')
    )
    monkeypatch.setattr(module, 'frames_from_docs', lambda _docs: frames)
    root = tmp_path / 'assets'
    root.mkdir()
    metadata = {
        'ocr': [
            {
                'meta': {
                    'page': p,
                    'imageSize': {'width': 64, 'height': 32},
                    'lines': [],
                    'lines_bboxes': [],
                },
                'words': [],
                'lines': [],
            }
            for p in [0, 1]
        ]
    }
    metadata_file = root / 'document.tif.meta.json'
    metadata_file.write_text(json.dumps(metadata))
    monkeypatch.setattr(
        module,
        'prepare_asset_directory',
        lambda **_kwargs: (str(root), str(root / 'frames'), str(metadata_file)),
    )
    monkeypatch.setattr(
        annotator_module, 'route_llm_engine', lambda *_args: MagicMock()
    )
    monkeypatch.setattr(module, 'torch_gc', lambda: None)

    async def annotate(annotator, _document, _frames):
        assert 'Custom instructions' in annotator.prompt_template.render(
            {'OCR_TEXT': 'OCR'}
        )
        assert '"payer"' in annotator.prompt_template.render({'OCR_TEXT': 'OCR'})
        _write_single_result(
            None,
            'prompt',
            '/frames/00001.png',
            'task-1',
            '{"payer":"A"}',
            annotator.output_dir,
            'json',
        )
        _write_single_result(
            None,
            'prompt',
            '/frames/00002.png',
            'task-2',
            '{"payer":"B"}',
            annotator.output_dir,
            'json',
        )

    monkeypatch.setattr(LLMAnnotator, 'aannotate', annotate)
    writes = []
    monkeypatch.setattr(
        module.StorageManager,
        'write',
        lambda data, uri, **_kwargs: writes.append((data.read(), uri)) or True,
    )
    return executor, root, writes


@pytest.mark.asyncio
async def test_endpoint_persists_result_for_exact_dag_without_shared_outputs(
    runtime,
) -> None:
    executor, root, writes = runtime
    docs = [SimpleNamespace(asset_key='s3://bucket/doc.tif', pages=[2, 5])]
    parameters = {
        'job_id': 'node-1',
        'dag_id': 'dag-1',
        'ref_id': 'Doc.tif',
        'ref_type': 'extractor_document',
        'payload': {'op_params': {'layout': 'claims', 'key': 'extract'}},
    }
    first = await executor.annotator_schema(docs, parameters)
    second = await executor.annotator_schema(docs, {**parameters, 'dag_id': 'dag-2'})
    assert first['status'] == second['status'] == 'success'
    assert writes[0][1].endswith('/extractor_document/doc/extraction/dag-1.json')
    result = json.loads(writes[0][0])
    assert result['dag_id'] == 'dag-1'
    assert result['model'] == 'configured-model'
    assert [size['page'] for size in result['pages']['sizes']] == [2, 5]
    assert [value['value'] for value in result['values']] == ['A', 'B']
    assert not (root / 'agent-output').exists()


@pytest.mark.asyncio
async def test_endpoint_concurrent_runs_preserve_independent_outputs(
    runtime, monkeypatch
) -> None:
    executor, root, writes = runtime

    async def annotate(annotator, _document, _frames):
        _write_single_result(
            None,
            'prompt',
            '/frames/00001.png',
            'task-1',
            json.dumps({'payer': annotator.dag_id}),
            annotator.output_dir,
            'json',
        )
        await asyncio.sleep(0)
        _write_single_result(
            None,
            'prompt',
            '/frames/00002.png',
            'task-2',
            json.dumps({'payer': annotator.dag_id}),
            annotator.output_dir,
            'json',
        )

    monkeypatch.setattr(LLMAnnotator, 'aannotate', annotate)
    parameters = {
        'job_id': 'node-1',
        'ref_id': 'Doc.tif',
        'ref_type': 'extractor_document',
        'payload': {'op_params': {'layout': 'claims', 'key': 'extract'}},
    }
    await asyncio.gather(
        *[
            executor.annotator_schema(
                [SimpleNamespace(asset_key='s3://bucket/doc.tif', pages=[2, 5])],
                {**parameters, 'dag_id': dag_id},
            )
            for dag_id in ['dag-1', 'dag-2']
        ]
    )
    assert {json.loads(data)['dag_id'] for data, _ in writes} == {'dag-1', 'dag-2'}
    for data, _ in writes:
        result = json.loads(data)
        assert [value['value'] for value in result['values']] == [result['dag_id']] * 2


def test_default_workbench_profile_uses_shared_schema_prompt() -> None:
    root = Path(__file__).resolve().parents[3] / 'config/extract'
    conf, prompt, schema = module._load_schema_definition(
        str(root), 'workbench', 'extract'
    )
    assert conf['name'] == 'extract'
    assert prompt.file_path == str(root / 'base/schema-extract.j2')
    rendered = prompt.render({'OCR_TEXT': 'Title: Statement'})
    assert 'Title: Statement' in rendered
    assert json.dumps(schema, ensure_ascii=False) in rendered


def test_saved_definition_rejects_path_keys_before_using_config(
    definition: Path,
) -> None:
    config = definition / 'TID-claims/annotator/config.yml'
    config.write_text(config.read_text().replace('  extract:', '  ../escape:'))
    with pytest.raises(ValueError, match='key'):
        module._load_schema_definition(str(definition), 'claims', '../escape')


def test_refinement_prompt_retains_internal_schema_binding(
    monkeypatch, definition: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        annotator_module, 'route_llm_engine', lambda *_args: MagicMock()
    )
    (definition / 'TID-claims/annotator/refine.j2').write_text(
        '{{ EXTRACTOR_SCHEMA }} {{ PREVIOUS_EXTRACTION }}'
    )
    conf, prompt, _ = module._load_schema_definition(
        str(definition), 'claims', 'extract'
    )
    conf['model_config'].update(
        {'refine_passes': 1, 'refine_prompt_path': './refine.j2'}
    )
    annotator = LLMAnnotator(
        str(tmp_path),
        conf,
        {'layout_id': 'claims'},
        prompt_template=prompt,
        prompt_dir=str(definition / 'TID-claims/annotator'),
    )
    assert '"payer"' in annotator.refine_prompt_template.render(
        {'PREVIOUS_EXTRACTION': '{}'}
    )


@pytest.mark.asyncio
async def test_endpoint_propagates_failed_upload(runtime, monkeypatch) -> None:
    executor, _, _ = runtime
    monkeypatch.setattr(module.StorageManager, 'write', lambda *_args, **_kwargs: False)
    with pytest.raises(RuntimeError, match='persist|write'):
        await executor.annotator_schema(
            [SimpleNamespace(asset_key='s3://bucket/doc.tif', pages=[2, 5])],
            {
                'job_id': 'node-1',
                'dag_id': 'dag-1',
                'ref_id': 'Doc.tif',
                'ref_type': 'extractor_document',
                'payload': {'op_params': {'layout': 'claims', 'key': 'extract'}},
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'overrides',
    [
        {'dag_id': None},
        {'dag_id': '../escape'},
        {'ref_type': 'other'},
        {'ref_id': '../escape.tif'},
    ],
)
async def test_endpoint_requires_run_and_extractor_document_identity(
    runtime, overrides
) -> None:
    executor, _, _ = runtime
    parameters = {
        'job_id': 'node-1',
        'dag_id': 'dag-1',
        'ref_id': 'Doc.tif',
        'ref_type': 'extractor_document',
        'payload': {'op_params': {'layout': 'claims', 'key': 'extract'}},
        **overrides,
    }
    with pytest.raises(ValueError):
        await executor.annotator_schema(
            [SimpleNamespace(asset_key='s3://bucket/doc.tif', pages=[2, 5])], parameters
        )


@pytest.mark.asyncio
async def test_endpoint_annotation_failure_cannot_persist_success(
    runtime, monkeypatch
) -> None:
    executor, _, writes = runtime

    async def annotate(_annotator, _document, _frames):
        raise RuntimeError('model request failed')

    monkeypatch.setattr(LLMAnnotator, 'aannotate', annotate)
    with pytest.raises(RuntimeError, match='model request failed'):
        await executor.annotator_schema(
            [SimpleNamespace(asset_key='s3://bucket/doc.tif', pages=[2, 5])],
            {
                'job_id': 'node-1',
                'dag_id': 'dag-1',
                'ref_id': 'Doc.tif',
                'ref_type': 'extractor_document',
                'payload': {'op_params': {'layout': 'claims', 'key': 'extract'}},
            },
        )
    assert writes == []


@pytest.mark.asyncio
@pytest.mark.parametrize('suffixes', [[''], ['_t0', '_t1']])
async def test_per_page_extraction_rejects_missing_selected_page(
    runtime, monkeypatch, suffixes: list[str]
) -> None:
    executor, _, writes = runtime

    async def annotate(annotator, _document, _frames):
        for index, suffix in enumerate(suffixes):
            _write_single_result(
                None,
                'prompt',
                '/frames/00001.png',
                f'task-{index}',
                '{"payer":"A"}',
                annotator.output_dir,
                'json',
                suffix,
            )

    monkeypatch.setattr(LLMAnnotator, 'aannotate', annotate)
    with pytest.raises(ValueError, match='Missing extraction output.*5'):
        await executor.annotator_schema(
            [SimpleNamespace(asset_key='s3://bucket/doc.tif', pages=[2, 5])],
            {
                'job_id': 'node-1',
                'dag_id': 'dag-1',
                'ref_id': 'Doc.tif',
                'ref_type': 'extractor_document',
                'payload': {'op_params': {'layout': 'claims', 'key': 'extract'}},
            },
        )
    assert writes == []


@pytest.mark.asyncio
async def test_per_table_extraction_preserves_multiple_units_on_selected_page(
    runtime, monkeypatch, definition: Path
) -> None:
    executor, _, writes = runtime
    profile = definition / 'TID-claims/annotator/config.yml'
    profile.write_text(profile.read_text().replace('mode: per-page', 'mode: per-table'))

    async def annotate(annotator, _document, _frames):
        for index, payer in enumerate(['A', 'B']):
            _write_single_result(
                None,
                'prompt',
                '/frames/00001.png',
                f'task-{index}',
                json.dumps({'payer': payer}),
                annotator.output_dir,
                'json',
                f'_t{index}',
            )

    monkeypatch.setattr(LLMAnnotator, 'aannotate', annotate)
    response = await executor.annotator_schema(
        [SimpleNamespace(asset_key='s3://bucket/doc.tif', pages=[2, 5])],
        {
            'job_id': 'node-1',
            'dag_id': 'dag-1',
            'ref_id': 'Doc.tif',
            'ref_type': 'extractor_document',
            'payload': {'op_params': {'layout': 'claims', 'key': 'extract'}},
        },
    )
    assert response['status'] == 'success'
    result = json.loads(writes[0][0])
    assert [value['value'] for value in result['values']] == ['A', 'B']
    assert result['pages']['extracted'] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [None, 'annotation', 'adapter', 'upload'])
async def test_endpoint_isolates_and_cleans_temporary_run_artifacts(
    runtime, monkeypatch, failure: str | None
) -> None:
    from marie.extract.extractor import schema_extraction_result

    executor, root, _ = runtime
    output_paths = []
    original_annotate = LLMAnnotator.aannotate
    original_prepare = module.prepare_asset_directory

    def prepare(**kwargs):
        assert not kwargs.get('restore_dirs')
        return original_prepare(**kwargs)

    async def annotate(annotator, document, frames):
        output = Path(annotator.output_dir)
        output_paths.append(output)
        assert not output.is_relative_to(root)
        assert annotator.name == 'extract'
        await original_annotate(annotator, document, frames)
        scratch = Path(annotator._scratch_root())
        scratch.mkdir(parents=True)
        (scratch / 'private-context.json').write_text('{}')
        if failure == 'annotation':
            raise RuntimeError('injected annotation failure')

    def fail_adapter(*_args):
        raise RuntimeError('injected adapter failure')

    monkeypatch.setattr(module, 'prepare_asset_directory', prepare)
    monkeypatch.setattr(LLMAnnotator, 'aannotate', annotate)
    if failure == 'adapter':
        monkeypatch.setattr(
            schema_extraction_result, 'build_schema_extraction_result', fail_adapter
        )
    if failure == 'upload':
        monkeypatch.setattr(
            module.StorageManager, 'write', lambda *_args, **_kwargs: False
        )
    docs = [SimpleNamespace(asset_key='s3://bucket/doc.tif', pages=[2, 5])]
    parameters = {
        'job_id': 'node-1',
        'dag_id': 'dag-1',
        'ref_id': 'Doc.tif',
        'ref_type': 'extractor_document',
        'payload': {'op_params': {'layout': 'claims', 'key': 'extract'}},
    }
    if failure:
        with pytest.raises(RuntimeError, match='injected|persist'):
            await executor.annotator_schema(docs, parameters)
    else:
        assert (await executor.annotator_schema(docs, parameters))[
            'status'
        ] == 'success'
    assert output_paths
    assert all(not path.parent.exists() for path in output_paths)
    assert not (root / 'agent-output').exists()


@pytest.mark.asyncio
async def test_blocking_result_adapter_allows_event_loop_to_advance(
    runtime, monkeypatch
) -> None:
    from marie.extract.extractor import schema_extraction_result

    executor, _, _ = runtime
    original_adapter = schema_extraction_result.build_schema_extraction_result
    loop = asyncio.get_running_loop()
    adapter_started = asyncio.Event()
    release_adapter = threading.Event()

    def blocking_adapter(*args):
        loop.call_soon_threadsafe(adapter_started.set)
        if not release_adapter.wait(timeout=2):
            raise RuntimeError('event loop could not advance while adapter blocked')
        return original_adapter(*args)

    monkeypatch.setattr(
        schema_extraction_result, 'build_schema_extraction_result', blocking_adapter
    )
    task = asyncio.create_task(
        executor.annotator_schema(
            [SimpleNamespace(asset_key='s3://bucket/doc.tif', pages=[2, 5])],
            {
                'job_id': 'node-1',
                'dag_id': 'dag-1',
                'ref_id': 'Doc.tif',
                'ref_type': 'extractor_document',
                'payload': {'op_params': {'layout': 'claims', 'key': 'extract'}},
            },
        )
    )
    try:
        await asyncio.wait_for(adapter_started.wait(), timeout=3)
        await asyncio.sleep(0)
        release_adapter.set()
        assert (await task)['status'] == 'success'
    finally:
        release_adapter.set()


def test_prompt_without_schema_placeholder_still_receives_the_schema(
    definition: Path,
) -> None:
    (definition / 'TID-claims/annotator/claim.j2').write_text(
        'Hand-written fields. {{ OCR_TEXT }}'
    )
    _, prompt, _ = module._load_schema_definition(str(definition), 'claims', 'extract')
    rendered = prompt.render({'OCR_TEXT': 'Acme'})
    assert rendered.startswith('Hand-written fields. Acme')
    assert rendered.count('"payer"') == 1


@pytest.mark.parametrize('placeholder', ['{{ EXTRACTOR_SCHEMA }}', '{{EXTRACTOR_SCHEMA}}'])
def test_prompt_with_schema_placeholder_receives_the_schema_once(
    definition: Path, placeholder: str
) -> None:
    (definition / 'TID-claims/annotator/claim.j2').write_text(
        f'Fields: {placeholder} {{{{ OCR_TEXT }}}}'
    )
    _, prompt, _ = module._load_schema_definition(str(definition), 'claims', 'extract')
    assert prompt.render({'OCR_TEXT': 'Acme'}).count('"payer"') == 1
