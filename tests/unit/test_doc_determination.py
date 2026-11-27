from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tifffile import TiffFile, imwrite

from marie.executor.pipeline import (
    doc_determination_pipeline_executor as executor_module,
)
from marie.executor.pipeline.doc_determination_pipeline_executor import (
    doc_determination_collation,
    filter_pages_by_classifier_results,
    get_pipeline_pages,
    split_assets,
)


def _classification(group: str, pages: dict[int, tuple[str, float]]) -> dict:
    return {
        'group': group,
        'classification': {
            'pages': {
                str(page): {
                    'best': {
                        'page': page,
                        'classification': label,
                        'score': score,
                    }
                }
                for page, (label, score) in pages.items()
            }
        },
    }


def test_filter_include_honors_min_confidence() -> None:
    metadata = {
        'classifications': [
            _classification(
                'split-boundary-group',
                {0: (1, 0.999), 1: ('1', 0.9), 2: ('0', 0.999)},
            )
        ]
    }

    pages = filter_pages_by_classifier_results(
        [
            {
                'type': 'classification',
                'method': 'include',
                'group': 'split-boundary-group',
                'classifications': ['1'],
                'min_conf': 0.995,
            }
        ],
        metadata,
    )

    assert pages == {0: [0]}


def test_filter_exclude_removes_only_confident_matches() -> None:
    metadata = {
        'classifications': [
            _classification(
                'medical-page-classifier',
                {0: ('OTHER', 0.999), 1: ('OTHER', 0.9), 2: ('CHECK-FRONT', 0.999)},
            )
        ]
    }

    pages = filter_pages_by_classifier_results(
        [
            {
                'type': 'classification',
                'method': 'exclude',
                'group': 'medical-page-classifier',
                'classifications': ['OTHER'],
                'min_conf': 0.995,
            }
        ],
        metadata,
    )

    assert pages == {0: [1, 2]}


def test_get_pipeline_pages_uses_all_pages_when_metadata_is_missing(
    monkeypatch, tmp_path
) -> None:
    missing_meta_path = tmp_path / 'missing.meta.json'
    monkeypatch.setattr(
        executor_module,
        'download_asset',
        lambda *args, **kwargs: str(missing_meta_path),
    )

    pages = get_pipeline_pages(
        'document-id',
        'document-type',
        {
            'pages': [
                {
                    'type': 'classification',
                    'method': 'include',
                    'group': 'classifier',
                    'classifications': ['keep'],
                }
            ]
        },
        {},
    )

    assert pages == {0: None}


def test_collation_preserves_contiguous_page_range_and_metadata() -> None:
    metadata = {
        'rotation': {
            'pages': {
                '1': {'rotate': 90},
                '2': {'rotate': 0},
                '3': {'rotate': 0},
            }
        },
        'classifications': [
            _classification(
                'doc-determination-classifier',
                {1: ('PatpayCard', 0.999), 3: ('PatpayCard', 0.999)},
            ),
            _classification(
                'medical-page-classifier',
                {1: ('CHECK-FRONT', 0.99), 2: ('CHECK-BACK', 0.99)},
            ),
        ],
    }

    result = doc_determination_collation(metadata)

    document = result['docs'][0]
    assert document['page-count'] == 3
    assert [page['page'] for page in document['pages']] == [1, 2, 3]
    assert document['pages'][0]['rotation'] == 90
    assert document['pages'][1]['medical-page-classification'] == 'CHECK-BACK'


def test_collate_and_split_without_rotation(tmp_path: Path) -> None:
    metadata = {
        'pages': '4',
        'indexes': [
            {
                'task': 'code',
                'index': {
                    'pages': {
                        str(page): {
                            'best': {'page': page, 'indexing': f'X{page}'},
                            'details': [],
                        }
                        for page in [1, 3]
                    }
                },
            }
        ],
        'classifications': [
            _classification('doc-determination-classifier', {1: ('A', 1.0)}),
            _classification('doc-determination-classifier', {3: ('B', 1.0)}),
        ],
    }
    (tmp_path / 'input.meta.json').write_text(json.dumps(metadata))
    burst = tmp_path / 'burst'
    burst.mkdir()
    for page in range(4):
        imwrite(burst / f'page_{page}.tif', np.full((4, 4), page, dtype=np.uint8))
    collation = doc_determination_collation(metadata)
    assert [doc['doc-classification'] for doc in collation['docs']] == ['A', 'B']
    split_assets(collation, str(tmp_path), 'input')
    for start in (0, 2):
        folder = tmp_path / 'splits' / f'{start}-{start + 1}'
        child = json.loads((folder / '__CHILD_REFID__.meta.json').read_text())
        assert child['parent_pages'] == [start, start + 1]
        index = child['indexes'][0]['index']['pages']
        assert index == {
            '1': {'best': {'page': 1, 'indexing': f'X{start + 1}'}, 'details': []}
        }
        with TiffFile(folder / '__CHILD_REFID__.tif') as tif:
            assert [p.asarray()[0, 0] for p in tif.pages] == child['parent_pages']


def test_filters_split_in_source_order_and_can_select_no_pages() -> None:
    metadata = {
        'classifications': [
            _classification(
                'kind',
                {
                    3: ('keep', 1.0),
                    0: ('keep', 1.0),
                    2: ('keep', 1.0),
                    1: ('end', 1.0),
                },
            )
        ]
    }
    rule = {
        'type': 'classification',
        'method': 'split',
        'classifications': ['end'],
        'group': 'kind',
    }
    assert filter_pages_by_classifier_results([rule], metadata) == {
        0: [0, 1],
        1: [2, 3],
    }
    rule.update(method='include', classifications=['missing'])
    assert filter_pages_by_classifier_results([rule], metadata) == {0: []}
    split = {**rule, 'method': 'split', 'classifications': ['end']}
    assert filter_pages_by_classifier_results([rule, split], metadata) == {0: []}


def test_merging_filtered_indexes_keeps_other_tasks_and_page_sets() -> None:
    from marie.pipe.components import update_existing_meta

    def entry(task: str, page: int, value: str) -> dict:
        return {
            'group': 'index',
            'task': task,
            'index': {'page_numbers': [page], 'value': value},
        }

    first = entry('a', 1, 'first')
    other = entry('b', 2, 'other')
    latest = entry('a', 2, 'latest')
    merged = update_existing_meta(
        {'indexes': [first, other, entry('a', 2, 'old')]}, {'indexes': [latest]}
    )
    assert merged['indexes'] == [first, other, latest]


def test_collation_requires_document_classifications() -> None:
    with pytest.raises(ValueError, match='doc-determination-classifier'):
        doc_determination_collation({'pages': '3'})


def test_rotation_rejects_partial_source_before_rewriting_assets() -> None:
    from types import SimpleNamespace

    from marie.executor.rotation.rotation_executor import RotationExecutor

    with pytest.raises(ValueError, match='whole document'):
        RotationExecutor.rotate_frames(
            SimpleNamespace(),
            [SimpleNamespace(asset_key='s3://test/doc.tif', pages=[1])],
            {'job_id': 'job', 'ref_id': 'doc', 'ref_type': 'test', 'payload': {}},
        )


@pytest.mark.parametrize("backup_ok", [True, False])
def test_rotation_backs_up_an_external_tiff_before_replacing_it(
    tmp_path, monkeypatch, backup_ok
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from marie.executor.rotation import rotation_executor as module

    source = 's3://input/original.tif'
    original = b'original color TIFF bytes'
    docs = [SimpleNamespace(asset_key=source, pages=None)]
    writes = []

    def read(uri, path, **kwargs):
        assert uri == source
        Path(path).write_bytes(original)
        return True

    def write(path, uri, **kwargs):
        writes.append((uri, Path(path).read_bytes()))
        return backup_ok

    monkeypatch.setattr(module.StorageManager, 'read_to_file', read)
    monkeypatch.setattr(module.StorageManager, 'write', write)
    monkeypatch.setattr(module, 'get_frames_from_docs', lambda docs: [np.ones((3, 3))])
    monkeypatch.setattr(module, 'create_working_dir', lambda *a, **kw: str(tmp_path))
    monkeypatch.setattr(module, 's3_asset_path', lambda *a: 's3://output/doc')
    monkeypatch.setattr(module, 'restore_assets', lambda *a, **kw: 's3://output/doc')
    monkeypatch.setattr(module, 'store_assets', lambda *a, **kw: ['asset'])
    monkeypatch.setattr(module, 'rotate_frames', lambda *a, **kw: ({'pages': {}}, True))
    monkeypatch.setattr(module, 'burst_frames', Mock())
    monkeypatch.setattr(
        module,
        'merge_tiff_frames_ifd',
        lambda folder, path, **kw: Path(path).write_bytes(b'rotated'),
    )
    monkeypatch.setattr(module, 'ocr_frames', lambda *a, **kw: [])
    executor = SimpleNamespace(
        logger=Mock(), rotation_api=Mock(), ocr_engines={}, runtime_info={}
    )
    params = {'ref_id': 'doc.tif', 'ref_type': 'test', 'job_id': 'job', 'payload': {}}
    if backup_ok:
        module.RotationExecutor.rotate_frames(executor, docs, params)
        assert writes[-1] == (source, b'rotated')
    else:
        with pytest.raises(OSError):
            module.RotationExecutor.rotate_frames(executor, docs, params)
        assert len(writes) == 1
    assert writes[0][0].startswith('s3://output/doc/doc_prerotation_')
    assert writes[0][1] == original


def test_missing_remote_metadata_processes_all_pages(monkeypatch) -> None:
    monkeypatch.setattr(executor_module, 'download_asset', lambda *a, **kw: None)
    rule = {'type': 'classification', 'method': 'include', 'classifications': ['A']}
    assert get_pipeline_pages('input', 'test', {'pages': [rule]}, {}) == {0: None}


def test_filtered_pipeline_keeps_checkpoints_and_full_document_ocr(
    tmp_path, monkeypatch
):
    from unittest.mock import Mock

    from marie.pipe import llm_pipeline as module

    pipe = object.__new__(module.LLMPipeline)
    pipe.pipeline_name = 'selection'
    pipe.logger = Mock()
    engine = Mock()
    engine.extract.side_effect = lambda frames, *args, **kw: [
        {'text': int(frame[0, 0])} for frame in frames
    ]
    pipe.ocr_engines = {'default': engine}
    monkeypatch.setattr(module, 'restore_assets', lambda *args, **kw: None)
    monkeypatch.setattr(module, 'store_assets', lambda *args, **kw: [])
    monkeypatch.setattr(module, 'burst_frames', Mock())
    observed = []

    def classify(frames, metadata, ocr, config, pages, root):
        observed.append([entry['text'] for entry in ocr])
        metadata['classifications'] = [
            {'group': 'g', 'classification': {'page_numbers': pages}}
        ]

    monkeypatch.setattr(pipe, 'execute_llm_pipeline', classify)

    def process(pages):
        frames = [np.full((4, 4), p, dtype=np.uint8) for p in pages]
        pipe.execute_frames_pipeline(
            'input.tif', 'test', frames, str(tmp_path), 'job', {}, pages=pages
        )

    process([0, 2])
    process([1, 3])
    assert observed == [[0, 2], [1, 3]]
    result_file = tmp_path / 'results' / 'input.json'
    assert not result_file.exists()
    checkpoint = json.loads((tmp_path / 'selection/input.tif.meta.json').read_text())
    assert [
        x['classification']['page_numbers'] for x in checkpoint['classifications']
    ] == [[0, 2], [1, 3]]
    result_file.write_text(json.dumps([{'text': n + 10} for n in range(4)]))
    process([3, 1])
    assert observed[-1] == [13, 11]
    assert engine.extract.call_count == 2


def test_asset_page_selection_reaches_classification_and_collation(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    executor = SimpleNamespace(
        logger=Mock(),
        runtime_info={},
        pipeline=Mock(),
        resolve_runtime_and_pipeline_configs=lambda features: ({}, {}),
    )
    docs = [SimpleNamespace(pages=[1, 3])]
    params = {'ref_id': 'doc', 'ref_type': 'test', 'job_id': 'job', 'payload': {}}
    fetch = Mock(return_value=[])
    monkeypatch.setattr(executor_module, 'get_frames_from_docs', fetch)
    monkeypatch.setattr(
        executor_module, 'create_working_dir', Mock(return_value='/tmp')
    )
    executor_module.DocDeterminationPipelineExecutor.run_llm_pipeline(
        executor, docs, params
    )
    assert fetch.call_args.args[1] == [1, 3]
    assert executor.pipeline.execute_frames_pipeline.call_args.kwargs['pages'] == [1, 3]
    with pytest.raises(ValueError, match='whole document'):
        executor_module.DocDeterminationPipelineExecutor.collate(executor, docs, params)


def test_selected_pages_keep_per_task_index_results(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from marie.api.docs import DOC_KEY_INDEXER
    from marie.components.document_indexer.llm_task import PageResult
    from marie.pipe.llm_pipeline import LLMPipeline

    indexer = Mock()
    indexer.run.return_value = [
        SimpleNamespace(tags={DOC_KEY_INDEXER: {'other': ('Y', None)}}),
        SimpleNamespace(
            tags={DOC_KEY_INDEXER: {'code': PageResult(page=1, value='X')}}
        ),
    ]
    pipeline = object.__new__(LLMPipeline)
    pipeline.logger = Mock()
    pipeline.pipeline_name = 'test'
    pipeline.classifier_groups = {}
    pipeline.indexer_groups = {
        'test': {
            'llm_tasks': {'code': True, 'other': True, 'disabled': False},
            'indexers': {'example': {'indexer': indexer, 'group': 'test'}},
        }
    }

    def execute(components, *args, **kwargs):
        meta = components[0].index_documents(indexer.run.return_value)
        return {'indexes': pipeline.get_indexer_results(meta)}

    monkeypatch.setattr(pipeline, 'execute_pipeline', execute)
    result = {}
    pipeline.execute_llm_pipeline(
        [], result, [], {}, pages=[1, 3], root_asset_path=str(tmp_path)
    )
    assert indexer.run.call_args.kwargs['tasks'] == ['code', 'other']
    assert indexer.run.call_args.kwargs['root_asset_path'] == str(tmp_path)
    pipeline.execute_llm_pipeline(
        [], {}, [], {'llm_tasks': {'disabled': {'enabled': True}}}
    )
    assert indexer.run.call_args.kwargs['tasks'] == ['code', 'other', 'disabled']
    tasks = {entry['task']: entry for entry in result['indexes']}
    assert tasks['other']['index']['pages'][1]['best']['indexing'] == 'Y'
    item = tasks['code']
    assert list(item['index']['pages']) == [3]
    assert item['index']['pages'][3]['best']['indexing'] == 'X'
    assert item['index']['pages'][3]['details'][0]['page'] == 3
