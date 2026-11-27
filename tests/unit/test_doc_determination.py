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
                {0: ('1', 0.999), 1: ('1', 0.9), 2: ('0', 0.999)},
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


def test_missing_remote_metadata_processes_all_pages(monkeypatch) -> None:
    monkeypatch.setattr(executor_module, 'download_asset', lambda *a, **kw: None)
    rule = {'type': 'classification', 'method': 'include', 'classifications': ['A']}
    assert get_pipeline_pages('input', 'test', {'pages': [rule]}, {}) == {0: None}


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
            'llm_tasks': ['code', 'other'],
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
    tasks = {entry['task']: entry for entry in result['indexes']}
    assert tasks['other']['index']['pages'][1]['best']['indexing'] == 'Y'
    item = tasks['code']
    assert list(item['index']['pages']) == [3]
    assert item['index']['pages'][3]['best']['indexing'] == 'X'
    assert item['index']['pages'][3]['details'][0]['page'] == 3
    assert indexer.run.call_args.kwargs['root_asset_path'] == str(tmp_path)
