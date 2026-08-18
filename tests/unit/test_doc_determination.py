from __future__ import annotations

from marie.executor.pipeline.doc_determination_pipeline_executor import (
    doc_determination_collation,
    filter_pages_by_classifier_results,
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
