from __future__ import annotations

from unittest.mock import patch

import pytest


def page(page_index: int, *lines: tuple[float, list[str]]) -> dict:
    return {
        'page': page_index,
        'words': [
            {
                'text': text,
                'box': {
                    'x1': 0.05 + index * 0.1,
                    'y1': y,
                    'x2': 0.13 + index * 0.1,
                    'y2': y + 0.02,
                },
            }
            for y, texts in lines
            for index, text in enumerate(texts)
        ],
    }


def test_normalizes_pixel_xywh_and_accepts_wrapped_ocr() -> None:
    from marie.extract.extractor.grounding import parse_ocr_meta

    raw = {
        'meta': {'imageSize': {'width': 1000, 'height': 500}, 'page': 3},
        'words': [{'text': 'ACME', 'confidence': 0.9, 'box': [100, 50, 200, 25]}],
    }
    expected = [
        {
            'page': 3,
            'width': 1000,
            'height': 500,
            'words': [
                {'text': 'ACME', 'box': {'x1': 0.1, 'y1': 0.1, 'x2': 0.3, 'y2': 0.15}}
            ],
        }
    ]
    assert parse_ocr_meta([raw]) == expected
    assert parse_ocr_meta({'ocr': [raw]}) == expected
    assert parse_ocr_meta([{'meta': {'page': 0}, 'words': []}, raw]) == expected
    assert parse_ocr_meta(None) == []


def test_skips_invalid_boxes_and_clamps_to_page_frame() -> None:
    from marie.extract.extractor.grounding import parse_ocr_meta

    raw = {
        'meta': {'imageSize': {'width': 100, 'height': 100}},
        'words': [
            {'text': 'edge', 'box': [-5, 90, 20, 20]},
            {'text': 'bad', 'box': [0, 0, float('nan'), 1]},
            {'text': 'reversed', 'box': [10, 10, -5, 1]},
            {'text': 'outside', 'box': [150, 0, 10, 1]},
        ],
    }
    assert parse_ocr_meta([raw])[0]['words'] == [
        {'text': 'edge', 'box': {'x1': 0, 'y1': 0.9, 'x2': 0.15, 'y2': 1}},
    ]


@pytest.mark.parametrize(
    ('value', 'printed', 'match'),
    [
        ('Total', ['Total'], 'exact'),
        ('acme inc', ['ACME,', 'Inc.'], 'exact'),
        ('714.57', ['$714.57'], 'exact'),
        ('-25.00', ['(25.00)'], 'exact'),
        ('1234.56', ['1,234.56'], 'exact'),
        ('Acme Incorporated', ['Acme', 'Incorporatd'], 'fuzzy'),
        ('714.57', ['71,457'], None),
        ('-50', ['50'], None),
        ('-50.00', ['50.00'], None),
        ('1/23', ['12/3'], None),
        ('Acme 123', ['Acme', '124'], None),
        ('Nowhere', ['Total'], None),
        ('', ['Total'], None),
    ],
)
def test_grounding_preserves_numbers_and_only_fuzzes_text(
    value: str, printed: list[str], match: str | None
) -> None:
    from marie.extract.extractor.grounding import ground_text

    location = ground_text(value, [page(0, (0.1, printed))], None)
    if match is None:
        assert location is None
    else:
        assert location['match'] == match
        assert location['candidates'] == 1
        assert location['page'] == 0


def test_multiword_match_returns_union_box() -> None:
    from marie.extract.extractor.grounding import ground_text

    location = ground_text(
        'ACME, Inc.', [page(0, (0.1, ['Payer:', 'ACME,', 'Inc.']))], None
    )
    assert location['bbox'] == pytest.approx(
        {'x1': 0.15, 'y1': 0.1, 'x2': 0.33, 'y2': 0.12}
    )


def test_duplicate_spans_are_ambiguous_and_page_hint_preserves_source_page() -> None:
    from marie.extract.extractor.grounding import ground_text

    first = page(0, (0.1, ['Date', '03/18/2023']), (0.5, ['Service', '03/18/2023']))
    second = page(7, (0.6, ['03/18/2023']))
    location = ground_text('03/18/2023', [first, second], None)
    assert location['match'] == 'ambiguous'
    assert location['candidates'] == 3
    assert location['bbox']['y1'] == 0.1
    assert ground_text('03/18/2023', [first, second], 7) == {
        'page': 7,
        'bbox': {'x1': 0.05, 'y1': 0.6, 'x2': 0.13, 'y2': 0.62},
        'match': 'exact',
        'candidates': 1,
    }


def test_exact_match_wins_over_fuzzy_and_overlapping_spans_count_once() -> None:
    from marie.extract.extractor.grounding import ground_text

    ocr = page(0, (0.1, ['Acme', 'Incorporatd']), (0.5, ['Acme', 'Incorporated']))
    assert ground_text('Acme Incorporated', [ocr], None)['match'] == 'exact'
    assert ground_text('Acme Incorporated', [ocr], None)['candidates'] == 1


def test_ground_values_keeps_value_identity() -> None:
    from marie.extract.extractor.grounding import ground_values

    values = [
        {'id': 'a', 'value': '714.57', 'page': 0},
        {'id': 'b', 'value': 'missing', 'page': None},
    ]
    grounded = ground_values(values, [page(0, (0.2, ['Total', '$714.57']))])
    assert grounded[0]['id'] == 'a'
    assert grounded[0]['location']['match'] == 'exact'
    assert grounded[1]['location'] is None
    assert 'location' not in values[0]


@pytest.mark.parametrize(
    ('value', 'printed', 'comparison'),
    [
        (
            'Acme Incorporated',
            ['Acme', 'Incorporatd'],
            ('acmeincorporatd', 'acmeincorporated'),
        ),
        (
            'Metropolitan Services',
            ['Metropolitan', 'Servi'],
            ('metropolitanservi', 'metropolitanservices'),
        ),
    ],
)
def test_fuzzy_scan_skips_impossible_lengths_and_keeps_threshold_boundary(
    value: str, printed: list[str], comparison: tuple[str, str]
) -> None:
    from marie.extract.extractor import grounding

    with patch.object(
        grounding, '_levenshtein', wraps=grounding._levenshtein
    ) as distance:
        location = grounding.ground_text(value, [page(0, (0.1, printed))])

    assert location['match'] == 'fuzzy'
    assert location['candidates'] == 1
    distance.assert_called_once_with(*comparison)
