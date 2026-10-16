import json
from pathlib import Path

import pytest
from marie_parsebench.models import (
    ArtifactIdentity,
    DocumentArtifact,
    Element,
    PageArtifact,
    ProducerIdentity,
    SourceTransform,
    Usage,
    normalize_bbox_xyxy,
)
from pydantic import ValidationError

IDENTITY = {
    'run_id': 'run-1',
    'example_id': 'example-1',
    'pipeline_name': 'default',
    'source_sha256': 'a' * 64,
    'config_sha256': 'b' * 64,
    'ref_id': 'ref-1',
    'ref_type': 'document',
    'dag_id': 'dag-1',
}
PRODUCER = {'name': 'marie', 'version': '1.0.0'}


def make_transform(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        'source_width': 200.0,
        'source_height': 100.0,
        'source_unit': 'pixels',
        'crop_xyxy': (0.0, 0.0, 200.0, 100.0),
        'rotation_degrees': 0,
        'render_width_px': 200,
        'render_height_px': 100,
    }
    values.update(overrides)
    return values


def make_element(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        'element_id': 'element-1',
        'kind': 'text',
        'content': 'Body',
        'content_format': 'markdown',
        'bbox_xyxy_normalized': (0.1, 0.2, 0.3, 0.4),
        'geometry_source': 'ocr',
    }
    values.update(overrides)
    return values


def make_page(page_index: int = 0, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        'format_version': 'marie.parsebench.page.v1',
        'identity': IDENTITY,
        'producer': PRODUCER,
        'page_id': f'page-{page_index}',
        'page_index': page_index,
        'width_px': 200,
        'height_px': 100,
        'source_transform': make_transform(),
        'status': 'complete',
        'elements': [make_element(element_id=f'element-{page_index}')],
    }
    values.update(overrides)
    return values


def make_document(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        'format_version': 'marie.parsebench.document.v1',
        'identity': IDENTITY,
        'producer': PRODUCER,
        'status': 'complete',
        'source_page_count': 3,
        'expected_page_indices': [2, 0],
        'expected_page_count': 2,
        'pages': [make_page(2), make_page(0)],
    }
    values.update(overrides)
    return values


def test_serialized_page_and_document_round_trip() -> None:
    document = DocumentArtifact.model_validate_json(json.dumps(make_document()))

    restored = DocumentArtifact.model_validate_json(document.model_dump_json())

    assert restored == document
    assert restored.expected_page_indices == [2, 0]
    assert restored.pages[0].elements[0].bbox_xyxy_normalized == (0.1, 0.2, 0.3, 0.4)
    assert isinstance(
        json.loads(restored.model_dump_json())['pages'][0]['elements'][0][
            'bbox_xyxy_normalized'
        ],
        list,
    )


@pytest.mark.parametrize(
    ('status', 'elements'),
    [('complete', [make_element()]), ('blank', []), ('failed', [])],
)
def test_page_status_accepts_valid_element_inventory(
    status: str, elements: list[dict[str, object]]
) -> None:
    page = PageArtifact.model_validate(make_page(status=status, elements=elements))
    assert page.status == status


@pytest.mark.parametrize(
    ('status', 'elements'),
    [('complete', []), ('blank', [make_element()])],
)
def test_page_status_rejects_invalid_element_inventory(
    status: str, elements: list[dict[str, object]]
) -> None:
    with pytest.raises(ValidationError):
        PageArtifact.model_validate(make_page(status=status, elements=elements))


def test_identity_usage_and_unknown_fields_are_strict() -> None:
    with pytest.raises(ValidationError):
        ArtifactIdentity.model_validate({**IDENTITY, 'run_id': ' ', 'extra': 'value'})
    with pytest.raises(ValidationError):
        ArtifactIdentity.model_validate({**IDENTITY, 'source_sha256': 'A' * 64})
    with pytest.raises(ValidationError):
        Usage.model_validate({'input_tokens': True})
    with pytest.raises(ValidationError):
        Usage.model_validate({'cost_usd': float('inf')})


@pytest.mark.parametrize('kind', ['table', 'chart'])
def test_html_and_markdown_content_is_preserved_exactly(kind: str) -> None:
    content_format = 'html' if kind == 'table' else 'markdown'
    content = (
        '<table><tr><td rowspan="2"> A </td></tr></table>'
        if kind == 'table'
        else '  **Q1**\n\n- 10\n- 20  '
    )
    element = Element.model_validate(
        make_element(kind=kind, content=content, content_format=content_format)
    )
    assert element.content == content


def test_element_geometry_pair_and_content_rules_are_enforced() -> None:
    with pytest.raises(ValidationError):
        Element.model_validate(make_element(geometry_source=None))
    with pytest.raises(ValidationError):
        Element.model_validate(make_element(bbox_xyxy_normalized=None))
    with pytest.raises(ValidationError):
        Element.model_validate(make_element(geometry_source='synthetic'))
    with pytest.raises(ValidationError):
        Element.model_validate(make_element(bbox_xyxy_normalized=(0.3, 0.2, 0.1, 0.4)))
    with pytest.raises(ValidationError):
        Element.model_validate(make_element(content=' ', kind='text'))
    with pytest.raises(ValidationError):
        Element.model_validate(make_element(kind='table', content_format='markdown'))

    picture = Element.model_validate(make_element(kind='picture', content=''))
    assert picture.content == ''

    ungrounded = Element.model_validate(
        make_element(bbox_xyxy_normalized=None, geometry_source=None)
    )
    assert ungrounded.bbox_xyxy_normalized is None


def test_page_rejects_duplicate_element_ids() -> None:
    with pytest.raises(ValidationError):
        PageArtifact.model_validate(
            make_page(elements=[make_element(), make_element()])
        )


@pytest.mark.parametrize(
    'mutation',
    [
        {'expected_page_count': 1},
        {'expected_page_indices': [2, 2]},
        {'expected_page_indices': [2, 3]},
        {'pages': [make_page(0), make_page(2)]},
        {'pages': [make_page(2), make_page(0, page_id='page-2')]},
        {
            'pages': [
                make_page(2),
                make_page(0, identity={**IDENTITY, 'run_id': 'other'}),
            ]
        },
        {'pages': [make_page(2), make_page(0, status='failed', elements=[])]},
    ],
)
def test_document_rejects_invalid_inventory_identity_or_status(
    mutation: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        DocumentArtifact.model_validate(make_document(**mutation))


@pytest.mark.parametrize(
    ('rotation', 'render_size', 'bbox', 'expected'),
    [
        (0, (200, 100), (20, 20, 60, 40), (0.1, 0.2, 0.3, 0.4)),
        (90, (100, 200), (20, 40, 60, 120), (0.2, 0.3, 0.4, 0.5)),
        (180, (200, 100), (140, 60, 180, 80), (0.1, 0.2, 0.3, 0.4)),
        (270, (100, 200), (40, 80, 80, 120), (0.4, 0.4, 0.6, 0.8)),
    ],
)
def test_rendered_geometry_undoes_every_right_angle_rotation(
    rotation: int,
    render_size: tuple[int, int],
    bbox: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
) -> None:
    crop = (0.0, 0.0, 200.0, 100.0)
    if rotation == 90:
        crop = (20.0, 10.0, 120.0, 60.0)
    transform = SourceTransform.model_validate(
        make_transform(
            crop_xyxy=crop,
            rotation_degrees=rotation,
            render_width_px=render_size[0],
            render_height_px=render_size[1],
        )
    )
    assert normalize_bbox_xyxy(bbox, transform) == pytest.approx(expected)


def test_geometry_supports_source_points_and_full_rendered_frame() -> None:
    points = SourceTransform.model_validate(
        make_transform(
            source_width=612.0,
            source_height=792.0,
            source_unit='points',
            crop_xyxy=(0.0, 0.0, 612.0, 792.0),
            render_width_px=1224,
            render_height_px=1584,
        )
    )
    assert normalize_bbox_xyxy(
        (61.2, 79.2, 306.0, 396.0), points, coordinate_space='source_units'
    ) == pytest.approx((0.1, 0.1, 0.5, 0.5))
    assert normalize_bbox_xyxy((0.0, 0.0, 1224.0, 1584.0), points) == pytest.approx(
        (0.0, 0.0, 1.0, 1.0)
    )


@pytest.mark.parametrize(
    ('bbox', 'coordinate_space'),
    [
        ((0.0, 0.0, 0.0, 1.0), 'rendered_pixels'),
        ((-1.0, 0.0, 1.0, 1.0), 'rendered_pixels'),
        ((0.0, 0.0, 201.0, 1.0), 'rendered_pixels'),
        ((0.0, 0.0, float('nan'), 1.0), 'source_units'),
    ],
)
def test_geometry_rejects_invalid_boxes(
    bbox: tuple[float, float, float, float], coordinate_space: str
) -> None:
    transform = SourceTransform.model_validate(make_transform())
    with pytest.raises(ValueError):
        normalize_bbox_xyxy(bbox, transform, coordinate_space=coordinate_space)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    'mutation',
    [
        {'source_width': 0.0},
        {'source_height': float('inf')},
        {'crop_xyxy': (0.0, 0.0, 0.0, 10.0)},
        {'crop_xyxy': (-1.0, 0.0, 10.0, 10.0)},
        {'crop_xyxy': (0.0, 0.0, 201.0, 100.0)},
        {'render_width_px': 0},
        {'rotation_degrees': 45},
    ],
)
def test_source_transform_rejects_invalid_values(mutation: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SourceTransform.model_validate(make_transform(**mutation))


@pytest.mark.parametrize('rotation', [False, 0.0])
def test_source_transform_json_requires_integer_rotation(rotation: object) -> None:
    payload = make_transform(rotation_degrees=rotation)

    with pytest.raises(ValidationError):
        SourceTransform.model_validate_json(json.dumps(payload))


def test_packaged_schema_matches_model_schema() -> None:
    schema_path = (
        Path(__file__).parents[1]
        / 'src'
        / 'marie_parsebench'
        / 'document-v1.schema.json'
    )
    assert (
        json.loads(schema_path.read_text(encoding='utf-8'))
        == DocumentArtifact.model_json_schema()
    )
