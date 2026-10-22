"""Runtime artifact contracts for normalized ParseBench output."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonblankString = Annotated[str, StringConstraints(min_length=1, pattern=r'.*\S.*')]
Sha256 = Annotated[str, StringConstraints(pattern=r'^[0-9a-f]{64}$')]
FiniteNonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]
NormalizedBox = tuple[
    Annotated[float, Field(allow_inf_nan=False, strict=True)],
    Annotated[float, Field(allow_inf_nan=False, strict=True)],
    Annotated[float, Field(allow_inf_nan=False, strict=True)],
    Annotated[float, Field(allow_inf_nan=False, strict=True)],
]


class ContractModel(BaseModel):
    """Strict base configuration shared by all wire models."""

    model_config = ConfigDict(extra='forbid', strict=True)


class ArtifactIdentity(ContractModel):
    run_id: NonblankString
    example_id: NonblankString
    pipeline_name: NonblankString
    source_sha256: Sha256
    config_sha256: Sha256
    ref_id: NonblankString
    ref_type: NonblankString
    dag_id: NonblankString


class ProducerIdentity(ContractModel):
    name: NonblankString
    version: NonblankString
    job_id: NonblankString | None = None


class Usage(ContractModel):
    input_tokens: Annotated[int, Field(ge=0, strict=True)] | None = None
    output_tokens: Annotated[int, Field(ge=0, strict=True)] | None = None
    cost_usd: FiniteNonnegativeFloat | None = None
    latency_ms: FiniteNonnegativeFloat | None = None


class SourceTransform(ContractModel):
    source_width: PositiveFiniteFloat
    source_height: PositiveFiniteFloat
    source_unit: Literal['pixels', 'points']
    crop_xyxy: tuple[float, float, float, float]
    rotation_degrees: Literal[0, 90, 180, 270]
    render_width_px: Annotated[int, Field(gt=0, strict=True)]
    render_height_px: Annotated[int, Field(gt=0, strict=True)]

    @field_validator('rotation_degrees', mode='before')
    @classmethod
    def require_integer_rotation(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError('rotation must be an integer')
        return value

    @model_validator(mode='after')
    def validate_crop(self) -> SourceTransform:
        x1, y1, x2, y2 = self.crop_xyxy
        if not all(math.isfinite(value) for value in self.crop_xyxy):
            raise ValueError('crop coordinates must be finite')
        if not (
            0 <= x1 < x2 <= self.source_width and 0 <= y1 < y2 <= self.source_height
        ):
            raise ValueError('crop must have positive area inside the source')
        return self


ElementKind = Literal[
    'text',
    'heading',
    'list',
    'table',
    'chart',
    'picture',
    'caption',
    'formula',
    'page_header',
    'page_footer',
    'footnote',
    'page_number',
    'code',
    'checkbox',
]
GeometrySource = Literal['ocr', 'layout_model', 'native_source', 'vlm_localization']


class Element(ContractModel):
    element_id: NonblankString
    kind: ElementKind
    content: str
    content_format: Literal['markdown', 'html']
    bbox_xyxy_normalized: NormalizedBox | None = None
    geometry_source: GeometrySource | None = None
    attributes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_content_and_geometry(self) -> Element:
        if self.kind != 'picture' and not self.content.strip():
            raise ValueError('content must be nonblank unless the element is a picture')
        if self.kind == 'table' and self.content_format != 'html':
            raise ValueError('table content must use HTML')
        if (self.bbox_xyxy_normalized is None) != (self.geometry_source is None):
            raise ValueError(
                'geometry box and provenance must both be present or absent'
            )
        if self.bbox_xyxy_normalized is not None:
            x1, y1, x2, y2 = self.bbox_xyxy_normalized
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError('normalized box must have positive area inside [0, 1]')
        return self


class PageArtifact(ContractModel):
    format_version: Literal['marie.parsebench.page.v1']
    identity: ArtifactIdentity
    producer: ProducerIdentity
    page_id: NonblankString
    page_index: Annotated[int, Field(ge=0, strict=True)]
    original_source_page_index: Annotated[int, Field(ge=0, strict=True)] | None = None
    width_px: Annotated[int, Field(gt=0, strict=True)]
    height_px: Annotated[int, Field(gt=0, strict=True)]
    source_transform: SourceTransform
    status: Literal['complete', 'blank', 'failed']
    elements: list[Element]
    warnings: list[str] = Field(default_factory=list)
    usage: Usage | None = None

    @model_validator(mode='after')
    def validate_elements(self) -> PageArtifact:
        element_ids = [element.element_id for element in self.elements]
        if len(element_ids) != len(set(element_ids)):
            raise ValueError('element IDs must be unique within a page')
        if self.status == 'complete' and not self.elements:
            raise ValueError('a complete page must contain at least one element')
        if self.status == 'blank' and self.elements:
            raise ValueError('a blank page must not contain elements')
        return self


class DocumentArtifact(ContractModel):
    format_version: Literal['marie.parsebench.document.v1']
    identity: ArtifactIdentity
    producer: ProducerIdentity
    status: Literal['complete']
    source_page_count: Annotated[int, Field(gt=0, strict=True)]
    expected_page_indices: list[Annotated[int, Field(ge=0, strict=True)]]
    expected_page_count: Annotated[int, Field(gt=0, strict=True)]
    pages: list[PageArtifact]
    warnings: list[str] = Field(default_factory=list)
    usage: Usage | None = None

    @model_validator(mode='after')
    def validate_page_inventory(self) -> DocumentArtifact:
        expected = self.expected_page_indices
        if self.expected_page_count != len(expected) or len(expected) != len(
            self.pages
        ):
            raise ValueError(
                'expected page count, indices and pages must have equal lengths'
            )
        if len(expected) != len(set(expected)):
            raise ValueError('expected page indices must be unique')
        if any(index >= self.source_page_count for index in expected):
            raise ValueError('expected page index is outside the source document')
        if [page.page_index for page in self.pages] != expected:
            raise ValueError('pages must match expected indices in selection order')
        page_ids = [page.page_id for page in self.pages]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError('document page IDs must be unique')
        if any(page.identity != self.identity for page in self.pages):
            raise ValueError('page identity must match document identity')
        if any(page.status == 'failed' for page in self.pages):
            raise ValueError('a complete document must not contain failed pages')
        return self


def normalize_bbox_xyxy(
    bbox: tuple[float, float, float, float],
    transform: SourceTransform,
    *,
    coordinate_space: Literal['rendered_pixels', 'source_units'] = 'rendered_pixels',
) -> tuple[float, float, float, float]:
    """Convert a source-unit or rendered-frame box to canonical coordinates."""
    if len(bbox) != 4 or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in bbox
    ):
        raise ValueError('bbox must contain four numbers')
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError('bbox coordinates must be finite')
    x1, y1, x2, y2 = bbox
    if not (x1 < x2 and y1 < y2):
        raise ValueError('bbox must have positive area')

    if coordinate_space == 'source_units':
        if not (
            0 <= x1 < x2 <= transform.source_width
            and 0 <= y1 < y2 <= transform.source_height
        ):
            raise ValueError('source-unit bbox is outside the source')
        return (
            x1 / transform.source_width,
            y1 / transform.source_height,
            x2 / transform.source_width,
            y2 / transform.source_height,
        )
    if coordinate_space != 'rendered_pixels':
        raise ValueError(f'unknown coordinate space: {coordinate_space}')
    if not (
        0 <= x1 < x2 <= transform.render_width_px
        and 0 <= y1 < y2 <= transform.render_height_px
    ):
        raise ValueError('rendered bbox is outside the rendered frame')

    crop_x1, crop_y1, crop_x2, crop_y2 = transform.crop_xyxy
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    rotated_width = (
        crop_width if transform.rotation_degrees in (0, 180) else crop_height
    )
    rotated_height = (
        crop_height if transform.rotation_degrees in (0, 180) else crop_width
    )
    scaled_x1 = x1 * rotated_width / transform.render_width_px
    scaled_x2 = x2 * rotated_width / transform.render_width_px
    scaled_y1 = y1 * rotated_height / transform.render_height_px
    scaled_y2 = y2 * rotated_height / transform.render_height_px

    corners = (
        (scaled_x1, scaled_y1),
        (scaled_x1, scaled_y2),
        (scaled_x2, scaled_y1),
        (scaled_x2, scaled_y2),
    )
    if transform.rotation_degrees == 0:
        unrotated = corners
    elif transform.rotation_degrees == 90:
        unrotated = tuple((y, crop_height - x) for x, y in corners)
    elif transform.rotation_degrees == 180:
        unrotated = tuple((crop_width - x, crop_height - y) for x, y in corners)
    else:
        unrotated = tuple((crop_width - y, x) for x, y in corners)

    source_x = [crop_x1 + x for x, _ in unrotated]
    source_y = [crop_y1 + y for _, y in unrotated]
    return (
        min(source_x) / transform.source_width,
        min(source_y) / transform.source_height,
        max(source_x) / transform.source_width,
        max(source_y) / transform.source_height,
    )
