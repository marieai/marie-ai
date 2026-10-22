from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import (
    ArtifactIdentity,
    Element,
    ElementKind,
    PageArtifact,
    ProducerIdentity,
    SourceTransform,
    Usage,
    normalize_bbox_xyxy,
)

ANNOTATOR_NAME = 'parsebench-page'
RESPONSE_HANDLER = 'marie_parsebench.page_parsing.handle_page_response'
HANDLER_PROTOCOL_VERSION = 'marie.parsebench.page-handler.v5'
SOURCE_CONTEXT_SCHEMA_VERSION = 'marie.parsebench.source-context.v1'
MAX_TRANSPORT_ATTEMPTS = 3
_RESPONSE_FENCE = re.compile(
    r'\A```[ \t]*(?:json)?[ \t]*\r?\n(?P<payload>.*?)\r?\n```[ \t]*\Z',
    re.IGNORECASE | re.DOTALL,
)
RESERVED_ATTRIBUTES = {
    'marie.element_id',
    'marie.page_id',
    'marie.geometry_source',
    'marie.original_source_page_index',
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class PageBinding(StrictModel):
    local_page_number: Annotated[int, Field(gt=0, strict=True)]
    page_index: Annotated[int, Field(ge=0, strict=True)]
    original_source_page_index: Annotated[int, Field(ge=0, strict=True)] | None = None
    width_px: Annotated[int, Field(gt=0, strict=True)]
    height_px: Annotated[int, Field(gt=0, strict=True)]
    source_transform: SourceTransform

    @field_validator('source_transform', mode='before')
    @classmethod
    def accept_json_transform(cls, value: object) -> object:
        if isinstance(value, Mapping):
            value = dict(value)
            crop = value.get('crop_xyxy')
            if isinstance(crop, list):
                value['crop_xyxy'] = tuple(crop)
        return value

    @model_validator(mode='after')
    def validate_render_dimensions(self) -> PageBinding:
        if (
            self.width_px != self.source_transform.render_width_px
            or self.height_px != self.source_transform.render_height_px
        ):
            raise ValueError(
                'page dimensions must match source transform render dimensions'
            )
        return self


class ParseBenchParameters(StrictModel):
    identity: ArtifactIdentity
    producer: ProducerIdentity
    pages: Annotated[list[PageBinding], Field(min_length=1)]

    @model_validator(mode='after')
    def validate_page_inventory(self) -> ParseBenchParameters:
        local_pages = [page.local_page_number for page in self.pages]
        if local_pages != list(range(1, len(self.pages) + 1)):
            raise ValueError('local page numbers must cover 1..N in frame order')
        source_pages = [page.page_index for page in self.pages]
        if len(source_pages) != len(set(source_pages)):
            raise ValueError('source page indices must be unique')
        return self


CandidateBox = tuple[float, float, float, float]

_HTML_VOID_ELEMENTS = {
    'area',
    'base',
    'br',
    'col',
    'embed',
    'hr',
    'img',
    'input',
    'link',
    'meta',
    'param',
    'source',
    'track',
    'wbr',
}


class _TableFragmentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.has_table = False
        self.has_table_header = False
        self.is_balanced = True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == 'table':
            self.has_table = True
        elif tag == 'th' and 'table' in self.stack:
            self.has_table_header = True
        if tag not in _HTML_VOID_ELEMENTS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {'table', 'th'}:
            self.is_balanced = False

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            self.is_balanced = False
            return
        self.stack.pop()

    def valid_table(self, *, require_header: bool) -> bool:
        return (
            self.is_balanced
            and not self.stack
            and self.has_table
            and (self.has_table_header or not require_header)
        )


def _contains_balanced_table(content: str, *, require_header: bool) -> bool:
    parser = _TableFragmentParser()
    try:
        parser.feed(content)
        parser.close()
    except Exception:
        return False
    return parser.valid_table(require_header=require_header)


class CandidateElement(StrictModel):
    kind: ElementKind
    content: str
    content_format: Literal['markdown', 'html']
    bbox_xyxy: CandidateBox | None = None
    attributes: dict[str, str] = Field(default_factory=dict)

    @field_validator('bbox_xyxy')
    @classmethod
    def validate_box(cls, value: CandidateBox | None) -> CandidateBox | None:
        if value is None:
            return value
        if not all(math.isfinite(coordinate) for coordinate in value):
            raise ValueError('bbox coordinates must be finite')
        x1, y1, x2, y2 = value
        if not (x1 < x2 and y1 < y2):
            raise ValueError('bbox must have positive area')
        return value

    @field_validator('attributes')
    @classmethod
    def reject_host_attributes(cls, value: dict[str, str]) -> dict[str, str]:
        reserved = RESERVED_ATTRIBUTES.intersection(value)
        if reserved:
            raise ValueError(
                f'candidate attributes are host-owned: {", ".join(sorted(reserved))}'
            )
        return value

    @model_validator(mode='after')
    def validate_semantics(self) -> CandidateElement:
        if self.kind != 'picture' and not self.content.strip():
            raise ValueError('content must be nonblank unless the element is a picture')
        if self.kind == 'table':
            if self.content_format != 'html':
                raise ValueError('table content must use HTML')
            if not _contains_balanced_table(self.content, require_header=False):
                raise ValueError('table content must contain an HTML table')
        elif self.kind == 'chart':
            if self.content_format != 'html' or not _contains_balanced_table(
                self.content, require_header=True
            ):
                raise ValueError('chart content must be a labeled HTML table')
        elif self.content_format != 'markdown':
            raise ValueError('non-table and non-chart content must use Markdown')
        return self


class PageCandidate(StrictModel):
    status: Literal['complete', 'blank']
    elements: list[CandidateElement]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_status(self) -> PageCandidate:
        if self.status == 'complete' and not self.elements:
            raise ValueError('a complete page candidate must contain elements')
        if self.status == 'blank' and self.elements:
            raise ValueError('a blank page candidate must not contain elements')
        return self


def _response_candidate_json(response: str) -> str:
    stripped = response.strip()
    if not stripped.startswith('```'):
        return response
    match = _RESPONSE_FENCE.fullmatch(stripped)
    if match is None:
        raise ValueError(
            'model response code fence must be complete, outer, and json or unlabelled'
        )
    return match.group('payload')


@dataclass(frozen=True, slots=True)
class LoadedPageAnnotatorConfig:
    annotator_conf: dict[str, Any]
    prompt_template: str
    config_sha256: str


def _candidate_schema() -> dict[str, Any]:
    return PageCandidate.model_json_schema()


def output_candidate_schema_json() -> str:
    return json.dumps(_candidate_schema(), sort_keys=True, separators=(',', ':'))


def _validate_annotator_config(annotator_conf: Mapping[str, Any]) -> None:
    allowed = {
        'name',
        'annotator_type',
        'mode',
        'model_config',
        'response_handler',
        'parser',
        'validators',
    }
    extras = set(annotator_conf) - allowed
    if extras:
        raise ValueError(f'unsupported annotator settings: {sorted(extras)}')
    expected = {
        'name': ANNOTATOR_NAME,
        'annotator_type': 'llm',
        'mode': 'per-page',
        'response_handler': RESPONSE_HANDLER,
        'parser': 'noop',
        'validators': [],
    }
    for key, value in expected.items():
        if annotator_conf.get(key) != value:
            raise ValueError(f'{key} must be {value!r}')

    model = annotator_conf.get('model_config')
    if not isinstance(model, Mapping):
        raise ValueError('model_config must be an object')
    allowed_model = {
        'model_name',
        'multimodal',
        'prompt_path',
        'system_prompt_text',
        'top_p',
        'frequency_penalty',
        'presence_penalty',
        'temperature',
        'expect_output',
        'max_tokens',
        'mini_batch_size',
        'refine_passes',
        'min_pixels',
        'max_pixels',
    }
    model_extras = set(model) - allowed_model
    if model_extras:
        raise ValueError(f'unsupported model settings: {sorted(model_extras)}')
    fixed = {
        'model_name': 'qwen_v3_30b_instruct',
        'multimodal': True,
        'top_p': 1.0,
        'frequency_penalty': 0,
        'presence_penalty': 0,
        'temperature': 0.0,
        'expect_output': 'json',
        'max_tokens': 8192,
        'mini_batch_size': 1,
        'refine_passes': 0,
        'min_pixels': 401408,
        'max_pixels': 2007040,
    }
    for key, value in fixed.items():
        actual = model.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f'{key} must be {value!r}')
    if (
        not isinstance(model.get('prompt_path'), str)
        or not model['prompt_path'].strip()
    ):
        raise ValueError('prompt_path must be a nonblank relative path')
    if (
        not isinstance(model.get('system_prompt_text'), str)
        or not model['system_prompt_text'].strip()
    ):
        raise ValueError('system_prompt_text must be nonblank')


def config_sha256(
    annotator_conf: Mapping[str, Any],
    *,
    prompt_template: str,
    candidate_schema: Mapping[str, Any] | None = None,
) -> str:
    _validate_annotator_config(annotator_conf)
    if not isinstance(prompt_template, str) or not prompt_template:
        raise ValueError('prompt_template must be nonblank')
    normalized = json.loads(json.dumps(annotator_conf))
    normalized['model_config'].pop('prompt_path', None)
    payload = {
        'handler_protocol_version': HANDLER_PROTOCOL_VERSION,
        'source_context_schema_version': SOURCE_CONTEXT_SCHEMA_VERSION,
        'max_transport_attempts': MAX_TRANSPORT_ATTEMPTS,
        'annotator': normalized,
        'prompt_template': prompt_template,
        'candidate_schema': candidate_schema or _candidate_schema(),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_page_annotator_config(config_path: str | Path) -> LoadedPageAnnotatorConfig:
    import yaml

    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(raw, Mapping):
        raise ValueError('annotator config must be an object')
    annotators = raw.get('annotators')
    if not isinstance(annotators, Mapping) or set(annotators) != {ANNOTATOR_NAME}:
        raise ValueError(f'config must define only the {ANNOTATOR_NAME} annotator')
    section = annotators[ANNOTATOR_NAME]
    if not isinstance(section, Mapping):
        raise ValueError(f'{ANNOTATOR_NAME} config must be an object')
    annotator_conf = json.loads(json.dumps(section))
    annotator_conf['name'] = ANNOTATOR_NAME
    _validate_annotator_config(annotator_conf)

    prompt_path = Path(annotator_conf['model_config']['prompt_path'])
    if prompt_path.is_absolute() or '..' in prompt_path.parts:
        raise ValueError('prompt_path must stay inside the annotator config directory')
    prompt_template = (path.parent / prompt_path).read_text(encoding='utf-8').strip()
    return LoadedPageAnnotatorConfig(
        annotator_conf=annotator_conf,
        prompt_template=prompt_template,
        config_sha256=config_sha256(annotator_conf, prompt_template=prompt_template),
    )


def parse_run_parameters(run_context: Any) -> ParseBenchParameters:
    if run_context is None:
        raise ValueError('ParseBench requires RunContext')
    parameters = getattr(run_context, 'parameters', None)
    if not isinstance(parameters, Mapping):
        raise ValueError('RunContext.parameters must be an object')
    nested = parameters.get('parsebench')
    if not isinstance(nested, Mapping):
        raise ValueError("RunContext.parameters['parsebench'] must be an object")
    return ParseBenchParameters.model_validate(nested)


def _page_binding(unit: Any, parameters: ParseBenchParameters) -> PageBinding:
    page_number = getattr(unit, 'page_number', None)
    if type(page_number) is not int or not 1 <= page_number <= len(parameters.pages):
        raise ValueError('processing unit has an invalid local page number')
    binding = parameters.pages[page_number - 1]
    unit_binding = PageBinding.model_validate(getattr(unit, 'data', None))
    if unit_binding != binding:
        raise ValueError('processing unit page binding does not match RunContext')
    return binding


def _validate_document_inventory(
    document: Any, parameters: ParseBenchParameters
) -> None:
    if getattr(document, 'page_count', None) != len(parameters.pages):
        raise ValueError('document page count does not match ParseBench page bindings')
    source_metadata = getattr(document, 'source_metadata', None)
    if not isinstance(source_metadata, Mapping):
        raise ValueError('document source metadata must be an object')
    if source_metadata.get('pages') != len(parameters.pages):
        raise ValueError('document source page coverage does not match page bindings')
    ocr = source_metadata.get('ocr')
    if ocr is not None and (
        not isinstance(ocr, list) or len(ocr) != len(parameters.pages)
    ):
        raise ValueError('source OCR page coverage does not match page bindings')


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f'{label} must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{label} must be finite')
    return result


def source_context(document: Any, binding: PageBinding) -> dict[str, Any]:
    metadata = document.source_metadata
    extraction = metadata.get('extraction')
    if isinstance(extraction, Mapping) and extraction.get('ocr_invoked') is False:
        raise ValueError('source OCR cannot come from a non-OCR extraction')
    ocr_pages = metadata.get('ocr')
    lines: list[dict[str, Any]] = []
    if ocr_pages is not None:
        page = ocr_pages[binding.local_page_number - 1]
        if not isinstance(page, Mapping):
            raise ValueError('source OCR page must be an object')
        meta = page.get('meta')
        if not isinstance(meta, Mapping):
            raise ValueError('source OCR meta must be an object')
        if meta.get('page') != binding.local_page_number - 1:
            raise ValueError('source OCR local page does not match page binding')
        size = meta.get('imageSize')
        if not isinstance(size, Mapping) or size != {
            'width': binding.width_px,
            'height': binding.height_px,
        }:
            raise ValueError('source OCR dimensions do not match page binding')
        if meta.get('format') != 'xywh':
            raise ValueError('source OCR boxes must use xywh format')
        raw_lines = page.get('lines', [])
        if not isinstance(raw_lines, list):
            raise ValueError('source OCR lines must be an array')
        for raw_line in raw_lines:
            if not isinstance(raw_line, Mapping):
                raise ValueError('source OCR line must be an object')
            line_number = raw_line.get('line')
            text = raw_line.get('text')
            bbox = raw_line.get('bbox')
            if type(line_number) is not int or line_number < 0:
                raise ValueError('source OCR line number must be a nonnegative integer')
            if not isinstance(text, str):
                raise ValueError('source OCR line text must be a string')
            if (
                not isinstance(bbox, Sequence)
                or isinstance(bbox, (str, bytes))
                or len(bbox) != 4
            ):
                raise ValueError('source OCR line bbox must contain four numbers')
            x, y, width, height = (
                _number(value, 'source OCR line bbox') for value in bbox
            )
            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > binding.width_px
                or y + height > binding.height_px
            ):
                raise ValueError('source OCR line bbox is outside the rendered page')
            lines.append(
                {
                    'line_number': line_number,
                    'text': text,
                    'bbox_xywh': [x, y, width, height],
                    'provenance': 'source_ocr',
                }
            )
    return {
        'schema_version': SOURCE_CONTEXT_SCHEMA_VERSION,
        'local_page_number': binding.local_page_number,
        'source_page_index': binding.page_index,
        'original_source_page_index': binding.original_source_page_index,
        'width_px': binding.width_px,
        'height_px': binding.height_px,
        'bbox_format': 'xywh',
        'ocr_lines': lines,
    }


def prompt_variables(document: Any, binding: PageBinding) -> dict[str, str]:
    return {
        'OUTPUT_CANDIDATE_SCHEMA_JSON': output_candidate_schema_json(),
        'SOURCE_CONTEXT_JSON': json.dumps(
            source_context(document, binding),
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
        ),
    }


def render_page_prompt(
    prompt_template: str, document: Any, binding: PageBinding
) -> str:
    variables = prompt_variables(document, binding)
    checked_template = prompt_template
    markers: dict[str, str] = {}
    for name in variables:
        marker = '{{ ' + name + ' }}'
        if checked_template.count(marker) != 1:
            raise ValueError(f'prompt template must contain exactly one {marker}')
        checked_template = checked_template.replace(marker, '')
        markers[name] = marker
    if '{{' in checked_template or '{%' in checked_template or '{#' in checked_template:
        raise ValueError('prompt template contains unsupported Jinja syntax')

    rendered = prompt_template
    for name, value in variables.items():
        marker = markers[name]
        rendered = rendered.replace(marker, value)
    return rendered


def validate_document_bindings(document: Any, parameters: ParseBenchParameters) -> None:
    _validate_document_inventory(document, parameters)
    for binding in parameters.pages:
        source_context(document, binding)


def _usage(attempts: Sequence[Any]) -> Usage | None:
    if not attempts:
        return None

    def total(field: str) -> int | float | None:
        values = [getattr(attempt, field, None) for attempt in attempts]
        return sum(values) if all(value is not None for value in values) else None

    return Usage(
        input_tokens=total('input_tokens'),
        output_tokens=total('output_tokens'),
        cost_usd=total('cost_usd'),
        latency_ms=sum(float(attempt.elapsed_ms) for attempt in attempts),
    )


def _page_id(identity: ArtifactIdentity, page_index: int) -> str:
    return f'{identity.example_id}:page:{page_index}'


def _artifact(
    candidate: PageCandidate,
    parameters: ParseBenchParameters,
    binding: PageBinding,
    attempts: Sequence[Any],
) -> PageArtifact:
    page_id = _page_id(parameters.identity, binding.page_index)
    elements = []
    for ordinal, candidate_element in enumerate(candidate.elements, 1):
        box = candidate_element.bbox_xyxy
        elements.append(
            Element(
                element_id=f'{page_id}:element:{ordinal:04d}',
                kind=candidate_element.kind,
                content=candidate_element.content,
                content_format=candidate_element.content_format,
                bbox_xyxy_normalized=(
                    normalize_bbox_xyxy(
                        box,
                        binding.source_transform,
                        coordinate_space='rendered_pixels',
                    )
                    if box is not None
                    else None
                ),
                geometry_source='vlm_localization' if box is not None else None,
                attributes=candidate_element.attributes,
            )
        )
    return PageArtifact(
        format_version='marie.parsebench.page.v1',
        identity=parameters.identity,
        producer=parameters.producer,
        page_id=page_id,
        page_index=binding.page_index,
        original_source_page_index=binding.original_source_page_index,
        width_px=binding.width_px,
        height_px=binding.height_px,
        source_transform=binding.source_transform,
        status=candidate.status,
        elements=elements,
        warnings=candidate.warnings,
        usage=_usage(attempts),
    )


def _failed_artifact(
    error: Exception,
    parameters: ParseBenchParameters,
    binding: PageBinding,
    attempts: Sequence[Any],
) -> PageArtifact:
    return PageArtifact(
        format_version='marie.parsebench.page.v1',
        identity=parameters.identity,
        producer=parameters.producer,
        page_id=_page_id(parameters.identity, binding.page_index),
        page_index=binding.page_index,
        original_source_page_index=binding.original_source_page_index,
        width_px=binding.width_px,
        height_px=binding.height_px,
        source_transform=binding.source_transform,
        status='failed',
        elements=[],
        warnings=[f'{type(error).__name__}: {error}'],
        usage=_usage(attempts),
    )


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _attempt_metadata(attempt: Any) -> dict[str, Any]:
    if is_dataclass(attempt):
        values = asdict(attempt)
    else:
        values = vars(attempt)
    return {
        key: values.get(key)
        for key in (
            'task_id',
            'attempt_number',
            'finish_reason',
            'response_model',
            'input_tokens',
            'output_tokens',
            'elapsed_ms',
            'error',
            'cost_usd',
        )
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def handle_page_response(
    *,
    response: str | None,
    prompt: str,
    prompt_template: str,
    image_path: str,
    image: Any,
    unit: Any,
    document: Any,
    run_context: Any,
    annotator_conf: Mapping[str, Any],
    attempts: Sequence[Any],
    error: Exception | None,
    output_path: str,
) -> dict[str, Any]:
    parameters = parse_run_parameters(run_context)
    binding = _page_binding(unit, parameters)
    stem = Path(image_path).stem
    try:
        validate_document_bindings(document, parameters)
        if tuple(image.size) != (binding.width_px, binding.height_px):
            raise ValueError('runtime image dimensions do not match page binding')
        if len(attempts) > MAX_TRANSPORT_ATTEMPTS:
            raise ValueError('transport attempt count exceeds the recorded limit')
        effective_hash = config_sha256(annotator_conf, prompt_template=prompt_template)
        if effective_hash != parameters.identity.config_sha256:
            raise ValueError('effective configuration hash does not match identity')
        expected_prompt = render_page_prompt(prompt_template, document, binding)
        if expected_prompt != prompt:
            raise ValueError(
                'rendered prompt contains inputs outside the source contract'
            )

        _write_json_once(
            Path(output_path) / 'evidence' / f'{stem}.json',
            {
                'handler_protocol_version': HANDLER_PROTOCOL_VERSION,
                'source_context_schema_version': SOURCE_CONTEXT_SCHEMA_VERSION,
                'config_sha256': effective_hash,
                'prompt_template_sha256': _sha256_text(prompt_template),
                'rendered_prompt_sha256': _sha256_text(prompt),
                'candidate_schema_sha256': _sha256_text(output_candidate_schema_json()),
                'image': {'width_px': image.width, 'height_px': image.height},
                'attempts': [_attempt_metadata(attempt) for attempt in attempts],
            },
        )
        if error is not None:
            raise error
        if response is None:
            raise ValueError('model response is missing')
        candidate = PageCandidate.model_validate_json(
            _response_candidate_json(response)
        )
        return _artifact(candidate, parameters, binding, attempts).model_dump(
            mode='json'
        )
    except Exception as exc:
        _write_json_once(
            Path(output_path) / 'failed' / f'{stem}.json',
            _failed_artifact(exc, parameters, binding, attempts).model_dump(
                mode='json'
            ),
        )
        raise
