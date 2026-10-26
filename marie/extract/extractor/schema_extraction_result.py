from __future__ import annotations

import copy
import json
from collections import defaultdict
from typing import Any

from jsonschema import SchemaError, ValidationError
from jsonschema.validators import validator_for

from marie.extract.extractor.grounding import ground_text, parse_ocr_meta


def _field_path(path: Any) -> str:
    return '.'.join(part for part in path if isinstance(part, str))


def _missing_paths(error: ValidationError) -> set[str] | None:
    if error.validator == 'required':
        prefix = _field_path(error.absolute_path)
        return {
            f'{prefix}.{name}' if prefix else name
            for name in error.validator_value
            if name not in error.instance
        }
    if error.validator in ('anyOf', 'oneOf') and error.context:
        branches = defaultdict(list)
        for child in error.context:
            branches[child.schema_path[0]].append(child)
        alternatives = []
        for errors in branches.values():
            missing = [_missing_paths(child) for child in errors]
            if all(paths is not None for paths in missing):
                alternatives.append(set().union(*missing))
        if alternatives:
            return min(alternatives, key=lambda paths: (len(paths), sorted(paths)))
    return None


def _validate(validator: Any, data: dict, context: str) -> set[str]:
    missing = set()
    for error in validator.iter_errors(data):
        paths = _missing_paths(error)
        if paths is None:
            path = _field_path(error.absolute_path) or '<root>'
            raise ValueError(
                f'Invalid extraction output {context} at {path}: violates {error.validator}'
            )
        missing.update(paths)
    return missing


def _merge(target: dict, source: dict) -> None:
    for name, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(name), dict):
            _merge(target[name], value)
        elif isinstance(value, list) and isinstance(target.get(name), list):
            target[name].extend(copy.deepcopy(value))
        else:
            target[name] = copy.deepcopy(value)


def build_schema_extraction_result(
    schema: dict,
    page_results: list[dict],
    ocr_metadata: object,
    dag_id: str,
    model: str,
) -> dict:
    """Validate, merge and ground schema-shaped page extraction results.

    Missing required properties are reported after merging. Other schema errors
    raise ValueError. Plain schema values carry no model confidence.
    """
    validator_type = validator_for(schema)
    try:
        validator_type.check_schema(schema)
    except SchemaError as error:
        raise ValueError(f'Invalid extraction schema: {error.message}') from error
    validator = validator_type(schema, format_checker=validator_type.FORMAT_CHECKER)
    for result in page_results:
        page = result.get('page')
        if not isinstance(page, int) or isinstance(page, bool) or page < 0:
            raise ValueError('Extraction page must be an absolute zero-based integer')
        data = result.get('data')
        if not isinstance(data, dict):
            raise ValueError(
                f'Invalid extraction output on page {page}: expected an object'
            )
        try:
            json.dumps(data, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'Invalid extraction output on page {page}: expected finite JSON values'
            ) from error
        _validate(validator, data, f'on page {page}')

    ocr_pages = parse_ocr_meta(ocr_metadata)
    values = []
    occurrences: dict[str, int] = defaultdict(int)
    array_offsets: dict[str, int] = defaultdict(int)
    row_offsets: dict[tuple[str, int], int] = defaultdict(int)

    def flatten(
        data: Any,
        field_id: str,
        path: str,
        page: int,
        row_index: int | None,
        array_depth: int = 0,
    ) -> None:
        if isinstance(data, dict):
            for name, value in data.items():
                flatten(
                    value,
                    f'{field_id}.{name}' if field_id else name,
                    f'{path}.{name}' if path else name,
                    page,
                    row_index,
                    array_depth,
                )
        elif isinstance(data, list):
            start = array_offsets[path]
            array_offsets[path] += len(data)
            for index, value in enumerate(data):
                row_key = (field_id, array_depth)
                row = row_offsets[row_key]
                row_offsets[row_key] += 1
                flatten(
                    value,
                    field_id,
                    f'{path}[{start + index}]',
                    page,
                    row,
                    array_depth + 1,
                )
        elif data is not None:
            occurrence = occurrences[field_id]
            occurrences[field_id] += 1
            value = (
                data
                if isinstance(data, str)
                else json.dumps(data, ensure_ascii=False, allow_nan=False)
            )
            values.append(
                {
                    'id': f'{field_id}:{row_index if row_index is not None else ""}:{occurrence}',
                    'field_id': field_id,
                    'path': path,
                    'value': value,
                    'confidence': None,
                    'row_index': row_index,
                    'occurrence_index': occurrence,
                    'location': ground_text(
                        value,
                        ocr_pages,
                        page,
                        numeric=isinstance(data, (int, float))
                        and not isinstance(data, bool),
                    ),
                }
            )

    merged: dict[str, Any] = {}
    for result in sorted(page_results, key=lambda result: result['page']):
        flatten(result['data'], '', '', result['page'], None)
        _merge(merged, result['data'])
    missing = _validate(validator, merged, 'after merging pages')
    extracted_pages = {result['page'] for result in page_results}
    all_pages = extracted_pages | {page['page'] for page in ocr_pages}
    sizes = {
        page['page']: {
            'page': page['page'],
            'width': page['width'],
            'height': page['height'],
        }
        for page in ocr_pages
    }
    return {
        'dag_id': dag_id,
        'model': model,
        'pages': {
            'extracted': len(extracted_pages),
            'total': len(all_pages),
            'sizes': [sizes[page] for page in sorted(sizes)],
        },
        'values': values,
        'missing_required': sorted(missing),
    }
