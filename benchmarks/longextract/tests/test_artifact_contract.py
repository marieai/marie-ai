from __future__ import annotations

from pathlib import Path

import pytest
from marie_longextract.tools.artifacts import (
    read_json,
    require_benchmark_metadata,
    write_json,
)


class FakeStorage:
    values: dict[str, bytes] = {}

    @classmethod
    def read(cls, uri: str) -> bytes:
        return cls.values[uri]

    @classmethod
    def write(cls, source: bytes, destination: str, overwrite: bool = False) -> bool:
        assert overwrite is True
        cls.values[destination] = source
        return True


def _metadata(**benchmark_overrides) -> dict:
    benchmark = {
        'schema_uri': 's3://bucket/schema.json',
        'output_uri': 's3://bucket/result.json',
        **benchmark_overrides,
    }
    return {'content_type': 'application/pdf', 'benchmark': benchmark}


@pytest.mark.parametrize(
    ('metadata', 'message'),
    [
        ({'content_type': 'application/pdf'}, 'metadata.benchmark'),
        ({'benchmark': {}}, 'metadata.content_type'),
        (
            {'content_type': 'text/plain', 'benchmark': {}},
            'Unsupported benchmark content type',
        ),
        (_metadata(schema_uri=''), 'schema_uri'),
        (_metadata(output_uri=''), 'output_uri'),
    ],
)
def test_invalid_metadata_fails(metadata: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        require_benchmark_metadata(metadata)


def test_missing_work_uri_derives_sibling_prefix() -> None:
    assert require_benchmark_metadata(_metadata()) == (
        's3://bucket/schema.json',
        's3://bucket/result.json',
        's3://bucket/work/',
    )


def test_json_artifacts_use_requested_uri() -> None:
    FakeStorage.values = {}
    write_json('s3://bucket/result.json', {'answer': 42}, storage=FakeStorage)
    assert read_json('s3://bucket/result.json', storage=FakeStorage) == {'answer': 42}


def test_read_json_accepts_local_path(tmp_path: Path) -> None:
    schema_path = tmp_path / 'schema.json'
    schema_path.write_text('{"type": "object"}', encoding='utf-8')

    assert read_json(str(schema_path)) == {'type': 'object'}


def test_runtime_source_never_references_benchmark_answers() -> None:
    source_root = Path(__file__).resolve().parents[1] / 'src'
    forbidden_name = 'ground' + '_truth'
    assert not any(
        forbidden_name in path.read_text(encoding='utf-8')
        for path in source_root.rglob('*.py')
    )
