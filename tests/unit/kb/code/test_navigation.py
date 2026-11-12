import hashlib

import pytest

from marie.kb.code.navigation import parse_file


def parse(tmp_path, text, suffix='.py', **options):
    path = tmp_path / f'code{suffix}'
    payload = text.encode('utf-8')
    path.write_bytes(payload)
    return parse_file(path, hashlib.sha256(payload).hexdigest(), **options)


def test_real_parser_preserves_nested_symbols_and_original_lines(tmp_path):
    result = parse(
        tmp_path,
        '# é\r\nclass Box:\r\n    def same(self):\r\n        return same()\r\n\r\ndef same():\r\n    pass\r\n',
    )
    assert result['status'] == 'ready'
    assert [s['qualified_name'] for s in result['symbols']] == [
        'Box',
        'Box.same',
        'Box.same.self',
        'same',
    ]
    assert result['symbols'][1]['span']['start_line'] == 3
    assert any(
        ref['name'] == 'same' and ref['line'] == 4 for ref in result['references']
    )


def test_real_tsx_uses_the_tsx_grammar(tmp_path):
    result = parse(
        tmp_path, 'export function Panel() { return <div>Hello</div>; }\n', '.tsx'
    )
    assert result['status'] == 'ready'
    assert result['language'] == 'tsx'
    assert result['symbols'][0]['name'] == 'Panel'


def test_empty_supported_file_has_a_truthful_empty_outline(tmp_path):
    result = parse(tmp_path, '# no declarations\n')
    assert result['status'] == 'ready'
    assert result['symbols'] == []


def test_broken_syntax_is_never_reported_as_a_complete_parse(tmp_path):
    assert parse(tmp_path, 'def unfinished(:\n    return 3\n')['status'] == 'partial'


def test_unsupported_format_retains_exact_read_availability(tmp_path):
    result = parse(tmp_path, 'plain text', '.txt')
    assert result['status'] == 'unsupported'
    assert result['symbols'] == []


def test_unavailable_grammar_is_reported_instead_of_failing_capture(
    tmp_path, monkeypatch
):
    from marie_plugins.document_extraction.providers.tree_sitter import (
        TreeSitterProvider,
    )

    monkeypatch.setattr(TreeSitterProvider, 'is_ready', lambda *a: False)
    assert parse(tmp_path, 'def hello():\n    pass\n')['status'] == 'unavailable'


def test_capture_digest_is_checked_before_parsing(tmp_path):
    path = tmp_path / 'changed.py'
    path.write_text('changed = True\n')
    with pytest.raises(ValueError, match='digest'):
        parse_file(path, 'a' * 64)


def test_bounded_symbol_output_is_explicitly_truncated(tmp_path):
    result = parse(
        tmp_path, ''.join(f'def f{i}():\n    pass\n' for i in range(25)), max_records=3
    )
    assert result['status'] == 'truncated'
    assert len(result['symbols']) <= 3


def test_navigation_drops_oversized_names_and_does_not_persist_docstrings(tmp_path):
    long_name = 'a' * 250
    result = parse(
        tmp_path,
        f'def {long_name}():\n    pass\n\ndef short():\n    """'
        + 'd' * 50000
        + '"""\n    pass\n',
    )
    assert result['status'] == 'truncated'
    assert [s['name'] for s in result['symbols']] == ['short']
    assert 'docstring' not in result['symbols'][0]
    assert all(len(s['name']) <= 200 for s in result['symbols'])


def test_partial_parse_remains_explicit_when_record_limit_is_reached(tmp_path):
    result = parse(
        tmp_path,
        'def first():\n    pass\ndef second():\n    pass\ndef broken(:\n',
        max_records=1,
    )
    assert result['status'] == 'truncated'
    assert result['parse_status'] == 'partial'
    assert result['symbols'][0]['parse_status'] == 'partial'
