"""Real tree-sitter provider checks for source-code formats."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from marie_plugins.document_extraction.detection import detect_format
from marie_plugins.document_extraction.dispatch import extract_document
from marie_plugins.document_extraction.providers.tree_sitter import TreeSitterProvider

PLUGIN_DIR = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / 'fixtures'

# language: (fixture, expected qualified name, expected kind, fixture has imports)
CODE_CASES = {
    'python': ('sample.py', 'InvoiceParser.parse', 'method', True),
    'typescript': ('sample.ts', 'InvoiceParser.parse', 'method', True),
    'javascript': ('sample.js', 'totalAmount', 'function', True),
    'go': ('sample.go', 'TotalAmount', 'function', True),
    'java': ('sample.java', 'InvoiceParser.totalAmount', 'method', True),
    'rust': ('sample.rs', 'total_amount', 'function', True),
    'c': ('sample.c', 'total_amount', 'function', True),
    'cpp': ('sample.cpp', 'InvoiceParser', 'class', True),
    'csharp': ('sample.cs', 'Invoicing.InvoiceParser.TotalAmount', 'method', True),
    'ruby': ('sample.rb', 'Invoicing.InvoiceParser.parse', 'method', False),
    'php': ('sample.php', 'total_amount', 'function', False),
    'kotlin': ('sample.kt', 'InvoiceParser.parse', 'method', True),
    'swift': ('sample.swift', 'totalAmount', 'function', True),
}


def test_fixture_covered_languages_are_advertised():
    assert frozenset(CODE_CASES) <= TreeSitterProvider().formats


@pytest.mark.parametrize('language', sorted(TreeSitterProvider().formats))
def test_every_advertised_language_is_ready_and_reachable(language):
    from marie_plugins.document_extraction.detection import EXTENSIONS

    provider = TreeSitterProvider()
    assert provider.is_ready(language), f'{language}: grammar or query broken'
    assert language in set(EXTENSIONS.values()), f'{language}: no extension mapping'


@pytest.mark.parametrize(('language', 'case'), CODE_CASES.items())
def test_code_symbols_json_artifact(language, case, tmp_path):
    fixture, qualified_name, kind, has_imports = case
    result = extract_document(
        path=str(FIXTURES / fixture),
        output_dir=str(tmp_path),
        output_format='json',
    )
    assert result['outcome'] == 'success'
    assert result['result_kind'] == 'structured_document'
    assert result['provenance']['provider'] == 'tree-sitter'
    assert result['provenance']['canonical_format'] == language
    assert result['metadata']['symbol_count'] > 0

    body = json.loads(Path(result['artifact']['path']).read_text())
    schema = json.loads((PLUGIN_DIR / 'schemas' / 'code-symbols-v1.json').read_text())
    Draft202012Validator(schema).validate(body)
    assert body['language'] == language
    assert bool(body['imports']) == has_imports
    symbols = {item['qualified_name']: item for item in body['symbols']}
    assert symbols[qualified_name]['kind'] == kind
    assert symbols[qualified_name]['span']['start_line'] >= 1
    assert 'references' not in body


def test_include_references_option(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.py'),
        output_dir=str(tmp_path),
        output_format='json',
        provider_options={'include_references': True},
    )
    body = json.loads(Path(result['artifact']['path']).read_text())
    schema = json.loads((PLUGIN_DIR / 'schemas' / 'code-symbols-v1.json').read_text())
    Draft202012Validator(schema).validate(body)
    assert {item['name'] for item in body['references']} >= {'loads', 'sum'}
    assert all(item['line'] >= 1 for item in body['references'])


def test_code_markdown_outline(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.py'),
        output_dir=str(tmp_path),
    )
    assert result['outcome'] == 'success'
    assert result['result_kind'] == 'semantic_document'
    outline = Path(result['artifact']['path']).read_text()
    assert '**class** `InvoiceParser`' in outline
    assert 'Parse invoice documents into line items.' in outline
    assert '`import json`' in outline


def test_jsdoc_comment_is_captured_via_doc_capture(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.js'),
        output_dir=str(tmp_path),
        output_format='json',
    )
    body = json.loads(Path(result['artifact']['path']).read_text())
    symbols = {item['qualified_name']: item for item in body['symbols']}
    assert (
        symbols['totalAmount']['docstring'] == 'Sum the amount field across line items.'
    )


# language: (qualified name, expected variable-level kind)
VARIABLE_CASES = {
    'typescript': ('LineItem.description', 'property'),
    'javascript': ('InvoiceParser.parse.path', 'parameter'),
    'go': ('Parse.data', 'variable'),
    'java': ('InvoiceParser.totalAmount.items', 'parameter'),
    'rust': ('InvoiceParser.currency', 'property'),
    'c': ('total_amount.count', 'parameter'),
    'cpp': ('InvoiceParser.totalAmount.amounts', 'parameter'),
    'csharp': ('Invoicing.InvoiceParser.TotalAmount.total', 'variable'),
    'ruby': ('Invoicing.InvoiceParser.parse.path', 'parameter'),
    'php': ('InvoiceParser.parse.path', 'parameter'),
    'kotlin': ('InvoiceParser.currency', 'property'),
    'swift': ('InvoiceParser.parse.path', 'parameter'),
}


@pytest.mark.parametrize(('language', 'case'), VARIABLE_CASES.items())
def test_variable_level_symbols_across_languages(language, case, tmp_path):
    fixture = CODE_CASES[language][0]
    qualified_name, kind = case
    result = extract_document(
        path=str(FIXTURES / fixture),
        output_dir=str(tmp_path),
        output_format='json',
    )
    body = json.loads(Path(result['artifact']['path']).read_text())
    hits = [s for s in body['symbols'] if s['qualified_name'] == qualified_name]
    assert any(s['kind'] == kind for s in hits), hits


def test_variables_parameters_and_properties_are_captured(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.py'),
        output_dir=str(tmp_path),
        output_format='json',
    )
    body = json.loads(Path(result['artifact']['path']).read_text())
    symbols = {item['qualified_name']: item for item in body['symbols']}
    assert symbols['InvoiceParser.parse.path']['kind'] == 'parameter'
    assert symbols['InvoiceParser.__init__.currency']['kind'] == 'property'
    assert symbols['InvoiceParser.parse.path']['parent'] == 'parse'
    # module-level assignments stay constants, not variables
    assert symbols['DEFAULT_CURRENCY']['kind'] == 'constant'


def test_module_level_constants_are_captured(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.py'),
        output_dir=str(tmp_path),
        output_format='json',
    )
    body = json.loads(Path(result['artifact']['path']).read_text())
    constants = {s['name'] for s in body['symbols'] if s['kind'] == 'constant'}
    assert 'DEFAULT_CURRENCY' in constants


def test_python_docstrings_are_captured(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.py'),
        output_dir=str(tmp_path),
        output_format='json',
    )
    body = json.loads(Path(result['artifact']['path']).read_text())
    symbols = {item['qualified_name']: item for item in body['symbols']}
    assert (
        symbols['InvoiceParser.parse']['docstring']
        == 'Read one invoice file and return its line items.'
    )


def test_cst_output_format(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.py'),
        output_dir=str(tmp_path),
        output_format='cst',
    )
    assert result['outcome'] == 'success'
    assert result['result_kind'] == 'structured_document'
    assert result['artifact']['media_type'] == 'text/plain'
    artifact = Path(result['artifact']['path'])
    assert artifact.suffix == '.txt'
    lines = artifact.read_text().splitlines()
    assert lines[0].startswith('0:0-') and lines[0].endswith(' module')
    assert not any(line != line.lstrip() for line in lines)
    assert any('function_definition' in line for line in lines)
    assert any('`InvoiceParser`' in line for line in lines)
    assert not any(line.split(' ', 1)[1].startswith('"') for line in lines)


def test_nodes_output_is_queryable_without_parsing(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.py'),
        output_dir=str(tmp_path),
        output_format='nodes',
    )
    assert result['outcome'] == 'success'
    assert result['artifact']['media_type'] == 'application/x-ndjson'
    artifact = Path(result['artifact']['path'])
    assert artifact.suffix == '.jsonl'

    rows = [json.loads(line) for line in artifact.read_text().splitlines()]
    by_id = {row['id']: row for row in rows}
    assert rows[0]['type'] == 'module' and rows[0]['parent'] is None
    assert all(row['parent'] is None or row['parent'] < row['id'] for row in rows)

    # structural query with plain dict lookups, no parser involved:
    # every method defined inside the InvoiceParser class
    class_row = next(row for row in rows if row['type'] == 'class_definition')

    def ancestors(row):
        while row['parent'] is not None:
            row = by_id[row['parent']]
            yield row

    methods = [
        row
        for row in rows
        if row['type'] == 'function_definition'
        and any(a['id'] == class_row['id'] for a in ancestors(row))
    ]
    assert len(methods) == 2
    leaf_texts = {row.get('text') for row in rows if 'text' in row}
    assert 'InvoiceParser' in leaf_texts


def test_cst_output_with_anonymous_tokens(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.py'),
        output_dir=str(tmp_path),
        output_format='cst',
        provider_options={'include_anonymous': True},
    )
    lines = Path(result['artifact']['path']).read_text().splitlines()
    assert any(line.split(' ', 1)[1] == '"def"' for line in lines)


def test_include_cst_alongside_symbols(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.py'),
        output_dir=str(tmp_path),
        output_format='json',
        provider_options={'include_cst': True},
    )
    body = json.loads(Path(result['artifact']['path']).read_text())
    schema = json.loads((PLUGIN_DIR / 'schemas' / 'code-symbols-v1.json').read_text())
    Draft202012Validator(schema).validate(body)
    assert body['symbols']
    assert body['cst'].splitlines()[0].endswith(' module')


def test_all_representations_in_one_artifact(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.py'),
        output_dir=str(tmp_path),
        output_format='json',
        provider_options={
            'include_cst': True,
            'include_markdown': True,
            'include_references': True,
        },
    )
    body = json.loads(Path(result['artifact']['path']).read_text())
    schema = json.loads((PLUGIN_DIR / 'schemas' / 'code-symbols-v1.json').read_text())
    Draft202012Validator(schema).validate(body)
    assert body['symbols']
    assert body['references']
    assert body['cst'].splitlines()[0].endswith(' module')
    assert '**class** `InvoiceParser`' in body['markdown']


def test_cst_output_unavailable_for_document_formats(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.docx'),
        output_dir=str(tmp_path),
        output_format='cst',
    )
    assert result['outcome'] == 'not_extractable'
    assert result['reason'] == 'no_ready_provider'


def test_code_detection_by_extension_and_shebang(tmp_path):
    assert detect_format(str(FIXTURES / 'sample.py')).canonical_format == 'python'
    assert detect_format(str(FIXTURES / 'sample.go')).canonical_format == 'go'

    script = tmp_path / 'tool'
    script.write_text('#!/usr/bin/env python3\nprint("hi")\n')
    detected = detect_format(str(script))
    assert detected.canonical_format == 'python'
    assert detected.evidence == ('content',)


def test_code_file_without_symbols_is_not_extractable(tmp_path):
    source = tmp_path / 'empty.py'
    source.write_text('# only a comment\n')
    result = extract_document(path=str(source), output_dir=str(tmp_path))
    assert result['outcome'] == 'not_extractable'
    assert result['reason'] == 'providers_exhausted'
