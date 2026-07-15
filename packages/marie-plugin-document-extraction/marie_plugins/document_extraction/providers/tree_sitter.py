"""Tree-sitter source code extraction provider.

Symbol extraction is driven by the vendored per-language ``queries/*-tags.scm``
files (standard tree-sitter tags queries); provider formats derive from that
directory. Imports are not tagged by the standard queries, so a small
per-language node-type table covers them.
"""

from __future__ import annotations

import json
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

from ..models import ProviderDocument, ResultKind
from .base import (
    ProviderNotExtractableError,
    ProviderUnavailableError,
)

SYMBOLS_SCHEMA_VERSION = '1.0'

_QUERIES_DIR = Path(__file__).resolve().parents[1] / 'queries'

_IMPORT_NODES = {
    'python': {'import_statement', 'import_from_statement'},
    'javascript': {'import_statement'},
    'typescript': {'import_statement'},
    'go': {'import_declaration'},
    'java': {'import_declaration'},
    'rust': {'use_declaration'},
    'c': {'preproc_include'},
    'cpp': {'preproc_include'},
    'csharp': {'using_directive'},
    'ruby': set(),
    'php': {'namespace_use_declaration'},
    'kotlin': {'import_header'},
    'swift': {'import_declaration'},
}


def _query_languages() -> frozenset[str]:
    return frozenset(
        path.name.removesuffix('-tags.scm') for path in _QUERIES_DIR.glob('*-tags.scm')
    )


_READINESS: dict[str, bool] = {}


def _grammar_ready(canonical_format: str) -> bool:
    """Probe once per language that the grammar loads and its query compiles."""
    cached = _READINESS.get(canonical_format)
    if cached is not None:
        return cached
    try:
        from tree_sitter import Query
        from tree_sitter_language_pack import get_language

        Query(
            get_language(canonical_format),
            (_QUERIES_DIR / f'{canonical_format}-tags.scm').read_text(),
        )
        ready = True
    except Exception:
        ready = False
    _READINESS[canonical_format] = ready
    return ready


class TreeSitterProvider:
    provider_id = 'tree-sitter'
    formats = _query_languages()
    output_formats = frozenset({'markdown', 'json', 'cst', 'nodes'})

    def is_ready(self, canonical_format: str) -> bool:
        return (
            canonical_format in self.formats
            and find_spec('tree_sitter_language_pack') is not None
            and _grammar_ready(canonical_format)
        )

    def extract(
        self,
        path: str,
        canonical_format: str,
        options: dict | None = None,
        output_format: str = 'markdown',
    ) -> ProviderDocument:
        if not self.is_ready(canonical_format):
            raise ProviderUnavailableError(
                f'Tree-sitter is not ready for {canonical_format}'
            )
        if output_format not in self.output_formats:
            raise ValueError(f'Tree-sitter cannot produce {output_format!r} output')

        from tree_sitter import Query, QueryCursor
        from tree_sitter_language_pack import get_language, get_parser

        source = Path(path)
        code = source.read_bytes()
        tree = get_parser(canonical_format).parse(code)
        scm = (_QUERIES_DIR / f'{canonical_format}-tags.scm').read_text()
        matches = QueryCursor(Query(get_language(canonical_format), scm)).matches(
            tree.root_node
        )

        symbols = _symbols_from_matches(matches, code)
        imports = _collect_imports(tree.root_node, code, canonical_format)
        references = _references_from_matches(matches, code)
        request_options = options or {}
        include_references = bool(request_options.get('include_references'))
        include_cst = bool(request_options.get('include_cst'))
        include_markdown = bool(request_options.get('include_markdown'))
        include_anonymous = bool(request_options.get('include_anonymous'))
        if output_format != 'cst' and not symbols and not imports:
            raise ProviderNotExtractableError(
                f'Tree-sitter found no symbols or imports in {source.name}'
            )

        if output_format == 'cst':
            content = _render_cst(tree.root_node, code, include_anonymous)
            media_type = 'text/plain'
            result_kind = ResultKind.STRUCTURED_DOCUMENT
        elif output_format == 'nodes':
            content = _render_nodes(tree.root_node, code, include_anonymous)
            media_type = 'application/x-ndjson'
            result_kind = ResultKind.STRUCTURED_DOCUMENT
        elif output_format == 'json':
            body = {
                'schema_version': SYMBOLS_SCHEMA_VERSION,
                'language': canonical_format,
                'file_name': source.name,
                'symbols': symbols,
                'imports': imports,
            }
            if include_references:
                body['references'] = references
            if include_cst:
                body['cst'] = _render_cst(tree.root_node, code, include_anonymous)
            if include_markdown:
                body['markdown'] = _render_markdown(
                    source.name, canonical_format, symbols, imports
                )
            content = json.dumps(body, ensure_ascii=False, indent=2)
            media_type = 'application/json'
            result_kind = ResultKind.STRUCTURED_DOCUMENT
        else:
            content = _render_markdown(source.name, canonical_format, symbols, imports)
            media_type = 'text/markdown'
            result_kind = ResultKind.SEMANTIC_DOCUMENT

        return ProviderDocument(
            content=content,
            media_type=media_type,
            result_kind=result_kind,
            provider=self.provider_id,
            provider_version=metadata.version('tree-sitter-language-pack'),
            backend=f'{canonical_format}-tags.scm',
            metadata={
                'language': canonical_format,
                'symbol_count': len(symbols),
                'import_count': len(imports),
                'reference_count': len(references),
            },
        )


_WEAK_KINDS = frozenset({'variable'})


def _symbols_from_matches(matches: list, code: bytes) -> list[dict]:
    by_key: dict[tuple[int, int, str], dict] = {}
    for _pattern, capture_map in matches:
        definition = next(
            (name for name in capture_map if name.startswith('definition.')), None
        )
        if definition is None:
            continue
        kind = definition.removeprefix('definition.')
        node = capture_map[definition][0]
        name_nodes = (
            capture_map.get(f'name.definition.{kind}') or capture_map.get('name') or []
        )
        if not name_nodes:
            continue
        doc = None
        if capture_map.get('doc'):
            doc = _clean_doc(_text(capture_map['doc'][0], code))
        for name_node in name_nodes:
            name = _text(name_node, code)
            if name == '_':
                continue
            key = (node.start_byte, node.end_byte, name)
            existing = by_key.get(key)
            if existing is not None and not (
                existing['kind'] in _WEAK_KINDS and kind not in _WEAK_KINDS
            ):
                continue
            by_key[key] = {
                'kind': kind,
                'name': name,
                'node': node,
                'name_node': name_node,
                'doc': doc,
                'start': node.start_byte,
                'end': node.end_byte,
            }
    entries = sorted(by_key.values(), key=lambda entry: (entry['start'], -entry['end']))

    def container(entry):
        best = None
        for other in entries:
            strictly_larger = (other['end'] - other['start']) > (
                entry['end'] - entry['start']
            )
            if (
                strictly_larger
                and other['start'] <= entry['start']
                and entry['end'] <= other['end']
                and (
                    best is None
                    or (other['end'] - other['start']) < (best['end'] - best['start'])
                )
            ):
                best = other
        return best

    symbols = []
    for entry in entries:
        parent = container(entry)
        ancestry = []
        cursor = parent
        while cursor is not None:
            ancestry.append(cursor)
            cursor = container(cursor)
        ancestry.reverse()

        kind = entry['kind']
        if (
            kind == 'function'
            and parent is not None
            and parent['kind']
            in {
                'class',
                'interface',
                'enum',
            }
        ):
            kind = 'method'
        node = entry['node']
        symbols.append(
            {
                'kind': kind,
                'name': entry['name'],
                'qualified_name': '.'.join(
                    [item['name'] for item in ancestry] + [entry['name']]
                ),
                'signature': _text(node, code).split('\n', 1)[0].strip()[:200],
                'span': {
                    'start_line': node.start_point[0] + 1,
                    'start_col': node.start_point[1],
                    'end_line': node.end_point[0] + 1,
                    'end_col': node.end_point[1],
                },
                'name_span': {
                    'line': entry['name_node'].start_point[0] + 1,
                    'col': entry['name_node'].start_point[1],
                },
                'docstring': entry['doc'] or _docstring(node, code),
                'parent': parent['name'] if parent else None,
            }
        )
    return symbols


def _references_from_matches(matches: list, code: bytes) -> list[dict]:
    references = []
    seen: set[tuple[str, str, int]] = set()
    for _pattern, capture_map in matches:
        reference = next(
            (name for name in capture_map if name.startswith('reference.')), None
        )
        if reference is None:
            continue
        kind = reference.removeprefix('reference.')
        name_nodes = (
            capture_map.get(f'name.reference.{kind}') or capture_map.get('name') or []
        )
        for node in name_nodes:
            item = (
                kind,
                _text(node, code),
                node.start_point[0] + 1,
                node.start_point[1],
            )
            if item in seen:
                continue
            seen.add(item)
            references.append(
                {'kind': item[0], 'name': item[1], 'line': item[2], 'col': item[3]}
            )
    references.sort(key=lambda item: (item['line'], item['col'], item['name']))
    return references


def _clean_doc(text: str) -> str | None:
    lines = []
    for line in text.splitlines():
        stripped = line.strip().strip('/*#').strip()
        if stripped:
            lines.append(stripped)
    return ' '.join(lines) or None


def _collect_imports(root, code: bytes, canonical_format: str) -> list[str]:
    import_nodes = _IMPORT_NODES.get(canonical_format, set())
    imports: list[str] = []

    def walk(node):
        for child in node.named_children:
            if child.type in import_nodes:
                imports.append(_text(child, code).split('\n', 1)[0].strip())
            else:
                walk(child)

    walk(root)
    return imports


def _render_cst(root, code, include_anonymous: bool = False) -> str:
    """Render the syntax tree, one line per node in pre-order.

    ``row:col-row:col type``; named leaf nodes carry their source text in
    backticks (newlines escaped, long literals truncated). Named nodes only
    by default — anonymous tokens (punctuation, keywords) are implied by the
    grammar and included only on request, matching the tree-sitter CLI's
    default-vs-``--cst`` split. Nesting is not written out: parent/child
    structure is fully recoverable from the spans (a child's range is
    contained in the nearest preceding wider range).
    """
    lines: list[str] = []

    def visit(node):
        if node.is_named:
            start, end = node.start_point, node.end_point
            line = f'{start[0]}:{start[1]}-{end[0]}:{end[1]} {node.type}'
            if not any(child.is_named for child in node.children):
                leaf = _leaf_text(node, code)
                if leaf:
                    line += f' `{leaf}`'
            lines.append(line)
        elif include_anonymous:
            start, end = node.start_point, node.end_point
            lines.append(f'{start[0]}:{start[1]}-{end[0]}:{end[1]} "{node.type}"')
        for child in node.children:
            visit(child)

    try:
        visit(root)
    except RecursionError:
        raise ProviderNotExtractableError(
            'Source is too deeply nested to serialize its syntax tree'
        ) from None
    return '\n'.join(lines) + '\n'


def _render_nodes(root, code, include_anonymous: bool = False) -> str:
    """Render the syntax tree as one JSON object per line (JSONEachRow).

    Columns: ``id`` (pre-order index), ``parent`` (id or null), ``type``,
    ``start``/``end`` as ``[row, col]``, ``bytes`` as ``[start, end]``, and
    ``text`` on leaf nodes. Loads directly into ClickHouse/DuckDB for SQL
    queries over code structure without any parsing.
    """
    lines: list[str] = []
    next_id = 0

    def visit(node, parent_id):
        nonlocal next_id
        include = node.is_named or include_anonymous
        node_id = None
        if include:
            node_id = next_id
            next_id += 1
            entry = {
                'id': node_id,
                'parent': parent_id,
                'type': node.type,
                'start': [node.start_point[0], node.start_point[1]],
                'end': [node.end_point[0], node.end_point[1]],
                'bytes': [node.start_byte, node.end_byte],
            }
            if not any(child.is_named for child in node.children):
                leaf = _leaf_text(node, code)
                if leaf:
                    entry['text'] = leaf
            lines.append(json.dumps(entry, ensure_ascii=False))
        for child in node.children:
            visit(child, node_id if include else parent_id)

    try:
        visit(root, None)
    except RecursionError:
        raise ProviderNotExtractableError(
            'Source is too deeply nested to serialize its syntax tree'
        ) from None
    return '\n'.join(lines) + '\n'


def _leaf_text(node, code) -> str:
    text = _text(node, code).replace('\n', '\\n')
    return text if len(text) <= 120 else text[:119] + '…'


def _text(node, code) -> str:
    return code[node.start_byte : node.end_byte].decode('utf-8', 'replace')


def _docstring(node, code) -> str | None:
    body = node.child_by_field_name('body')
    if body is None or not body.named_children:
        return None
    first = body.named_children[0]
    if first.type == 'expression_statement' and first.named_children:
        first = first.named_children[0]
    if first.type != 'string':
        return None
    content = next(
        (child for child in first.named_children if child.type == 'string_content'),
        None,
    )
    if content is not None:
        return _text(content, code).strip()
    return _text(first, code).strip('\'"').strip()


def _render_markdown(file_name, language, symbols, imports) -> str:
    lines = [f'# {file_name} ({language})', '']
    if imports:
        lines.append('## Imports')
        lines.extend(f'- `{item}`' for item in imports)
        lines.append('')
    outline_symbols = [
        symbol
        for symbol in symbols
        if symbol['kind'] not in {'variable', 'parameter', 'property'}
    ]
    if outline_symbols:
        lines.append('## Symbols')
        for symbol in outline_symbols:
            span = symbol['span']
            lines.append(
                f"- **{symbol['kind']}** `{symbol['qualified_name']}` "
                f"(lines {span['start_line']}-{span['end_line']}): "
                f"`{symbol['signature']}`"
            )
            if symbol['docstring']:
                lines.append(f"  - {symbol['docstring'].splitlines()[0]}")
    return '\n'.join(lines) + '\n'
