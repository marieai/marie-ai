"""Execute bounded code-query branches against authorized captured sources."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from typing import TYPE_CHECKING, Any
from uuid import UUID

from marie.kb.code.capture import code_lines

if TYPE_CHECKING:
    from marie.kb.code.store import CodeIndexStore

FIELDS = {'repo', 'file', 'lang', 'rev', 'sym', 'context', 'content'}


def validate_branches(value: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError('Use one to eight code-query alternatives')
    count = 0
    for branch in value:
        if not isinstance(branch, list) or not 1 <= len(branch) <= 64:
            raise ValueError('Use one to 64 terms per query alternative')
        count += len(branch)
        for atom in branch:
            if (
                not isinstance(atom, dict)
                or set(atom) != {'field', 'value', 'negated', 'regex'}
                or not isinstance(atom['field'], str)
                or atom['field'] not in FIELDS
                or not isinstance(atom['value'], str)
                or not 1 <= len(atom['value']) <= 1000
                or any(ord(c) < 32 for c in atom['value'])
                or type(atom['negated']) is not bool
                or type(atom['regex']) is not bool
                or (atom['field'] not in ('content', 'file') and atom['regex'])
            ):
                raise ValueError('Invalid compiled code-query term')
    if count > 128:
        raise ValueError('Simplify the query to at most 128 expanded terms')
    return value


def native_query(handle: str, terms: list[dict[str, Any]], case_sensitive: bool) -> str:
    if not re.fullmatch('[a-f0-9]{64}', handle):
        raise ValueError('Invalid code index handle')
    encoded = []
    for atom in terms:
        if atom['field'] not in ('file', 'lang', 'content'):
            raise ValueError(
                'Native search only accepts file, language and content terms'
            )
        value = atom['value']
        if atom['field'] == 'content' and not atom['regex']:
            value = re.escape(value)
        encoded.append(
            ('-' if atom['negated'] else '')
            + atom['field']
            + ':'
            + json.dumps(value, ensure_ascii=False)
        )
    expression = ' '.join(encoded) or 'file:".*"'
    return f'repo:^{handle}$ case:{"yes" if case_sensitive else "no"} ({expression})'


def _metadata_matches(
    atom: dict[str, Any],
    source: Any,
    capture: dict[str, Any],
    scope: dict[str, Any],
    current: str,
    case_sensitive: bool,
) -> bool:
    field, value = atom['field'], atom['value']
    if field == 'repo':
        candidates = [
            source['name'],
            source['repository_name'],
            str(scope['source_id']),
            capture['repositoryId'],
        ]
        matches = any(
            value in str(item) if case_sensitive else value.lower() in str(item).lower()
            for item in candidates
        )
    elif field == 'context':
        matches = any(
            value in str(item) if case_sensitive else value.lower() in str(item).lower()
            for item in [source['index_name'], scope['index_id']]
        )
    else:
        matches = (
            (value.upper() == 'HEAD' or value == capture['branch'])
            and capture['handle'] == current
        ) or (
            bool(re.fullmatch('[a-fA-F0-9]{7,64}', value))
            and capture['commit'].lower().startswith(value.lower())
        )
    return matches != atom['negated']


async def _symbols(
    connection: Any,
    scope: dict[str, Any],
    handle: str,
    names: list[str],
    excluded_names: list[str],
    case_sensitive: bool,
) -> tuple[list[Any], set[str], set[str], bool]:
    rows = await connection.fetch(
        '''SELECT n.path,n.digest,n.status,entry.value AS record FROM marie_scheduler.kb_code_navigation n
        CROSS JOIN LATERAL jsonb_array_elements(n.symbols) entry(value)
        WHERE n.tenant_id=$1 AND n.index_id=$2 AND n.source_id=$3 AND n.handle=$4
        AND ((CASE WHEN $6 THEN entry.value->>'name' ELSE lower(entry.value->>'name') END)=ANY($5::text[])
          OR (CASE WHEN $6 THEN entry.value->>'qualified_name' ELSE lower(entry.value->>'qualified_name') END)=ANY($5::text[]))
        ORDER BY n.path,COALESCE(entry.value->>'line',entry.value->'span'->>'start_line')::int LIMIT 501''',
        UUID(scope['tenant_id']),
        UUID(scope['index_id']),
        UUID(scope['source_id']),
        handle,
        names if case_sensitive else [name.lower() for name in names],
        case_sensitive,
    )
    coverage = await connection.fetch(
        '''SELECT n.path,n.status,EXISTS(
            SELECT 1 FROM jsonb_array_elements(n.symbols) entry(value)
            WHERE (CASE WHEN $6 THEN entry.value->>'name' ELSE lower(entry.value->>'name') END)=ANY($5::text[])
               OR (CASE WHEN $6 THEN entry.value->>'qualified_name' ELSE lower(entry.value->>'qualified_name') END)=ANY($5::text[])
        ) AS excluded FROM marie_scheduler.kb_code_navigation n
        WHERE n.tenant_id=$1 AND n.index_id=$2 AND n.source_id=$3 AND n.handle=$4''',
        UUID(scope['tenant_id']),
        UUID(scope['index_id']),
        UUID(scope['source_id']),
        handle,
        excluded_names if case_sensitive else [name.lower() for name in excluded_names],
        case_sensitive,
    )
    return (
        list(rows[:500]),
        {row['path'] for row in coverage if row['status'] == 'ready'},
        {row['path'] for row in coverage if row['excluded']},
        len(rows) > 500 or any(row['status'] != 'ready' for row in coverage),
    )


async def advanced_search(
    store: CodeIndexStore,
    connection: Any,
    scope: dict[str, Any],
    source: Any,
    branches: Any,
    case_sensitive: bool,
    limit: int,
) -> dict[str, Any]:
    branches = validate_branches(branches)
    if not 1 <= limit <= 50 or type(case_sensitive) is not bool:
        raise ValueError('Invalid code search options')
    current, _ = store._manifest(source, scope)
    captures = [current, *((source['metadata'] or {}).get('codeRevisions') or [])]
    captures = list({capture['handle']: capture for capture in captures}.values())
    results: dict[tuple[str, str, int], dict[str, Any]] = {}
    warnings: set[str] = set()
    truncated = False
    searched = set()
    try:
        async with asyncio.timeout(8):
            for capture in captures:
                for branch in branches:
                    if capture['handle'] != current['handle'] and not any(
                        atom['field'] == 'rev' and not atom['negated']
                        for atom in branch
                    ):
                        continue
                    if not all(
                        _metadata_matches(
                            atom,
                            source,
                            capture,
                            scope,
                            current['handle'],
                            case_sensitive,
                        )
                        for atom in branch
                        if atom['field'] in ('repo', 'context', 'rev')
                    ):
                        continue
                    try:
                        selected, manifest = store._manifest(
                            source, scope, capture['handle']
                        )
                    except ValueError:
                        warnings.add(
                            'Some requested captures are no longer available. Refresh the source or choose a retained capture.'
                        )
                        continue
                    if selected['handle'] != capture['handle']:
                        continue
                    if not await store.zoekt.usable(
                        capture['handle'], capture['files']
                    ):
                        warnings.add('A requested captured index is unavailable.')
                        continue
                    searched.add(capture['handle'])
                    terms = [
                        atom
                        for atom in branch
                        if atom['field'] in ('content', 'file', 'lang')
                    ]
                    symbols = [atom for atom in branch if atom['field'] == 'sym']
                    symbol_rows = []
                    positive_symbols = [atom for atom in symbols if not atom['negated']]
                    if symbols:
                        (
                            symbol_rows,
                            ready_paths,
                            excluded_paths,
                            incomplete,
                        ) = await _symbols(
                            connection,
                            scope,
                            capture['handle'],
                            [atom['value'] for atom in positive_symbols],
                            [atom['value'] for atom in symbols if atom['negated']],
                            case_sensitive,
                        )
                        incomplete |= len(ready_paths) < len(manifest['files'])
                        if incomplete:
                            warnings.add(
                                'Symbol results have incomplete parser coverage or exceeded the definition budget in this capture.'
                            )
                        allowed = set(manifest['files'])
                        if any(atom['negated'] for atom in symbols):
                            allowed &= ready_paths - excluded_paths
                        for atom in positive_symbols:

                            def matches(row: Any) -> bool:
                                values = [
                                    row['record']['name'],
                                    row['record'].get('qualified_name', ''),
                                ]
                                return (
                                    atom['value'] in values
                                    if case_sensitive
                                    else atom['value'].lower()
                                    in [value.lower() for value in values]
                                )

                            paths = {row['path'] for row in symbol_rows if matches(row)}
                            allowed &= paths
                        if not allowed:
                            continue
                        if len(allowed) > 200:
                            truncated = True
                            allowed = set(sorted(allowed)[:200])
                        pattern = (
                            '^(?:'
                            + '|'.join(re.escape(path) for path in sorted(allowed))
                            + ')$'
                        )
                        if len(pattern) > 64000:
                            raise ValueError(
                                'Narrow the symbol query to fewer file paths'
                            )
                        terms.append(
                            {
                                'field': 'file',
                                'value': pattern,
                                'regex': True,
                                'negated': False,
                            }
                        )
                    found = await store.zoekt.search(
                        capture['handle'],
                        '',
                        'query',
                        limit,
                        case_sensitive=case_sensitive,
                        terms=terms,
                    )
                    truncated |= bool(
                        found.get('FilesSkipped')
                        or found.get('ShardsSkipped')
                        or found.get('MatchCount', 0) > limit
                        or len(found.get('Files') or []) > limit
                    )
                    for file in (found.get('Files') or [])[:limit]:
                        path = file['FileName']
                        if (
                            file['Repository'] != capture['handle']
                            or path not in manifest['files']
                        ):
                            raise PermissionError(
                                'Code query returned a file outside its captured scope'
                            )
                        if positive_symbols:
                            rows = [
                                row
                                for row in symbol_rows
                                if row['path'] == path
                                and any(
                                    (
                                        (
                                            atom['value']
                                            in [
                                                row['record']['name'],
                                                row['record'].get('qualified_name', ''),
                                            ]
                                        )
                                        if case_sensitive
                                        else (
                                            atom['value'].lower()
                                            in [
                                                row['record']['name'].lower(),
                                                row['record']
                                                .get('qualified_name', '')
                                                .lower(),
                                            ]
                                        )
                                    )
                                    for atom in positive_symbols
                                )
                            ]
                            content = await asyncio.to_thread(
                                (
                                    store.zoekt.directory(capture['handle'])
                                    / 'files'
                                    / path
                                ).read_bytes
                            )
                            if hashlib.sha256(content).hexdigest() != manifest['files'][
                                path
                            ]['sha256'] or any(
                                row['digest'] != manifest['files'][path]['sha256']
                                for row in rows
                            ):
                                raise PermissionError(
                                    'Symbol artifact differs from captured file'
                                )
                            lines = code_lines(content.decode('utf-8'))
                            hits = []
                            for row in rows:
                                line = (
                                    row['record'].get('line')
                                    or row['record']['span']['start_line']
                                )
                                if (
                                    not 1 <= line <= len(lines)
                                    or len(lines[line - 1]) > 8000
                                ):
                                    warnings.add(
                                        'Some symbol lines are too large to quote. Narrow the query or open the file.'
                                    )
                                    continue
                                hits.append(
                                    {
                                        'LineNumber': line,
                                        'text': lines[line - 1],
                                        'before': ''.join(
                                            lines[max(0, line - 4) : line - 1]
                                        ),
                                        'after': ''.join(lines[line : line + 3]),
                                    }
                                )
                        else:
                            hits = [
                                {
                                    'LineNumber': row['LineNumber'],
                                    'text': base64.b64decode(row['Line']).decode(
                                        'utf-8'
                                    ),
                                    'before': base64.b64decode(
                                        row.get('Before') or ''
                                    ).decode('utf-8'),
                                    'after': base64.b64decode(
                                        row.get('After') or ''
                                    ).decode('utf-8'),
                                }
                                for row in (
                                    file.get('LineMatches')
                                    or [{'LineNumber': 1, 'Line': ''}]
                                )[:10]
                            ]
                        for hit in hits:
                            line = max(1, hit['LineNumber'])
                            context = hit['before'] + hit['text'] + hit['after']
                            results[(capture['handle'], path, line)] = {
                                'path': path,
                                'language': file.get('Language') or 'Unknown',
                                'line': line,
                                'preview': hit['text'][:1000],
                                'context': context[:4000],
                                'contextStartLine': max(
                                    1, line - hit['before'].count('\n')
                                ),
                                'contextTruncated': len(context) > 4000,
                                'citation': {
                                    'kind': 'code',
                                    'sourceId': scope['source_id'],
                                    'repositoryId': capture['repositoryId'],
                                    'handle': capture['handle'],
                                    'branch': capture['branch'],
                                    'commit': capture['commit'],
                                    'path': path,
                                    'sourceDigest': manifest['files'][path]['sha256'],
                                    'startLine': line,
                                    'endLine': line,
                                },
                            }
    except TimeoutError:
        warnings.add(
            'The query time budget was reached; these results cover completed source checks.'
        )
    ordered = [value for _, value in sorted(results.items())]
    return {
        'results': ordered[:limit],
        'truncated': truncated or len(ordered) > limit or bool(warnings),
        'warnings': sorted(warnings),
        'indexed': {**current, 'searchedCaptures': sorted(searched)},
        'backend': 'zoekt',
    }
