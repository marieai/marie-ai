"""Bounded source evidence for page boundaries."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_TABLE_TITLE = re.compile(
    r'^(?:#{1,6}\s+)?table\s+(?P<identifier>[a-z0-9][a-z0-9._-]*)(?:\s|$)',
    re.I,
)
_PAGE_COUNTER = re.compile(r'\b\d+\s*(?:of|/)\s*\d+\b', re.I)


def _table_identifier(title: str) -> str | None:
    match = _TABLE_TITLE.match(title.strip())
    return match['identifier'].rstrip('.').casefold() if match else None


def _normalized_title(title: str) -> str:
    title = re.sub(r'--(?:Continues|Table Ends)\s*$', '', title.strip(), flags=re.I)
    return ' '.join(title.split()).casefold()


def audit_table_continuations(
    tables: list[dict[str, Any]],
    page_text: str,
    prior_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Correct an incoming flag only when a single printed table ID changed."""
    previous = prior_context.get('prior_table_title_candidates', [])
    previous_page = prior_context.get('previous_page') or {}
    previous = [
        title
        for title in previous
        if title['page_number'] == previous_page.get('page_number')
    ]
    titles = [
        line.strip()
        for line in page_text.splitlines()
        if _TABLE_TITLE.match(line.strip())
    ]
    if (
        len(tables) != 1
        or len(titles) != 1
        or len(previous) != 1
        or previous[0]['line_number'] != 1
        or _PAGE_COUNTER.search(page_text)
    ):
        return tables
    current_id = _table_identifier(titles[0])
    if (
        current_id != _table_identifier(tables[0].get('name', ''))
        or current_id == _table_identifier(previous[-1]['text'])
        or _normalized_title(tables[0].get('name', '')) != _normalized_title(titles[0])
    ):
        return tables
    continuation = tables[0].get('continuation', {})
    if continuation.get('is_continuation') is not True:
        return tables
    return [
        {
            **tables[0],
            'continuation': {
                **continuation,
                'is_continuation': False,
                'from_table_name': '',
                'continuation_rationale': (
                    f"Printed source identity changed from {previous[-1]['text']!r} "
                    f"to {titles[0]!r}; an outgoing footer does not establish "
                    'incoming continuation.'
                ),
            },
        }
    ]


def _page_excerpt(text: str) -> tuple[list[str], bool]:
    lines = [f'{number} | {line}' for number, line in enumerate(text.splitlines(), 1)]
    if len('\n'.join(lines)) <= 3500:
        return lines, False

    head = []
    head_size = 0
    for line in lines[:12]:
        if head_size + len(line) + 1 > 900:
            break
        head.append(line)
        head_size += len(line) + 1
    tail = []
    remaining = 3500 - head_size - 40
    for line in reversed(lines[len(head) :]):
        if len(line) + 1 > remaining:
            break
        tail.append(line)
        remaining -= len(line) + 1
    return [*head, '[middle source lines omitted]', *reversed(tail)], True


def build_previous_page_context(
    page_number: int, page_texts: Mapping[int, str]
) -> dict[str, Any]:
    """Keep prior printed table titles and the preceding page's opening and tail."""
    context: dict[str, Any] = {
        'previous_page': None,
        'prior_table_title_candidates': [],
    }
    if page_number <= 1:
        return context

    lines, truncated = _page_excerpt(page_texts[page_number - 1])
    context['previous_page'] = {
        'page_number': page_number - 1,
        'lines': lines,
        'truncated': truncated,
    }
    titles = []
    for source_page in sorted(page_texts):
        if source_page >= page_number:
            continue
        for line_number, line in enumerate(page_texts[source_page].splitlines(), 1):
            title = line.strip()
            if len(title) <= 400 and _TABLE_TITLE.match(title):
                titles.append(
                    {
                        'page_number': source_page,
                        'line_number': line_number,
                        'text': title,
                    }
                )
    context['prior_table_title_candidates'] = titles[-3:]
    return context
