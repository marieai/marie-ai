from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, TypedDict

_FUZZY_THRESHOLD = 0.85


class NormBox(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float


class OcrWord(TypedDict):
    text: str
    box: NormBox


class OcrPage(TypedDict):
    page: int
    width: float
    height: float
    words: list[OcrWord]


class ValueLocation(TypedDict):
    page: int
    bbox: NormBox
    match: Literal['exact', 'fuzzy', 'ambiguous']
    candidates: int


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_ocr_meta(metadata: object) -> list[OcrPage]:
    """Normalize extract-pipeline OCR pixel boxes into page coordinates."""
    raw_pages = metadata.get('ocr') if isinstance(metadata, dict) else metadata
    if not isinstance(raw_pages, list):
        return []

    pages = []
    for index, raw in enumerate(raw_pages):
        if not isinstance(raw, dict) or not isinstance(raw.get('meta'), dict):
            continue
        meta = raw['meta']
        size = meta.get('imageSize')
        if not isinstance(size, dict):
            continue
        width, height = _number(size.get('width')), _number(size.get('height'))
        if width is None or height is None or width <= 0 or height <= 0:
            continue
        page = meta.get('page', index)
        if not isinstance(page, int) or isinstance(page, bool) or page < 0:
            page = index
        words = []
        for word in raw.get('words', []) if isinstance(raw.get('words'), list) else []:
            if not isinstance(word, dict) or not isinstance(word.get('text'), str):
                continue
            box = word.get('box')
            if not isinstance(box, list) or len(box) < 4:
                continue
            dimensions = [_number(value) for value in box[:4]]
            if any(value is None for value in dimensions):
                continue
            x, y, w, h = dimensions
            if w <= 0 or h <= 0:
                continue
            normalized = {
                'x1': min(1, max(0, x / width)),
                'y1': min(1, max(0, y / height)),
                'x2': min(1, max(0, (x + w) / width)),
                'y2': min(1, max(0, (y + h) / height)),
            }
            if (
                normalized['x2'] > normalized['x1']
                and normalized['y2'] > normalized['y1']
            ):
                words.append({'text': word['text'], 'box': normalized})
        pages.append({'page': page, 'width': width, 'height': height, 'words': words})
    return pages


def _tokenize(text: str, numeric: bool = False) -> list[tuple[str, str]]:
    tokens = []
    for raw in text.split():
        if re.search(r'\d', raw):
            value = re.sub(r'[.,;:]+$', '', raw)
            if value.startswith('(') and value.endswith(')'):
                value = '-' + value[1:-1]
            value = re.sub(r'^(-?)[$€£¥]', r'\1', value)
            if re.fullmatch(r'-?\d{1,3}(,\d{3})+(\.\d+)?', value):
                value = value.replace(',', '')
            if numeric and re.fullmatch(r'-?\d+(\.\d+)?', value):
                value = format(Decimal(value), 'f')
                if '.' in value:
                    value = value.rstrip('0').rstrip('.')
            tokens.append(('num', value))
        else:
            value = ''.join(char for char in raw.lower() if char.isalnum())
            if value:
                tokens.append(('text', value))
    return tokens


def _levenshtein(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, 1):
        current = [i]
        for j, right in enumerate(second, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


@dataclass
class _Candidate:
    page: int
    start: int
    end: int
    exact: bool
    score: float
    bbox: NormBox


def _candidates_on_page(
    value: list[tuple[str, str]], page: OcrPage, numeric: bool = False
) -> list[_Candidate]:
    positions = [
        (token, word['box'])
        for word in page['words']
        for token in _tokenize(word['text'], numeric)
    ]
    numbers = [text for kind, text in value if kind == 'num']
    text = ''.join(text for kind, text in value if kind == 'text')
    candidates = []
    for length in range(max(1, len(value) - 1), len(value) + 2):
        for start in range(len(positions) - length + 1):
            span = positions[start : start + length]
            tokens = [token for token, _ in span]
            exact = tokens == value
            score = 1.0 if exact else 0.0
            if not exact:
                # Numeric punctuation, dates and identifiers are never fuzzy.
                if (
                    not text
                    or [text for kind, text in tokens if kind == 'num'] != numbers
                ):
                    continue
                candidate_text = ''.join(
                    text for kind, text in tokens if kind == 'text'
                )
                longest = max(len(candidate_text), len(text))
                if (
                    1 - abs(len(candidate_text) - len(text)) / longest
                    < _FUZZY_THRESHOLD
                ):
                    continue
                score = 1 - _levenshtein(candidate_text, text) / longest
                if score < _FUZZY_THRESHOLD:
                    continue
            candidates.append(
                _Candidate(
                    page=page['page'],
                    start=start,
                    end=start + length,
                    exact=exact,
                    score=score,
                    bbox={
                        'x1': min(box['x1'] for _, box in span),
                        'y1': min(box['y1'] for _, box in span),
                        'x2': max(box['x2'] for _, box in span),
                        'y2': max(box['y2'] for _, box in span),
                    },
                )
            )
    return candidates


def ground_text(
    text: str,
    pages: list[OcrPage],
    page_hint: int | None = None,
    *,
    numeric: bool = False,
) -> ValueLocation | None:
    """Find the best OCR span, preserving numeric values and match ambiguity."""
    value = _tokenize(text, numeric)
    if not value:
        return None
    candidates = [
        candidate
        for page in pages
        if page['page'] == page_hint
        for candidate in _candidates_on_page(value, page, numeric)
    ]
    if not candidates:
        candidates = [
            candidate
            for page in pages
            for candidate in _candidates_on_page(value, page, numeric)
        ]
    if not candidates:
        return None
    pool = [candidate for candidate in candidates if candidate.exact] or candidates
    top_score = max(candidate.score for candidate in pool)
    best = sorted(
        (candidate for candidate in pool if top_score - candidate.score < 1e-9),
        key=lambda candidate: (candidate.page, candidate.start, candidate.end),
    )
    chosen = []
    for candidate in best:
        if not any(
            existing.page == candidate.page
            and candidate.start < existing.end
            and existing.start < candidate.end
            for existing in chosen
        ):
            chosen.append(candidate)
    first = chosen[0]
    return {
        'page': first.page,
        'bbox': first.bbox,
        'match': (
            'ambiguous' if len(chosen) > 1 else 'exact' if first.exact else 'fuzzy'
        ),
        'candidates': len(chosen),
    }


def ground_values(
    values: list[dict[str, Any]], pages: list[OcrPage]
) -> list[dict[str, Any]]:
    """Attach OCR locations without changing extracted values or confidence."""
    return [
        {**value, 'location': ground_text(value['value'], pages, value.get('page'))}
        for value in values
    ]
