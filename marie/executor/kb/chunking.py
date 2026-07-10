"""Deterministic character-based text chunker used by VectorStoreExecutor.

No token-aware/sentence-aware splitter is vendored in this codebase today
(marie.core.text_splitter imports a marie.core.node_parser package that does
not exist in this checkout), so this implements the minimal character-window
splitter the DAG's EMBED stage needs.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class TextChunk:
    text: str
    start: int
    end: int


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Split ``text`` into overlapping, fixed-size character windows.

    :param text: text to split. Empty/falsy input returns [].
    :param chunk_size: maximum number of characters per chunk.
    :param chunk_overlap: number of characters shared between consecutive chunks.
    :return: chunks in document order, each carrying its [start, end) offset
        into ``text`` so callers can map chunks back to source locations
        (e.g. OCR page boundaries).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size "
            f"({chunk_size})"
        )

    if not text:
        return []

    step = chunk_size - chunk_overlap
    text_len = len(text)
    chunks = []
    start = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(TextChunk(text=text[start:end], start=start, end=end))
        if end == text_len:
            break
        start += step
    return chunks
