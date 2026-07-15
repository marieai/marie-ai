"""Input-aware document format detection for the extraction plugin."""

from __future__ import annotations

import csv
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

ALIASES = {
    'htm': 'html',
    'md': 'markdown',
    'tex': 'latex',
}

EXTENSIONS = {
    'pdf': 'pdf',
    'docx': 'docx',
    'xlsx': 'xlsx',
    'pptx': 'pptx',
    'html': 'html',
    'htm': 'html',
    'md': 'markdown',
    'markdown': 'markdown',
    'csv': 'csv',
    'tsv': 'tsv',
    'odt': 'odt',
    'ods': 'ods',
    'odp': 'odp',
    'tex': 'latex',
    'latex': 'latex',
    'eml': 'eml',
    'msg': 'msg',
    'epub': 'epub',
    'py': 'python',
    'js': 'javascript',
    'mjs': 'javascript',
    'ts': 'typescript',
    'go': 'go',
    'java': 'java',
    'rs': 'rust',
    'c': 'c',
    'h': 'c',
    'cpp': 'cpp',
    'cc': 'cpp',
    'cxx': 'cpp',
    'hpp': 'cpp',
    'cs': 'csharp',
    'rb': 'ruby',
    'php': 'php',
    'kt': 'kotlin',
    'kts': 'kotlin',
    'swift': 'swift',
    'ino': 'arduino',
    'sh': 'bash',
    'bash': 'bash',
    'chatito': 'chatito',
    'clj': 'clojure',
    'cljs': 'clojure',
    'cljc': 'clojure',
    'lisp': 'commonlisp',
    'cl': 'commonlisp',
    'd': 'd',
    'dart': 'dart',
    'el': 'elisp',
    'ex': 'elixir',
    'exs': 'elixir',
    'elm': 'elm',
    'f90': 'fortran',
    'f95': 'fortran',
    'f03': 'fortran',
    'f': 'fortran',
    'gleam': 'gleam',
    'hs': 'haskell',
    'hcl': 'hcl',
    'tf': 'hcl',
    'lua': 'lua',
    'm': 'matlab',
    'ml': 'ocaml',
    'mli': 'ocaml_interface',
    'pony': 'pony',
    'properties': 'properties',
    'ql': 'ql',
    'r': 'r',
    'rkt': 'racket',
    'scala': 'scala',
    'sc': 'scala',
    'sol': 'solidity',
    'rules': 'udev',
    'zig': 'zig',
}

MIME_TYPES = {
    'application/pdf': 'pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
    'text/html': 'html',
    'text/markdown': 'markdown',
    'text/csv': 'csv',
    'text/tab-separated-values': 'tsv',
    'application/vnd.oasis.opendocument.text': 'odt',
    'application/vnd.oasis.opendocument.spreadsheet': 'ods',
    'application/vnd.oasis.opendocument.presentation': 'odp',
    'application/x-latex': 'latex',
    'message/rfc822': 'eml',
    'application/vnd.ms-outlook': 'msg',
    'application/epub+zip': 'epub',
    'text/x-python': 'python',
    'text/javascript': 'javascript',
    'text/x-typescript': 'typescript',
    'text/x-go': 'go',
    'text/x-java-source': 'java',
}


@dataclass(frozen=True)
class DetectionResult:
    canonical_format: str
    evidence: tuple[str, ...]


def detect_format(
    path: str, *, format_hint: str | None = None, mime_type: str | None = None
) -> DetectionResult:
    """Detect a supported semantic format and retain the agreeing evidence."""
    if not path:
        raise ValueError("extract requires a 'path'")
    source = Path(os.path.expanduser(path)).resolve()
    if not source.is_file():
        raise FileNotFoundError(f'No such input file: {source}')

    evidence: list[tuple[str, str]] = []
    if format_hint:
        normalized = format_hint.lower().lstrip('.')
        evidence.append(('hint', ALIASES.get(normalized, normalized)))
    if mime_type:
        canonical = MIME_TYPES.get(mime_type.lower().split(';', 1)[0].strip())
        if canonical:
            evidence.append(('mime', canonical))
    extension = EXTENSIONS.get(source.suffix.lower().lstrip('.'))
    if extension:
        evidence.append(('extension', extension))
    content = _content_format(source)
    if content:
        evidence.append(('content', content))

    if not evidence:
        raise ValueError(f'Unsupported document format: {source.name}')

    content_values = {value for kind, value in evidence if kind == 'content'}
    declared_values = {value for kind, value in evidence if kind != 'content'}
    if (
        content_values
        and declared_values
        and content_values.isdisjoint(declared_values)
    ):
        raise ValueError(f'Conflicting format evidence: {evidence!r}')

    canonical = next(
        (value for kind, value in evidence if kind == 'content'), evidence[0][1]
    )
    agreeing = tuple(kind for kind, value in evidence if value == canonical)
    return DetectionResult(canonical_format=canonical, evidence=agreeing)


def _content_format(path: Path) -> str | None:
    with path.open('rb') as source:
        prefix = source.read(8192)
    if prefix.startswith(b'%PDF-'):
        return 'pdf'
    if prefix.startswith(b'PK\x03\x04'):
        return _zip_format(path)

    text = prefix.decode('utf-8', 'ignore').lstrip()
    lowered = text.lower()
    if lowered.startswith('#!'):
        interpreter = lowered.splitlines()[0]
        if 'python' in interpreter:
            return 'python'
        if 'node' in interpreter:
            return 'javascript'
        if interpreter.rstrip().endswith(('/sh', '/bash', ' sh', ' bash')):
            return 'bash'
    if lowered.startswith(('<!doctype html', '<html')):
        return 'html'
    if lowered.startswith(('from:', 'return-path:', 'received:')) and '\n' in text:
        return 'eml'
    if path.suffix.lower() in {'.csv', '.tsv'} and text:
        try:
            csv.Sniffer().sniff(text, delimiters=',\t;|')
            return 'tsv' if '\t' in text.splitlines()[0] else 'csv'
        except csv.Error:
            return None
    return None


def _zip_format(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if 'word/document.xml' in names:
                return 'docx'
            if 'xl/workbook.xml' in names:
                return 'xlsx'
            if 'ppt/presentation.xml' in names:
                return 'pptx'
            if 'mimetype' in names:
                member = archive.getinfo('mimetype')
                if member.file_size > 256:
                    return None
                mime = archive.read(member).decode('ascii', 'ignore').strip()
                return MIME_TYPES.get(mime)
    except (OSError, zipfile.BadZipFile, KeyError):
        return None
    return None
