"""CLI and atomic import operation for Studio-generated candidates."""

from __future__ import annotations

import argparse
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from marie.logging_core.logger import MarieLogger
from marie_document_config_library.package_validation import (
    CandidateValidationError,
    ValidatedCandidate,
    validate_candidate,
)

_logger = MarieLogger('marie_document_config_library.import_candidate')


@dataclass(frozen=True)
class ImportResult:
    """Candidate import result."""

    candidate: ValidatedCandidate
    destination: Path
    status: str


def import_candidate(source: str | Path, destination: str | Path) -> ImportResult:
    """Validate and atomically import a candidate into benchmark ownership.

    Args:
        source: Studio-generated candidate directory.
        destination: Benchmark import root.

    Returns:
        Import result with ``imported`` or ``unchanged`` status.

    Raises:
        CandidateValidationError: If source or existing content is invalid or conflicts.
    """
    source_candidate = validate_candidate(source)
    destination_root = Path(destination).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    target = destination_root / source_candidate.configuration_id
    if destination_root not in target.resolve().parents:
        raise CandidateValidationError('Candidate destination escapes import root')

    if target.exists():
        existing = validate_candidate(target)
        if existing.package_digest != source_candidate.package_digest:
            raise CandidateValidationError(
                f'Conflicting candidate already exists for {source_candidate.configuration_id}'
            )
        return ImportResult(candidate=existing, destination=target, status='unchanged')

    temporary = (
        destination_root
        / f'.{source_candidate.configuration_id}.tmp-{uuid.uuid4().hex}'
    )
    try:
        shutil.copytree(source_candidate.root, temporary, symlinks=False)
        copied = validate_candidate(temporary)
        if copied.package_digest != source_candidate.package_digest:
            raise CandidateValidationError(
                'Candidate changed while it was being imported'
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return ImportResult(
        candidate=validate_candidate(target), destination=target, status='imported'
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description='Import a Studio document configuration candidate'
    )
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()

    result = import_candidate(args.source, args.destination)
    _logger.info(
        f'Candidate {result.candidate.configuration_id} {result.status} at {result.destination} '
        f'(package_sha256={result.candidate.package_digest})'
    )


if __name__ == '__main__':
    _main()
