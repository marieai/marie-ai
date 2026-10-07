"""Fail-closed qualification prerequisite checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from marie.logging_core.logger import MarieLogger
from marie_document_config_library.package_validation import validate_candidate

_logger = MarieLogger('marie_document_config_library.preflight')


@dataclass(frozen=True)
class PreflightIssue:
    """One unresolved qualification prerequisite."""

    code: str
    message: str


@dataclass(frozen=True)
class PreflightResult:
    """Qualification preflight result."""

    ready: bool
    configuration_id: str
    package_digest: str
    issues: tuple[PreflightIssue, ...]


def run_preflight(
    candidate_root: str | Path,
    config_root: str | Path,
    evidence_root: str | Path,
) -> PreflightResult:
    """Check runtime layout and qualification evidence without executing a model.

    Args:
        candidate_root: Imported candidate package.
        config_root: Marie extraction configuration root.
        evidence_root: Mutable qualification evidence directory for this candidate.

    Returns:
        Readiness result. Any missing gate leaves ``ready`` false.
    """
    candidate = validate_candidate(candidate_root)
    evidence = Path(evidence_root).expanduser().resolve()
    layout = _layout(candidate.query_plan)
    config_path = (
        Path(config_root).expanduser().resolve()
        / f'TID-{layout}'
        / 'annotator'
        / 'config.yml'
    )
    issues: list[PreflightIssue] = []
    if not config_path.is_file():
        issues.append(
            PreflightIssue(
                code='runtime_layout_missing',
                message=f'Runtime extraction layout is not installed: {config_path}',
            )
        )
    else:
        installed = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        if not isinstance(installed, dict) or installed.get('layout_id') != layout:
            issues.append(
                PreflightIssue(
                    code='runtime_layout_mismatch',
                    message=f'Installed runtime extraction layout does not identify {layout}',
                )
            )

    expected_output = evidence / 'expected-output.json'
    if not expected_output.is_file():
        issues.append(
            PreflightIssue(
                code='reviewed_gold_missing',
                message=f'A reviewed expected-output.json is required before qualification: {expected_output}',
            )
        )

    qualification_result = evidence / 'qualification-result.json'
    if not qualification_result.is_file():
        issues.append(
            PreflightIssue(
                code='qualification_result_missing',
                message='No real Marie runtime qualification result exists for this candidate',
            )
        )

    return PreflightResult(
        ready=not issues,
        configuration_id=candidate.configuration_id,
        package_digest=candidate.package_digest,
        issues=tuple(issues),
    )


def _layout(query_plan: object) -> str:
    nodes = getattr(query_plan, 'nodes')
    layouts = {
        node.definition.params.get('layout')
        for node in nodes
        if isinstance(node.definition.params, dict)
        and node.definition.params.get('layout')
    }
    if len(layouts) != 1:
        raise ValueError('Validated query plan did not retain one layout')
    return str(next(iter(layouts)))


def _main() -> None:
    parser = argparse.ArgumentParser(
        description='Check document configuration qualification prerequisites'
    )
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--config-root', required=True, type=Path)
    parser.add_argument('--evidence-root', required=True, type=Path)
    args = parser.parse_args()

    result = run_preflight(args.candidate, args.config_root, args.evidence_root)
    _logger.info(json.dumps(asdict(result), sort_keys=True))
    if not result.ready:
        raise SystemExit(2)


if __name__ == '__main__':
    _main()
