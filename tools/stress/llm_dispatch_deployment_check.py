"""Verify all three retained real-deployment cases against the current source."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tools.stress.llm_dispatch_deployment import (
    assert_scheduler_recovery,
    require,
    source_inventory,
    verify_output_files,
)


def validate_manifest(path: Path, canonical: list[dict[str, str]]) -> str:
    manifest = json.loads(path.read_text())
    rows = manifest.get('files')
    require(
        bool(canonical) and isinstance(rows, list) and rows == canonical,
        'exact_canonical_inventory',
    )
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    require(manifest.get('digest') == digest, 'manifest_digest')
    return digest


def validate_launch_manifests(
    report: dict, row: dict, canonical: list[dict[str, str]]
) -> str:
    initial = report['startup_source']
    producer = (
        row['before_outage']['producer'] if row['name'] == 'S03' else row['producer']
    )
    executors = [
        item
        for item in report['processes']
        if item['role'] == 'executor'
        and item.get('deployment_pid', item['pid']) == producer['ppid']
    ]
    require(
        len(executors) == 1 and executors[0]['source_manifest'] == initial,
        'used_initial_executor_manifest',
    )
    gateways = [
        item
        for item in report['processes']
        if item['role'] == 'gateway' and item['source_manifest'] == initial
    ]
    require(len(gateways) == 1, 'used_initial_gateway_manifest')
    digest = validate_manifest(Path(initial), canonical)
    if row['name'] != 'S03':
        require(
            row['gateway_parent']['source_manifest'] == initial,
            'surviving_gateway_manifest',
        )
        replacement = row.get('replacement', {}).get('source_manifest')
        require(
            isinstance(replacement, str) and bool(replacement),
            'used_replacement_manifest',
        )
        replacement_launches = [
            item
            for item in report['processes']
            if item['role'] == 'executor'
            and item.get('deployment_pid', item['pid']) == row['new_producer']['ppid']
        ]
        require(
            len(replacement_launches) == 1
            and replacement_launches[0]['pid'] == row['replacement']['pid']
            and replacement_launches[0]['source_manifest'] == replacement,
            'used_replacement_launch',
        )
        require(
            validate_manifest(Path(replacement), canonical) == digest,
            'replacement_source_digest',
        )
    return digest


def verify(paths: list[Path]) -> dict:
    require(len(paths) == 3, 'three_case_reports')
    source = Path(__file__).resolve().parents[2]
    canonical = source_inventory(source)
    cases = {}
    digests = set()
    for path in paths:
        report = json.loads(path.read_text())
        require(
            not report['failures'] and len(report['scenarios']) == 1,
            'successful_case_report',
        )
        row = report['scenarios'][0]
        require(
            row['status'] == 'passed' and row['name'] not in cases,
            'unique_passed_cases',
        )
        digests.add(validate_launch_manifests(report, row, canonical))
        if row['name'] == 'S03':
            verify_output_files(row['writes'], path.parent)
            assert_scheduler_recovery(
                row,
                report['ordinary_executor_result'][-1]['job_id'],
                report['submission']['dag_id'],
            )
            require(
                len(row['receipts']) == 9
                and sorted(item['index'] for item in row['receipts']) == list(range(9)),
                'nine_provider_acceptances',
            )
            require(
                row['during_outage']['caller_pending']
                and row['during_outage']['endpoint']['circuit'] == 'open',
                'observed_circuit_outage',
            )
            first = row['before_outage']['first_output']
            require(
                hashlib.sha256(Path(first['path']).read_bytes()).hexdigest()
                == first['sha256'],
                'first_output_preserved',
            )
        else:
            require(
                row['cleanup']['producer_keys']
                == row['cleanup']['request_records']
                == 0,
                'dead_producer_drained',
            )
            require(
                row['new_producer']['producer_id'] != row['producer']['producer_id'],
                'new_producer',
            )
            fresh = row['replacement_annotation']
            verify_output_files(fresh['ordinary_writes'], path.parent)
            require(
                not set(row['cleanup']['old_attempts'])
                & {item['attempt_id'] for item in fresh['ordinary_admissions']},
                'no_adoption',
            )
            if row['name'] == 'OOM':
                require(
                    row['oom_after']['events']['oom_kill']
                    > row['oom_before']['events']['oom_kill'],
                    'kernel_oom_evidence',
                )
        cases[row['name']] = {
            'report': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    require(
        set(cases) == {'S03', 'SIGKILL', 'OOM'} and len(digests) == 1,
        'complete_current_source_matrix',
    )
    return {'status': 'passed', 'source_digest': digests.pop(), 'cases': cases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'status': 'failed'}
    try:
        result = verify(args.reports)
        return 0
    except Exception as error:
        result['failure_category'] = type(error).__name__
        return 1
    finally:
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
