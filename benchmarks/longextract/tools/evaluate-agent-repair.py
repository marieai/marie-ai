from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from longextract_bench.grading import grade
from marie_longextract.repair_eval import run_repair


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object in {path}')
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run and score the LongExtract boundary repair agent.'
    )
    parser.add_argument('--asset-dir', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--page', required=True, type=int)
    parser.add_argument('--record-index', type=int, default=0)
    parser.add_argument('--api-base', default=os.environ.get('LXBENCH_REPAIR_API_BASE'))
    parser.add_argument(
        '--api-key', default=os.environ.get('LXBENCH_REPAIR_API_KEY', 'EMPTY')
    )
    parser.add_argument('--model', default='qwen_v3_30b_instruct')
    parser.add_argument('--request-timeout-seconds', type=float, default=300.0)
    parser.add_argument('--schema', required=True, type=Path)
    parser.add_argument('--ground-truth', required=True, type=Path)
    args = parser.parse_args()
    if not args.api_base:
        parser.error(
            '--api-base or the LXBENCH_REPAIR_API_BASE environment variable is required'
        )
    if args.request_timeout_seconds <= 0:
        parser.error('--request-timeout-seconds must be positive')

    output_dir = args.out_dir.expanduser().resolve()
    report = asyncio.run(
        run_repair(
            asset_dir=args.asset_dir.expanduser().resolve(),
            output_dir=output_dir,
            page_number=args.page,
            record_index=args.record_index,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    )
    prediction = _load_object(output_dir / 'parsed-result' / 'longextract-result.json')
    report['score'] = grade(
        prediction,
        _load_object(args.ground_truth.expanduser().resolve()),
        _load_object(args.schema.expanduser().resolve()),
    )
    report_path = output_dir / 'repair-evaluation.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
