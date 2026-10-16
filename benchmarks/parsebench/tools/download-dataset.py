from __future__ import annotations

import argparse
import json
from pathlib import Path

from marie_parsebench.download import download_dataset


def main() -> None:
    """Download and verify the pinned ParseBench source dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True, type=Path)
    parser.add_argument('--revision', required=True)
    parser.add_argument(
        '--inventory',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'config/dataset-inventory.json',
    )
    args = parser.parse_args()
    report = download_dataset(args.dataset_dir, args.inventory, args.revision)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
