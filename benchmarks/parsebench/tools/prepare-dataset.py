from __future__ import annotations

import argparse
from pathlib import Path

from marie_parsebench.dataset import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Verify ParseBench inputs and create manifests'
    )
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    input_path, smoke_path = prepare_dataset(
        args.dataset_dir, args.output_dir, args.inventory, args.revision
    )
    print(input_path)
    print(smoke_path)


if __name__ == '__main__':
    main()
