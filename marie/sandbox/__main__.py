"""
CLI entry point: ``python -m marie.sandbox seed [options]``

Dispatches to the seed sub-command.  Additional sub-commands (capture, restore)
will be added in later slices.
"""

from __future__ import annotations

import sys


def _main() -> None:
    import argparse

    top = argparse.ArgumentParser(
        prog='python -m marie.sandbox',
        description='Marie sandbox management commands.',
    )
    sub = top.add_subparsers(dest='command', required=True)

    # -- seed sub-command ---------------------------------------------------
    from marie.sandbox.seed import _cli_seed
    from marie.sandbox.seed import build_parser as _seed_parser

    seed_p = _seed_parser()
    sub.add_parser(
        'seed',
        parents=[seed_p],
        add_help=False,
        description=seed_p.description,
        help='Wave-1 idempotent seed for a Marie sandbox.',
    )

    args = top.parse_args()

    if args.command == 'seed':
        _cli_seed(args)
    else:
        top.print_help()
        sys.exit(1)


if __name__ == '__main__':
    _main()
