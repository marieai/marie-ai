"""
CLI entry point: ``python -m marie.sandbox <command> [options]``

Sub-commands
------------
seed              Wave-1 idempotent seed (org / workspace / admin / API key).
install-blueprint Wave-2 blueprint install (connectors + query plans).
install-plugins   Wave-3 plugin / extension install.

Additional sub-commands (capture, restore) will be added in later slices.
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

    # -- install-blueprint sub-command --------------------------------------
    from marie.sandbox.install_blueprint import _cli_install_blueprint
    from marie.sandbox.install_blueprint import build_parser as _bp_parser

    bp_p = _bp_parser()
    sub.add_parser(
        'install-blueprint',
        parents=[bp_p],
        add_help=False,
        description=bp_p.description,
        help='Wave-2 blueprint install (connectors + query plans).',
    )

    # -- install-plugins sub-command ----------------------------------------
    from marie.sandbox.install_plugins import _cli_install_plugins
    from marie.sandbox.install_plugins import build_parser as _plugins_parser

    plugins_p = _plugins_parser()
    sub.add_parser(
        'install-plugins',
        parents=[plugins_p],
        add_help=False,
        description=plugins_p.description,
        help='Wave-3 plugin / extension install.',
    )

    args = top.parse_args()

    if args.command == 'seed':
        _cli_seed(args)
    elif args.command == 'install-blueprint':
        _cli_install_blueprint(args)
    elif args.command == 'install-plugins':
        _cli_install_plugins(args)
    else:
        top.print_help()
        sys.exit(1)


if __name__ == '__main__':
    _main()
