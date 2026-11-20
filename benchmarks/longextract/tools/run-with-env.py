from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from dotenv import dotenv_values


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Run a benchmark command with its dedicated environment file.'
    )
    parser.add_argument(
        '--env-file',
        type=Path,
        default=Path(__file__).resolve().parents[1] / '.env',
    )
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Provide a command after --')
    env_file = args.env_file.expanduser()
    if not env_file.is_file():
        parser.error(f'Environment file not found: {env_file}')
    values = dotenv_values(env_file, interpolate=False)
    for name in ('OPENAI_API_BASE', 'OPENAI_API_KEY'):
        if not (values.get(name) or '').strip():
            parser.error(f'Set {name} in {env_file}')
    env = dict(os.environ)
    env.update({name: value for name, value in values.items() if value is not None})
    env['MARIE_SKIP_LOAD_ENV'] = 'true'
    return subprocess.call(command, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
