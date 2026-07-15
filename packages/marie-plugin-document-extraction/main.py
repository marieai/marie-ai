"""Document extraction plugin daemon entrypoint."""

from __future__ import annotations

import argparse
import json
import sys

from marie_plugins.document_extraction.handler import dispatch_request
from marie_plugins.runtime import run


def _parse_option(item: str) -> tuple[str, object]:
    key, separator, value = item.partition('=')
    if not key or not separator:
        raise SystemExit(f'invalid --option {item!r}; expected KEY=VALUE')
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def _one_shot(argv: list[str]) -> int:
    """Run one request through the real handler and print its frames."""
    parser = argparse.ArgumentParser(
        prog='main.py',
        description=(
            'One-shot debug invocation; run without arguments to serve '
            'daemon requests over stdin/stdout.'
        ),
    )
    subparsers = parser.add_subparsers(dest='action', required=True)
    subparsers.add_parser('capabilities', help='print the capability snapshot')
    extract = subparsers.add_parser('extract', help='extract one document')
    extract.add_argument('path', help='input document path')
    extract.add_argument('output_dir', help='directory for the result artifact')
    extract.add_argument(
        '--provider',
        help='prefer one provider (e.g. docling, markitdown) ahead of the default order',
    )
    extract.add_argument(
        '--no-fallback',
        action='store_true',
        help='fail instead of trying the remaining providers',
    )
    extract.add_argument(
        '--option',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='provider-specific option, repeatable; VALUE is parsed as JSON when possible',
    )
    extract.add_argument(
        '--output-format',
        default='markdown',
        help=(
            'result serialization: markdown (default), html, text, json, '
            'or cst (source-code syntax tree)'
        ),
    )
    args = parser.parse_args(argv)

    data = {'action': args.action}
    if args.action == 'extract':
        data.update(path=args.path, output_dir=args.output_dir)
        if args.provider:
            data['provider'] = args.provider
        if args.no_fallback:
            data['fallback'] = False
        if args.option:
            data['provider_options'] = dict(_parse_option(item) for item in args.option)
        if args.output_format != 'markdown':
            data['output_format'] = args.output_format

    frames = dispatch_request(
        {'session_id': 'one-shot', 'event': 'request', 'data': data}
    )
    for frame in frames:
        print(json.dumps(frame, indent=2))
    return 0 if all(frame['data']['type'] != 'error' for frame in frames) else 1


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(_one_shot(sys.argv[1:]))
    run(dispatch_request)


if __name__ == '__main__':
    main()
