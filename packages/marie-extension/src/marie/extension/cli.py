import argparse
import sys

from marie.extension.validator import validate_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marie-extension")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--path", required=True)

    args = parser.parse_args(argv)
    if args.command == "validate":
        result = validate_package(args.path)
        if result.ok and result.package:
            print(f"valid: {result.package.manifest.metadata.id}")
            print(f"digest: {result.package.digest}")
            return 0
        for error in result.errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
