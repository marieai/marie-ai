"""Daemon entrypoint for the LongExtract package's agent actions."""

from marie_longextract.agents.plugin_handler import dispatch_request
from marie_plugins.runtime import run


def main() -> None:
    run(dispatch_request)


if __name__ == '__main__':
    main()
