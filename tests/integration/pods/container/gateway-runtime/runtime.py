import sys

from marie.parsers import set_gateway_parser
from marie.parsers.helper import _update_gateway_args
from marie.serve.runtimes.gateway import GatewayRuntime


def run(*args, **kwargs):
    runtime_cls = GatewayRuntime
    print(f' args {args}')
    runtime_args = set_gateway_parser().parse_args(args)
    print(f' protocol {runtime_args.protocol}')
    _update_gateway_args(runtime_args)

    print(f' runtime_cls {runtime_cls}')
    with runtime_cls(runtime_args) as runtime:
        print(f' Lets run forever')
        runtime.run_forever()


if __name__ == '__main__':
    run(*sys.argv[1:])
