import sys

from marie.parsers import set_pod_parser
from marie.serve.runtimes.head import HeadRuntime


def run(*args, **kwargs):
    runtime_args = set_pod_parser().parse_args(args)
    with HeadRuntime(runtime_args) as runtime:
        runtime.run_forever()


if __name__ == '__main__':
    run(*sys.argv[1:])
