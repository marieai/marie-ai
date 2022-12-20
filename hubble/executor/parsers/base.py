import argparse


def set_base_parser():
    from hubble.parsers.helper import _chf

    parser = argparse.ArgumentParser(
        description='Push/Pull an Executor to/from Jina Hub',
        formatter_class=_chf,
    )

    return parser
