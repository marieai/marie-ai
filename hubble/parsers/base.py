import argparse


def set_base_parser():
    from .helper import _chf

    parser = argparse.ArgumentParser(
        description=f'Jina Auth CLI helps you login in to Jina AI Ecosystem.',  # noqa F501
        formatter_class=_chf,
    )

    return parser
