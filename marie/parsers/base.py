"""Module containing the base parser for arguments of Marie."""
import argparse

from marie.parsers.helper import _chf


def set_base_parser():
    """Set the base parser

    :return: the parser
    """
    from marie import __version__
    from marie.helper import colored, format_full_version_info, get_full_version

    # create the top-level parser
    urls = {
        'Code': ('💻', 'https://github.com/gregbugaj/marie-ai'),
        'Docs': ('📖', 'https://github.com/gregbugaj/marie-ai/tree/main/docs'),
        'Help': ('💬', 'https://github.com/gregbugaj/marie-ai'),
    }
    url_str = '\n'.join(
        f'- {v[0]:<10} {k:10.10}\t{colored(v[1], "cyan", attrs=["underline"])}'
        for k, v in urls.items()
    )

    parser = argparse.ArgumentParser(
        epilog=f'''
Marie 🦊 (v{colored(__version__, "green")}) is the Document processing framework powered by deep learning.

{url_str}

''',
        formatter_class=_chf,
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=__version__,
        help='Show Marie version',
    )

    parser.add_argument(
        '-vf',
        '--version-full',
        action='version',
        version=format_full_version_info(*get_full_version()),
        help='Show Marie and all dependencies\' versions',
    )
    return parser
