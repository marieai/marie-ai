"""Argparser module for server"""

from marie.parsers.base import set_base_parser
from marie.parsers.helper import add_arg_group, KVAppendAction


def set_server_parser(parser=None):
    """Set the parser for `server`

    :param parser: an existing parser to build upon
    :return: the parser
    """
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, title='Server Feature')

    gp.add_argument(
        '--start',
        action='store_true',
        default=False,
        help='Start the server if it is not running',
    )

    gp.add_argument(
        '--purge',
        action='store_true',
        default=False,
        help='Purge all temporary data. This will delete all data in the workspace directory.',
    )

    gp.add_argument(
        '--uses',
        type=str,
        help='The YAML path represents a marie-ai config. It can be either a local file path or a URL.',
    )

    gp.add_argument(
        '--env',
        action=KVAppendAction,
        metavar='KEY: VALUE',
        nargs='*',
        help='The map of environment variables that are available inside runtime',
    )
    #

    gp.add_argument(
        '--env-file',
        type=str,
        help='You can set default values for multiple environment variables, in an environment file.',
    )

    sp = add_arg_group(parser, title='status')

    sp.add_argument(
        '--status',
        type=str,
        choices=['all', 'storage', 'messaging', 'flow'],
        help='The target type to check. Checks the readiness of the individual service components.',
        default='all',
    )

    return parser
