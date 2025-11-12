"""Argparser module for Marie scaffold command"""


def set_scaffold_parser(parser=None):
    """Set the parser for the scaffold command

    :param parser: an optional existing parser to build upon
    :return: the parser
    """
    if not parser:
        from marie.parsers.base import set_base_parser

        parser = set_base_parser()

    # Subcommand for scaffolding
    sp = parser.add_subparsers(
        dest="scaffold_type",
        required=True,
        help="Type of component to scaffold",
    )

    # Asset scaffolding
    asset_parser = sp.add_parser(
        "asset",
        help="Scaffold asset tracking for an executor",
        description="Generate asset tracking boilerplate for an executor file",
    )

    asset_parser.add_argument(
        "path",
        type=str,
        help="Path to the executor file to scaffold",
    )

    asset_parser.add_argument(
        "--template",
        "-t",
        type=str,
        default="single",
        choices=["single", "multi-asset", "ocr", "classification", "extraction"],
        help="Template to use for scaffolding (default: single)",
    )

    asset_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: same as input)",
    )

    asset_parser.add_argument(
        "--config",
        "-c",
        action="store_true",
        help="Generate YAML configuration instead of Python code",
    )

    asset_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the generated code without writing",
    )

    asset_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing file",
    )

    return parser
