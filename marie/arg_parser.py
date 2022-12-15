"""
This module parses the arguments given through the command-line. This is used by model-server
at runtime.
"""

import argparse


# noinspection PyTypeChecker
class ArgParser(object):
    """
    Argument parser for marie commands
    """

    @staticmethod
    def server_parser() -> object:
        """
        Argument parser for marie start service
        """
        parser = argparse.ArgumentParser(prog="marie", description="MarieAI Server")

        sub_parse = parser.add_mutually_exclusive_group(required=False)
        sub_parse.add_argument(
            "-v", "--version", action="store_true", help="Return Version"
        )
        sub_parse.add_argument(
            "--start", action="store_true", help="Start the model-server"
        )
        sub_parse.add_argument(
            "--stop", action="store_true", help="Stop the model-server"
        )

        parser.add_argument(
            "--model-store",
            required=False,
            dest="model_store",
            help="Model store location from where local or default models can be loaded",
        )

        parser.add_argument(
            "--workflow-store",
            required=False,
            dest="workflow_store",
            help="Workflow store location from where local or default workflows can be loaded",
        )

        parser.add_argument(
            "--enable-crypto",
            required=False,
            dest="enable_crypto",
            help="Enable encryption",
        )

        parser.add_argument(
            "--tls-cert", required=False, dest="tls_cert", help="Certificate location"
        )

        parser.add_argument(
            "--config",
            type=str,
            # default="./config/marie-debug.yml",
            default="/etc/marie/marie.yml",
            help="Configuration file",
        )

        return parser

    @staticmethod
    def extract_args(args=None):
        parser = ArgParser.server_parser()
        return parser.parse_args(args) if args else parser.parse_args()
