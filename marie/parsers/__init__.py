from marie.helper import GATEWAY_NAME
from marie.parsers.helper import _SHOW_ALL_ARGS
from marie.parsers.logging import mixin_suppress_root_logging_parser
from marie.parsers.orchestrate.pod import (
    mixin_discovery_parser,
    mixin_gateway_job_scheduler_parser,
    mixin_gateway_kv_store_parser,
)
from marie.parsers.orchestrate.runtimes.container import mixin_container_runtime_parser
from marie.parsers.orchestrate.runtimes.grpc_channel import (
    mixin_grpc_channel_options_parser,
)
from marie.parsers.orchestrate.runtimes.head import mixin_head_parser


def set_pod_parser(parser=None, default_name=None):
    """Set the parser for the Pod

    :param parser: an optional existing parser to build upon
    :param default_name: default pod name
    :return: the parser
    """
    if not parser:
        from marie.parsers.base import set_base_parser

        parser = set_base_parser()

    from marie.parsers.orchestrate.base import mixin_scalable_deployment_parser
    from marie.parsers.orchestrate.pod import mixin_pod_parser
    from marie.parsers.orchestrate.runtimes.container import (
        mixin_container_runtime_parser,
    )
    from marie.parsers.orchestrate.runtimes.remote import mixin_remote_runtime_parser
    from marie.parsers.orchestrate.runtimes.worker import mixin_worker_runtime_parser

    mixin_scalable_deployment_parser(parser, default_name=default_name)
    mixin_worker_runtime_parser(parser)
    mixin_container_runtime_parser(parser)
    mixin_remote_runtime_parser(parser)
    mixin_pod_parser(parser)
    # mixin_hub_pull_options_parser(parser)
    mixin_head_parser(parser)

    return parser


def set_deployment_parser(parser=None):
    """Set the parser for the Deployment

    :param parser: an optional existing parser to build upon
    :return: the parser
    """
    if not parser:
        from marie.parsers.base import set_base_parser

        parser = set_base_parser()

    set_pod_parser(parser, default_name="executor")

    from marie.parsers.orchestrate.deployment import mixin_base_deployment_parser

    mixin_base_deployment_parser(parser)

    return parser


def set_gateway_parser(parser=None):
    """Set the parser for the gateway arguments

    :param parser: an optional existing parser to build upon
    :return: the parser
    """
    if not parser:
        from marie.parsers.base import set_base_parser

        parser = set_base_parser()

    from marie.parsers.orchestrate.base import mixin_base_deployment_parser
    from marie.parsers.orchestrate.pod import mixin_pod_parser
    from marie.parsers.orchestrate.runtimes.remote import (
        mixin_gateway_parser,
        mixin_graphql_parser,
        mixin_http_gateway_parser,
        mixin_prefetch_parser,
    )

    mixin_base_deployment_parser(parser)
    mixin_container_runtime_parser(parser, pod_type="gateway")
    mixin_prefetch_parser(parser)
    mixin_http_gateway_parser(parser)
    mixin_graphql_parser(parser)
    mixin_gateway_parser(parser)
    mixin_pod_parser(parser, pod_type="gateway")

    from marie.enums import DeploymentRoleType

    parser.set_defaults(
        name=GATEWAY_NAME,
        runtime_cls="GatewayRuntime",
        deployment_role=DeploymentRoleType.GATEWAY,
    )

    return parser


def set_gateway_runtime_args_parser(parser=None):
    """Set the parser for the gateway runtime arguments

    :param parser: an optional existing parser to build upon
    :return: the parser
    """
    if not parser:
        from marie.parsers.base import set_base_parser

        parser = set_base_parser()

    from marie.parsers.orchestrate.pod import mixin_pod_runtime_args_parser
    from marie.parsers.orchestrate.runtimes.remote import (
        _add_host,
        mixin_gateway_streamer_parser,
        mixin_prefetch_parser,
    )

    mixin_gateway_streamer_parser(parser)
    mixin_discovery_parser(parser)
    mixin_pod_runtime_args_parser(parser, pod_type="gateway")
    mixin_prefetch_parser(parser)
    _add_host(parser)

    return parser


def set_client_cli_parser(parser=None):
    """Set the parser for the cli client

    :param parser: an optional existing parser to build upon
    :return: the parser
    """
    if not parser:
        from marie.parsers.base import set_base_parser

        parser = set_base_parser()

    from marie.parsers.client import (
        mixin_client_features_parser,
        mixin_client_protocol_parser,
    )
    from marie.parsers.orchestrate.runtimes.remote import (
        mixin_client_gateway_parser,
        mixin_prefetch_parser,
    )

    mixin_client_gateway_parser(parser)
    mixin_client_features_parser(parser)
    mixin_client_protocol_parser(parser)
    mixin_grpc_channel_options_parser(parser)
    mixin_prefetch_parser(parser)
    mixin_suppress_root_logging_parser(parser)

    return parser


def set_help_parser(parser=None):
    """Set the parser for the jina help lookup

    :param parser: an optional existing parser to build upon
    :return: the parser
    """

    if not parser:
        from marie.parsers.base import set_base_parser

        parser = set_base_parser()

    parser.add_argument(
        "query",
        type=str,
        help="Look up usage & mention of argument name in Marie API. The name can be fuzzy",
    )
    return parser


def get_main_parser():
    """The main parser for Jina

    :return: the parser
    """
    from marie.parsers.base import set_base_parser
    from marie.parsers.create import set_new_project_parser
    from marie.parsers.export import set_export_parser
    from marie.parsers.flow import set_flow_parser
    from marie.parsers.helper import _SHOW_ALL_ARGS, _chf
    from marie.parsers.ping import set_ping_parser
    from marie.parsers.server import set_server_parser

    # create the top-level parser
    parser = set_base_parser()

    sp = parser.add_subparsers(
        dest="cli",
        required=True,
    )

    set_pod_parser(
        sp.add_parser(
            "executor",
            help="Start an Executor",
            description="Start an Executor. Jina uses Executors process Documents",
            formatter_class=_chf,
        )
    )

    set_flow_parser(
        sp.add_parser(
            "flow",
            description="Start a Flow. Jina uses Flows to streamline and distribute Executors",
            help="Start a Flow",
            formatter_class=_chf,
        )
    )

    set_ping_parser(
        sp.add_parser(
            "ping",
            help="Ping an Executor/Flow",
            description="Ping a remote Executor or a Flow.",
            formatter_class=_chf,
        )
    )

    set_export_parser(
        sp.add_parser(
            "export",
            help="Export Jina API/Flow",
            description="Export Jina API and Flow to JSONSchema, Kubernetes YAML, or SVG flowchart.",
            formatter_class=_chf,
        )
    )

    set_new_project_parser(
        sp.add_parser(
            "new",
            help="Create a new Marie project",
            description="Create a new Marie project with a predefined template",
            formatter_class=_chf,
        )
    )

    set_gateway_parser(
        sp.add_parser(
            "gateway",
            description="Start a Gateway to receive client Requests via gRPC/RESTful interface",
            **(dict(help="Start a Gateway")) if _SHOW_ALL_ARGS else {},
            formatter_class=_chf,
        )
    )

    from hubble.executor.parsers import get_main_parser as get_hub_parser
    from hubble.parsers import get_main_parser as get_auth_parser

    get_auth_parser(
        sp.add_parser(
            "auth",
            description="Login to Marie AI with your GitHub/Google/Email account",
            formatter_class=_chf,
            help="Login to Marie AI",
        )
    )

    get_hub_parser(
        sp.add_parser(
            "hub",
            help="Manage Executor on Marie Hub",
            description="Push/Pull an Executor to/from Marie Hub",
            formatter_class=_chf,
        )
    )

    set_server_parser(
        sp.add_parser(
            "server",
            help="Manage Marie Server",
            description="Manage Marie Server (e.g. start/stop/purge)",
            formatter_class=_chf,
        )
    )

    set_help_parser(
        sp.add_parser(
            "help",
            help="Show help text of a CLI argument",
            description="Show help text of a CLI argument",
            formatter_class=_chf,
        )
    )
    # Below are low-level / internal / experimental CLIs, hidden from users by default

    set_pod_parser(
        sp.add_parser(
            "pod",
            description="Start a Pod. "
            "You should rarely use this directly unless you "
            "are doing low-level orchestration",
            formatter_class=_chf,
            **(dict(help="Start a Pod")) if _SHOW_ALL_ARGS else {},
        )
    )

    set_deployment_parser(
        sp.add_parser(
            "deployment",
            description="Start a Deployment. "
            "You should rarely use this directly unless you "
            "are doing low-level orchestration",
            formatter_class=_chf,
            **(dict(help="Start a Deployment")) if _SHOW_ALL_ARGS else {},
        )
    )

    set_client_cli_parser(
        sp.add_parser(
            "client",
            description="Start a Python client that connects to a Marie Gateway",
            formatter_class=_chf,
            **(dict(help="Start a Client")) if _SHOW_ALL_ARGS else {},
        )
    )

    return parser
