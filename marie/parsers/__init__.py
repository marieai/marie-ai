from marie.parsers.helper import _SHOW_ALL_ARGS


def set_pod_parser(parser=None):
    """Set the parser for the Pod

    :param parser: an optional existing parser to build upon
    :return: the parser
    """
    if not parser:
        from marie.parsers.base import set_base_parser

        parser = set_base_parser()

    from marie.parsers.orchestrate.base import mixin_base_ppr_parser
    from marie.parsers.orchestrate.pod import mixin_pod_parser
    from marie.parsers.orchestrate.runtimes.container import (
        mixin_container_runtime_parser,
    )
    from marie.parsers.orchestrate.runtimes.distributed import (
        mixin_distributed_feature_parser,
    )
    from marie.parsers.orchestrate.runtimes.remote import mixin_remote_runtime_parser
    from marie.parsers.orchestrate.runtimes.worker import mixin_worker_runtime_parser

    mixin_base_ppr_parser(parser)
    mixin_worker_runtime_parser(parser)
    mixin_container_runtime_parser(parser)
    mixin_remote_runtime_parser(parser)
    mixin_distributed_feature_parser(parser)
    mixin_pod_parser(parser)
    # mixin_hub_pull_options_parser(parser)

    return parser


def set_deployment_parser(parser=None):
    """Set the parser for the Deployment

    :param parser: an optional existing parser to build upon
    :return: the parser
    """
    if not parser:
        from marie.parsers.base import set_base_parser

        parser = set_base_parser()

    set_pod_parser(parser)

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

    from marie.parsers.orchestrate.base import mixin_base_ppr_parser
    from marie.parsers.orchestrate.pod import mixin_pod_parser
    from marie.parsers.orchestrate.runtimes.remote import (
        mixin_gateway_parser,
        mixin_graphql_parser,
        mixin_http_gateway_parser,
        mixin_prefetch_parser,
    )
    from marie.parsers.orchestrate.runtimes.worker import mixin_worker_runtime_parser

    mixin_base_ppr_parser(parser)
    mixin_worker_runtime_parser(parser)
    mixin_prefetch_parser(parser)
    mixin_http_gateway_parser(parser)
    mixin_graphql_parser(parser)
    # mixin_comm_protocol_parser(parser)
    mixin_gateway_parser(parser)
    mixin_pod_parser(parser)

    from marie.enums import DeploymentRoleType

    parser.set_defaults(
        name="gateway",
        runtime_cls="GRPCGatewayRuntime",
        deployment_role=DeploymentRoleType.GATEWAY,
    )

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
        help="Lookup the usage & mention of the argument name in Marie API. The name can be fuzzy",
    )
    return parser


def get_main_parser():
    """The main parser for Marie

    :return: the parser
    """
    from marie.parsers.base import set_base_parser
    from marie.parsers.flow import set_flow_parser
    from marie.parsers.helper import _SHOW_ALL_ARGS, _chf
    from marie.parsers.ping import set_ping_parser

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
            description="Start an Executor. Executor is how Marie processes Document.",
            formatter_class=_chf,
        )
    )

    set_flow_parser(
        sp.add_parser(
            "flow",
            description="Start a Flow. Flow is how Marie streamlines and distributes Executors.",
            help="Start a Flow",
            formatter_class=_chf,
        )
    )

    set_ping_parser(
        sp.add_parser(
            "ping",
            help="Ping an Executor",
            description="Ping a Deployment and check its network connectivity.",
            formatter_class=_chf,
        )
    )

    set_gateway_parser(
        sp.add_parser(
            "gateway",
            description="Start a Gateway that receives client Requests via gRPC/REST interface",
            **(dict(help="Start a Gateway")) if _SHOW_ALL_ARGS else {},
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

    return parser
