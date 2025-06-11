"""Argparser module for Pod runtimes"""

import argparse
from dataclasses import dataclass
from typing import Dict

from marie.enums import PodRoleType, ProtocolType, ProviderType
from marie.helper import random_port
from marie.parsers.helper import (
    _SHOW_ALL_ARGS,
    CastPeerPorts,
    CastToIntAction,
    KVAppendAction,
    add_arg_group,
)


@dataclass
class PodTypeParams:
    """Data Class representing possible parameters for each pod type"""

    runtime_cls: str
    role_type: PodRoleType


POD_PARAMS_MAPPING: Dict[str, PodTypeParams] = {
    "worker": PodTypeParams(runtime_cls="WorkerRuntime", role_type=PodRoleType.WORKER),
    "head": PodTypeParams(runtime_cls="HeadRuntime", role_type=PodRoleType.HEAD),
    "gateway": PodTypeParams(
        runtime_cls="GatewayRuntime", role_type=PodRoleType.GATEWAY
    ),
}


def mixin_pod_parser(parser, pod_type: str = "worker"):
    """Mixing in arguments required by :class:`Pod` into the given parser.
    :param parser: the parser instance to which we add arguments
    :param pod_type: the pod_type configured by the parser. Can be either 'worker' for WorkerRuntime or 'gateway' for GatewayRuntime
    """

    gp = add_arg_group(parser, title="Pod")

    gp.add_argument(
        "--runtime-cls",
        type=str,
        default=POD_PARAMS_MAPPING[pod_type].runtime_cls,
        help="The runtime class to run inside the Pod",
    )

    gp.add_argument(
        "--timeout-ready",
        type=int,
        default=600000,
        help="The timeout in milliseconds of a Pod waits for the runtime to be ready, -1 for waiting "
        "forever",
    )

    gp.add_argument(
        "--env",
        action=KVAppendAction,
        metavar="KEY: VALUE",
        nargs="*",
        help="The map of environment variables that are available inside runtime",
    )

    gp.add_argument(
        "--env-from-secret",
        action=KVAppendAction,
        metavar="KEY: VALUE",
        nargs="*",
        help=(
            "The map of environment variables that are read from kubernetes cluster secrets"
            if _SHOW_ALL_ARGS
            else argparse.SUPPRESS
        ),
    )
    gp.add_argument(
        "--image-pull-secrets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "List of ImagePullSecrets that the Kubernetes Pods need to have access to in order to pull the image. Used in `to_kubernetes_yaml`"
            if _SHOW_ALL_ARGS
            else argparse.SUPPRESS
        ),
    )

    # hidden CLI used for internal only

    gp.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help=(
            "defines the shard identifier for the executor. It is used as suffix for the workspace path of the executor`"
            if _SHOW_ALL_ARGS
            else argparse.SUPPRESS
        ),
    )

    gp.add_argument(
        "--pod-role",
        type=PodRoleType.from_string,
        choices=list(PodRoleType),
        default=POD_PARAMS_MAPPING[pod_type].role_type,
        help=(
            "The role of this Pod in a Deployment"
            if _SHOW_ALL_ARGS
            else argparse.SUPPRESS
        ),
    )

    gp.add_argument(
        "--noblock-on-start",
        action="store_true",
        default=False,
        help=(
            "If set, starting a Pod/Deployment does not block the thread/process. It then relies on "
            "`wait_start_success` at outer function for the postpone check."
            if _SHOW_ALL_ARGS
            else argparse.SUPPRESS
        ),
    )

    gp.add_argument(
        "--floating",
        action="store_true",
        default=False,
        help="If set, the current Pod/Deployment can not be further chained, "
        "and the next `.add()` will chain after the last Pod/Deployment not this current one.",
    )

    gp.add_argument(
        "--replica-id",
        type=int,
        default=0,
        help=(
            "defines the replica identifier for the executor. It is used when `stateful` is set to true"
            if _SHOW_ALL_ARGS
            else argparse.SUPPRESS
        ),
    )

    if pod_type != "gateway":
        gp.add_argument(
            "--reload",
            action="store_true",
            default=False,
            help="If set, the Executor will restart while serving if YAML configuration source or Executor modules "
            "are changed. If YAML configuration is changed, the whole deployment is reloaded and new "
            "processes will be restarted. If only Python modules of the Executor have changed, they will be "
            "reloaded to the interpreter without restarting process.",
        )
        gp.add_argument(
            "--install-requirements",
            action="store_true",
            default=False,
            help="If set, try to install `requirements.txt` from the local Executor if exists in the Executor folder. If using Hub, install `requirements.txt` in the Hub Executor bundle to local.",
        )
    else:
        gp.add_argument(
            "--reload",
            action="store_true",
            default=False,
            help="If set, the Gateway will restart while serving if YAML configuration source is changed.",
        )
    mixin_pod_runtime_args_parser(gp, pod_type=pod_type)
    mixin_stateful_parser(gp)


def mixin_pod_runtime_args_parser(arg_group, pod_type="worker"):
    """Mixin for runtime arguments of pods
    :param arg_group: the parser instance or args group to which we add arguments
    :param pod_type: the pod_type configured by the parser. Can be either 'worker' for WorkerRuntime or 'gateway' for GatewayRuntime
    """
    alias = ["--port", "--ports"]
    if pod_type != "gateway":
        port_description = (
            "The port for input data to bind to, default is a random port between [49152, 65535]. "
            "In the case of an external Executor (`--external` or `external=True`) this can be a list of ports. "
            "Then, every resulting address will be considered as one replica of the Executor."
        )
    else:
        port_description = (
            "The port for input data to bind the gateway server to, by default, random ports between range [49152, 65535] will be assigned. "
            "The port argument can be either 1 single value in case only 1 protocol is used or multiple values when "
            "many protocols are used."
        )
        alias.extend(["--port-expose", "--port-in"])
    arg_group.add_argument(
        *alias,
        action=CastToIntAction,
        type=str,
        nargs="+",
        default=[random_port()],
        help=port_description,
    )

    server_name = "Gateway" if pod_type == "gateway" else "Executor"
    arg_group.add_argument(
        "--protocol",
        "--protocols",
        nargs="+",
        type=ProtocolType.from_string,
        choices=list(ProtocolType),
        default=[ProtocolType.GRPC],
        help=f"Communication protocol of the server exposed by the {server_name}. This can be a single value or a list of protocols, depending on your chosen Gateway. Choose the convenient protocols from: {[protocol.to_string() for protocol in list(ProtocolType)]}.",
    )

    arg_group.add_argument(
        "--provider",
        type=ProviderType.from_string,
        choices=list(ProviderType),
        default=[ProviderType.NONE],
        help=f"If set, Executor is translated to a custom container compatible with the chosen provider. Choose the convenient providers from: {[provider.to_string() for provider in list(ProviderType)]}.",
    )

    arg_group.add_argument(
        "--provider-endpoint",
        type=str,
        default=None,
        help=f"If set, Executor endpoint will be explicitly chosen and used in the custom container operated by the provider.",
    )

    arg_group.add_argument(
        "--monitoring",
        action="store_true",
        default=False,
        help="If set, spawn an http server with a prometheus endpoint to expose metrics",
    )

    arg_group.add_argument(
        "--port-monitoring",
        type=str,
        nargs="+",
        default=[random_port()],
        action=CastToIntAction,
        dest="port_monitoring",
        help=f"The port on which the prometheus server is exposed, default is a random port between [49152, 65535]",
    )

    arg_group.add_argument(
        "--retries",
        type=int,
        default=-1,
        dest="retries",
        help=f"Number of retries per gRPC call. If <0 it defaults to max(3, num_replicas)",
    )

    arg_group.add_argument(
        "--tracing",
        action="store_true",
        default=False,
        help="If set, the sdk implementation of the OpenTelemetry tracer will be available and will be enabled for automatic tracing of requests and customer span creation. "
        "Otherwise a no-op implementation will be provided.",
    )

    arg_group.add_argument(
        "--traces-exporter-host",
        type=str,
        default=None,
        help="If tracing is enabled, this hostname will be used to configure the trace exporter agent.",
    )

    arg_group.add_argument(
        "--traces-exporter-port",
        type=int,
        default=None,
        help="If tracing is enabled, this port will be used to configure the trace exporter agent.",
    )

    arg_group.add_argument(
        "--metrics",
        action="store_true",
        default=False,
        help="If set, the sdk implementation of the OpenTelemetry metrics will be available for default monitoring and custom measurements. "
        "Otherwise a no-op implementation will be provided.",
    )

    arg_group.add_argument(
        "--metrics-exporter-host",
        type=str,
        default=None,
        help="If tracing is enabled, this hostname will be used to configure the metrics exporter agent.",
    )

    arg_group.add_argument(
        "--metrics-exporter-port",
        type=int,
        default=None,
        help="If tracing is enabled, this port will be used to configure the metrics exporter agent.",
    )


def mixin_stateful_parser(parser):
    """Mixing in arguments required to work with Stateful Executors into the given parser.
    :param parser: the parser instance to which we add arguments
    """

    gp = add_arg_group(parser, title="Stateful Executor")

    gp.add_argument(
        "--stateful",
        action="store_true",
        default=False,
        help="If set, start consensus module to make sure write operations are properly replicated between all the replicas",
    )
    gp.add_argument(
        "--peer-ports",
        type=str,
        default=None,
        help="When using --stateful option, it is required to tell the cluster what are the cluster configuration. This is important"
        "when the Deployment is restarted. It indicates the ports to which each replica of the cluster binds."
        " It is expected to be a single list if shards == 1 or a dictionary if shards > 1.",
        action=CastPeerPorts,
        nargs="+",
    )


def mixin_discovery_parser(parser):
    """Add the arguments for the Gateway / Deployment

    :param parser: the parser configured
    """
    parser.add_argument(
        "--discovery",
        action="store_true",
        default=False,
        help="If set, service discovery will be enabled.",
    )

    parser.add_argument(
        "--discovery-host",
        type=str,
        default="127.0.0.1",
        help="If discovery is enabled, this can be a single hostname or a comma-separated list of endpoints (e.g., '127.0.0.1,10.0.0.1:2379'). Default is '127.0.0.1'.",
    )

    parser.add_argument(
        "--discovery-port",
        type=int,
        default=2379,
        help="If discovery is enabled, this port will be used to configure the discovery agent. Default is 2379.",
    )

    parser.add_argument(
        "--discovery-watchdog-interval",
        type=int,
        default=None,
        help="DEPRECATED! Time interval (seconds) between sending system health checks. Use `discovery-heartbeat-sec` instead.",
    )

    parser.add_argument(
        "--discovery-service-name",
        type=str,
        default="gateway/marie",
        help="If discovery is enabled, this service name will be used to configure the discovery agent. Default is 'gateway/marie'.",
    )

    parser.add_argument(
        "--discovery-namespace",
        type=str,
        default="marie",
        help="The namespace to be used in the discovery system. Default is 'marie'.",
    )

    parser.add_argument(
        "--discovery-lease-sec",
        type=int,
        default=6,
        help="The lease duration in seconds for the discovery registration. Default is 6 seconds.",
    )

    parser.add_argument(
        "--discovery-heartbeat-sec",
        type=float,
        default=2.0,
        help="Time interval (seconds, can be a float) for sending heartbeat signals. Default is 2.0 seconds.",
    )

    parser.add_argument(
        "--discovery-timeout-sec",
        type=int,
        default=10,
        help="The timeout duration in seconds for discovery operations. Default is 10 seconds.",
    )

    parser.add_argument(
        "--discovery-retry-times",
        type=int,
        default=5,
        help="Number of retry attempts for discovery operations. Default is 5 retries.",
    )

    parser.add_argument(
        "--discovery-ca-cert",
        type=str,
        default=None,
        help="Path to the CA certificate for secure communication with the discovery service.",
    )

    parser.add_argument(
        "--discovery-cert-key",
        type=str,
        default=None,
        help="Path to the client certificate key for secure communication with the discovery service.",
    )

    parser.add_argument(
        "--discovery-cert-cert",
        type=str,
        default=None,
        help="Path to the client certificate for secure communication with the discovery service.",
    )

    parser.add_argument(
        "--discovery-grpc-options",
        type=str,
        default=None,
        help="Comma-separated gRPC options for advanced configurations, e.g., 'grpc.keepalive_time_ms:30000,grpc.keepalive_timeout_ms:5000'.",
    )


def mixin_gateway_kv_store_parser(parser):
    """Add the arguments for the Gateway

    :param parser: the parser configured
    """

    parser.add_argument(
        "--kv-store-kwargs",
        action=KVAppendAction,
        metavar="KEY: VALUE",
        nargs="*",
        help="""Dictionary of kwargs arguments that will be passed to Job Monitoring service.
        """,
    )


def mixin_gateway_job_scheduler_parser(parser):
    """Add the arguments for the Gateway

    :param parser: the parser configured
    """

    parser.add_argument(
        "--job-scheduler-kwargs",
        action=KVAppendAction,
        metavar="KEY: VALUE",
        nargs="*",
        help="""Dictionary of kwargs arguments that will be passed to Job scheduling service.
        """,
    )
