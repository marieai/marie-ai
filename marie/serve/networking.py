import asyncio
import contextlib
import ipaddress
import os
from typing import Optional

from marie.serve.runtimes.asyncio import DataRequest, ControlRequest
from marie.types.request import Request


def in_docker():
    """
    Checks if the current process is running inside Docker
    :return: True if the current process is running inside Docker
    """
    path = "/proc/self/cgroup"
    if os.path.exists("/.dockerenv"):
        return True
    if os.path.isfile(path):
        with open(path) as file:
            return any("docker" in line for line in file)
    return False


def host_is_local(hostname):
    """
    Check if hostname is point to localhost
    :param hostname: host to check
    :return: True if hostname means localhost, False otherwise
    """
    import socket

    fqn = socket.getfqdn(hostname)
    if fqn in ("localhost", "0.0.0.0") or hostname == "0.0.0.0":
        return True

    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class GrpcConnectionPool:
    """
    Manages a list of grpc connections.

    :param logger: the logger to use
    :param compression: The compression algorithm to be used by this GRPCConnectionPool when sending data to GRPC
    """

    @staticmethod
    async def send_request_async(
        request: Request,
        target: str,
        timeout: float = 1.0,
        tls: bool = False,
        root_certificates: Optional[str] = None,
    ) -> Request:
        """
        Sends a request asynchronously to the target via grpc

        :param request: the request to send
        :param target: where to send the request to, like 127.0.0.1:8080
        :param timeout: timeout for the send
        :param tls: if True, use tls for the grpc channel
        :param root_certificates: the path to the root certificates for tls, only used if tls is True

        :returns: the response request
        """
        print('Request **')
        print(type(request))
        if type(request) == DataRequest:
            stub = jina_pb2_grpc.JinaSingleDataRequestRPCStub(channel)

            return await stub.process_single_data(request, timeout=timeout)
        elif type(request) == ControlRequest:
            raise NotImplemented()

