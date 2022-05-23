import asyncio
import contextlib
import ipaddress
import os


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

    pass
