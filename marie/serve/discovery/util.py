import asyncio
import functools
import time
from typing import Callable
from urllib.parse import quote as _quote
from urllib.parse import unquote

import etcd3
import grpc
from etcd3 import etcdrpc
from etcd3.client import EtcdTokenCallCredentials

from marie.logging_core.predefined import default_logger as logger

log = logger

quote = functools.partial(_quote, safe="")


def form_service_key(service_name: str, service_addr: str):
    """Return service's key in etcd."""
    # validate service_addr format meets the requirement of host:port or ip:port or scheme://host:port
    # if not re.match(r'^[a-zA-Z]+://[a-zA-Z0-9.]+:\d+$', service_addr):
    #     raise ValueError(f"Invalid service address: {service_addr}")

    return '/'.join((service_name, service_addr))


def make_dict_from_pairs(key_prefix, pairs, path_sep="/"):
    result = {}
    len_prefix = len(key_prefix)
    if isinstance(pairs, dict):
        iterator = pairs.items()
    else:
        iterator = pairs
    for k, v in iterator:
        if not k.startswith(key_prefix):
            continue
        subkey = k[len_prefix:]
        if subkey.startswith(path_sep):
            subkey = subkey[1:]
        path_components = subkey.split("/")
        parent = result
        for p in path_components[:-1]:
            p = unquote(p)
            if p not in parent:
                parent[p] = {}
            if p in parent and not isinstance(parent[p], dict):
                root = parent[p]
                parent[p] = {"": root}
            parent = parent[p]
        parent[unquote(path_components[-1])] = v
    return result


def _slash(v: str):
    return v.rstrip("/") + "/" if len(v) > 0 else ""


def reauthenticate(etcd_sync, creds, executor):
    # This code is taken from the constructor of etcd3.client.Etcd3Client class.
    # Related issue: kragniz/python-etcd3#580
    etcd_sync.auth_stub = etcdrpc.AuthStub(etcd_sync.channel)
    auth_request = etcdrpc.AuthenticateRequest(
        name=creds["user"],
        password=creds["password"],
    )
    resp = etcd_sync.auth_stub.Authenticate(auth_request, etcd_sync.timeout)
    etcd_sync.metadata = (("token", resp.token),)
    etcd_sync.call_credentials = grpc.metadata_call_credentials(
        EtcdTokenCallCredentials(resp.token)
    )


def reconn_reauth_adaptor(meth: Callable):
    """
    Retry connection and authentication for the given method.

    :param meth: The method to be wrapped.
    :return: The wrapped method.
    """

    @functools.wraps(meth)
    def wrapped(self, *args, **kwargs):
        num_reauth_tries = 0
        num_reconn_tries = 0

        while True:
            try:
                return meth(self, *args, **kwargs)
            except etcd3.exceptions.ConnectionFailedError:
                if num_reconn_tries >= 20:
                    log.warning(
                        "etcd3 connection failed more than %d times. retrying after 1 sec...",
                        num_reconn_tries,
                    )
                else:
                    log.debug("etcd3 connection failed. retrying after 1 sec...")
                time.sleep(1.0)
                num_reconn_tries += 1
                continue
            except grpc.RpcError as e:
                if (
                    e.code() == grpc.StatusCode.UNAUTHENTICATED
                    or (
                        e.code() == grpc.StatusCode.UNKNOWN
                        and "invalid auth token" in e.details()
                    )
                ) and self._creds:
                    if num_reauth_tries > 0:
                        raise
                    reauthenticate(self.client, self._creds, None)
                    log.debug("etcd3 reauthenticated due to auth token expiration.")
                    num_reauth_tries += 1
                    continue
                else:
                    raise

    return wrapped


def async_reconn_reauth_adaptor(meth: Callable):
    """
    Asynchronous retry and re-authentication for the given async method.

    :param meth: The async method to be wrapped.
    :return: The wrapped async method.
    """

    @functools.wraps(meth)
    async def wrapped(self, *args, **kwargs):
        num_reauth_tries = 0
        num_reconn_tries = 0

        while True:
            try:
                # Await the async method
                return await meth(self, *args, **kwargs)
            except etcd3.exceptions.ConnectionFailedError:
                if num_reconn_tries >= 20:
                    log.warning(
                        "etcd3 connection failed more than %d times. Retrying after 1 sec...",
                        num_reconn_tries,
                    )
                else:
                    log.debug("etcd3 connection failed. Retrying after 1 sec...")

                # Use non-blocking sleep
                await asyncio.sleep(1.0)
                num_reconn_tries += 1
                continue
            except grpc.RpcError as e:
                if (
                    e.code() == grpc.StatusCode.UNAUTHENTICATED
                    or (
                        e.code() == grpc.StatusCode.UNKNOWN
                        and "invalid auth token" in e.details()
                    )
                ) and self._creds:
                    if num_reauth_tries > 0:
                        raise

                    # Assuming reauthenticate is synchronous, run it in an executor
                    # If it can be made async, await it directly.
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None, reauthenticate, self.client, self._creds, None
                    )

                    log.debug("etcd3 reauthenticated due to auth token expiration.")
                    num_reauth_tries += 1
                    continue
                else:
                    raise

    return wrapped


def parse_netloc(netloc: str) -> tuple[str, int]:
    """
    Parse endpoint netloc from format 'host:port' into host and port components.

    Args:
        netloc: String in format "host:port"

    Returns:
        Tuple of (host, port)

    Raises:
        ValueError: If netloc format is invalid
    """
    if not netloc:
        raise ValueError("Empty netloc string")

    # Handle IPv6 addresses like [::1]:2379 or [2001:db8::1]:2379
    if netloc.startswith('['):
        if ']:' not in netloc:
            raise ValueError(f"Invalid IPv6 netloc format: {netloc}")
        host, port_str = netloc.rsplit(']:', 1)
        host = host[1:]  # Remove leading '['
    elif ':' in netloc:
        # IPv4 or hostname format: host:port
        # Use rsplit to handle cases where host might contain ':'
        host, port_str = netloc.rsplit(':', 1)
    else:
        raise ValueError(f"No port found in netloc: {netloc}")

    try:
        port = int(port_str)
        if not (1 <= port <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got: {port}")
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Invalid port number in netloc: {netloc}")
        raise

    if not host:
        raise ValueError(f"Empty host in netloc: {netloc}")

    return (host, port)
