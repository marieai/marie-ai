import functools
import logging
import time
from collections import namedtuple
from typing import Callable, Dict, Mapping, Union
from urllib.parse import quote as _quote
from urllib.parse import unquote

import etcd3
import grpc
from etcd3 import etcdrpc
from etcd3.client import EtcdTokenCallCredentials
from grpc._channel import _Rendezvous

__all__ = ["EtcdClient", "Event"]

from marie.excepts import RuntimeFailToStart

Event = namedtuple("Event", "key event value")
log = logging.getLogger(__name__)

quote = functools.partial(_quote, safe="")


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


# https://github.com/qqq-tech/backend.ai-common/blob/main/src/ai/backend/common/etcd.py
class EtcdClient(object):
    """A etcd client proxy."""

    _suffer_status_code = (
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.ABORTED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
    )

    def __init__(
        self,
        etcd_host,
        etcd_port,
        namespace="marie",
        credentials=None,
        encoding="utf8",
        retry_times=10,
    ):
        self.client = None  # type: etcd3.client
        self._host = etcd_host
        self._port = etcd_port
        self._client_idx = 0
        self._cluster = None
        self.encoding = encoding
        self.retry_times = 3  # retry_times
        self.ns = namespace
        self._creds = credentials

        self.connect()

    def _mangle_key(self, key: str) -> bytes:
        if key.startswith("/"):
            key = key[1:]
        return f"/{self.ns}/{key}".encode(self.encoding)

    def _demangle_key(self, k: Union[bytes, str]) -> str:
        if isinstance(k, bytes):
            k = k.decode(self.encoding)
        prefix = f"/{self.ns}/"
        if k.startswith(prefix):
            k = k[len(prefix) :]
        return k

    def call(self, method, *args, **kwargs):
        """Etcd operation proxy method."""
        if self._cluster is None:
            raise RuntimeFailToStart("Etcd client not initialized.")

        times = 0
        while times < self.retry_times:
            client = self._cluster[self._client_idx]
            try:
                ret = getattr(client, method)(*args, **kwargs)
                return ret
            except _Rendezvous as e:
                if e.code() in self._suffer_status_code:
                    times += 1
                    self._client_idx = (self._client_idx + 1) % len(self._cluster)
                    log.info(f"Failed with exception {e}, retry after 1 second.")
                    time.sleep(1)
                raise e  # raise exception if not in suffer status code
            except Exception as e:
                times += 1
                log.info(f"Failed with exception {e}, retry after 1 second.")
                time.sleep(1)

        raise ValueError(f"Failed after {times} times.")

    def _watch(
        self, raw_key: bytes, event_callback: Callable, prefix: bool = False, **kwargs
    ) -> int:
        """Watch a key in etcd."""
        log.info(
            f"Watching raw key: {raw_key}",
        )

        def _watch_callback(response: etcd3.watch.WatchResponse):
            if isinstance(response, grpc.RpcError):
                if response.code() == grpc.StatusCode.UNAVAILABLE or (
                    response.code() == grpc.StatusCode.UNKNOWN
                    and "invalid auth token" not in response.details()
                ):
                    # server restarting or terminated
                    return
                else:
                    raise RuntimeError(f"Unexpected RPC Error: {response}")

            for ev in response.events:
                log.info(f"Received etcd event: {ev}")
                if isinstance(ev, etcd3.events.PutEvent):
                    ev_type = "put"
                elif isinstance(ev, etcd3.events.DeleteEvent):
                    ev_type = "delete"
                else:
                    raise TypeError("Not recognized etcd event type.")
                # etcd3 library uses a separate thread for its watchers.
                event = Event(
                    self._demangle_key(ev.key),
                    ev_type,
                    ev.value.decode(self.encoding),
                )
                event_callback(self._demangle_key(ev.key), event)

        try:
            if prefix:
                watch_id = self.client.add_watch_prefix_callback(
                    raw_key, _watch_callback, **kwargs
                )
            else:
                watch_id = self.client.add_watch_callback(
                    raw_key, _watch_callback, **kwargs
                )
            return watch_id
        except Exception as ex:
            raise ex

    @reconn_reauth_adaptor
    def watch(self, key: str, callback, **kwargs):
        scope_prefix = ""
        mangled_key = self._mangle_key(f"{_slash(scope_prefix)}{key}")
        return self._watch(mangled_key, callback, **kwargs)

    @reconn_reauth_adaptor
    def add_watch_prefix_callback(self, key_prefix: str, callback: Callable, **kwargs):
        scope_prefix = ""
        mangled_key = self._mangle_key(f"{_slash(scope_prefix)}{key_prefix}")
        return self._watch(mangled_key, callback, prefix=True, **kwargs)

    @reconn_reauth_adaptor
    def get(self, key: str) -> tuple:
        """
        Get a single key from the etcd.
        Returns ``None`` if the key does not exist.
        The returned value may be an empty string if the value is a zero-length string.

        :param key: The key. This must be quoted by the caller as needed.
        :return:
        """

        mangled_key = self._mangle_key(key)
        value, _ = self.client.get(mangled_key)
        return value.decode(self.encoding) if value is not None else None

    @reconn_reauth_adaptor
    def get_all(self):
        return self.client.get_all()

    @reconn_reauth_adaptor
    def put(self, key: str, val: str, lease=None) -> tuple:
        """
        Put a single key-value pair to the etcd.

        :param key: The key. This must be quoted by the caller as needed.
        :param val: The value.
        :param lease: The lease ID.
        :return: The key and value.
        """

        scope_prefix = ""
        mangled_key = self._mangle_key(f"{_slash(scope_prefix)}{key}")
        val = self.client.put(mangled_key, str(val).encode(self.encoding), lease=lease)
        return mangled_key, val

    @reconn_reauth_adaptor
    def put_prefix(self, key: str, dict_obj: Mapping[str, str]):
        """
        Put a nested dict object under the given key prefix.
        All keys in the dict object are automatically quoted to avoid conflicts with the path separator.

        :param key:  Prefix to put the given data. This must be quoted by the caller as needed.
        :param dict_obj: Nested dictionary representing the data.
        :return:
        """
        scope_prefix = ""
        flattened_dict: Dict[str, str] = {}

        def _flatten(prefix: str, inner_dict: Mapping[str, str]) -> None:
            for k, v in inner_dict.items():
                if k == "":
                    flattened_key = prefix
                else:
                    flattened_key = prefix + "/" + quote(k)
                if isinstance(v, dict):
                    _flatten(flattened_key, v)
                else:
                    flattened_dict[flattened_key] = v

        _flatten(key, dict_obj)

        return self.client.transaction(
            [],
            [
                self.client.transactions.put(
                    self._mangle_key(f"{_slash(scope_prefix)}{k}"),
                    str(v).encode(self.encoding),
                )
                for k, v in flattened_dict.items()
            ],
            [],
        )

    @reconn_reauth_adaptor
    def get_prefix(self, key_prefix: str, sort_order=None, sort_target="key") -> dict:
        """
        Retrieves all key-value pairs under the given key prefix as a nested dictionary.
        All dictionary keys are automatically unquoted.
        If a key has a value while it is also used as path prefix for other keys,
        the value directly referenced by the key itself is included as a value in a dictionary
        with the empty-string key.
        :param key_prefix: Prefix to get the data. This must be quoted by the caller as needed.
        :return: A dict object representing the data.
        """
        scope_prefix = ""
        mangled_key_prefix = self._mangle_key(f"{_slash(scope_prefix)}{key_prefix}")
        results = self.client.get_prefix(
            mangled_key_prefix, sort_order=sort_order, sort_target=sort_target
        )
        pair_sets = {
            self._demangle_key(k.key): v.decode(self.encoding) for v, k in results
        }

        return make_dict_from_pairs(
            f"{_slash(scope_prefix)}{key_prefix}", pair_sets, "/"
        )

    @reconn_reauth_adaptor
    def lease(self, ttl, lease_id=None):
        """Create a new lease."""
        return self.client.lease(ttl, lease_id=lease_id)

    @reconn_reauth_adaptor
    def delete(self, key: str):
        scope_prefix = ""
        mangled_key = self._mangle_key(f"{_slash(scope_prefix)}{key}")
        return self.client.delete(mangled_key)

    @reconn_reauth_adaptor
    def delete_prefix(self, key_prefix: str):
        scope_prefix = ""
        mangled_key_prefix = self._mangle_key(f"{_slash(scope_prefix)}{key_prefix}")
        return self.client.delete_prefix(mangled_key_prefix)

    @reconn_reauth_adaptor
    def replace(self, key: str, initial_val: str, new_val: str):
        scope_prefix = ""
        mangled_key = self._mangle_key(f"{_slash(scope_prefix)}{key}")
        return self.client.replace(mangled_key, initial_val, new_val)

    @reconn_reauth_adaptor
    def cancel_watch(self, watch_id):
        return self.client.cancel_watch(watch_id)

    @reconn_reauth_adaptor
    def reconnect(self) -> bool:
        """
        Reconnect to etcd. This method is used to recover from a connection failure.
        :return: True if reconnected successfully. False otherwise.
        """
        log.warning("Reconnecting to etcd.")
        try:
            connected = self.connect()
        except Exception as e:
            log.error(f"Failed to reconnect to etcd. {e}")
            connected = False
        log.warning(f"Reconnected to etcd. {connected}")
        return connected

    def connect(self) -> bool:
        addr = f"{self._host}:{self._port}"
        times = 0
        last_ex = None

        while times < self.retry_times:
            try:
                self.client = etcd3.client(
                    host=self._host,
                    port=self._port,
                    user=self._creds.get("user") if self._creds else None,
                    password=self._creds.get("password") if self._creds else None,
                )
                self._cluster = [member._etcd_client for member in self.client.members]
                break
            except grpc.RpcError as e:
                times += 1
                last_ex = e
                if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNKNOWN):
                    log.error(
                        f"etcd3 connection failed. retrying after 1 sec, attempt # {times} of {self.retry_times}"
                    )
                    time.sleep(1)
                    continue
                raise e
        if times >= self.retry_times:
            raise RuntimeFailToStart(
                f"Initialize etcd client failed failed after {self.retry_times} times. Due to {last_ex}"
            )
        log.info(f'using etcd cluster from {addr} with namespace "{self.ns}"')
        return True


if __name__ == "__main__":
    etcd_client = EtcdClient("localhost", 2379)
    etcd_client.put("key", "Value XYZ")

    kv = etcd_client.get("key")
    print(etcd_client.get("key"))
    # etcd_client.delete('key')
    print(etcd_client.get("key"))

    kv = {"key1": "Value 1", "key2": "Value 2", "key3": "Value 3"}

    etcd_client.put_prefix("prefix", kv)

    print(etcd_client.get_prefix("prefix"))

    print("------ GET ALL ---------")
    for kv in etcd_client.get_all():
        v = kv[0].decode("utf8")
        k = kv[1].key.decode("utf8")
        print(k, v)
