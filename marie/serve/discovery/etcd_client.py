import functools
import logging
import time
from collections import namedtuple
from typing import Callable, Union

import etcd3
import grpc
from etcd3 import etcdrpc
from etcd3.client import EtcdTokenCallCredentials
from grpc._channel import _Rendezvous

__all__ = ['EtcdClient']

Event = namedtuple('Event', 'key event value')
log = logging.getLogger(__name__)


def reauthenticate(etcd_sync, creds, executor):
    # This code is taken from the constructor of etcd3.client.Etcd3Client class.
    # Related issue: kragniz/python-etcd3#580
    etcd_sync.auth_stub = etcdrpc.AuthStub(etcd_sync.channel)
    auth_request = etcdrpc.AuthenticateRequest(
        name=creds['user'],
        password=creds['password'],
    )
    resp = etcd_sync.auth_stub.Authenticate(auth_request, etcd_sync.timeout)
    etcd_sync.metadata = (('token', resp.token),)
    etcd_sync.call_credentials = grpc.metadata_call_credentials(
        EtcdTokenCallCredentials(resp.token)
    )


def reconn_reauth_adaptor(meth: Callable):
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
                        'etcd3 connection failed more than %d times. retrying after 1 sec...',
                        num_reconn_tries,
                    )
                else:
                    log.debug('etcd3 connection failed. retrying after 1 sec...')
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
                    log.debug('etcd3 reauthenticated due to auth token expiration.')
                    num_reauth_tries += 1
                    continue
                else:
                    raise

    return wrapped


def _slash(v: str):
    return v.rstrip('/') + '/' if len(v) > 0 else ''


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
        namespace='marie',
        credentials=None,
        encoding='utf8',
        retry_times=10,
    ):
        self._host = etcd_host
        self._port = etcd_port
        self._client_idx = 0
        self._cluster = None
        self.encoding = encoding
        self.retry_times = retry_times
        self.ns = namespace
        self._creds = credentials

        addr = f'{etcd_host}:{etcd_port}'
        times = 0
        while times < self.retry_times:
            try:
                self.client = etcd3.client(
                    host=self._host,
                    port=self._port,
                    user=credentials.get('user') if credentials else None,
                    password=credentials.get('password') if credentials else None,
                )
                self._cluster = [member._etcd_client for member in self.client.members]
                break
            except grpc.RpcError as e:
                if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNKNOWN):
                    log.debug('etcd3 connection failed. retrying after 1 sec...')
                    time.sleep(1)
                    continue
                raise e
        if times >= self.retry_times:
            raise ValueError(
                f"Initialize etcd client failed failed after {self.retry_times} times."
            )
        log.info('using etcd cluster from {} with namespace "{}"', addr, namespace)

    def _mangle_key(self, k: str) -> bytes:
        if k.startswith('/'):
            k = k[1:]
        return f'/{self.ns}/{k}'.encode(self.encoding)

    def _demangle_key(self, k: Union[bytes, str]) -> str:
        if isinstance(k, bytes):
            k = k.decode(self.encoding)
        prefix = f'/{self.ns}/'
        if k.startswith(prefix):
            k = k[len(prefix) :]
        return k

    def call(self, method, *args, **kwargs):
        """Etcd operation proxy method."""
        if self._cluster is None:
            raise ValueError("Etcd client not initialized.")

        print("Method:", method, args, kwargs)
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
        self, raw_key: bytes, event_callback: callable, prefix: bool = False, **kwargs
    ) -> int:
        """Watch a key in etcd."""

        print("Watching key:", raw_key)

        def _watch_callback(response: etcd3.watch.WatchResponse):
            print("watch_response: ", response)
            if isinstance(response, grpc.RpcError):
                if response.code() == grpc.StatusCode.UNAVAILABLE or (
                    response.code() == grpc.StatusCode.UNKNOWN
                    and "invalid auth token" not in response.details()
                ):
                    # server restarting or terminated
                    return
                else:
                    raise RuntimeError(f'Unexpected RPC Error: {response}')

            for ev in response.events:
                log.info(f"Received etcd event: {ev}")
                if isinstance(ev, etcd3.events.PutEvent):
                    ev_type = 'put'
                elif isinstance(ev, etcd3.events.DeleteEvent):
                    ev_type = 'delete'
                else:
                    raise TypeError('Not recognized etcd event type.')

                # etcd3 library uses a separate thread for its watchers.
                event = Event(
                    self._demangle_key(ev.key),
                    ev_type,
                    ev.value.decode(self.encoding),
                )
                event_callback(raw_key, event)

        try:
            if prefix:
                watch_id = self.call(
                    'add_watch_prefix_callback', raw_key, _watch_callback, **kwargs
                )
            else:
                watch_id = self.call(
                    'add_watch_callback', raw_key, _watch_callback, **kwargs
                )
            return watch_id
        except Exception as ex:
            raise ex

    def watch(self, key: str, callback, **kwargs):
        scope_prefix = ""
        mangled_key = self._mangle_key(f'{_slash(scope_prefix)}{key}')
        return self._watch(mangled_key, callback, **kwargs)

    def add_watch_prefix_callback(self, key_prefix: str, callback, **kwargs):
        scope_prefix = ""
        mangled_key = self._mangle_key(f'{_slash(scope_prefix)}{key_prefix}')
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

    def get_all(self):
        return self.call('get_all')

    def get_prefix(self, key_prefix: str, sort_order=None, sort_target='key'):
        mangled_key_prefix = self._mangle_key(key_prefix)
        return self.call(
            'get_prefix',
            mangled_key_prefix,
            sort_order=sort_order,
            sort_target=sort_target,
        )

    @reconn_reauth_adaptor
    def put(self, key: str, val: str, lease=None) -> tuple:
        """
        Put a single key-value pair to the etcd.

        :param key: The key. This must be quoted by the caller as needed.
        :param val: The value.
        :return: The key and value.
        """

        scope_prefix = ""
        mangled_key = self._mangle_key(f'{_slash(scope_prefix)}{key}')
        val = self.client.put(mangled_key, str(val).encode(self.encoding), lease=lease)
        return mangled_key, val

    def lease(self, ttl, lease_id=None):
        """Create a new lease."""
        return self.call('lease', ttl, lease_id=lease_id)

    def delete(self, key: str):
        scope_prefix = ""
        mangled_key = self._mangle_key(f'{_slash(scope_prefix)}{key}')
        return self.call('delete', mangled_key)

    def delete_prefix(self, key_prefix: str):
        scope_prefix = ""
        mangled_key_prefix = self._mangle_key(f'{_slash(scope_prefix)}{key}')
        return self.call('delete_prefix', mangled_key_prefix)

    def replace(self, key: str, initial_val: str, new_val: str):
        scope_prefix = ""
        mangled_key = self._mangle_key(f'{_slash(scope_prefix)}{key}')
        return self.call('replace', mangled_key, initial_val, new_val)

    def cancel_watch(self, watch_id):
        return self.call('cancel_watch', watch_id)


if __name__ == '__main__':
    etcd_client = EtcdClient('localhost', 2379)
    etcd_client.put('key', 'Value XYZ')

    kv = etcd_client.get('key')
    print(etcd_client.get('key'))
    # etcd_client.delete('key')
    print(etcd_client.get('key'))

    for value, key in etcd_client.get_all():
        print(key, value.decode())
