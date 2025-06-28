import functools
import queue
import threading
import time
import traceback
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union
from urllib.parse import quote as _quote
from urllib.parse import unquote

import etcd3
import grpc
from etcd3 import MultiEndpointEtcd3Client, etcdrpc
from etcd3.client import Etcd3Client, EtcdTokenCallCredentials
from grpc._channel import _Rendezvous

__all__ = ["EtcdClient", "Event"]

from marie.excepts import RuntimeFailToStart
from marie.serve.discovery.state_tracker import StateTracker

Event = namedtuple("Event", "key event value")

from marie.logging_core.predefined import default_logger as logger

log = logger

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


@dataclass
class WatchState:
    watch_id: int
    raw_key: bytes
    callback: Callable
    prefix: bool
    kwargs: dict
    last_revision: Optional[int] = None
    error_count: int = 0
    last_successful_event: float = 0
    re_establishment_count: int = 0


class ConnectionState(Enum):
    """Enum representing etcd connection states"""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class ConnectionEvent:
    """Event object for connection state changes"""

    def __init__(
        self,
        old_state: ConnectionState,
        new_state: ConnectionState,
        error: Exception = None,
    ):
        self.old_state = old_state
        self.new_state = new_state
        self.error = error
        self.timestamp = time.time()


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
        etcd_host: Optional[str] = None,
        etcd_port: Optional[int] = None,
        endpoints: Optional[List[Union[str, Tuple[str, int]]]] = None,
        namespace: str = "marie",
        credentials: Optional[Dict[str, str]] = None,
        encoding: str = "utf8",
        retry_times: int = 10,
        ca_cert: Optional[str] = None,
        cert_key: Optional[str] = None,
        cert_cert: Optional[str] = None,
        timeout: float = 5.0,
        grpc_options: Optional[List[Tuple[str, Any]]] = None,
    ):
        try:
            raise Exception
        except Exception as e:
            print(
                f"Initializing EtcdClient with parameters: "
                f"etcd_host={etcd_host}, etcd_port={etcd_port}, "
                f"endpoints={endpoints}, namespace={namespace}, "
                f"credentials={credentials}, encoding={encoding}, "
                f"retry_times={retry_times}, ca_cert={ca_cert}, "
                f"cert_key={cert_key}, cert_cert={cert_cert}, "
                f"timeout={timeout}, grpc_options={grpc_options}"
            )
            traceback.print_stack()

        self.client: Optional[Union[Etcd3Client, MultiEndpointEtcd3Client]] = None
        self._host: Optional[str] = etcd_host
        self._port: Optional[int] = etcd_port
        self._endpoints: List[Tuple[str, int]] = self._normalize_endpoints(
            endpoints, etcd_host, etcd_port
        )
        self._client_idx: int = 0
        self._cluster: Optional[List[Any]] = None
        self.encoding: str = encoding
        self.retry_times: int = retry_times
        self.ns: str = namespace
        self._creds: Optional[Dict[str, str]] = credentials
        self._ca_cert: Optional[str] = ca_cert
        self._cert_key: Optional[str] = cert_key
        self._cert_cert: Optional[str] = cert_cert
        self._timeout: float = timeout
        self._grpc_options: List[Tuple[str, Any]] = grpc_options or []
        self._is_multi_endpoint: bool = len(self._endpoints) > 1

        # convert endpoint to  etcd3.Endpoint
        if self._is_multi_endpoint:
            secure = (
                True if self._ca_cert or self._cert_key or self._cert_cert else False
            )
            self._endpoints = [
                etcd3.Endpoint(host=host, port=port, secure=secure)
                for host, port in self._endpoints
            ]

        # Watch monitoring
        self._watch_states: Dict[int, WatchState] = {}
        self._active_watches: Set[int] = set()
        self._watch_lock = threading.RLock()
        self._monitor_thread = None
        self._monitor_running = False
        self._monitor_interval = 10  # seconds
        self._max_errors_before_reestablish = 3
        self._max_time_without_events = 60  # second

        self.event_queue = queue.Queue()
        self._processor_thread = None
        self._processor_running = False
        self.state_tracker = StateTracker(ttl=60)  # 60 seconds TTL for state tracking

        self._connection_state = ConnectionState.DISCONNECTED
        self._connection_state_lock = threading.RLock()
        self._connection_event_handlers: Dict[
            ConnectionState, List[Callable[[ConnectionEvent], None]]
        ] = {
            ConnectionState.CONNECTED: [],
            ConnectionState.DISCONNECTED: [],
            ConnectionState.RECONNECTING: [],
            ConnectionState.FAILED: [],
        }
        self._connection_monitor_thread = None
        self._connection_monitor_running = False
        self._last_successful_operation = time.time()

        self._reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = 10

        self._start_event_processor()
        self.connect()

    def add_connection_event_handler(
        self, state: ConnectionState, handler: Callable[[ConnectionEvent], None]
    ):
        """Add an event handler for connection state changes"""
        with self._connection_state_lock:
            self._connection_event_handlers[state].append(handler)

    def remove_connection_event_handler(
        self, state: ConnectionState, handler: Callable[[ConnectionEvent], None]
    ):
        """Remove an event handler for connection state changes"""
        with self._connection_state_lock:
            if handler in self._connection_event_handlers[state]:
                self._connection_event_handlers[state].remove(handler)

    def _set_connection_state(
        self, new_state: ConnectionState, error: Exception = None
    ):
        """Set the connection state and notify handlers"""
        with self._connection_state_lock:
            logger.info(f"Setting connection state: {new_state} (error: {error})")
            old_state = self._connection_state

            if old_state != new_state:
                self._connection_state = new_state
                event = ConnectionEvent(old_state, new_state, error)
                handlers = self._connection_event_handlers.get(new_state, [])

                for handler in handlers:
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(f"Error in connection event handler: {e}")

                logger.info(
                    f"EtcdClient connection state changed: {old_state.value} -> {new_state.value}"
                )

                if new_state == ConnectionState.DISCONNECTED:
                    self._attempt_immediate_reconnection()
                elif new_state == ConnectionState.CONNECTED:
                    self._reconnect_attempts = 0

    def _attempt_immediate_reconnection(self):
        """Attempt reconnection immediately when disconnected"""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(
                f"Maximum reconnection attempts ({self._max_reconnect_attempts}) reached"
            )
            self._set_connection_state(ConnectionState.FAILED)
            return

        delay = min(5 * (1.5**self._reconnect_attempts), 60)
        self._reconnect_attempts += 1

        logger.info(
            f"Scheduling reconnection attempt #{self._reconnect_attempts} in {delay:.1f} seconds"
        )

        # this will prevent the asyncio loop from being blocked
        timer = threading.Timer(delay, self._do_reconnection_attempt)
        timer.daemon = True
        timer.start()

    def _do_reconnection_attempt(self):
        """Execute reconnection attempt"""
        current_state = self.get_connection_state()
        if current_state != ConnectionState.DISCONNECTED:
            return

        try:
            logger.info(
                f"Attempting automatic reconnection #{self._reconnect_attempts}..."
            )
            self.reconnect()
            self._last_successful_operation = time.time()

        except Exception as e:
            logger.error(
                f"Automatic reconnection attempt #{self._reconnect_attempts} failed: {e}"
            )
            self._set_connection_state(ConnectionState.DISCONNECTED, e)

    def get_connection_state(self) -> ConnectionState:
        """Get the current connection state"""
        with self._connection_state_lock:
            return self._connection_state

    def _start_connection_monitor(self):
        """Start monitoring connection health"""
        if (
            self._connection_monitor_thread
            and self._connection_monitor_thread.is_alive()
        ):
            return

        self._connection_monitor_running = True
        self._connection_monitor_thread = threading.Thread(
            target=self._monitor_connection_health,
            name="etcd-connection-monitor",
            daemon=True,
        )
        self._connection_monitor_thread.start()
        logger.info("Started etcd connection monitor")

    def _stop_connection_monitor(self):
        """Stop monitoring connection health"""
        self._connection_monitor_running = False
        if self._connection_monitor_thread:
            self._connection_monitor_thread.join(timeout=5)

    def _monitor_connection_health(self):
        """Monitor connection health and update state accordingly"""
        check_interval = 5.0  # Check every 5 seconds
        failure_threshold = (
            15.0  # Consider failed after 15 seconds without successful operation
        )

        while self._connection_monitor_running:
            try:
                current_time = time.time()
                time_since_last_success = current_time - self._last_successful_operation

                # Check if we should consider the connection failed
                if time_since_last_success > failure_threshold:
                    current_state = self.get_connection_state()
                    if current_state == ConnectionState.CONNECTED:
                        self._set_connection_state(ConnectionState.DISCONNECTED)

                # Perform a lightweight health check
                try:
                    self.get("__health_check__")
                    self._last_successful_operation = current_time

                    current_state = self.get_connection_state()
                    if current_state in [
                        ConnectionState.DISCONNECTED,
                        ConnectionState.FAILED,
                    ]:
                        self._set_connection_state(ConnectionState.CONNECTED)

                except Exception as e:
                    current_state = self.get_connection_state()
                    if current_state == ConnectionState.CONNECTED:
                        self._set_connection_state(ConnectionState.DISCONNECTED, e)

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error in connection health monitor: {e}")
                time.sleep(check_interval)

    def _normalize_endpoints(self, endpoints, etcd_host, etcd_port):
        """Normalize various input formats to a list of (host, port) tuples."""
        if endpoints:
            # Handle different endpoint formats
            normalized = []
            for endpoint in endpoints:
                if isinstance(endpoint, tuple):
                    # Already a tuple (host, port)
                    normalized.append((endpoint[0], int(endpoint[1])))
                elif isinstance(endpoint, str):
                    # String format like "host:port" or just "host"
                    host, port = self._parse_single_endpoint(endpoint)
                    normalized.append((host, port))
                else:
                    raise ValueError(f"Invalid endpoint format: {endpoint}")
            return normalized
        elif etcd_host:
            # Handle etcd_host parameter
            if isinstance(etcd_host, str) and ',' in etcd_host:
                # Comma-separated string format
                normalized = []
                endpoint_strings = [
                    ep.strip() for ep in etcd_host.split(',') if ep.strip()
                ]
                for endpoint_str in endpoint_strings:
                    host, port = self._parse_single_endpoint(endpoint_str)
                    normalized.append((host, port))
                return normalized
            else:
                # Single endpoint from host/port
                port = etcd_port if etcd_port is not None else 2379
                return [(etcd_host, int(port))]
        else:
            raise ValueError(
                "Either 'endpoints' or 'etcd_host'/'etcd_port' must be provided"
            )

    def _parse_single_endpoint(self, endpoint_str):
        """
        Parse a single endpoint string into (host, port) tuple.

        Args:
            endpoint_str: String like "host:port" or "host"

        Returns:
            Tuple of (host, port)
        """
        endpoint_str = endpoint_str.strip()

        if not endpoint_str:
            raise ValueError("Empty endpoint string")

        # Handle IPv6 addresses like [::1]:2379 or [2001:db8::1]:2379
        if endpoint_str.startswith('['):
            if ']:' not in endpoint_str:
                raise ValueError(f"Invalid IPv6 endpoint format: {endpoint_str}")
            host, port_str = endpoint_str.rsplit(']:', 1)
            host = host[1:]  # Remove leading '['
        elif ':' in endpoint_str:
            # IPv4 or hostname format: host:port
            # Use rsplit to handle cases where host might contain ':'
            host, port_str = endpoint_str.rsplit(':', 1)
        else:
            # No port specified, use default
            host = endpoint_str
            port_str = "2379"

        # Validate and convert port
        try:
            port = int(port_str)
            if not (1 <= port <= 65535):
                raise ValueError(f"Port must be between 1 and 65535, got: {port}")
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid port number in endpoint: {endpoint_str}")
            raise

        if not host:
            raise ValueError(f"Empty host in endpoint: {endpoint_str}")

        return (host, port)

    def _create_client(self) -> Union[Etcd3Client, MultiEndpointEtcd3Client]:
        """Create the appropriate etcd client based on endpoint configuration."""
        try:
            client_kwargs = {
                'timeout': self._timeout,
                'grpc_options': self._grpc_options,
            }

            # Add credentials
            if self._creds:
                client_kwargs['user'] = self._creds.get("user")
                client_kwargs['password'] = self._creds.get("password")

            # Add TLS certificates if provided
            if self._ca_cert:
                client_kwargs['ca_cert'] = self._ca_cert
            if self._cert_key:
                client_kwargs['cert_key'] = self._cert_key
            if self._cert_cert:
                client_kwargs['cert_cert'] = self._cert_cert

            if self._is_multi_endpoint:
                client_kwargs['endpoints'] = self._endpoints
                # we need to pop the grpc_options from client_kwargs as they are not part of the constructor
                client_kwargs.pop('grpc_options')

                return MultiEndpointEtcd3Client(**client_kwargs)
            else:
                # Use single endpoint client
                host, port = self._endpoints[0]
                client_kwargs['host'] = host
                client_kwargs['port'] = port
                return etcd3.client(**client_kwargs)
        except Exception as e:
            log.error(f"Failed to create etcd client: {e}")
            raise

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
        """Etcd operation proxy method with improved failover."""
        if self._cluster is None:
            raise RuntimeFailToStart("Etcd client not initialized.")

        times = 0
        while times < self.retry_times:
            if self._is_multi_endpoint:
                # For multi-endpoint client, use the client directly (it handles failover)
                client = self.client
            else:
                # For single endpoint, use the cluster failover logic
                client = self._cluster[self._client_idx]

            try:
                ret = getattr(client, method)(*args, **kwargs)
                return ret
            except _Rendezvous as e:
                if e.code() in self._suffer_status_code:
                    times += 1
                    if not self._is_multi_endpoint:
                        # Only cycle through cluster members for single endpoint
                        self._client_idx = (self._client_idx + 1) % len(self._cluster)
                    log.info(f"Failed with exception {e}, retry after 1 second.")
                    time.sleep(1)
                    continue
                raise e  # raise exception if not in suffer status code
            except Exception as e:
                times += 1
                if not self._is_multi_endpoint:
                    # Only cycle through cluster members for single endpoint
                    self._client_idx = (self._client_idx + 1) % len(self._cluster)
                log.info(f"Failed with exception {e}, retry after 1 second.")
                time.sleep(1)

        raise ValueError(f"Failed after {times} times.")

    def start_watch_monitor(self):
        """Start the watch monitoring thread"""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return

        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_watches, daemon=True, name="etcd-watch-monitor"
        )
        self._monitor_thread.start()
        log.info("Watch monitor started")

    def stop_watch_monitor(self):
        """Stop the watch monitoring thread"""
        self._monitor_running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        log.info("Watch monitor stopped")

    def _monitor_watches(self):
        """Background thread to monitor watch health"""
        while self._monitor_running:
            try:
                with self._watch_lock:
                    watches_to_check = list(self._watch_states.items())

                for watch_id, watch_state in watches_to_check:
                    if not self.monitor_watch_state(watch_id, watch_state):
                        log.warning(
                            f"Watch {watch_id} is unhealthy, attempting re-establishment"
                        )
                        self._reestablish_watch(watch_id, watch_state)

                time.sleep(self._monitor_interval)

            except Exception as e:
                log.error(f"Watch monitor error: {e}")
                time.sleep(5)

    def monitor_watch_state(
        self, watch_id: int, watch_state: Optional[WatchState] = None
    ) -> bool:
        """Monitor watch state through gRPC stream status and activity"""
        try:
            # Get watch state if not provided
            if watch_state is None:
                with self._watch_lock:
                    watch_state = self._watch_states.get(watch_id)
                    if not watch_state:
                        return False

            try:
                self.client.status()
            except Exception as e:
                log.error(f"etcd connection check failed for watch {watch_id}: {e}")
                watch_state.error_count += 1
                return False

            if watch_state.error_count >= self._max_errors_before_reestablish:
                log.warning(
                    f"Watch {watch_id} has {watch_state.error_count} errors, needs re-establishment"
                )
                return False

            if (
                watch_state.last_successful_event > 0
            ):  # Only check if we've had events before
                time_since_last_event = time.time() - watch_state.last_successful_event
                if time_since_last_event > self._max_time_without_events:
                    log.warning(
                        f"Watch {watch_id} inactive for {time_since_last_event:.1f}s, needs re-establishment"
                    )
                    return False

            if watch_id not in self._active_watches:
                log.warning(
                    f"Watch {watch_id} no longer in active watches, needs re-establishment"
                )
                return False

            return True

        except Exception as e:
            log.error(f"Error monitoring watch {watch_id}: {e}")
            if watch_state:
                watch_state.error_count += 1
            return False

    def _reestablish_watch(self, old_watch_id: int, watch_state: WatchState):
        """Re-establish a watch from a known good revision"""
        try:
            with self._watch_lock:
                # Cancel the old watch
                try:
                    if old_watch_id in self._active_watches:
                        self.client.cancel_watch(old_watch_id)
                        self._active_watches.discard(old_watch_id)
                except Exception as e:
                    log.warning(f"Failed to cancel old watch {old_watch_id}: {e}")

                start_revision = self._get_reestablish_revision(watch_state)
                kwargs = watch_state.kwargs.copy()

                if start_revision:
                    kwargs['start_revision'] = start_revision

                try:
                    new_watch_id = self._watch_internal(
                        watch_state.raw_key,
                        watch_state.callback,
                        watch_state.prefix,
                        **kwargs,
                    )

                    if new_watch_id:
                        watch_state.watch_id = new_watch_id
                        watch_state.error_count = 0
                        watch_state.re_establishment_count += 1

                        if old_watch_id in self._watch_states:
                            del self._watch_states[old_watch_id]
                        self._watch_states[new_watch_id] = watch_state

                        log.info(
                            f"Watch re-established: {old_watch_id} -> {new_watch_id} "
                            f"(attempt #{watch_state.re_establishment_count})"
                        )

                        return new_watch_id
                    else:
                        log.error(
                            f"Failed to re-establish watch for key {watch_state.raw_key}"
                        )

                except Exception as e:
                    log.error(f"Error re-establishing watch: {e}")
                    watch_state.error_count += 1

        except Exception as e:
            log.error(f"Critical error in watch re-establishment: {e}")

        return None

    def _get_reestablish_revision(self, watch_state: WatchState) -> Optional[int]:
        """Determine the revision to start from when re-establishing a watch"""
        try:
            if watch_state.last_revision:
                log.info(
                    f"Re-establishing watch from last known revision: {watch_state.last_revision}"
                )
                return watch_state.last_revision

            # This ensures we don't miss any recent changes
            status = self.client.status()
            current_revision = status.header.revision
            log.info(f"Re-establishing watch from current revision: {current_revision}")
            return current_revision

        except Exception as e:
            log.error(f"Failed to determine re-establishment revision: {e}")
            return None

    def _create_watch_callback(
        self,
        raw_key: bytes,
        event_callback: Callable,
        return_type=None,
        use_queue=False,
        watch_state=None,
    ):
        """Create a watch callback function with common event processing logic"""

        def _watch_callback(response):
            try:
                if isinstance(response, grpc.RpcError):
                    if response.code() == grpc.StatusCode.UNAVAILABLE or (
                        response.code() == grpc.StatusCode.UNKNOWN
                        and "invalid auth token" not in response.details()
                    ):
                        # server restarting or terminated
                        if watch_state:
                            watch_state.error_count += 1
                        return
                    else:
                        if watch_state:
                            watch_state.error_count += 1
                        raise RuntimeError(f"Unexpected RPC Error: {response}")

                for ev in response.events:
                    try:
                        log.debug(f"Received etcd event: {ev}")
                        if watch_state:
                            watch_state.last_successful_event = time.time()
                            watch_state.error_count = 0

                        if isinstance(ev, etcd3.events.PutEvent):
                            ev_type = "put"
                        elif isinstance(ev, etcd3.events.DeleteEvent):
                            ev_type = "delete"
                        else:
                            raise TypeError("Not recognized etcd event type.")

                        key = self._demangle_key(ev.key)
                        value = ev.value.decode(self.encoding) if ev.value else None

                        if return_type == 'dict':
                            scope_prefix = ""
                            raw_key_str = raw_key.decode("utf-8")
                            key_prefix = self._demangle_key(
                                f"{_slash(scope_prefix)}{raw_key_str}"
                            )
                            pair_sets = {key: value}
                            pairs = make_dict_from_pairs(
                                f"{_slash(scope_prefix)}{key_prefix}", pair_sets, "/"
                            )
                            value = pairs

                        state_key = f"{key}:{ev_type}"
                        event = Event(key, ev_type, value)

                        if self.state_tracker.has_state_changed(state_key, value):
                            log.debug(
                                f"State changed for {key}: {ev_type} -> fired event"
                            )
                            if use_queue:
                                event_data = {
                                    'key': self._demangle_key(ev.key),
                                    'event': event,
                                    'callback': event_callback,
                                    'watch_id': (
                                        watch_state.watch_id if watch_state else None
                                    ),
                                }
                                self.event_queue.put(event_data)
                            else:
                                event_callback(self._demangle_key(ev.key), event)
                        else:
                            log.debug(
                                f"State unchanged for {key}: {ev_type} -> skipped event"
                            )

                    except Exception as e:
                        log.error(f"Error processing watch event: {e}")
                        if watch_state:
                            watch_state.error_count += 1

            except Exception as e:
                log.error(f"Error in watch callback: {e}")
                if watch_state:
                    watch_state.error_count += 1

        return _watch_callback

    def _watch_internal(
        self, raw_key: bytes, event_callback: Callable, prefix: bool = False, **kwargs
    ) -> int:
        """Internal watch method that handles the actual etcd watch creation"""
        log.info(f"Creating internal watch for key: {raw_key}, prefix: {prefix}")

        return_type = kwargs.get('return_type')
        callback = self._create_watch_callback(
            raw_key=raw_key,
            event_callback=event_callback,
            return_type=return_type,
            use_queue=False,
            watch_state=None,
        )

        try:
            if prefix:
                watch_id = self.client.add_watch_prefix_callback(
                    raw_key, callback, **kwargs
                )
            else:
                watch_id = self.client.add_watch_callback(raw_key, callback, **kwargs)

            log.info(f"Internal watch created with ID: {watch_id} for key: {raw_key}")
            if watch_id is None:
                log.error(f"Invalid watch_id received: {watch_id} for key: {raw_key}")
                return None

            return watch_id
        except Exception as ex:
            log.error(f"Failed to create internal watch for key {raw_key}: {ex}")
            raise RuntimeError(
                f"Failed to create internal watch for key: {raw_key}"
            ) from ex

    def _watch(
        self, raw_key: bytes, event_callback: Callable, prefix: bool = False, **kwargs
    ) -> int:
        """Enhanced watch method with monitoring support"""
        log.info(f"Creating watch for key: {raw_key}, prefix: {prefix}")

        # return_type = kwargs.get('return_type')
        return_type = 'dict'  #

        # Create watch state for monitoring
        watch_state = WatchState(
            watch_id=None,  # Will be updated when watch is created
            raw_key=raw_key,
            callback=event_callback,
            prefix=prefix,
            kwargs=kwargs,
            last_successful_event=time.time(),
        )

        # Create callback with queue and monitoring
        callback = self._create_watch_callback(
            raw_key=raw_key,
            event_callback=event_callback,
            return_type=return_type,
            use_queue=True,
            watch_state=watch_state,
        )

        try:
            # Create the actual watch using the underlying etcd client
            if prefix:
                watch_id = self.client.add_watch_prefix_callback(
                    raw_key, callback, **kwargs
                )
            else:
                watch_id = self.client.add_watch_callback(raw_key, callback, **kwargs)

            log.info(f"Etcd client returned watch_id: {watch_id} for key: {raw_key}")

            # Validate watch_id - only None is invalid, 0 is valid
            if watch_id is None:
                log.error(f"Invalid watch_id received: {watch_id} for key: {raw_key}")
                return None

            # Update watch state and tracking
            watch_state.watch_id = watch_id

            with self._watch_lock:
                self._active_watches.add(watch_id)
                self._watch_states[watch_id] = watch_state

            # Start monitor if not running
            if not self._monitor_running:
                self.start_watch_monitor()

            log.info(f"Watch established with ID: {watch_id} for key: {raw_key}")
            return watch_id

        except Exception as ex:
            log.error(f"Failed to create watch for key {raw_key}: {ex}")
            raise RuntimeError(f"Failed to create watch for key: {raw_key}") from ex

    def is_watch_valid(self, watch_id: int) -> bool:
        """Check if a watch ID is valid"""
        return watch_id is not None and watch_id >= 0

    def is_watch_active(self, watch_id: int) -> bool:
        """Check if a watch is currently active"""
        if not self.is_watch_valid(watch_id):
            return False

        with self._watch_lock:
            return watch_id in self._active_watches

    def cancel_watch(self, watch_id: int) -> bool:
        """Enhanced cancel method with state cleanup"""
        try:
            with self._watch_lock:
                # Cancel the watch
                success = False
                if watch_id in self._active_watches:
                    self.client.cancel_watch(watch_id)
                    self._active_watches.remove(watch_id)
                    success = True

                # Clean up state
                if watch_id in self._watch_states:
                    del self._watch_states[watch_id]

                return success

        except Exception as e:
            log.error(f"Error canceling watch {watch_id}: {e}")
            return False

    def _start_event_processor(self):
        """Start single event processor thread"""
        if self._processor_running:
            return

        self._processor_running = True
        self._processor_thread = threading.Thread(
            target=self._process_events, daemon=True, name="etcd-event-processor"
        )
        self._processor_thread.start()
        log.info("Event processor started")

    def _process_events(self):
        """Process events from the queue"""
        while self._processor_running:
            try:
                event_data = self.event_queue.get(timeout=1)

                key = event_data['key']
                event = event_data['event']
                callback = event_data['callback']
                watch_id = event_data.get('watch_id')

                if watch_id and watch_id in self._watch_states:
                    watch_state = self._watch_states[watch_id]
                    watch_state.last_successful_event = time.time()
                    watch_state.error_count = 0

                try:
                    queue_size = self.event_queue.qsize()
                    log.debug(f"Event queue size: {queue_size} : {datetime.now()}")
                    callback(key, event)
                except Exception as e:
                    log.error(f"Callback error for key {key}: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Event processor error: {e}")

    def get_watch_stats(self) -> Dict:
        """Get statistics about all watches"""
        with self._watch_lock:
            stats = {
                "total_watches": len(self._watch_states),
                "active_watches": len(self._active_watches),
                "watches": {},
            }

            for watch_id, state in self._watch_states.items():
                stats["watches"][watch_id] = {
                    "key": state.raw_key.decode('utf-8', errors='ignore'),
                    "prefix": state.prefix,
                    "error_count": state.error_count,
                    "re_establishment_count": state.re_establishment_count,
                    "last_revision": state.last_revision,
                    "time_since_last_event": (
                        time.time() - state.last_successful_event
                        if state.last_successful_event
                        else 0
                    ),
                }

            return stats

    @reconn_reauth_adaptor
    def watch(self, key: str, callback, **kwargs):
        log.info(f"Adding watch callback for: {key}")
        try:
            scope_prefix = ""
            mangled_key = self._mangle_key(f"{_slash(scope_prefix)}{key}")
            watch_id = self._watch(mangled_key, callback, **kwargs)
            if watch_id is not None:
                log.info(f"Successfully created prefix watch {watch_id} for: {key}")
                return watch_id
            else:
                log.error(
                    f"Failed to create prefix watch for: {key} - got None watch_id"
                )
                raise RuntimeError(f"Failed to create prefix watch for: {key}")

        except Exception as e:
            log.error(f"Error creating prefix watch for {key}: {e}")
            raise

    @reconn_reauth_adaptor
    def add_watch_prefix_callback(self, key_prefix: str, callback: Callable, **kwargs):
        """Watch all keys with a given prefix"""
        log.info(f"Adding watch prefix callback for: {key_prefix}")

        try:
            scope_prefix = ""
            mangled_key = self._mangle_key(f"{_slash(scope_prefix)}{key_prefix}")
            watch_id = self._watch(mangled_key, callback, prefix=True, **kwargs)
            if watch_id is not None:
                log.info(
                    f"Successfully created prefix watch {watch_id} for: {key_prefix}"
                )
                return watch_id
            else:
                log.error(
                    f"Failed to create prefix watch for: {key_prefix} - got None watch_id"
                )
                raise RuntimeError(f"Failed to create prefix watch for: {key_prefix}")

        except Exception as e:
            log.error(f"Error creating prefix watch for {key_prefix}: {e}")
            raise

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
    def reconnectXXXX(self) -> bool:
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

    @reconn_reauth_adaptor
    def reconnect(self):
        """
        Reconnect to etcd with proper state management.
        This method is called when explicit reconnection is needed.
        """
        try:
            self._set_connection_state(ConnectionState.RECONNECTING)
            log.info("Explicitly reconnecting to etcd...")
            if self.client:
                try:
                    self.client.close()
                except:
                    pass

            # Reset client index for multi-endpoint setups
            if self._is_multi_endpoint:
                self._client_idx = 0

            connected = self.connect()  # Recreate the client
            if not connected:
                self._set_connection_state(ConnectionState.FAILED)
                return False
            log.warning(f"Reconnected to etcd.")

            try:
                self.get("__reconnect_test__")
                log.info("Reconnection test successful")
            except Exception as test_error:
                log.warning(f"Reconnection test failed: {test_error}")

            logger.info("Successfully reconnected to etcd")
            self._set_connection_state(ConnectionState.CONNECTED)
            self._last_successful_operation = time.time()

            return True

        except Exception as e:
            logger.error(f"Failed to reconnect to etcd: {e}")
            self._set_connection_state(ConnectionState.FAILED, e)
            raise

    def connect(self) -> bool:
        """Connect to etcd using the appropriate client type with connection state management."""
        times = 0
        last_ex = None

        # Set initial state to reconnecting (connecting for the first time)
        self._set_connection_state(ConnectionState.RECONNECTING)

        while times < self.retry_times:
            try:
                self.client = self._create_client()

                if self._is_multi_endpoint:
                    log.info(
                        f'Connected to etcd using multi-endpoint client: {self._endpoints}'
                    )
                    # For multi-endpoint client, we use the client directly
                    self._cluster = [self.client]
                else:
                    host, port = self._endpoints[0]
                    log.info(
                        f'Connected to etcd using single-endpoint client: {host}:{port}'
                    )
                    # Build cluster from members for single endpoint
                    self._cluster = [
                        member._etcd_client for member in self.client.members
                    ]

                # Connection successful - update state and start monitoring
                self._set_connection_state(ConnectionState.CONNECTED)
                self._last_successful_operation = time.time()

                # Start connection monitoring
                self._start_connection_monitor()

                break

            except grpc.RpcError as e:
                times += 1
                last_ex = e

                # Update connection state based on error type
                if times == 1:  # First failure
                    self._set_connection_state(ConnectionState.DISCONNECTED, e)

                if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNKNOWN):
                    log.error(
                        f"etcd3 connection failed. retrying after 2 sec, attempt # {times} of {self.retry_times}"
                    )
                    # Still trying to reconnect
                    if times < self.retry_times:
                        self._set_connection_state(ConnectionState.RECONNECTING, e)
                    time.sleep(1)
                    continue
                else:
                    # Non-retryable gRPC error
                    self._set_connection_state(ConnectionState.FAILED, e)
                    raise e

            except Exception as e:
                times += 1
                last_ex = e

                # Update connection state on first failure
                if times == 1:
                    self._set_connection_state(ConnectionState.DISCONNECTED, e)

                log.error(
                    f"etcd3 connection failed with {e}. retrying after 2 sec, attempt # {times} of {self.retry_times}"
                )

                if times < self.retry_times:
                    self._set_connection_state(ConnectionState.RECONNECTING, e)
                time.sleep(1)

        if times >= self.retry_times:
            # All retry attempts failed
            self._set_connection_state(ConnectionState.FAILED, last_ex)
            raise RuntimeFailToStart(
                f"Initialize etcd client failed after {self.retry_times} times. Due to {last_ex}"
            )

        log.info(f'Connected to etcd with namespace "{self.ns}"')
        return True

    def close(self):
        """Close the client"""
        try:
            log.info("Closing EtcdClient...")
            self._set_connection_state(ConnectionState.DISCONNECTED)
            self._stop_connection_monitor()

            with self._connection_state_lock:
                total_handlers = sum(
                    len(handlers)
                    for handlers in self._connection_event_handlers.values()
                )
                if total_handlers > 0:
                    log.debug(f"Clearing {total_handlers} connection event handlers")

                for state in list(self._connection_event_handlers.keys()):
                    self._connection_event_handlers[state].clear()

            self._processor_running = False
            if self._processor_thread:
                self._processor_thread.join(timeout=5)

            self.stop_watch_monitor()
            with self._watch_lock:
                for watch_id in list(self._active_watches):
                    try:
                        self.client.cancel_watch(watch_id)
                    except:
                        pass

                self._active_watches.clear()
                self._watch_states.clear()

            # Clear event queue
            try:
                while not self.event_queue.empty():
                    try:
                        self.event_queue.get_nowait()
                    except:
                        break
            except Exception as e:
                log.warning(f"Error clearing event queue: {e}")

            if hasattr(self.client, 'close'):
                self.client.close()
        except Exception as e:
            log.error(f"Error during EtcdClient close: {e}")
            self._connection_state = ConnectionState.DISCONNECTED
            raise


def client_examples():
    # These all work the same way
    etcd_client = EtcdClient("localhost", 2379, namespace="marie")
    etcd_client = EtcdClient(etcd_host="localhost", etcd_port=2379, namespace="marie")

    # List of endpoints as strings in t etcd_host format
    etcd_client = EtcdClient(
        etcd_host="etcd-node1.example.com:2379, etcd-node2.example.com,etcd-node3.example.com:2379",
    )

    # List of tuples
    etcd_client = EtcdClient(
        endpoints=[
            ("etcd-node1.example.com", 2379),
            ("etcd-node2.example.com", 2379),
            ("etcd-node3.example.com", 2379),
        ],
        namespace="marie",
    )

    # List of strings with ports
    etcd_client = EtcdClient(
        endpoints=[
            "etcd-node1.example.com:2379",
            "etcd-node2.example.com:2379",
            "etcd-node3.example.com:2379",
        ],
        namespace="marie",
    )

    # Mixed formats (defaults to port 2379 if not specified)
    etcd_client = EtcdClient(
        endpoints=[
            "etcd-node1.example.com",  # Uses port 2379
            "etcd-node2.example.com:2380",  # Uses port 2380
            ("etcd-node3.example.com", 2381),  # Uses port 2381
        ],
        namespace="marie",
    )


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
