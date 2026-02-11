import functools
import queue
import threading
import time
from collections import namedtuple
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union
from urllib.parse import quote as _quote

import etcd3
import grpc
from etcd3 import MultiEndpointEtcd3Client
from etcd3 import transactions as tx
from etcd3.client import Etcd3Client

from marie.excepts import RuntimeFailToStart
from marie.logging_core.predefined import default_logger as logger
from marie.serve.discovery.base import ConnectionEvent, ConnectionState
from marie.serve.discovery.state_tracker import StateTracker
from marie.serve.discovery.timeout_utils import OperationTimeoutError, run_with_timeout
from marie.serve.discovery.util import (
    _slash,
    make_dict_from_pairs,
    parse_netloc,
    reconn_reauth_adaptor,
)

__all__ = ["EtcdClient", "Event"]
Event = namedtuple("Event", "key event value")
quote = functools.partial(_quote, safe="")


def _cmp_op(cmp_obj, op: str, rhs):
    """
    Map string operator to etcd3 transactions comparator.
    Supported by etcd: '==', '!=', '<', '>'
    """
    if op == "==":
        return cmp_obj == rhs
    elif op == "!=":
        return cmp_obj != rhs
    elif op == "<":
        return cmp_obj < rhs
    elif op == ">":
        return cmp_obj > rhs
    else:
        raise ValueError(
            f"Unsupported comparator op for etcd: {op!r}. "
            "Use one of '==', '!=', '<', '>'"
        )


def _prefix_end_key(start_key_bytes: bytes) -> bytes:
    """
    Compute the exclusive range end for a key prefix, as required by etcd range deletes.
    Standard trick: bump the last byte (or append 0x00 if all 0xFF).
    """
    s = bytearray(start_key_bytes)
    for i in range(len(s) - 1, -1, -1):
        if s[i] != 0xFF:
            s[i] += 1
            return bytes(s[: i + 1])
    # all 0xFF
    return start_key_bytes + b'\x00'


def _pref_end(b: bytes) -> bytes:
    """Compute range_end for prefix range deletes."""
    s = bytearray(b)
    for i in range(len(s) - 1, -1, -1):
        if s[i] != 0xFF:
            s[i] += 1
            return bytes(s[: i + 1])
    return b + b"\x00"


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
        self.client: Optional[Union[Etcd3Client, MultiEndpointEtcd3Client]] = None
        self._host: Optional[str] = etcd_host
        self._port: Optional[int] = etcd_port
        self._client_idx: int = 0
        self._cluster: Optional[List[Any]] = None
        self.encoding: str = encoding
        self.retry_times: int = retry_times
        self.ns: str = namespace
        self._creds: Optional[Dict[str, str]] = credentials
        self._ca_cert: Optional[str] = ca_cert
        self._cert_key: Optional[str] = cert_key
        self._cert_cert: Optional[str] = cert_cert
        self._timeout: float = timeout or 30

        default_grpc_options = [
            # Send a keepalive ping only every 2 minutes (tune based on LB/NAT idle timeout)
            ('grpc.keepalive_time_ms', 120000),  # 2m
            ('grpc.keepalive_timeout_ms', 10000),  # 10s
            # Do NOT ping when there are no active calls unless you must keep a hole in a NAT/LB.
            ('grpc.keepalive_permit_without_calls', False),
            # Limit pings when there’s no data; never leave this at 0 (unlimited).
            ('grpc.http2.max_pings_without_data', 1),
            # Back off the rate of pings (must be >= server min). 60s is usually safe.
            ('grpc.http2.min_time_between_pings_ms', 60000),
            # Extra guard in C-core for idle periods (supported by python gRPC C-core):
            ('grpc.http2.min_ping_interval_without_data_ms', 60000),
            # Message size limits (keep as-is if you need them)
            ('grpc.max_receive_message_length', 1 * 1024 * 1024),
            ('grpc.max_send_message_length', 1 * 1024 * 1024),
            # Avoid forcing frequent reconnects unless you have a reason.
            # You can remove these three or make them generous:
            # ('grpc.max_connection_idle_ms', 300000),     # consider removing
            # ('grpc.max_connection_age_ms', 300000),      # consider removing
            # ('grpc.max_connection_age_grace_ms', 15000), # consider removing
        ]

        if grpc_options:
            if isinstance(grpc_options, dict):
                grpc_options = [(k, v) for k, v in grpc_options.items()]

            # Merge with defaults (user options override defaults)
            option_keys = [opt[0] for opt in grpc_options]
            filtered_defaults = [
                opt for opt in default_grpc_options if opt[0] not in option_keys
            ]
            self._grpc_options = filtered_defaults + grpc_options
        else:
            self._grpc_options = default_grpc_options

        # convert endpoint to  etcd3.Endpoint if need to
        normal_endpoints: List[Tuple[str, int]] = self._normalize_endpoints(
            endpoints, etcd_host, etcd_port
        )
        self._is_multi_endpoint: bool = len(normal_endpoints) > 1
        secure = True if self._ca_cert or self._cert_key or self._cert_cert else False
        self._endpoints = [
            etcd3.Endpoint(host=host, port=port, secure=secure)
            for host, port in normal_endpoints
        ]

        self._state_lock = threading.RLock()
        self._active_watches: Set[int] = set()

        self.event_queue = queue.Queue(maxsize=10000)
        self._processor_thread = None
        self._processor_running = False
        self.state_tracker = StateTracker(ttl=60)  # 60 seconds TTL for state tracking

        self._processor_ready = threading.Event()
        self._monitor_ready = threading.Event()

        self._connection_state = ConnectionState.DISCONNECTED
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
        self._shutting_down = False
        self._reconnect_timer = None
        self._monitor_stop = threading.Event()

        self._start_event_processor()

        if not self._processor_ready.wait(timeout=5.0):
            raise RuntimeFailToStart("Event processor failed to start within timeout")

        self._start_connection_monitor()

        if not self._monitor_ready.wait(timeout=5.0):
            logger.warning("Connection monitor did not start within timeout")

        self.connect()

    def add_connection_event_handler(
        self, state: ConnectionState, handler: Callable[[ConnectionEvent], None]
    ):
        """Add an event handler for connection state changes"""
        with self._state_lock:
            self._connection_event_handlers[state].append(handler)

    def remove_connection_event_handler(
        self, state: ConnectionState, handler: Callable[[ConnectionEvent], None]
    ):
        """Remove an event handler for connection state changes"""
        with self._state_lock:
            if handler in self._connection_event_handlers[state]:
                self._connection_event_handlers[state].remove(handler)

    def _set_connection_state(
        self, new_state: ConnectionState, error: Exception = None
    ):
        """Set the connection state and notify handlers"""
        # Prevent recursion during shutdown
        if self._shutting_down:
            return

        handlers_to_call = []
        old_state = None

        with self._state_lock:
            logger.info(f"Setting connection state: {new_state} (error: {error})")
            old_state = self._connection_state

            if old_state != new_state:
                self._connection_state = new_state
                event = ConnectionEvent(old_state, new_state, error)
                handlers = self._connection_event_handlers.get(new_state, [])
                handlers_to_call = list(handlers)

        for handler in handlers_to_call:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in connection event handler: {e}")

        if old_state != new_state:
            logger.info(
                f"EtcdClient connection state changed: {old_state.value} -> {new_state.value}"
            )

            if not self._shutting_down:
                if new_state == ConnectionState.DISCONNECTED:
                    self._attempt_immediate_reconnection()
                elif new_state == ConnectionState.CONNECTED:
                    self._reconnect_attempts = 0

    def _attempt_immediate_reconnection(self):
        """Attempt reconnection with exponential backoff when disconnected"""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(
                f"Maximum reconnection attempts ({self._max_reconnect_attempts}) reached. "
                "Manual intervention may be required."
            )
            self._set_connection_state(ConnectionState.FAILED)
            return

        if self._reconnect_timer and self._reconnect_timer.is_alive():
            logger.debug("Reconnection already scheduled, skipping")
            return

        delay = min(3 * (2**self._reconnect_attempts), 60)
        self._reconnect_attempts += 1

        logger.info(
            f"Scheduling reconnection attempt #{self._reconnect_attempts} in {delay:.0f}s"
        )

        self._reconnect_timer = threading.Timer(delay, self._do_reconnection_attempt)
        self._reconnect_timer.daemon = True
        self._reconnect_timer.start()

    def _do_reconnection_attempt(self):
        """Execute reconnection attempt"""
        current_state = self.get_connection_state()
        if current_state not in (ConnectionState.DISCONNECTED, ConnectionState.FAILED):
            logger.debug(f"Skipping reconnection - state is {current_state.name}")
            return

        try:
            logger.info(
                f"Attempting automatic reconnection #{self._reconnect_attempts}..."
            )
            self.reconnect()
            self._last_successful_operation = time.time()
            logger.info("Reconnection successful")

        except Exception as e:
            # Log concisely - the connect() method already logged details
            logger.warning(
                f"Reconnection attempt #{self._reconnect_attempts} failed, will retry"
            )
            self._set_connection_state(ConnectionState.DISCONNECTED, e)

    def get_connection_state(self) -> ConnectionState:
        """Get the current connection state"""
        with self._state_lock:
            return self._connection_state

    def _start_connection_monitor(self):
        """Start monitoring connection health"""
        if (
            self._connection_monitor_thread
            and self._connection_monitor_thread.is_alive()
        ):
            return

        self._monitor_ready.clear()
        self._connection_monitor_running = True
        self._connection_monitor_thread = threading.Thread(
            target=self._monitor_connection_health,
            name="etcd-connection-monitor",
            daemon=True,
        )
        self._connection_monitor_thread.start()

    def _monitor_connection_health(self):
        """Monitor connection health with timeout-protected health checks."""
        logger.info("Started etcd connection monitor")
        self._monitor_ready.set()

        # Balanced detection parameters - faster than before but not too aggressive
        check_interval = 3.0  # Was 5.0 - check every 3 seconds
        failure_threshold = 6.0  # Was 15.0 - 2 consecutive failed checks
        health_check_timeout = 2.0  # NEW - timeout for health check

        # Hysteresis counters
        consecutive_failures = 0
        consecutive_successes = 0

        # Monotonic baseline (avoid wall-clock jumps)
        last_ok_monotonic = time.monotonic()

        while self._connection_monitor_running and not self._monitor_stop.is_set():
            try:
                now_mono = time.monotonic()

                # If it's been a while since last success, consider flipping to DISCONNECTED
                if (now_mono - last_ok_monotonic) > failure_threshold:
                    if (
                        self.get_connection_state() == ConnectionState.CONNECTED
                        and consecutive_failures >= 1
                    ):
                        # Require at least 2 consecutive failures before declaring down
                        self._set_connection_state(ConnectionState.DISCONNECTED)

                try:
                    run_with_timeout(
                        lambda: self.get("__health_check__"),
                        timeout=health_check_timeout,
                        operation_name="health_check",
                    )

                    consecutive_successes += 1
                    consecutive_failures = 0
                    last_ok_monotonic = now_mono
                    self._last_successful_operation = time.time()

                    cur = self.get_connection_state()
                    if (
                        cur in (ConnectionState.DISCONNECTED, ConnectionState.FAILED)
                        and consecutive_successes >= 2
                    ):
                        self._set_connection_state(ConnectionState.CONNECTED)

                except OperationTimeoutError:
                    consecutive_failures += 1
                    consecutive_successes = 0
                    logger.warning(
                        f"Health check timed out after {health_check_timeout}s "
                        f"(failures: {consecutive_failures})"
                    )

                    cur = self.get_connection_state()
                    if cur == ConnectionState.CONNECTED and consecutive_failures >= 2:
                        self._set_connection_state(ConnectionState.DISCONNECTED)

                except Exception as e:
                    consecutive_failures += 1
                    consecutive_successes = 0

                    cur = self.get_connection_state()
                    # Only flip to DISCONNECTED from CONNECTED after 2 consecutive failures
                    if cur == ConnectionState.CONNECTED and consecutive_failures >= 2:
                        self._set_connection_state(ConnectionState.DISCONNECTED, e)

                # Cooperative wait so close() can stop us immediately
                self._monitor_stop.wait(check_interval)

            except Exception as e:
                logger.error("Error in connection health monitor", exc_info=True)
                self._monitor_stop.wait(check_interval)

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
                host, port = parse_netloc(self._endpoints[0].netloc)
                logger.info(f"Parsed endpoint netloc - host: {host}, port: {port}")

                client_kwargs['host'] = host
                client_kwargs['port'] = port

                return etcd3.client(**client_kwargs)
        except Exception as e:
            logger.error("Failed to create etcd client", exc_info=True)
            raise

    def _mangle_key(self, key: str) -> bytes:
        """
        Turn a logical key like 'foo/bar' into b'/{ns}/foo/bar'.

        Idempotent:
          - '/{ns}/foo' → b'/{ns}/foo'
          - 'foo'       → b'/{ns}/foo'
          - '/foo'      → b'/{ns}/foo'
        """
        if not isinstance(key, str):
            key = str(key)

        ns_prefix = f"/{self.ns}/"

        # If caller passed a full path beginning with '/{ns}/', strip it to avoid double namespace
        if key.startswith(ns_prefix):
            key = key[len(ns_prefix) :]

        # If caller passed any leading slash, drop it (we add ours)
        if key.startswith("/"):
            key = key[1:]

        return f"{ns_prefix}{key}".encode(self.encoding)

    def _demangle_key(self, k: Union[bytes, str]) -> str:
        if isinstance(k, bytes):
            k = k.decode(self.encoding)
        prefix = f"/{self.ns}/"
        if k.startswith(prefix):
            k = k[len(prefix) :]
        return k

    class Txn:
        """
        Fluent transaction builder for EtcdClient.
        Usage:
            with client.txn() as t:
                (t.if_missing("locks/mykey")
                   .put("locks/mykey", owner_id, lease=my_lease.id))
            ok, resp = t.commit()
        """

        def __init__(self, outer: "EtcdClient"):
            self._outer = outer
            self._cmp = []
            self._then = []
            self._else = []
            self._committed = False

        def if_value(
            self, key: str, op: str, value: Union[str, bytes]
        ) -> "EtcdClient.Txn":
            mk = self._outer._mangle_key(key)
            vb = (
                value
                if isinstance(value, (bytes, bytearray))
                else str(value).encode(self._outer.encoding)
            )
            c = tx.Value(mk)
            self._cmp.append(_cmp_op(c, op, vb))
            return self

        def if_version(self, key: str, op: str, ver: int) -> "EtcdClient.Txn":
            mk = self._outer._mangle_key(key)
            c = tx.Version(mk)
            self._cmp.append(_cmp_op(c, op, ver))
            return self

        def if_mod_revision(self, key: str, op: str, rev: int) -> "EtcdClient.Txn":
            mk = self._outer._mangle_key(key)
            c = tx.Mod(mk)
            self._cmp.append(_cmp_op(c, op, rev))
            return self

        def if_create_revision(self, key: str, op: str, rev: int) -> "EtcdClient.Txn":
            mk = self._outer._mangle_key(key)
            c = tx.Create(mk)
            self._cmp.append(_cmp_op(c, op, rev))
            return self

        def if_exists(self, key: str) -> "EtcdClient.Txn":
            # version(key) > 0 is a common idiom for "exists"
            return self.if_version(key, ">", 0)

        def if_missing(self, key: str) -> "EtcdClient.Txn":
            # version(key) == 0 means "not created yet"
            return self.if_version(key, "==", 0)

        def put(
            self,
            key: str,
            val: Union[str, bytes],
            *,
            lease=None,
            else_branch: bool = False,
        ) -> "EtcdClient.Txn":
            mk = self._outer._mangle_key(key)
            vb = (
                val
                if isinstance(val, (bytes, bytearray))
                else str(val).encode(self._outer.encoding)
            )
            op = tx.Put(mk, vb, lease=lease)
            (self._else if else_branch else self._then).append(op)
            return self

        def delete(self, key: str, *, else_branch: bool = False) -> "EtcdClient.Txn":
            mk = self._outer._mangle_key(key)
            op = tx.Delete(mk)
            (self._else if else_branch else self._then).append(op)
            return self

        def delete_prefix(
            self, key_prefix: str, *, else_branch: bool = False
        ) -> "EtcdClient.Txn":
            start = self._outer._mangle_key(key_prefix)
            end = _prefix_end_key(start)
            op = tx.Delete(start, range_end=end)
            (self._else if else_branch else self._then).append(op)
            return self

        def then(self) -> List[Any]:
            return self._then

        def otherwise(self) -> List[Any]:
            return self._else

        def commit(self) -> Tuple[bool, list]:
            if self._committed:
                raise RuntimeError("Transaction already committed.")
            self._committed = True
            return self._outer.transaction(self._cmp, self._then, self._else)

        def __enter__(self) -> "EtcdClient.Txn":
            return self

        def __exit__(self, exc_type, exc, tb):
            # do not auto-commit on exceptions
            return False

    def txn(self) -> "EtcdClient.Txn":
        """Create a new fluent transaction builder."""
        return EtcdClient.Txn(self)

    @reconn_reauth_adaptor
    def put_if_absent(self, key: str, val: Union[str, bytes], lease=None) -> bool:
        """
        Atomically create key if it does not exist.
        Returns True if created, False if already present.
        """
        with self.txn() as t:
            t.if_missing(key).put(key, val, lease=lease)
            ok, _ = t.commit()
            return bool(ok)

    @reconn_reauth_adaptor
    def compare_and_set(
        self, key: str, expected: Union[str, bytes], new: Union[str, bytes], lease=None
    ) -> bool:
        """
        CAS based on value equality.
        Returns True if swapped, False if current value != expected or key missing.
        """
        with self.txn() as t:
            t.if_value(key, "==", expected).put(key, new, lease=lease)
            ok, _ = t.commit()
            return bool(ok)

    @reconn_reauth_adaptor
    def update_if_unchanged(
        self, key: str, new: Union[str, bytes], mod_revision: int, lease=None
    ) -> bool:
        """
        CAS based on mod_revision (optimistic concurrency).
        Returns True if swapped, False if key changed since caller read it.
        """
        with self.txn() as t:
            t.if_mod_revision(key, "==", mod_revision).put(key, new, lease=lease)
            ok, _ = t.commit()
            return bool(ok)

    @reconn_reauth_adaptor
    def increment(
        self,
        key: str,
        delta: int = 1,
        *,
        create_default: int = 0,
        max_retries: int = 16,
    ) -> int:
        """
        Atomic integer increment implemented as a read+CAS loop using mod_revision.
        Returns the new integer value on success. Raises on exhaustion.
        """
        tries = 0
        while True:
            tries += 1
            val_bytes, meta = self.get(key, metadata=True, serializable=False)
            if val_bytes is None:
                # try to create
                target = int(create_default) + int(delta)
                if self.put_if_absent(key, str(target)):
                    return target
                # someone beat us, retry
            else:
                cur_s = val_bytes.decode(self.encoding).strip()
                try:
                    cur = int(cur_s)
                except ValueError:
                    raise ValueError(
                        f"increment() expects integer value at '{key}', found: {cur_s!r}"
                    )
                new_val = cur + int(delta)
                if self.update_if_unchanged(key, str(new_val), meta.mod_revision):
                    return new_val

            if tries >= max_retries:
                raise RuntimeError(
                    f"increment() failed after {max_retries} CAS attempts for key '{key}'"
                )
            time.sleep(0.01 * tries)  # small backoff

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
            except grpc.RpcError as e:
                if e.code() in self._suffer_status_code:
                    times += 1
                    if not self._is_multi_endpoint:
                        # Only cycle through cluster members for single endpoint
                        self._client_idx = (self._client_idx + 1) % len(self._cluster)
                    logger.info(f"Failed with exception {e}, retry after 1 second.")
                    time.sleep(1)
                    continue
                raise e  # raise exception if not in suffer status code
            except Exception as e:
                times += 1
                if not self._is_multi_endpoint:
                    # Only cycle through cluster members for single endpoint
                    self._client_idx = (self._client_idx + 1) % len(self._cluster)
                logger.info(f"Failed with exception {e}, retry after 1 second.")
                time.sleep(1)

        raise ValueError(f"Failed after {times} times.")

    def _create_watch_callback(
        self,
        raw_key: bytes,
        event_callback: Callable,
        return_type=None,
        use_queue=False,
    ):
        """Create a watch callback function with common event processing logic"""

        def _watch_callback(response):
            try:
                if isinstance(response, grpc.RpcError):
                    code = response.code()
                    if (
                        code == grpc.StatusCode.UNAVAILABLE
                        or code == grpc.StatusCode.INTERNAL
                        or (
                            code == grpc.StatusCode.UNKNOWN
                            and "invalid auth token" not in response.details()
                        )
                    ):
                        logger.error(f"Watch connection error detected: {response}")
                        self._set_connection_state(
                            ConnectionState.DISCONNECTED, response
                        )
                        return

                # Check if response has events attribute before accessing it
                if not hasattr(response, 'events'):
                    logger.error(
                        f"Response object does not have 'events' attribute: {type(response)}"
                    )
                    if isinstance(response, grpc.RpcError):
                        logger.error(
                            f"Unhandled gRPC error in watch callback: {response}"
                        )
                        self._set_connection_state(
                            ConnectionState.DISCONNECTED, response
                        )
                    return

                for ev in response.events:
                    try:
                        logger.debug(f"Received etcd event: {ev}")
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
                            logger.debug(
                                f"State changed for {key}: {ev_type} -> fired event"
                            )
                            if use_queue:
                                event_data = {
                                    'key': self._demangle_key(ev.key),
                                    'event': event,
                                    'callback': event_callback,
                                }

                                try:
                                    self.event_queue.put_nowait(event_data)
                                except queue.Full:
                                    logger.warning("Event queue full; dropping oldest")
                                    try:
                                        self.event_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                    self.event_queue.put_nowait(event_data)
                            else:
                                event_callback(self._demangle_key(ev.key), event)

                            self._last_successful_operation = time.time()
                        else:
                            logger.debug(
                                f"State unchanged for {key}: {ev_type} -> skipped event"
                            )
                    except Exception as e:
                        logger.error(f"Error processing watch event: {e}")

            except Exception as e:
                logger.error(f"Error in watch callback: {e}")
                if isinstance(e, (grpc.RpcError, ConnectionError)):
                    error_msg = str(e).lower()
                    if any(
                        keyword in error_msg
                        for keyword in ['unavailable', 'socket closed', 'connection']
                    ):
                        logger.warning(
                            f"Connection error detected in watch callback: {e}"
                        )
                        self._set_connection_state(ConnectionState.DISCONNECTED, e)

        return _watch_callback

    def _watch_internal(
        self, raw_key: bytes, event_callback: Callable, prefix: bool = False, **kwargs
    ) -> int:
        """Internal watch method that handles the actual etcd watch creation"""
        logger.info(f"Creating internal watch for key: {raw_key}, prefix: {prefix}")

        return_type = kwargs.get('return_type')
        callback = self._create_watch_callback(
            raw_key=raw_key,
            event_callback=event_callback,
            return_type=return_type,
            use_queue=False,
        )

        try:
            if prefix:
                watch_id = self.client.add_watch_prefix_callback(
                    raw_key, callback, **kwargs
                )
            else:
                watch_id = self.client.add_watch_callback(raw_key, callback, **kwargs)

            logger.info(
                f"Internal watch created with ID: {watch_id} for key: {raw_key}"
            )
            if watch_id is None:
                logger.error(
                    f"Invalid watch_id received: {watch_id} for key: {raw_key}"
                )
                return None

            return watch_id
        except Exception as ex:
            logger.error(f"Failed to create internal watch for key {raw_key}: {ex}")
            raise RuntimeError(
                f"Failed to create internal watch for key: {raw_key}"
            ) from ex

    def _watch(
        self, raw_key: bytes, event_callback: Callable, prefix: bool = False, **kwargs
    ) -> int:
        """Enhanced watch method with monitoring support"""
        logger.info(f"Creating watch for key: {raw_key}, prefix: {prefix}")

        # return_type = kwargs.get('return_type')
        return_type = 'dict'
        callback = self._create_watch_callback(
            raw_key=raw_key,
            event_callback=event_callback,
            return_type=return_type,
            use_queue=True,
        )

        try:
            if prefix:
                watch_id = self.client.add_watch_prefix_callback(
                    raw_key, callback, **kwargs
                )
            else:
                watch_id = self.client.add_watch_callback(raw_key, callback, **kwargs)

            logger.info(f"Etcd client returned watch_id: {watch_id} for key: {raw_key}")
            if watch_id is None:
                logger.error(
                    f"Invalid watch_id received: {watch_id} for key: {raw_key}"
                )
                return None

            with self._state_lock:
                self._active_watches.add(watch_id)

            logger.info(f"Watch established with ID: {watch_id} for key: {raw_key}")
            return watch_id

        except Exception as ex:
            logger.error(f"Failed to create watch for key {raw_key}: {ex}")
            raise RuntimeError(f"Failed to create watch for key: {raw_key}") from ex

    @reconn_reauth_adaptor
    def cancel_watch(self, watch_id):
        """Cancel a watch."""
        try:
            with self._state_lock:
                if watch_id in self._active_watches:
                    self.client.cancel_watch(watch_id)
                    self._active_watches.discard(watch_id)
                    logger.info(f"Cancelled watch {watch_id}")
                    return True
                else:
                    logger.warning(f"Watch {watch_id} not found in active watches")
                    return False
        except Exception as e:
            logger.error(f"Failed to cancel watch {watch_id}: {e}")
            return False

    def _start_event_processor(self):
        """Start single event processor thread"""
        if (
            self._processor_running
            and self._processor_thread
            and self._processor_thread.is_alive()
        ):
            return

        self._processor_running = True
        self._processor_ready.clear()

        self._processor_thread = threading.Thread(
            target=self._process_events, daemon=True, name="etcd-event-processor"
        )
        self._processor_thread.start()
        logger.info("Event processor thread started")

    def _process_events(self):
        """Process events from the queue"""
        try:
            self._processor_ready.set()
            logger.info("Event processor is ready")

            while self._processor_running:
                try:
                    event_data = self.event_queue.get(timeout=1)

                    key = event_data['key']
                    event = event_data['event']
                    callback = event_data['callback']

                    try:
                        queue_size = self.event_queue.qsize()
                        logger.debug(
                            f"Event queue size: {queue_size} : {datetime.now()}"
                        )
                        callback(key, event)
                    except Exception as e:
                        logger.error(f"Callback error for key {key}: {e}")

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Event processor error: {e}")

        except Exception as e:
            logger.error(f"Critical error in event processor: {e}")
            raise
        finally:
            logger.info("Event processor stopped")
            self._processor_running = False

    def get_watch_stats(self) -> Dict:
        """Get simple watch statistics."""
        with self._state_lock:
            return {
                "active_watches": len(self._active_watches),
                "watch_ids": list(self._active_watches),
                "event_queue_size": self.event_queue.qsize(),
            }

    @reconn_reauth_adaptor
    def watch(self, key: str, callback, **kwargs):
        logger.info(f"Adding watch callback for: {key}")
        scope_prefix = ""
        mangled_key = self._mangle_key(f"{_slash(scope_prefix)}{key}")
        watch_id = self._watch(mangled_key, callback, **kwargs)
        if watch_id is not None:
            logger.info(f"Successfully created prefix watch {watch_id} for: {key}")
            return watch_id
        else:
            logger.error(
                f"Failed to create prefix watch for: {key} - got None watch_id"
            )
            raise RuntimeError(f"Failed to create prefix watch for: {key}")

    @reconn_reauth_adaptor
    def add_watch_prefix_callback(self, key_prefix: str, callback: Callable, **kwargs):
        """Watch all keys with a given prefix"""
        logger.info(f"Adding watch prefix callback for: {key_prefix}")

        scope_prefix = ""
        mangled_key = self._mangle_key(f"{_slash(scope_prefix)}{key_prefix}")
        watch_id = self._watch(mangled_key, callback, prefix=True, **kwargs)
        if watch_id is not None:
            logger.info(
                f"Successfully created prefix watch {watch_id} for: {key_prefix}"
            )
            return watch_id
        else:
            logger.error(
                f"Failed to create prefix watch for: {key_prefix} - got None watch_id"
            )
            raise RuntimeError(f"Failed to create prefix watch for: {key_prefix}")

    @reconn_reauth_adaptor
    def get(
        self, key: str, metadata: bool = False, serializable: bool = True
    ) -> tuple | str | None:
        """
        Get a single key from the etcd.
        Returns ``None`` if the key does not exist.
        The returned value may be an empty string if the value is a zero-length string.

        - when metadata=True -> return (value_bytes, meta)
        - when metadata=False -> return decoded str (legacy behavior)

        :param key: The key. This must be quoted by the caller as needed.
        :param metadata: If True, return (value_bytes, meta)
        :param serializable: Force linearizable reads  etcd3-py: serializable=True allows stale reads; set it to False.
        :return:
        """

        mangled_key = self._mangle_key(key)
        value, meta = self.client.get(mangled_key, serializable=serializable)
        if metadata:
            return value, meta  # raw bytes + meta (or (None, meta))
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
    def get_prefix_dict(
        self, key_prefix: str, sort_order=None, sort_target="key"
    ) -> dict:
        """
        Returns a nested dict under key_prefix (your current get_prefix behavior).
        Keeps canonical get_prefix free for etcd3-style (value, meta) iterators if needed elsewhere.
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
    def transaction(self, compare, success, failure):
        return self.client.transaction(compare, success, failure)

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

    def replace(self, key: str, initial_val: str, new_val: str):
        scope_prefix = ""
        mangled_key = self._mangle_key(f"{_slash('')}{key}")
        return self.client.replace(
            mangled_key,
            str(initial_val).encode(self.encoding),
            str(new_val).encode(self.encoding),
        )

    @reconn_reauth_adaptor
    def reconnect(self):
        """
        Reconnect to etcd with proper state management.
        This method is called when explicit reconnection is needed.
        """
        try:
            self._set_connection_state(ConnectionState.RECONNECTING)
            logger.debug("Closing existing etcd client...")
            if self.client:
                try:
                    self.client.close()
                except:
                    pass

            if not self._is_multi_endpoint:
                self._client_idx = 0

            connected = self.connect()
            if not connected:
                self._set_connection_state(ConnectionState.FAILED)
                return False

            # Quick connectivity test
            try:
                self.get("__reconnect_test__")
            except Exception as test_error:
                logger.debug(
                    f"Reconnection test query failed (non-fatal): {test_error}"
                )

            self._set_connection_state(ConnectionState.CONNECTED)
            self._last_successful_operation = time.time()

            return True

        except Exception as e:
            # Don't log here - connect() already logged the error
            self._set_connection_state(ConnectionState.FAILED, e)
            raise

    def connect(self) -> bool:
        """Connect to etcd using the appropriate client type with connection state management."""
        times = 0
        last_ex = None
        # Use fewer retries for reconnection (3) vs initial connection
        max_retries = (
            min(self.retry_times, 3)
            if self._reconnect_attempts > 0
            else self.retry_times
        )

        # Set initial state to reconnecting (connecting for the first time)
        self._set_connection_state(ConnectionState.RECONNECTING)

        while times < max_retries:
            try:
                self.client = self._create_client()

                if self._is_multi_endpoint:
                    logger.info(
                        f'Connected to etcd using multi-endpoint client: {self._endpoints}'
                    )
                    self._cluster = [self.client]
                else:
                    host, port = parse_netloc(self._endpoints[0].netloc)
                    logger.info(
                        f'Connected to etcd using single-endpoint client: {host}:{port}'
                    )
                    self._cluster = (
                        [self.client]
                        if self._is_multi_endpoint
                        else [m._etcd_client for m in self.client.members]
                    )
                    if not self._cluster:
                        # Fallback to the client itself for calls/retries
                        self._cluster = [self.client]

                self._set_connection_state(ConnectionState.CONNECTED)
                self._last_successful_operation = time.time()

                break

            except grpc.RpcError as e:
                times += 1
                last_ex = e
                if times == 1:
                    self._set_connection_state(ConnectionState.DISCONNECTED, e)

                if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNKNOWN):
                    # Exponential backoff: 2s, 4s, 8s, capped at 15s
                    backoff = min(2 * (2 ** (times - 1)), 15)
                    if times < max_retries:
                        logger.warning(
                            f"etcd connection attempt {times}/{max_retries} failed, retrying in {backoff}s"
                        )
                        self._set_connection_state(ConnectionState.RECONNECTING, e)
                    time.sleep(backoff)
                    continue
                else:
                    self._set_connection_state(ConnectionState.FAILED, e)
                    raise e

            except Exception as e:
                times += 1
                last_ex = e

                if times == 1:
                    self._set_connection_state(ConnectionState.DISCONNECTED, e)

                # Exponential backoff: 2s, 4s, 8s, capped at 15s
                backoff = min(2 * (2 ** (times - 1)), 15)
                if times < max_retries:
                    logger.warning(
                        f"etcd connection attempt {times}/{max_retries} failed, retrying in {backoff}s"
                    )
                    self._set_connection_state(ConnectionState.RECONNECTING, e)
                time.sleep(backoff)

        if times >= max_retries:
            self._set_connection_state(ConnectionState.FAILED, last_ex)
            raise RuntimeFailToStart(
                f"Initialize etcd client failed after {max_retries} attempts. Due to {last_ex}"
            )

        logger.info(f'Connected to etcd with namespace "{self.ns}"')
        return True

    def close(self):
        """Close the client"""
        try:
            logger.info("Closing EtcdClient...")
            self._shutting_down = True

            with self._state_lock:
                self._connection_state = ConnectionState.DISCONNECTED
                self._processor_running = False
                self._connection_monitor_running = False
                self._processor_ready.clear()
                self._monitor_ready.clear()
                self._monitor_stop.set()

                total_handlers = sum(
                    len(handlers)
                    for handlers in self._connection_event_handlers.values()
                )
                if total_handlers > 0:
                    logger.debug(f"Clearing {total_handlers} connection event handlers")

                for state in list(self._connection_event_handlers.keys()):
                    self._connection_event_handlers[state].clear()

            if self._reconnect_timer and self._reconnect_timer.is_alive():
                self._reconnect_timer.cancel()

            for thread in [self._processor_thread, self._connection_monitor_thread]:
                if thread and thread.is_alive():
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        logger.error(f"Thread {thread.name} failed to terminate")

            with self._state_lock:
                for watch_id in list(self._active_watches):
                    try:
                        self.client.cancel_watch(watch_id)
                    except:
                        pass

                self._active_watches.clear()

            try:
                while not self.event_queue.empty():
                    try:
                        self.event_queue.get_nowait()
                    except:
                        break
            except Exception as e:
                logger.warning(f"Error clearing event queue: {e}")

            if self.client:
                try:
                    self.client.close()
                except:
                    pass
        except Exception as e:
            logger.error(f"Error during EtcdClient close: {e}")
            self._connection_state = ConnectionState.DISCONNECTED
            raise
