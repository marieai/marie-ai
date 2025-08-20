import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

from marie.logging_core.predefined import default_logger as logger
from marie.serve.discovery.address import PlainAddress
from marie.serve.discovery.base import ConnectionState, ServiceResolver
from marie.serve.discovery.etcd_client import EtcdClient, Event
from marie.serve.discovery.etcd_manager import convert_to_etcd_args, get_etcd_client
from marie.serve.discovery.util import form_service_key

__all__ = ["EtcdServiceResolver"]


@dataclass
class WatchState:
    """State information for a service watch"""

    service_name: str
    watch_id: Optional[int]
    callback: Callable[[str, Event], None]
    notify_on_start: bool = True
    max_retries: int = 3
    last_revision: Optional[int] = None
    error_count: int = 0
    last_successful_event: float = 0
    re_establishment_count: int = 0
    is_active: bool = True

    def __post_init__(self):
        self._state_lock = threading.Lock()

    def reset_for_reestablishment(self):
        """Reset state for watch re-establishment"""
        with self._state_lock:
            self.watch_id = None
            self.error_count = 0
            self.re_establishment_count += 1
            self.last_successful_event = time.time()

    def mark_successful(self):
        """Mark watch as having successful event"""
        with self._state_lock:
            self.error_count = 0
            self.last_successful_event = time.time()

    def mark_error(self):
        """Mark watch as having an error"""
        with self._state_lock:
            self.error_count += 1


class EtcdServiceResolver(ServiceResolver):
    """Service resolver based on ETCD service."""

    def __init__(
        self,
        etcd_host=None,
        etcd_port=None,
        etcd_client=None,
        start_listener=True,
        listen_timeout=5,
        addr_cls=None,
        namespace="marie",
    ):
        """Initialize etcd service resolver.

        :param etcd_host: (optional) etcd node host for :class:`client.EtcdClient`.
        :param etcd_port: (optional) etcd node port for :class:`client.EtcdClient`.
        :param etcd_client: (optional) A :class:`client.EtcdClient` object.
        :param start_listener: (optional) Indicate whether starting the resolver listen thread.
        :param listen_timeout: (optional) Resolver thread listen timeout.
        :param addr_cls: (optional) address format class.
        :param namespace: (optional) Etcd namespace.
        """
        super().__init__()

        if etcd_host is None and etcd_client is None:
            raise ValueError("etcd_host or etcd_client must be provided.")
        self._listening = False
        self._stopped = False
        self._listen_thread: Optional[threading.Thread] = None
        self._listen_timeout = listen_timeout
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        if etcd_client:
            if not isinstance(etcd_client, EtcdClient):
                raise TypeError("etcd_client must be an instance of EtcdClient.")
            self._etcd_client = etcd_client
        else:
            args_dict = {
                "discovery_host": etcd_host,
                "discovery_port": etcd_port,
                "namespace": namespace,
            }
            etcd_args = convert_to_etcd_args(args_dict)
            self._etcd_client = get_etcd_client(etcd_args)

        self._watch_states: Dict[str, WatchState] = {}
        self._names: Dict[str, List] = {}
        self._addr_cls = addr_cls or PlainAddress

        self._etcd_client.add_connection_event_handler(
            ConnectionState.CONNECTED, self._on_etcd_connected
        )
        self._etcd_client.add_connection_event_handler(
            ConnectionState.DISCONNECTED, self._on_etcd_disconnected
        )
        self._etcd_client.add_connection_event_handler(
            ConnectionState.RECONNECTING, self._on_etcd_reconnecting
        )
        self._etcd_client.add_connection_event_handler(
            ConnectionState.FAILED, self._on_etcd_failed
        )

        if start_listener:
            self.start_listener()

    def _on_etcd_connected(self, event):
        """Handle etcd connection established - re-establish all watches."""
        import random

        with self._lock:
            active_watches = [ws for ws in self._watch_states.values() if ws.is_active]
        logger.info(f"Etcd connected - re-establishing {len(active_watches)} watches")

        watches_to_reconcile: List[tuple[str, Callable[[str, Event], None]]] = []

        for watch_state in active_watches:
            try:
                time.sleep(random.uniform(0.05, 0.25))  # jitter

                logger.info(
                    f"Re-establishing watch for service: {watch_state.service_name} "
                    f"(attempt #{watch_state.re_establishment_count + 1})"
                )

                watch_state.reset_for_reestablishment()
                wrapped_cb = self._make_safe_callback(
                    watch_state.service_name, watch_state
                )

                watch_id = self._add_watch_with_options(watch_state, wrapped_cb)
                if watch_id is not None:
                    with watch_state._state_lock:
                        watch_state.watch_id = watch_id
                    watch_state.mark_successful()
                    logger.info(
                        f"Successfully re-established watch {watch_id} for "
                        f"{watch_state.service_name} "
                        f"(re-establishment #{watch_state.re_establishment_count})"
                    )
                    # Reconcile to current state on reconnect
                    watches_to_reconcile.append((watch_state.service_name, wrapped_cb))
                else:
                    watch_state.mark_error()
                    logger.error(
                        f"Failed to re-establish watch for service: {watch_state.service_name}"
                    )

            except Exception as e:
                watch_state.mark_error()
                logger.error(
                    f"Failed to re-establish watch for {watch_state.service_name}: {e}",
                    exc_info=True,
                )

        for service_name, wrapped_cb in watches_to_reconcile:
            try:
                # Fire initial events via the wrapped callback (so health/rev updates apply)
                self._fire_initial_events(service_name, wrapped_cb)
            except Exception as e:
                logger.error(
                    f"Failed to fire initial events for {service_name} during reconnection: {e}",
                    exc_info=True,
                )

    def _on_etcd_disconnected(self, event):
        """Handle etcd disconnection."""
        with self._lock:
            active_watch_count = sum(
                1
                for ws in self._watch_states.values()
                if ws.is_active and ws.watch_id is not None
            )
        logger.warning(
            f"Etcd disconnected - {active_watch_count} watches may be invalid"
        )

        with self._lock:
            for watch_state in self._watch_states.values():
                if watch_state.is_active:
                    with watch_state._state_lock:
                        watch_state.watch_id = None

    def _on_etcd_reconnecting(self, event):
        """Handle etcd reconnection attempt."""
        logger.info("Etcd reconnecting...")

    def _on_etcd_failed(self, event):
        """Handle etcd connection failure."""
        logger.error(f"Etcd connection failed: {getattr(event, 'error', 'unknown')}")
        with self._lock:
            for watch_state in self._watch_states.values():
                if watch_state.is_active:
                    with watch_state._state_lock:
                        watch_state.watch_id = None
                    watch_state.mark_error()

    def resolve(self, name: str) -> list:
        """Resolve service name."""
        with self._lock:
            cached = self._names.get(name)
        if cached is not None:
            return list(cached)

        addrs = self.get(name)
        with self._lock:
            # setdefault to avoid duplicate gets under contention
            self._names.setdefault(name, list(addrs))
            return list(self._names[name])

    def get(self, name: str):
        """Get values from Etcd.

        :param name: Etcd key prefix name.
        :rtype list: A collection of Etcd values.
        """
        keys = self._etcd_client.get_prefix(name)
        vals = []
        plain = self._addr_cls == PlainAddress

        for val, metadata in keys.items():
            if plain:
                vals.append(self._addr_cls.from_value(val))
            else:
                add, addr = self._addr_cls.from_value(val)
                if add:
                    vals.append(addr)

        return vals

    def update(self, **kwargs):
        """Add or delete service address.

        :param kwargs: Dictionary of ``'service_name': ((add-address, delete-address)).``
        """
        with self._lock:
            for name, (add, delete) in kwargs.items():
                current = list(self._names.get(name, []))
                # Use normalized map to dedupe and perform add/remove safely
                norm_map: Dict[str, object] = {
                    self._normalize_addr(v): v for v in current
                }
                for v in add:
                    norm_map[self._normalize_addr(v)] = v
                for v in delete:
                    norm_map.pop(self._normalize_addr(v), None)
                self._names[name] = list(norm_map.values())

    def listen(self):
        """Listen for change about service address."""
        while not self._stopped:
            # Snapshot names to check under lock
            with self._lock:
                names_to_check = list(self._names.keys())

            for name in names_to_check:
                try:
                    # Snapshot current before fetch to reduce oscillations
                    with self._lock:
                        current = list(self._names.get(name, []))
                    vals = self.get(name)
                except Exception as e:
                    logger.error(
                        f"Unexpected error getting values for {name}: {e}",
                        exc_info=True,
                    )
                    continue

                try:
                    # Normalize for diffing
                    norm_new: Set[str] = self._normalize_addr_list(vals)
                    norm_cur: Set[str] = self._normalize_addr_list(current)

                    added_norm = list(norm_new - norm_cur)
                    removed_norm = list(norm_cur - norm_new)

                    added = [v for v in vals if self._normalize_addr(v) in added_norm]
                    removed = [
                        v for v in current if self._normalize_addr(v) in removed_norm
                    ]

                    if added or removed:
                        try:
                            self.update(**{name: (list(added), list(removed))})
                        except Exception as e:
                            logger.error(
                                f"Failed to update cache for {name}: {e}", exc_info=True
                            )
                except Exception as e:
                    logger.error(f"Normalization error for {name}: {e}", exc_info=True)

            # Cooperative wait so stop() can wake us
            self._stop_event.wait(self._listen_timeout)

    def watch_service(
        self,
        service_name: str,
        event_callback: Callable[[str, Event], None],
        notify_on_start: bool = True,
        max_retries: int = 3,
    ):
        """Watch service event."""
        logger.info(f"Watching service : {service_name} for changes.")
        logger.info(f"Notify on start : {notify_on_start}")

        with self._lock:
            watch_state = WatchState(
                service_name=service_name,
                watch_id=None,
                callback=event_callback,
                notify_on_start=notify_on_start,
                max_retries=max_retries,
            )
            self._watch_states[service_name] = watch_state

        try:
            wrapped_cb = self._make_safe_callback(service_name, watch_state)
            watch_id = self._add_watch_with_options(watch_state, wrapped_cb)

            if watch_id is not None:
                with watch_state._state_lock:
                    watch_state.watch_id = watch_id
                watch_state.mark_successful()
                logger.info(
                    f"Successfully created watch {watch_id} for service: {service_name}"
                )
            else:
                watch_state.mark_error()
                logger.error(f"Failed to create watch for service: {service_name}")

        except Exception as e:
            watch_state.mark_error()
            logger.error(
                f"Exception while creating watch for service {service_name}: {e}",
                exc_info=True,
            )

        # Seed local cache so the listener can reconcile
        try:
            initial_vals = self.get(service_name)
            with self._lock:
                self._names[service_name] = list(initial_vals)
        except Exception as e:
            logger.warning(
                f"Failed to seed cache for {service_name}: {e}", exc_info=True
            )

        if notify_on_start:
            # Fire initial events via the wrapped callback, not the raw callback
            try:
                self._fire_initial_events(service_name, wrapped_cb)
            except Exception:
                logger.error(
                    f"Failed to dispatch initial events for {service_name}",
                    exc_info=True,
                )

    def _fire_initial_events(
        self, service_name: str, event_callback: Callable[[str, Event], None]
    ):
        """Fire initial events for current state of the service via a provided callback."""
        try:
            resolved = self._etcd_client.get_prefix(service_name)
            events_to_fire = []

            for val, metadata in resolved.items():
                logger.info(f"Resolved service: {service_name}, {val}, {metadata}")
                key = form_service_key(service_name, val)
                event = Event(key, "put", {key: metadata})
                events_to_fire.append((service_name, event))

            for svc_name, event in events_to_fire:
                try:
                    event_callback(svc_name, event)
                except Exception:
                    # The wrapped callback already logs/updates state
                    pass

        except Exception:
            logger.error(
                f"Failed to fire initial events for {service_name}", exc_info=True
            )

    def stop_watch_service(self, service_name: str = None) -> None:
        """Stop watching services."""
        with self._lock:
            if service_name:
                watch_state = self._watch_states.get(service_name)
                if watch_state:
                    with watch_state._state_lock:
                        wid = watch_state.watch_id
                        watch_state.is_active = False
                        watch_state.watch_id = None
                    if wid is not None:
                        self._etcd_client.cancel_watch(wid)
                        logger.info(f"Stop watching service: {service_name}, {wid}")
                    self._watch_states.pop(service_name, None)
            else:
                for s_name, watch_state in list(self._watch_states.items()):
                    with watch_state._state_lock:
                        wid = watch_state.watch_id
                        watch_state.is_active = False
                        watch_state.watch_id = None
                    if wid is not None:
                        self._etcd_client.cancel_watch(wid)
                        logger.info(f"Stop watching service: {s_name}, {wid}")
                self._watch_states.clear()

    def get_watch_statistics(self) -> Dict:
        """Get watch statistics."""
        with self._lock:
            stats = {
                "total_watches": len(self._watch_states),
                "active_watches": sum(
                    1
                    for ws in self._watch_states.values()
                    if ws.is_active and ws.watch_id is not None
                ),
                "failed_watches": sum(
                    1 for ws in self._watch_states.values() if ws.error_count > 0
                ),
                "watches_detail": {},
            }

            for service_name, watch_state in self._watch_states.items():
                with watch_state._state_lock:
                    stats["watches_detail"][service_name] = {
                        "watch_id": watch_state.watch_id,
                        "is_active": watch_state.is_active,
                        "error_count": watch_state.error_count,
                        "re_establishment_count": watch_state.re_establishment_count,
                        "last_successful_event": watch_state.last_successful_event,
                    }

            return stats

    def start_listener(self, daemon=True):
        """Start listen thread.

        :param daemon: Indicate whether start thread as a daemon.
        """
        if self._listening and self._listen_thread and self._listen_thread.is_alive():
            return

        try:
            thread_name = "etcd-resolver-listener"
            self._stop_event.clear()
            self._listen_thread = threading.Thread(
                target=self.listen, name=thread_name, daemon=daemon
            )
            self._listen_thread.start()
            self._listening = True
            logger.debug(f"Started listener thread: {thread_name}")
        except Exception as e:
            logger.error(f"Failed to start listener thread: {e}", exc_info=True)
            self._listening = False
            raise

    def stop(self):
        """Stop service resolver."""
        if self._stopped:
            return
        logger.info("Stopping EtcdServiceResolver...")

        self._stopped = True
        self._listening = False
        self._stop_event.set()

        if self._listen_thread and self._listen_thread.is_alive():
            logger.debug("Waiting for listen thread to stop...")
            self._listen_thread.join(timeout=self._listen_timeout + 2)
            if self._listen_thread.is_alive():
                logger.warning("Listen thread did not stop gracefully")

        self.stop_watch_service()

        try:
            self._etcd_client.remove_connection_event_handler(
                ConnectionState.CONNECTED, self._on_etcd_connected
            )
            self._etcd_client.remove_connection_event_handler(
                ConnectionState.DISCONNECTED, self._on_etcd_disconnected
            )
            self._etcd_client.remove_connection_event_handler(
                ConnectionState.RECONNECTING, self._on_etcd_reconnecting
            )
            self._etcd_client.remove_connection_event_handler(
                ConnectionState.FAILED, self._on_etcd_failed
            )
        except Exception as e:
            logger.warning(
                f"Error removing connection event handlers: {e}", exc_info=True
            )

    def __del__(self):
        # Be defensive: avoid complex teardown during interpreter shutdown.
        try:
            self.stop()
        except Exception:
            pass

    def _make_safe_callback(
        self, service_name: str, watch_state: WatchState
    ) -> Callable[[str, Event], None]:
        """
        Wrap user callback to:
          - prevent exceptions from killing the watcher,
          - track last_revision (prefer event.kv.mod_revision if available),
          - basic compaction/error recovery triggers,
          - update watch_state health and backoff where appropriate.
        """

        def _wrapped(svc_name: str, event: Event):
            try:
                try:
                    rev = None
                    if hasattr(event, "kv") and hasattr(event.kv, "mod_revision"):
                        rev = int(event.kv.mod_revision)
                    else:
                        # Fallback to metadata dict if provided by proxy
                        meta = (
                            next(iter(event.values.values()))
                            if getattr(event, "values", None)
                            else None
                        )
                        if isinstance(meta, dict):
                            rev = meta.get("mod_revision") or meta.get("MOD_REVISION")
                            if rev is not None:
                                rev = int(rev)

                    if rev is not None:
                        with watch_state._state_lock:
                            # Only move forward
                            if (
                                watch_state.last_revision is None
                                or rev > watch_state.last_revision
                            ):
                                watch_state.last_revision = rev
                            else:
                                # Drop stale/out-of-order events silently
                                return
                except Exception:
                    # best-effort revision handling
                    pass

                # Invoke original user callback
                watch_state.callback(svc_name, event)
                watch_state.mark_successful()

            except Exception as cb_err:
                # Callback failed: record and consider recovery
                watch_state.mark_error()
                logger.error(
                    f"Watch callback error for {service_name}: {cb_err}", exc_info=True
                )
                with watch_state._state_lock:
                    errors = watch_state.error_count
                    max_r = watch_state.max_retries
                if errors >= max_r > 0:
                    # Offload recovery so we don't block etcd watch thread
                    self._schedule_backoff_and_recover(service_name, watch_state)

        return _wrapped

    def _schedule_backoff_and_recover(self, service_name: str, watch_state: WatchState):
        """Run backoff + recovery in a background thread."""

        def _worker():
            import random

            with watch_state._state_lock:
                attempt = max(1, watch_state.re_establishment_count + 1)
            base = min(5 * (1.5 ** (attempt - 1)), 60.0)
            delay = base + random.uniform(0, 0.5)
            logger.warning(
                f"Backoff before recovering watch for {service_name}: {delay:.2f}s"
            )
            try:
                # Early exit if we've been stopped
                if self._stop_event.wait(delay):
                    return
            except Exception:
                pass

            try:
                # Reconcile current state and reset baseline revision
                self._reconcile_and_reset_revision(service_name, watch_state)
                # Re-add watch
                wrapped_cb = self._make_safe_callback(service_name, watch_state)
                watch_id = self._add_watch_with_options(watch_state, wrapped_cb)
                if watch_id is not None:
                    with watch_state._state_lock:
                        watch_state.watch_id = watch_id
                    watch_state.mark_successful()
                    logger.info(
                        f"Recovered watch for {service_name} with id {watch_id}"
                    )
                    # Optional: cap re-establishment count to avoid unbounded backoff
                    with watch_state._state_lock:
                        if watch_state.re_establishment_count > 5:
                            watch_state.re_establishment_count = 5
                else:
                    logger.error(f"Failed to recover watch for {service_name}")
            except Exception as e:
                logger.error(
                    f"Error recovering watch for {service_name}: {e}", exc_info=True
                )

        threading.Thread(
            target=_worker,
            name=f"recover-{service_name}",
            daemon=True,
        ).start()

    def _reconcile_and_reset_revision(self, service_name: str, watch_state: WatchState):
        """
        List current state for the prefix, fire synthetic puts via wrapped callback,
        and set the last_revision baseline to the max mod_revision seen.
        """
        try:
            resolved = self._etcd_client.get_prefix(service_name)
            max_rev = None
            for _, metadata in resolved.items():
                if isinstance(metadata, dict):
                    try:
                        rev = metadata.get("mod_revision") or metadata.get(
                            "MOD_REVISION"
                        )
                        if rev is not None:
                            rev = int(rev)
                            if max_rev is None or rev > max_rev:
                                max_rev = rev
                    except Exception:
                        pass

            # Use wrapped callback for reconciliation so state/health is updated consistently
            wrapped_cb = self._make_safe_callback(service_name, watch_state)
            self._fire_initial_events(service_name, wrapped_cb)

            if max_rev is not None:
                with watch_state._state_lock:
                    if (
                        watch_state.last_revision is None
                        or max_rev > watch_state.last_revision
                    ):
                        watch_state.last_revision = max_rev
        except Exception as e:
            logger.error(
                f"Failed to reconcile {service_name} before rewatch: {e}", exc_info=True
            )

    def _add_watch_with_options(
        self, watch_state: WatchState, callback: Callable[[str, Event], None]
    ) -> Optional[int]:
        """
        Add watch with start_revision/progress_notify when supported by the client.
        Falls back to plain add if EtcdClient doesn't accept those parameters.
        """
        start_rev = None
        with watch_state._state_lock:
            if watch_state.last_revision is not None:
                start_rev = int(watch_state.last_revision) + 1

        try:
            return self._etcd_client.add_watch_prefix_callback(
                watch_state.service_name,
                callback,
                start_revision=start_rev,
                progress_notify=True,
            )
        except TypeError:
            logger.debug(
                "EtcdClient.add_watch_prefix_callback does not support start_revision/progress_notify; falling back."
            )
            return self._etcd_client.add_watch_prefix_callback(
                watch_state.service_name,
                callback,
            )
        except Exception as e:
            logger.error(
                f"Failed to add watch with options for {watch_state.service_name}: {e}",
                exc_info=True,
            )
            return None

    def _normalize_addr(self, v) -> str:
        """Normalize an address/value to a stable string key for diffing."""
        try:
            s = v if isinstance(v, str) else str(v)
        except Exception:
            s = repr(v)
        return s.strip().lower().rstrip("/")

    def _normalize_addr_list(self, values) -> Set[str]:
        return {self._normalize_addr(v) for v in values}
