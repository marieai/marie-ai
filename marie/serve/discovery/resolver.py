import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

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
    callback: Callable
    notify_on_start: bool = True
    max_retries: int = 3
    last_revision: Optional[int] = None
    error_count: int = 0
    last_successful_event: float = 0
    re_establishment_count: int = 0
    is_active: bool = True

    def reset_for_reestablishment(self):
        """Reset state for watch re-establishment"""
        self.watch_id = None
        self.error_count = 0
        self.re_establishment_count += 1
        self.last_successful_event = time.time()

    def mark_successful(self):
        """Mark watch as having successful event"""
        self.error_count = 0
        self.last_successful_event = time.time()

    def mark_error(self):
        """Mark watch as having an error"""
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
        self._listen_thread = None
        self._listen_timeout = listen_timeout
        self._lock = threading.Lock()

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
        self._names = {}
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
        watches_to_reestablish = []

        with self._lock:
            active_watches = [ws for ws in self._watch_states.values() if ws.is_active]
            logger.info(
                f"Etcd connected - re-establishing {len(active_watches)} watches"
            )

            for watch_state in active_watches:
                try:
                    logger.info(
                        f"Re-establishing watch for service: {watch_state.service_name} "
                        f"(attempt #{watch_state.re_establishment_count + 1})"
                    )

                    watch_state.reset_for_reestablishment()
                    watch_id = self._etcd_client.add_watch_prefix_callback(
                        watch_state.service_name, watch_state.callback
                    )

                    if watch_id is not None:
                        watch_state.watch_id = watch_id
                        watch_state.mark_successful()
                        logger.info(
                            f"Successfully re-established watch {watch_id} for service: "
                            f"{watch_state.service_name} (re-establishment #{watch_state.re_establishment_count})"
                        )

                        if watch_state.notify_on_start:
                            watches_to_reestablish.append(
                                (watch_state.service_name, watch_state.callback)
                            )
                    else:
                        watch_state.mark_error()
                        logger.error(
                            f"Failed to re-establish watch for service: {watch_state.service_name}"
                        )

                except Exception as e:
                    watch_state.mark_error()
                    logger.error(
                        f"Failed to re-establish watch for {watch_state.service_name}: {e}"
                    )

        for service_name, callback in watches_to_reestablish:
            try:
                self._fire_initial_events(service_name, callback)
            except Exception as e:
                logger.error(
                    f"Failed to fire initial events for {service_name} during reconnection: {e}"
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

            for watch_state in self._watch_states.values():
                if watch_state.is_active:
                    watch_state.watch_id = None

    def _on_etcd_reconnecting(self, event):
        """Handle etcd reconnection attempt."""
        logger.info("Etcd reconnecting...")

    def _on_etcd_failed(self, event):
        """Handle etcd connection failure."""
        logger.error(f"Etcd connection failed: {event.error}")
        with self._lock:
            for watch_state in self._watch_states.values():
                if watch_state.is_active:
                    watch_state.watch_id = None
                    watch_state.mark_error()

    def resolve(self, name: str) -> list:
        """Resolve  service name."""
        with self._lock:
            try:
                return list(self._names[name])
            except KeyError:
                pass  # Fall through to get from etcd

        addrs = self.get(name)
        with self._lock:
            if name not in self._names:
                self._names[name] = addrs
            return list(self._names[name])

    def get(self, name: str):
        """Get values from Etcd.

        :param name: Etcd key prefix name.
        :rtype list: A collection of Etcd values.

        """
        keys = self._etcd_client.get_prefix(name)
        vals = []
        plain = True
        if self._addr_cls != PlainAddress:
            plain = False

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
                try:
                    self._names[name].extend(add)
                except KeyError:
                    self._names[name] = add

                for del_item in delete:
                    try:
                        self._names[name].remove(del_item)
                    except ValueError:
                        continue

    def listen(self):
        """Listen for change about service address."""
        while not self._stopped:
            with self._lock:
                names_to_check = list(self._names.keys())

            for name in names_to_check:
                try:
                    vals = self.get(name)
                except Exception as e:
                    logger.error(f"Unexpected error getting values for {name}: {e}")
                    continue
                else:
                    with self._lock:
                        if name in self._names:
                            self._names[name] = vals

            time.sleep(self._listen_timeout)

    def watch_service(
        self,
        service_name: str,
        event_callback: callable,
        notify_on_start=True,
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
                watch_id = self._etcd_client.add_watch_prefix_callback(
                    service_name, event_callback
                )

                if watch_id is not None:
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
                    f"Exception while creating watch for service {service_name}: {e}"
                )

        if notify_on_start and watch_state.watch_id is not None:
            self._fire_initial_events(service_name, event_callback)

    def _fire_initial_events(self, service_name: str, event_callback: callable):
        """Fire initial events for current state of the service."""
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
                except Exception as e:
                    logger.error(f"Error in initial event callback for {svc_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to fire initial events for {service_name}: {e}")

    def stop_watch_service(self, service_name: str = None) -> None:
        """Stop watching services."""
        with self._lock:
            if service_name:
                watch_state = self._watch_states.get(service_name)
                if watch_state:
                    if watch_state.watch_id is not None:
                        self._etcd_client.cancel_watch(watch_state.watch_id)
                        logger.info(
                            f"Stop watching service: {service_name}, {watch_state.watch_id}"
                        )

                    watch_state.is_active = False
                    del self._watch_states[service_name]
            else:
                for service_name, watch_state in list(self._watch_states.items()):
                    if watch_state.watch_id is not None:
                        self._etcd_client.cancel_watch(watch_state.watch_id)
                        logger.info(
                            f"Stop watching service: {service_name}, {watch_state.watch_id}"
                        )
                    watch_state.is_active = False
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
            self._listen_thread = threading.Thread(target=self.listen, name=thread_name)
            self._listen_thread.daemon = daemon
            self._listen_thread.start()
            self._listening = True
            logger.debug(f"Started listener thread: {thread_name}")
        except Exception as e:
            logger.error(f"Failed to start listener thread: {e}")
            self._listening = False
            raise

    def stop(self):
        """Stop service resolver."""
        if self._stopped:
            return
        logger.info("Stopping EtcdServiceResolver...")

        self._stopped = True
        self._listening = False

        if self._listen_thread and self._listen_thread.is_alive():
            logger.debug("Waiting for listen thread to stop...")
            self._listen_thread.join(timeout=self._listen_timeout + 1)
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
            logger.warning(f"Error removing connection event handlers: {e}")

    def __del__(self):
        self.stop()
