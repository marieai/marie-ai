import threading
import time
import traceback
from typing import Union

from marie.logging_core.predefined import default_logger as logger
from marie.serve.discovery.address import JsonAddress, PlainAddress
from marie.serve.discovery.base import ConnectionState, ServiceRegistry
from marie.serve.discovery.etcd_client import EtcdClient
from marie.serve.discovery.etcd_manager import convert_to_etcd_args, get_etcd_client
from marie.serve.discovery.timeout_utils import OperationTimeoutError, run_with_timeout
from marie.serve.discovery.util import form_service_key
from marie.utils.timing import exponential_backoff

__all__ = ["EtcdServiceRegistry"]


class EtcdServiceRegistry(ServiceRegistry):
    """service registry based on etcd."""

    def __init__(
        self,
        etcd_host: str,
        etcd_port: int,
        etcd_client: EtcdClient = None,
        heartbeat_time=-1,
    ):
        """Initialize etcd service registry.

        :param heartbeat_time:
        :param etcd_host: (optional) etcd node host for :class:`client.EtcdClient`.
        :param etcd_port: (optional) etcd node port for :class:`client.EtcdClient`.
        :param etcd_client: (optional) A :class:`client.EtcdClient` object.
        :param heartbeat_time: (optional) service registry heartbeat time interval, default -1. If -1, no heartbeat.

        """
        if etcd_host is None and etcd_client is None:
            raise ValueError("etcd_host or etcd_client must be provided.")

        if etcd_client and (etcd_host or etcd_port):
            logger.warning(
                "Both etcd_client and etcd_host/etcd_port are provided. Using etcd_client."
            )
        if etcd_client:
            if not isinstance(etcd_client, EtcdClient):
                raise TypeError("etcd_client must be an instance of EtcdClient.")
            self._etcd_client = etcd_client
        else:
            args_dict = {
                "discovery_host": etcd_host,
                "discovery_port": etcd_port,
            }
            etcd_args = convert_to_etcd_args(args_dict)
            self._etcd_client = get_etcd_client(etcd_args)

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

        self._lock = threading.RLock()
        self._heartbeat_thread = None
        self._shutdown_event = threading.Event()

        self._leases = {}
        self._services = {}
        self._heartbeat_time = heartbeat_time
        self._default_service_ttl = 6  # Match container.py ETCD_LEASE_SEC default
        self.setup_heartbeat_async()

    def _on_etcd_connected(self, event):
        """Handle etcd connection established - re-register all services."""
        logger.info("Etcd connected - re-registering all services")

        with self._lock:
            old_leases = dict(self._leases)  # snapshot
            self._leases.clear()

            services_to_reregister = []
            for service_addr, service_names in self._services.items():
                old_lease = old_leases.get(service_addr)
                ttl = None
                if old_lease is not None:
                    try:
                        ttl = old_lease.remaining_ttl
                    except AttributeError:
                        try:
                            ttl = old_lease.ttl
                        except AttributeError:
                            ttl = None
                if not ttl:
                    ttl = self._default_service_ttl
                if service_names:
                    services_to_reregister.append(
                        (list(service_names), service_addr, ttl)
                    )

        for service_names, service_addr, ttl in services_to_reregister:
            try:
                self.register(service_names, service_addr, ttl)
                logger.info(f"Re-registered services {service_names} at {service_addr}")
            except Exception as e:
                logger.error(
                    f"Failed to re-register services {service_names} at {service_addr}: {e}",
                    exc_info=True,
                )

    def _on_etcd_disconnected(self, event):
        logger.warning("Etcd disconnected - service registrations may be invalid")

    def _on_etcd_reconnecting(self, event):
        logger.info("Etcd reconnecting...")

    def _on_etcd_failed(self, event):
        try:
            err = event.error
        except Exception:
            err = "unknown"
        logger.error(f"Etcd connection failed: {err}")

    def _lease_ttl(self, lease, default=None) -> int:
        """Return TTL from lease or default (no hasattr/getattr)."""
        if lease is None:
            return default
        try:
            return lease.remaining_ttl
        except AttributeError:
            pass
        try:
            return lease.ttl
        except AttributeError:
            pass
        return default

    def get_lease(self, service_addr, service_ttl):
        """Get or (re)create a service lease from etcd."""
        with self._lock:
            lease = self._leases.get(service_addr)

            if lease is not None:
                try:
                    ttl = self._lease_ttl(lease)
                    if ttl and ttl > 0:
                        return lease
                except Exception:
                    # bad lease; drop it
                    try:
                        del self._leases[service_addr]
                    except Exception:
                        pass
                    lease = None

            # Let etcd assign lease id to avoid collisions
            lease = self._etcd_client.lease(service_ttl)
            self._leases[service_addr] = lease
            return lease

    def register(
        self,
        service_names: Union[str, list[str]],
        service_addr: str,
        service_ttl: int,
        addr_cls=None,
        metadata=None,
    ):
        """Register services with the same address. Returns the lease object."""
        with self._lock:
            if isinstance(service_names, str):
                service_names = [service_names]
            lease = self.get_lease(service_addr, service_ttl)
            addr_cls = addr_cls or PlainAddress

            for service_name in service_names:
                key = form_service_key(service_name, service_addr)
                resolved = self._etcd_client.get(key)
                if resolved:
                    logger.info(
                        f"Service already registered : {service_name}@{service_addr}"
                    )
                    # still ensure we track it locally
                    try:
                        self._services[service_addr].add(service_name)
                    except KeyError:
                        self._services[service_addr] = {service_name}
                    continue

                if addr_cls == JsonAddress:
                    addr_obj = addr_cls(service_addr, metadata=metadata)
                else:
                    addr_obj = addr_cls(service_addr)

                addr_val = addr_obj.add_value()
                put_key, _ = self._etcd_client.put(key, addr_val, lease=lease)
                logger.info(
                    f"Registering service : {service_name}@{service_addr} : {put_key}"
                )

                try:
                    self._services[service_addr].add(service_name)
                except KeyError:
                    self._services[service_addr] = {service_name}

        return lease

    def heartbeat(self, service_addr=None, service_ttl=None):
        """Service heartbeat with connection state awareness."""
        if service_ttl is None:
            service_ttl = self._default_service_ttl
        logger.debug(f"Heartbeat service_addr : {service_addr}")

        state = self._etcd_client.get_connection_state()
        if state in (
            ConnectionState.DISCONNECTED,
            ConnectionState.RECONNECTING,
            ConnectionState.FAILED,
        ):
            logger.debug(
                f"Skipping heartbeat - connection state: {state.name if state else 'Unknown'}"
            )
            return

        with self._lock:
            if service_addr:
                lease = self.get_lease(service_addr, service_ttl)
                leases = ((service_addr, lease),)
                registered_services = {
                    service_addr: self._services.get(service_addr, set())
                }
            else:
                leases = tuple(self._leases.items())
                registered_services = dict(self._services)

        for svc_addr, lease in leases:
            registered = registered_services.get(svc_addr)
            if not registered:
                continue
            try:
                ttl_before = self._lease_ttl(lease)
                logger.debug(f"Refreshing lease for: {svc_addr}, TTL: {ttl_before}")

                # Wrap lease.refresh() with timeout to prevent blocking
                try:
                    refresh_result = run_with_timeout(
                        lambda: lease.refresh(),
                        timeout=3.0,  # 3 second timeout for lease refresh
                        operation_name=f"lease_refresh_{svc_addr}",
                    )
                except OperationTimeoutError:
                    logger.warning(
                        f"Lease refresh timed out for {svc_addr}, will retry next heartbeat"
                    )
                    continue  # Skip to next lease instead of blocking

                # Normalize return (could be (resp,) or object)
                try:
                    resp = refresh_result[0]
                except (TypeError, IndexError):
                    resp = refresh_result

                ttl = None
                try:
                    ttl = resp.TTL
                except AttributeError:
                    try:
                        ttl = resp.ttl
                    except AttributeError:
                        ttl = None

                if ttl is not None and int(ttl) == 0:
                    logger.warning(
                        f"Lease expired (TTL=0) for {svc_addr}, re-registering services"
                    )
                    fallback_ttl = self._lease_ttl(lease, service_ttl) or service_ttl
                    self.register(list(registered), svc_addr, fallback_ttl)

            except Exception as e:
                logger.error(
                    f"Error during heartbeat for {svc_addr}: {e}", exc_info=True
                )

    def unregister(self, service_names, service_addr, addr_cls=None):
        """Unregister services with the same address."""
        addr_cls = addr_cls or PlainAddress
        etcd_delete = addr_cls == PlainAddress

        with self._lock:
            registered_services = self._services.get(service_addr, set())

            for service_name in service_names:
                logger.info(f"Unregistering service : {service_name}@{service_addr}")
                key = form_service_key(service_name, service_addr)
                try:
                    if etcd_delete:
                        self._etcd_client.delete(key)
                    else:
                        # For JsonAddress etc: write a delete marker/value at the same key
                        if addr_cls == PlainAddress:
                            raise ValueError(
                                'addr_cls must be JsonAddress not PlainAddress'
                            )

                        value = addr_cls(service_addr).delete_value()
                        self._etcd_client.put(key, value)
                except Exception as e:
                    logger.error(
                        f"Error unregistering {service_name}@{service_addr}: {e}",
                        exc_info=True,
                    )
                registered_services.discard(service_name)

            if registered_services:
                self._services[service_addr] = registered_services
            else:
                self._services.pop(service_addr, None)

    def setup_heartbeat_async(self):
        """
        Set up an asynchronous heartbeat process with connection state monitoring.
        :return: None
        """
        if self._heartbeat_time <= 0:
            return

        def _heartbeat_setup():
            logger.debug(
                f"Starting heartbeat thread, interval: {self._heartbeat_time}s"
            )

            failures = 0
            max_failures = 5
            initial_backoff = 2
            max_backoff = 30

            time.sleep(self._heartbeat_time)

            while not self._shutdown_event.is_set():
                try:
                    current_state = self._etcd_client.get_connection_state()
                    if (
                        current_state == ConnectionState.CONNECTED
                        or current_state is None
                    ):
                        self.heartbeat()
                        failures = 0
                    else:
                        # Skip heartbeat but don't treat as failure
                        logger.debug(
                            f"Skipping heartbeat - connection state: {current_state.name if current_state else 'Unknown'}"
                        )

                except Exception as e:
                    failures += 1
                    current_state = self._etcd_client.get_connection_state()
                    state_info = (
                        f" (connection: {current_state.name})" if current_state else ""
                    )

                    logger.error(
                        f"Error in heartbeat attempt {failures}/{max_failures}{state_info}: {e}"
                    )
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    backoff_time = exponential_backoff(
                        failures, initial_backoff, max_backoff
                    )

                    # Don't count failures during known disconnections
                    if current_state in [
                        ConnectionState.DISCONNECTED,
                        ConnectionState.RECONNECTING,
                        ConnectionState.FAILED,
                    ]:
                        failures = min(failures, max_failures - 1)
                        backoff_time = self._heartbeat_time
                        logger.debug("Not counting failure - connection is down")
                    else:
                        logger.warning(
                            f"Retrying heartbeat in {backoff_time:.2f} seconds"
                        )

                    if failures >= max_failures:
                        logger.error(
                            f"Max failures reached ({max_failures}). Heartbeat process will stop."
                        )
                        break

                    if not self._shutdown_event.wait(backoff_time):
                        continue
                    break

                if not self._shutdown_event.wait(self._heartbeat_time):
                    continue
                break

        self._heartbeat_thread = threading.Thread(
            target=_heartbeat_setup, daemon=True, name="etcd-registry-heartbeat"
        )
        self._heartbeat_thread.start()

    def shutdown(self):
        self._shutdown_event.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            logger.debug("Waiting for heartbeat thread to stop...")
            self._heartbeat_thread.join(timeout=self._heartbeat_time + 1)
            if self._heartbeat_thread.is_alive():
                logger.warning("Heartbeat thread did not stop gracefully")
