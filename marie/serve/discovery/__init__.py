from typing import Optional
import threading
import time

import requests

from marie.importer import ImportExtensions


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.next_call = time.time()
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self.next_call += self.interval
            self._timer = threading.Timer(self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


class DiscoveryServiceMixin:
    """Instrumentation mixin for Service Discovery handling"""

    def _setup_service_discovery(
        self,
        name: str,
        host: str,
        port: int,
        discovery: Optional[bool] = False,
        discovery_host: Optional[str] = '0.0.0.0',
        discovery_port: Optional[int] = 8500,
        discovery_scheme: Optional[str] = 'http',
        discovery_watchdog_interval: Optional[int] = 60,
    ) -> None:

        if self.logger is None:
            raise Exception("Expected logger to be configured")

        if discovery:
            with ImportExtensions(
                required=True,
                help_text='You need to install the `consul` to use the service discovery functionality of marie',
            ):
                import consul

                self.logger.info(
                    f"Initializing Service Discovery for : {discovery_scheme}://{discovery_host}:{discovery_port}"
                )

                def _watchdog_target():
                    return self.__start_discovery_watchdog(
                        name=name,
                        service_host=host,
                        service_port=port,
                        discovery_host=discovery_host,
                        discovery_port=discovery_port,
                        discovery_scheme=discovery_scheme,
                        interval=discovery_watchdog_interval,
                    )

                t = threading.Thread(target=_watchdog_target)
                t.daemon = True
                t.start()

    def __start_discovery_watchdog(
        self,
        name,
        service_host,
        service_port,
        discovery_host,
        discovery_port,
        discovery_scheme,
        interval,
    ):
        sid = f"{name}:{service_port}:{service_host}"

        def _register(_service_host, _service_port):
            nonlocal sid
            self.logger.info(
                "watchdog:Host, Port, ServiceId : %s, %s, %s",
                _service_host,
                _service_port,
                sid,
            )
            online = self.__verify_discovery_connection(
                discovery_host, discovery_port, discovery_scheme
            )
            self.logger.info("watchdog:consul online : %s", online)
            service_name = "traefik-system-ingress"

            if online:
                # node = getServiceByNameAndId(service_name, sid)
                # if node is None:
                #     sid = register(
                #         service_host=_service_host,
                #         service_port=_service_port,
                #         service_id=sid,
                #     )
                self.logger.info("watchdog:Re-registered service: %s", sid)

        self.logger.info("watchdog:starting with interval : %s", interval)
        rt = RepeatedTimer(interval, _register, service_host, service_port)

    def __verify_discovery_connection(
        self,
        discovery_host: Optional[str] = '0.0.0.0',
        discovery_port: Optional[int] = 6831,
        discovery_scheme: Optional[str] = 'http',
    ) -> bool:
        """Verify consul connection
        Exceptions throw such as ConnectionError will be captured
        """
        import consul

        self.logger.debug(
            "Verifying Consul connection to %s://%s:%s",
            discovery_scheme,
            discovery_host,
            discovery_port,
        )

        try:
            client = consul.Consul(
                host=discovery_host, port=discovery_port, scheme=discovery_scheme
            )
            client.agent.self()
            return True
        except (requests.exceptions.ConnectionError, ConnectionError) as e:
            pass
        except Exception as e:
            self.logger.warning("Unable to verify connection : {msg}".format(msg=e))

        return False
