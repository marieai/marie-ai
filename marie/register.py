import argparse
import json
import threading
import time

# import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Tuple, Union

import consul
import yaml
from consul.base import Check
from logger import setup_logger

from marie.utils.types import strtobool
from utils.network import find_open_port, get_ip_address

logger = setup_logger(__name__, "registry.log")
config = None
current_service_id = None


class EndpointConfig:
    Port: int
    Host: str
    Scheme: str

    def __str__(self):
        return self.Scheme + "://" + self.Host + ":" + str(self.Port)


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


class DebugWebServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(
            bytes("<html><head><title>Registry info</title></head>", "utf-8")
        )
        self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>Service status.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))


def start_webserver(hostname, server_port):
    server = HTTPServer((hostname, server_port), DebugWebServer)
    logger.info("Server started http://%s:%s" % (hostname, server_port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

    server.server_close()
    logger.info("Server stopped.")


def verify_connection(cfg: EndpointConfig) -> bool:
    """
    Verify consul connection
    Exceptions throw such as ConnectionError will be captured
    """
    if cfg is None:
        raise Exception("Configuration is required but None provided.")
    port = cfg.Port
    host = cfg.Host

    logger.debug("Verifying Consul connection to %s:%s", host, port)

    try:
        client = consul.Consul(host=host, port=port)
        client.agent.self()
        return True
    except Exception as e:
        logger.warning("Unable to verify connection : {msg}".format(msg=e))

    return False


def create_client(
    cfg: EndpointConfig, verify: bool = True
) -> Tuple[Union[consul.Consul, None], bool]:
    """
    Create new consul client
    """
    if cfg is None:
        raise Exception("Configuration is required but got None")

    try:
        port = cfg.Port
        host = cfg.Host
        logger.info("Consul Host: %s Port: %s ", host, port)

        client = consul.Consul(host=host, port=port)
        online = False
        if verify:
            online = verify_connection(cfg)
            logger.debug("Consul online : %s", online)
        return client, online
    except Exception:
        pass

    return None, False


def driver_version():
    return consul.__version__


def getServiceByNameAndId(service_name, service_id):
    c, online = create_client(config, True)
    if not online:
        return None
    try:
        index, nodes = c.health.service(service_name)
        for node in nodes:
            if node["Service"]["ID"] == service_id:
                return node
    except Exception as e:
        pass
    return None


def register(service_host, service_port, service_id=None) -> Union[None, str]:
    """
    Register new service in consul
    """
    logger.info("Registering ServiceHost: %s Port: %s ", service_host, service_port)

    service_name = "traefik-system-ingress"
    service_url = f"http://{service_host}:{service_port}/health/status"

    # TODO : Service ID generation needs to be configurable
    # Create new service id, otherwise we will re-register same id
    if service_id is None:
        # service_id = f'{service_name}@{service_port}#{uuid.uuid4()}'
        host = get_ip_address()
        service_id = f"{service_name}@{host}:{service_port}"

    logger.info("Service url: %s", service_url)
    logger.info("Service id: %s", service_id)

    c, online = create_client(config, True)
    if not online:
        logger.debug("Consul service is offline")
        return service_id

    # TODO: De-registration needs to be configurable

    c.agent.service.register(
        name=service_name,
        service_id=service_id,
        port=service_port,
        address=service_host,
        # check=Check.http(service_url, '10s', deregister='10m'),
        check=Check.http(service_url, "10s"),
        tags=[
            "traefik.enable=true",
            "traefik.consulcatalog.connect=false",
            "traefik.http.routers.traefik-system-ingress.entrypoints=marie",
            "traefik.http.routers.traefik-system-ingress.service=traefik-system-ingress",
            "traefik.http.routers.traefik-system-ingress.rule=HostRegexp(`{host:.+}`)",
            "traefik.http.services.traefik-system-ingress.loadbalancer.server.scheme=http",
        ],
    )

    return service_id


def start_watchdog(interval, service_host, service_port):
    sid = current_service_id

    def _register(_service_host, _service_port):
        nonlocal sid
        logger.info(
            "watchdog:Host, Port, ServiceId : %s, %s, %s",
            _service_host,
            _service_port,
            sid,
        )
        online = verify_connection(config)
        logger.info("watchdog:consul online : %s", online)
        service_name = "traefik-system-ingress"

        if online:
            node = getServiceByNameAndId(service_name, sid)
            if node is None:
                sid = register(
                    service_host=_service_host,
                    service_port=_service_port,
                    service_id=sid,
                )
                logger.info("watchdog:Re-registered service: %s", sid)

    logger.info("watchdog:starting with interval : %s", interval)
    rt = RepeatedTimer(interval, _register, service_host, service_port)


def _dispatch_command(msg):
    logger.info(f"ipc:dispatching : {msg}")

    if "command" in msg:
        command = msg["command"]
        if "status" == command:
            online = bool(strtobool(msg["online"])) if "online" in msg else False
            logger.info(f"online = {online}")
            sid = current_service_id
            if not online:
                c, _online = create_client(config, True)
                if not _online:
                    logger.debug("Consul service is offline")
                    return None

                c.agent.service.deregister(sid)
                logger.info("ipc:de-registered service: %s", sid)


def ipc_listener():
    from multiprocessing.connection import Listener

    running = True
    address = ("localhost", 6500)  # family is deduced to be 'AF_INET'
    listener = Listener(address, family="AF_INET", authkey=b"redfox")
    logger.info(f"ipc:starting listener on : {address}")

    while running:
        conn = listener.accept()
        logger.info(f"ipc:connection accepted from : {listener.last_accepted}")
        while conn.poll():
            msg = conn.recv()
            logger.info(f"ipc:message = {msg}")
            if msg == "CLOSE":
                conn.close()
                break
            elif msg == "EXIT":
                running = False
                conn.close()
                break
            else:
                logger.info(f"Processing msg : {msg}")
                if not isinstance(msg, dict):
                    raise ValueError(f"Command expected to be a 'dict' not {type(msg)}")
                _dispatch_command(msg)

    listener.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--debug-server', type=bool, default=False, required=False, help='Should we start debug webserver')
    # parser.add_argument('--port', type=int, default=-1, help='Port number to export (-1 dynamic)')
    # parser.add_argument('--ip', type=str, default='127.0.0.1', help='Service IP to expose, blank for dynamic')
    # parser.add_argument('--watchdog-interval', type=int, default=60, help='watchdog interval checkin seconds')
    parser.add_argument(
        "--config",
        type=str,
        default="./config/marie-debug.yml",
        help="Configuration file",
    )

    opt = parser.parse_args()

    # Load config
    with open(opt.config, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        logger.info(f"Config read successfully : {opt.config}")

    enabled = bool(data["RegistryEnabled"])
    if not enabled:
        logger.info("registry not enabled, exiting...")
        exit()

    config = EndpointConfig()
    config.Host = data["ConsulEndpoint"]["Host"]
    config.Port = int(data["ConsulEndpoint"]["Port"])
    config.Scheme = data["ConsulEndpoint"]["Scheme"]

    host_name = data["ServiceEndpoint"]["Host"]
    server_port = int(data["ServiceEndpoint"]["Port"])
    watchdog_interval = int(data["WatchdogInterval"])
    debug_server = bool(data["DebugWebserver"])

    if host_name is None or host_name == "":
        host_name = get_ip_address()

    if server_port == -1:
        server_port = find_open_port()

    with open("port.dat", "r", encoding="utf-8") as fsrc:
        server_port = int(fsrc.read())
        logger.info(f"port = {server_port}")

    current_service_id = register(
        service_host=host_name, service_port=server_port, service_id=None
    )
    logger.info("Registration service: %s", current_service_id)

    def _watchdog_target():
        return start_watchdog(
            watchdog_interval, service_host=host_name, service_port=server_port
        )

    threading.Thread(target=_watchdog_target, daemon=debug_server).start()

    # ipc_task = threading.Thread(target=ipc_listener, daemon=debug_server).start()

    if debug_server:
        start_webserver(host_name, server_port)
