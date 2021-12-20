from math import fabs
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
import consul
from consul.base import Check

import time
from time import sleep
import threading
from threading import Event, Thread
from logger import create_info_logger

logger = create_info_logger("consul", "consul.log")

class EndpointConfig:
    Port: int
    Host: str
    Scheme: str

    def __str__(self):
        return self.Scheme + "://"+self.Host + ":" + str(self.Port)

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

class TestServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>Registry info</title></head>", "utf-8"))
        self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>Service status.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))

def start_webserver(hostName, serverPort):
    
    webServer = HTTPServer((hostName, serverPort), TestServer)
    print("Server started http://%s:%s" % (hostName, serverPort))
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")

DEFAULT_PORT = 8500
DEFAULT_HOST = 'localhost'

def verify_connection():
    port = DEFAULT_PORT
    host = DEFAULT_HOST

    logger.debug('Verify Consul connection to %s:%s', host, port)

    try:
        client = consul.Consul(host=host, port=int(port))
        client.agent.self()
        return True
    except ValueError:
        pass

    return False

def createClient(cfg: EndpointConfig)->consul.Consul:
    """
    Establish connection to consul server
    """
    port = cfg.Port
    host = cfg.Host

    try:
        client = consul.Consul(host=host, port=int(port))
        return client
    except ValueError:
        pass

    return None

def driver_version():
    return consul.__version__


def getServiceByNameAndId(service_name, service_id):
    c = None
    if c is None:
        return 

    index, nodes = c.health.service(service_name)
    print(nodes)
    for node in nodes:
        _id = node['Node']['ID']
        print(_id)    
    

def register(service_host, service_port, service_id=None) -> str:
    host = DEFAULT_HOST
    port = DEFAULT_PORT

    logger.info('ConsulHost: %s Port: %s ', host, port)
    logger.info('ServiceHost: %s Port: %s ', service_host, service_port)

    online = verify_connection()
    if online is False:
        logger.debug('Consule service is offline')
        return

    c = consul.Consul(host=host, port=int(port))
    service_name = 'traefik-system-ingress'
    service_url = f'http://{service_host}:{service_port}/api'

    # Create new service id
    if service_id is None:
        service_id = f'{service_name}@{service_port}#{uuid.uuid4()}'

    logger.info('Service url: %s', service_url)
    logger.info('Service id: %s', service_id)

    c.agent.service.register(
        name=service_name,
        service_id=service_id,
        port=service_port,
        address=service_host,
        check=Check.http(service_url, '10s', deregister='1m'),
        tags=[
            "traefik.enable=true",
            "traefik.consulcatalog.connect=false",
            "traefik.http.routers.traefik-system-ingress.entrypoints=marie",
            "traefik.http.routers.traefik-system-ingress.service=traefik-system-ingress",
            "traefik.http.routers.traefik-system-ingress.rule=HostRegexp(`{host:.+}`)",
            "traefik.http.services.traefik-system-ingress.loadbalancer.server.scheme=http",
        ])

    time.sleep(80 / 1000.0)

    node = getServiceByNameAndId(service_name, service_id)
    print(node)

    return service_id

def find_open_port():
    """
    Find an open port
    """
    import socket
    sock = socket.socket()
    sock.bind(('', 0))

    _, port = sock.getsockname()

    return port

def start_watchdog(service_host, service_port, service_id):

    def _register(service_host, service_port, service_id):
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        logger.info("Time, Host, Port, ServiceId : %s, %s, %s, %s", current_time, service_host, service_port, service_id)
        online = verify_connection()
        logger.info('watchdog:consul online : %s', online)


    print ("starting...")
    rt = RepeatedTimer(1, _register, service_host, service_port, service_id)

if __name__ == "__main__":

    import yaml
    
    cfg = EndpointConfig()
    cfg.Host = '127.0.0.1'
    cfg.Port = '8500'
    cfg.Scheme = 'http'

    endpoint_info = {
        'ConsulEndpoint': {
            'Host': '127.0.0.1',
            'Port': 8500,
            'Scheme': 'http'
        }
    }
    with open("discovery.yaml", 'w') as yamlfile:
        data = yaml.dump(endpoint_info, yamlfile)
        print("Write successful")
        

    cfg = EndpointConfig()

    with open("discovery.yaml", "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Read successful")
    print(data)

    cfg.Host = data['ConsulEndpoint']['Host']
    cfg.Port = int(data['ConsulEndpoint']['Port'])
    cfg.Scheme = data['ConsulEndpoint']['Scheme']

    print(str(cfg))

    createClient(cfg=cfg)

    exit()
    logger.info('registering with consul : %s', driver_version())
    online = verify_connection()
    logger.info('consul online : %s', online)
    
    hostName = "127.0.0.1"
    serverPort = find_open_port()

    service_id = register(service_host=hostName, service_port=serverPort, service_id=None)
    watchdog_task = threading.Thread(target=lambda: start_watchdog(service_host=hostName, service_port=serverPort, service_id = service_id), daemon=True).start()

    start_webserver(hostName, serverPort)
