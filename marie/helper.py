import threading
from typing import Dict, TYPE_CHECKING, Optional, Tuple


# based on jina

def get_internal_ip():
    """
    Return the private IP address of the gateway for connecting from other machine in the same network.

    :return: Private IP address.
    """
    import socket

    ip = "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # doesn't even have to be reachable
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except Exception:
        pass
    return ip


def get_public_ip(timeout: float = 0.3):
    """
    Return the public IP address of the gateway for connecting from other machine in the public network.

    :param timeout: the seconds to wait until return None.

    :return: Public IP address.

    .. warn::
        Set `timeout` to a large number will block the Flow.

    """
    import urllib.request

    results = []

    def _get_ip(url):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as fp:
                _ip = fp.read().decode().strip()
                results.append(_ip)

        except:
            pass  # intentionally ignored, public ip is not showed

    ip_server_list = [
        "https://api.ipify.org",
        "https://ident.me",
        "https://checkip.amazonaws.com/",
    ]

    threads = []

    for idx, ip in enumerate(ip_server_list):
        t = threading.Thread(target=_get_ip, args=(ip,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout)

    for r in results:
        if r:
            return r


def convert_tuple_to_list(d: Dict):
    """
    Convert all the tuple type values from a dict to list.

    :param d: Dict type of data.
    """
    for k, v in d.items():
        if isinstance(v, tuple):
            d[k] = list(v)
        elif isinstance(v, dict):
            convert_tuple_to_list(v)


if TYPE_CHECKING:
    from fastapi import FastAPI


def extend_rest_interface(app: 'FastAPI') -> 'FastAPI':
    """Extend Marie built-in FastAPI instance with customized APIs, routing, etc.

    :param app: the built-in FastAPI instance given by Marie
    :return: the extended FastAPI instance

    .. highlight:: python
    .. code-block:: python

        def extend_rest_interface(app: 'FastAPI'):
            @app.get('/extension1')
            async def root():
                return {"message": "Hello World"}

            return app
    """
    return app


def get_full_version() -> Optional[Tuple[Dict, Dict]]:
    info = {
        'marie': "-1.-1.-1"
    }
    return info