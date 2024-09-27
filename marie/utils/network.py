import os
import socket


def find_open_port():
    """
    Find an open port
    """
    sock = socket.socket()
    sock.bind(("", 0))
    _, port = sock.getsockname()

    return port


def is_docker():
    path = "/proc/self/cgroup"
    return (
        os.path.exists("/.dockerenv")
        or os.path.isfile(path)
        and any("docker" in line for line in open(path))
    )


_cached_ip_address = None


def get_ip_address(flush_cache=False):
    """
    Get the IP address of the current machine. Caches the result for future calls.
    Set `flush_cache` to True to refresh the cached IP address.

    https://stackoverflow.com/questions/24196932/how-can-i-get-the-ip-address-from-nic-in-python
    """
    global _cached_ip_address

    if flush_cache or _cached_ip_address is None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                _cached_ip_address = s.getsockname()[0]
        except Exception as e:
            _cached_ip_address = "127.0.0.1"

    return _cached_ip_address
