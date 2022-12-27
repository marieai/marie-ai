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


def get_ip_address():
    """
    https://stackoverflow.com/questions/24196932/how-can-i-get-the-ip-address-from-nic-in-python
    """
    # TODO : Add support for IP detection
    # if there is an access to external network we can try this
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            sockname = s.getsockname()
            return s.getsockname()[0]
    except Exception as e:
        # raise e  # For debug
        pass

    return "127.0.0.1"
