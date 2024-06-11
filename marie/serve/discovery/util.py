import re


def form_service_key(service_name: str, service_addr: str):
    """Return service's key in etcd."""
    # validate service_addr format meets the requirement of host:port or ip:port
    if not re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}$', service_addr):
        raise ValueError(f"Invalid service address: {service_addr}")

    return '/'.join((service_name, service_addr))
