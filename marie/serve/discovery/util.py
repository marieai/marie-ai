import re


def form_service_key(service_name: str, service_addr: str):
    """Return service's key in etcd."""
    # validate service_addr format meets the requirement of host:port or ip:port or scheme://host:port
    # if not re.match(r'^[a-zA-Z]+://[a-zA-Z0-9.]+:\d+$', service_addr):
    #     raise ValueError(f"Invalid service address: {service_addr}")

    return '/'.join((service_name, service_addr))
