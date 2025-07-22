import time

from marie.logging_core.predefined import default_logger as logger


def main():
    import argparse

    parser = argparse.ArgumentParser(description="service discovery etcd cluster")
    parser.add_argument(
        "--host",
        help="the etcd host, default = 127.0.0.1",
        required=False,
        default="127.0.0.1XXX",
    )
    parser.add_argument(
        "--port",
        help="the etcd port, default = 2379",
        required=False,
        default=2379,
        type=int,
    )
    parser.add_argument("--ca-cert", help="the etcd ca-cert", required=False)
    parser.add_argument("--cert-key", help="the etcd cert key", required=False)
    parser.add_argument("--cert-cert", help="the etcd cert", required=False)
    parser.add_argument("--service-key", help="the service key", required=True)
    parser.add_argument(
        "--service-addr", help="the service address host:port ", required=True
    )
    parser.add_argument(
        "--lease-ttl",
        help="the lease ttl in seconds, default is 10",
        required=False,
        default=10,
        type=int,
    )
    parser.add_argument("--my-id", help="my identifier", required=True)
    parser.add_argument(
        "--timeout",
        help="the etcd operation timeout in seconds, default is 2",
        required=False,
        type=int,
        default=2,
    )
    args = parser.parse_args()

    params = {"host": args.host, "port": args.port, "timeout": args.timeout}
    if args.ca_cert:
        params["ca_cert"] = args.ca_cert
    if args.cert_key:
        params["cert_key"] = args.cert_key
    if args.cert_cert:
        params["cert_cert"] = args.cert_cert

    logger.info(f"args : {args}")

    from marie.serve.discovery import EtcdServiceRegistry

    etcd_registry = EtcdServiceRegistry(args.host, args.port, heartbeat_time=5)
    etcd_registry.register([args.service_key], args.service_addr, args.lease_ttl)

    try:
        while True:
            time.sleep(2)  # Keep the program running.
    except KeyboardInterrupt:
        etcd_registry.unregister([args.service_key], args.service_addr)
        print("Service unregistered.")


if __name__ == "__main__":
    main()
