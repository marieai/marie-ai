import time

from marie.logging_core.predefined import default_logger as logger
from marie.serve.discovery.resolver import EtcdServiceResolver


def main():
    import argparse

    parser = argparse.ArgumentParser(description="service discovery etcd cluster")
    parser.add_argument(
        "--host",
        help="the etcd host, default = 127.0.0.1",
        required=False,
        default="127.0.0.1",
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

    resolver = EtcdServiceResolver(
        args.host, args.port, namespace="marie", start_listener=False, listen_timeout=5
    )

    logger.info(f"Resolved : {resolver.resolve(args.service_key)}")

    resolver.watch_service(
        args.service_key,
        lambda service, event: logger.info(f"Event from service : {service}, {event}"),
    )

    try:
        while True:
            logger.info(f"Checking service address...")
            time.sleep(2)
    except KeyboardInterrupt:
        print("Service stopped.")


if __name__ == "__main__":
    main()
