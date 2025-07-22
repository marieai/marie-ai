from marie.serve.discovery.etcd_client import EtcdClient


def client_examples():
    # These all work the same way
    etcd_client = EtcdClient("localhost", 2379, namespace="marie")
    etcd_client = EtcdClient(etcd_host="localhost", etcd_port=2379, namespace="marie")

    # List of endpoints as strings in t etcd_host format
    etcd_client = EtcdClient(
        etcd_host="etcd-node1.example.com:2379, etcd-node2.example.com,etcd-node3.example.com:2379",
    )

    # List of tuples
    etcd_client = EtcdClient(
        endpoints=[
            ("etcd-node1.example.com", 2379),
            ("etcd-node2.example.com", 2379),
            ("etcd-node3.example.com", 2379),
        ],
        namespace="marie",
    )

    # List of strings with ports
    etcd_client = EtcdClient(
        endpoints=[
            "etcd-node1.example.com:2379",
            "etcd-node2.example.com:2379",
            "etcd-node3.example.com:2379",
        ],
        namespace="marie",
    )

    # Mixed formats (defaults to port 2379 if not specified)
    etcd_client = EtcdClient(
        endpoints=[
            "etcd-node1.example.com",  # Uses port 2379
            "etcd-node2.example.com:2380",  # Uses port 2380
            ("etcd-node3.example.com", 2381),  # Uses port 2381
        ],
        namespace="marie",
    )


if __name__ == "__main__":
    etcd_client = EtcdClient("localhost", 2379)
    etcd_client.put("key", "Value XYZ")

    kv = etcd_client.get("key")
    print(etcd_client.get("key"))
    # etcd_client.delete('key')
    print(etcd_client.get("key"))

    kv = {"key1": "Value 1", "key2": "Value 2", "key3": "Value 3"}

    etcd_client.put_prefix("prefix", kv)

    print(etcd_client.get_prefix("prefix"))

    print("------ GET ALL ---------")
    for kv in etcd_client.get_all():
        v = kv[0].decode("utf8")
        k = kv[1].key.decode("utf8")
        print(k, v)
