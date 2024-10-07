from request_handling_custom import GatewayRequestHandler
from server_gateway import MarieServerGateway

from marie import Flow


def main():
    print("Bootstrapping server gateway")
    # gateway --protocol http --discovery --discovery-host 127.0.0.1 --discovery-port 8500 --host 192.168.102.65 --port 5555

    # we could override the default GatewayRequestHandler with our custom GatewayRequestHandler
    # but for now we will do monkey patching in MarieGatewayServer
    if False:
        from marie.serve.runtimes.gateway import request_handling

        request_handling.GatewayRequestHandler = GatewayRequestHandler

    with (
        Flow(
            discovery=False,  # server gateway does not need discovery service
        ).config_gateway(
            uses=MarieServerGateway,
            protocols=["GRPC", "HTTP"],
            ports=[52000, 51000],
            kv_store_kwargs={
                "provider": "postgresql",
                "hostname": "0.0.0.0",
                "port": 5432,
                "username": "postgres",
                "password": "123456",
                "database": "postgres",
                "default_table": "kv_store_worker",
                "max_pool_size": 25,
                "max_connections": 25,
            },
            job_scheduler_kwargs={
                "provider": "postgresql",
                "hostname": "0.0.0.0",
                "port": 5432,
                "database": "postgres",
                "username": "postgres",
                "password": "123456",
                "default_table": "job",  # Unused as it will be provided by the gateway
                "max_pool_size": 25,
                "max_connections": 25,
            },
            # ETCD discovery service
            discovery=True,
            discovery_host="0.0.0.0",
            discovery_port=2379,
            discovery_watchdog_interval=2,
            discovery_service_name="gateway/marie",
        )
        # .add(tls=False, host="0.0.0.0", external=True, port=61000)
        as flow
    ):
        flow.save_config("/mnt/data/marie-ai/config/service/direct-flow-gateway.yml")
        flow.block()


if __name__ == "__main__":
    main()
