import time

from marie import Flow
from marie.serve.runtimes.servers.grpc import GRPCServer
from poc.custom_gateway.request_handling_custom import GatewayRequestHandler
from poc.custom_gateway.server_gateway import MarieServerGateway


def main():
    print("Bootstrapping server gateway")

    if False:
        ctrl_address = "0.0.0.0:61000"
        print('waiting for all servers to be ready at : ', ctrl_address)
        while True:
            print(f"checking is ready at {ctrl_address}")
            res = GRPCServer.is_ready(ctrl_address)

            print(f"res: {res}")
            if res:
                print(f"Gateway is ready at {ctrl_address}")

                break
            time.sleep(1)
        return

    # gateway --protocol http --discovery --discovery-host 127.0.0.1 --discovery-port 8500 --host 192.168.102.65 --port 5555

    # we could override the default GatewayRequestHandler with our custom GatewayRequestHandler
    # but for now we will do monkey patching in MarieGatewayServer
    if False:
        from marie.serve.runtimes.gateway import request_handling

        request_handling.GatewayRequestHandler = GatewayRequestHandler

    with (
        Flow(
            # server gateway does not need discovery service this will be available as runtime_args.discovery: bool
            discovery=False,
        ).config_gateway(
            uses=MarieServerGateway,
            protocols=["GRPC", "HTTP"],
            ports=[52000, 51000],
        )
        # .add(tls=False, host="0.0.0.0", external=True, port=61000)
        as flow
    ):
        flow.block()


if __name__ == "__main__":
    main()
