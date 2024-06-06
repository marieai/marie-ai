import time

from marie import Flow
from marie.serve.runtimes.servers.grpc import GRPCServer
from poc.custom_gateway.server_gateway import MarieServerGateway


def main():
    print("Bootstrapping server gateway")

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

    with (
        Flow().config_gateway(
            uses=MarieServerGateway, protocols=["GRPC", "HTTP"], ports=[52000, 51000]
        )
        # .add(tls=False, host="0.0.0.0", external=True, port=61000)
        as flow
    ):
        flow.block()


if __name__ == "__main__":
    main()
