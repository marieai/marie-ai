from docarray import DocList
from docarray.documents import TextDoc

from marie import Executor, Flow, requests
from poc.custom_gateway.server_gateway import MarieServerGateway


def main():
    print("Bootstrapping server gateway")
    with (
        Flow().config_gateway(
            uses=MarieServerGateway, protocols=["GRPC", "HTTP"], ports=[52000, 51000]
        )
        # .add(tls=False, host="0.0.0.0", external=True, port=61876)
        as flow
    ):
        flow.block()


if __name__ == "__main__":
    main()
