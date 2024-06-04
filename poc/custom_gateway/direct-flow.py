from marie import Executor, Flow, requests
from poc.custom_gateway.gateway import MarieGateway


class FirstExec(Executor):
    @requests(on="/")
    def func(self, docs, **kwargs):
        print("FirstExec func called")

        for doc in docs:
            doc.text += " First"


class SecondExec(Executor):
    @requests(on="/")
    def func(self, docs, **kwargs):
        print("SecondExec func called")

        for doc in docs:
            doc.text += " Second"


with (
    Flow(port=12345)
    .config_gateway(uses=MarieGateway, protocols=["GRPC", "HTTP"], ports=[52000, 51000])
    .add(uses=FirstExec) as flow
):
    flow.block()

#
# with (Flow(port=12345).config_gateway(uses=MarieGateway, protocols=["GRPC", "HTTP"], ports=[52000, 51000])
#               .add(uses=FirstExec, name="executor0"
#                    ).add(uses=SecondExec, name="executor1") as flow):
#     flow.block()

# curl -X GET "http://localhost:50975/endpoint?text=abc"
# https://docs.jina.ai/concepts/serving/gateway/customization/
