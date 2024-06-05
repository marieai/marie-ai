import inspect
import os

from docarray import DocList
from docarray.documents import TextDoc

from marie import Deployment, Executor, Flow, requests
from marie.conf.helper import load_yaml
from poc.custom_gateway.deployment_gateway import MariePodGateway


class TestExecutor(Executor):
    @requests(on="/")
    def func(
        self,
        docs: DocList[TextDoc],
        parameters: dict = {},
        *args,
        **kwargs,
    ):
        print(f"FirstExec func called : {len(docs)}")
        print(f"FirstExec funct : {parameters}")

        for doc in docs:
            doc.text += " First"

        return {
            "parameters": parameters,
            "data": "Data",
        }


def main_deployment():
    context = {"name": "test"}
    yml_config = "/mnt/data/marie-ai/config/service/deployment.yml"

    # Load the config file and set up the toast events
    config = load_yaml(yml_config, substitute=True, context=context)
    f = Deployment.load_config(
        config,
        extra_search_paths=[os.path.dirname(inspect.getfile(inspect.currentframe()))],
        substitute=True,
        context=context,
        include_gateway=False,
        noblock_on_start=False,
        prefetch=1,
        statefull=False,
    )

    with Deployment(
        uses=TestExecutor,
        timeout_ready=-1,
        protocol="grpc",
        port=61000,
        include_gateway=True,
        replicas=3,
    ):
        f.block()

    # with (Flow().add(uses=FirstExec) as f):
    #     f.block()


def main():
    context = {"name": "test"}
    yml_config = "/mnt/data/marie-ai/config/service/deployment.yml"

    # Load the config file and set up the toast events
    config = load_yaml(yml_config, substitute=True, context=context)

    with (
        Flow()
        .add(uses=TestExecutor)
        .config_gateway(
            uses=MariePodGateway,
            protocols=["GRPC", "HTTP"],
            ports=[61000, 61001],
        ) as f
    ):
        f.block()


if __name__ == "__main__":
    main()

# curl -X GET "http://localhost:51000/endpoint?text=abc"
# https://docs.jina.ai/concepts/serving/gateway/customization/
