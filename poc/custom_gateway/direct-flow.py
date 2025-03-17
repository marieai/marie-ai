import inspect
import os
import random
import time

from docarray import DocList
from docarray.documents import TextDoc

from marie import Deployment, Executor, Flow, requests
from marie.api import AssetKeyDoc
from marie.conf.helper import load_yaml
from marie.executor.marie_executor import MarieExecutor


class TestExecutor(MarieExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("TestExecutor init called")
        # emulate the long loading time
        # time.sleep(1

    @requests(on="/extract")
    async def func_extract(
        self,
        docs: DocList[AssetKeyDoc],
        parameters=None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = {}
        print(f"FirstExec func called : {len(docs)}, {parameters}")
        # randomly throw an error to test the error handling
        # if random.random() > 0.5:
        #     raise Exception("random error in FirstExec")
        #
        # for doc in docs:
        #     doc.text += " First Exec"
        sec = 5
        print(f"Sleeping for {sec} seconds : ", time.time())
        time.sleep(sec)
        # raise Exception("random error in FirstExec")

        print(f"Sleeping for {sec} seconds - done : ", time.time())
        return {
            "parameters": parameters,
            "data": "Data reply",
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
    # yml_config = "/mnt/data/marie-ai/config/service/deployment.yml"
    # # Load the config file and set up the toast events
    # config = load_yaml(yml_config, substitute=True, context=context)
    print("Bootstrapping Flow")
    with Flow(
        discovery=True,  # server gateway does not need discovery service
        discovery_host="0.0.0.0",
        discovery_port=2379,
        discovery_watchdog_interval=2,
        discovery_service_name="gateway/marie",
        kv_store_kwargs={
            "provider": "postgresql",
            "hostname": "127.0.0.1",
            "port": 5432,
            "username": "postgres",
            "password": "123456",
            "database": "postgres",
            "default_table": "kv_store_worker",
            "max_pool_size": 5,
            "max_connections": 5,
        },
    ).add(uses=TestExecutor, name="executor0", replicas=1) as f:
        f.save_config("/mnt/data/marie-ai/config/service/direct-flow.yml")
        f.block()


if __name__ == "__main__":
    main()

# curl -X GET "http://localhost:51000/endpoint?text=abc"
# https://docs.jina.ai/concepts/serving/gateway/customization/
