import pytest
from fastapi.testclient import TestClient

from fastapi.testclient import TestClient

from marie.logging.logger import MarieLogger
from marie.parsers import set_gateway_parser
from marie.serve.networking import GrpcConnectionPool
from marie.serve.runtimes.gateway import TopologyGraph
from marie.serve.runtimes.gateway.http import get_fastapi_app, HTTPGatewayRuntime


@pytest.mark.parametrize('runtime_cls', [HTTPGatewayRuntime])
def test_uvicorn_ssl_deprecated(runtime_cls):
    args = set_gateway_parser().parse_args(
        [ ]
    )
    with runtime_cls(args):
        pass


# @pytest.mark.asyncio
async def test_app_bootstrap():
    args = set_gateway_parser().parse_args([])
    logger = MarieLogger('')
    # app = get_fastapi_app(
    #     args, TopologyGraph({}), GrpcConnectionPool(logger=logger), logger
    # )
    # print(app.routes)
    # for route in app.routes:
    #     print(route.path)
    #

    gateway = HTTPGatewayRuntime(args)

    await gateway.async_run_forever()

