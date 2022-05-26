import argparse
import json
from typing import TYPE_CHECKING, Dict, List, Optional

from docarray import Document

from marie.version import __version__
from marie.clients.request import request_generator
from marie.enums import DataInputType
from marie.excepts import InternalNetworkError
from marie.helper import get_full_version
from marie.importer import ImportExtensions
from marie.logging.profile import used_memory_readable
from marie.serve.networking import GrpcConnectionPool

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

    from marie.serve.networking import GrpcConnectionPool
    # from jina.serve.runtimes.gateway.graph.topology_graph import TopologyGraph


def get_fastapi_app(
    args: 'argparse.Namespace',
    topology_graph: 'TopologyGraph',
    connection_pool: 'GrpcConnectionPool',
    logger: 'JinaLogger',
    metrics_registry: Optional['CollectorRegistry'] = None,
):
    """
    Get the app from FastAPI as the REST interface.

    :param args: passed arguments.
    :param topology_graph: topology graph that manages the logic of sending to the proper executors.
    :param connection_pool: Connection Pool to handle multiple replicas and sending to different of them
    :param logger: logger.
    :param metrics_registry: optional metrics registry for prometheus used if we need to expose metrics from the executor or from the data request handler
    :return: fastapi app
    """
    with ImportExtensions(required=True):
        from fastapi import FastAPI, Response, status
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse
        from starlette.requests import Request

        from marie.serve.runtimes.gateway.http.models import (
            JinaEndpointRequestModel,
            JinaRequestModel,
            JinaResponseModel,
            JinaStatusModel,
        )

    docs_url = "/docs"
    app = FastAPI(
        title=args.title or "Marie Service",
        description=args.description
        or "You can set `title` and `description` in your `Flow` or `Gateway` "
        "to customize this text.",
        version=__version__,
        docs_url=docs_url if args.default_swagger_ui else None,
    )

    if args.cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.warning("CORS is enabled. This service is now accessible from any website!")

    from marie.serve.runtimes.gateway.request_handling import RequestHandler
    from marie.serve.stream import RequestStreamer

    request_handler = RequestHandler(metrics_registry, args.name)

    streamer = RequestStreamer(
        args=args,
        request_handler=request_handler.handle_request(graph=None, connection_pool=connection_pool),
        result_handler=request_handler.handle_result(),
    )
    streamer.Call = streamer.stream

    @app.on_event("shutdown")
    async def _shutdown():
        await connection_pool.close()

    openapi_tags = []
    if not args.no_debug_endpoints:
        openapi_tags.append(
            {
                "name": "Debug",
                "description": "Debugging interface. In production, you should hide them by setting "
                "`--no-debug-endpoints` in `Flow`/`Gateway`.",
            }
        )

        from marie.serve.runtimes.gateway.http.models import JinaHealthModel

        @app.get(
            path="/",
            summary="Get the health of Marie service",
            response_model=JinaHealthModel,
        )
        async def _health():
            """
            Get the health of this Marie service.
            .. # noqa: DAR201

            """
            return {}

        @app.get(
            path="/status",
            summary="Get the status of Marie service",
            response_model=JinaStatusModel,
            tags=["Debug"],
        )
        async def _status():
            """
            Get the status of this Marie service.

            This is equivalent to running `marie -vf` from command line.

            .. # noqa: DAR201
            """
            _info = get_full_version()
            return {
                "marie": _info[0],
                "envs": _info[1],
                "used_memory": used_memory_readable(),
            }

        @app.post(
            path="/post",
            summary="Post a data request to some endpoint",
            response_model=JinaResponseModel,
            tags=["Debug"]
            # do not add response_model here, this debug endpoint should not restricts the response model
        )
        async def post(
            body: JinaEndpointRequestModel, response: Response
        ):  # 'response' is a FastAPI response, not a Jina response
            """
            Post a data request to some endpoint.

            This is equivalent to the following:

                from marie import Flow

                f = Flow().add(...)

                with f:
                    f.post(endpoint, ...)

            .. # noqa: DAR201
            .. # noqa: DAR101
            """
            # The above comment is written in Markdown for better rendering in FastAPI
            from marie.enums import DataInputType

            bd = body.dict()  # type: Dict
            req_generator_input = bd
            req_generator_input["data_type"] = DataInputType.DICT
            if bd["data"] is not None and "docs" in bd["data"]:
                req_generator_input["data"] = req_generator_input["data"]["docs"]

            try:
                result = await _get_singleton_result(request_generator(**req_generator_input))
            except InternalNetworkError as err:
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                result = bd  # send back the request
                result["header"] = _generate_exception_header(err)  # attach exception details to response header
                logger.error(f"Error while getting responses from deployments: {err.details()}")
            return result

    def _generate_exception_header(error: InternalNetworkError):
        import traceback

        exception_dict = {
            "name": str(error.__class__),
            "stacks": [str(x) for x in traceback.extract_tb(error.og_exception.__traceback__)],
            "executor": "",
        }
        status_dict = {
            "code": 3,  # status error
            "description": error.details() if error.details() else "",
            "exception": exception_dict,
        }
        header_dict = {"request_id": error.request_id, "status": status_dict}
        return header_dict

    def expose_executor_endpoint(exec_endpoint, http_path=None, **kwargs):
        """Exposing an executor endpoint to http endpoint
        :param exec_endpoint: the executor endpoint
        :param http_path: the http endpoint
        :param kwargs: kwargs accepted by FastAPI
        """

        # set some default kwargs for richer semantics
        # group flow exposed endpoints into `customized` group
        kwargs["tags"] = kwargs.get("tags", ["Customized"])
        kwargs["response_model"] = kwargs.get(
            "response_model",
            JinaResponseModel,  # use standard response model by default
        )
        kwargs["methods"] = kwargs.get("methods", ["POST"])

        @app.api_route(path=http_path or exec_endpoint, name=http_path or exec_endpoint, **kwargs)
        async def foo(body: JinaRequestModel):
            from marie.enums import DataInputType

            bd = body.dict() if body else {"data": None}
            bd["exec_endpoint"] = exec_endpoint
            req_generator_input = bd
            req_generator_input["data_type"] = DataInputType.DICT
            if bd["data"] is not None and "docs" in bd["data"]:
                req_generator_input["data"] = req_generator_input["data"]["docs"]

            result = await _get_singleton_result(request_generator(**req_generator_input))
            return result

    if openapi_tags:
        app.openapi_tags = openapi_tags

    if args.expose_endpoints:
        endpoints = json.loads(args.expose_endpoints)  # type: Dict[str, Dict]
        for k, v in endpoints.items():
            expose_executor_endpoint(exec_endpoint=k, **v)

    if not args.default_swagger_ui:

        async def _render_custom_swagger_html(req: Request) -> HTMLResponse:
            import urllib.request

            swagger_url = "https://api.marie##.ai/swagger"
            req = urllib.request.Request(swagger_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as f:
                return HTMLResponse(f.read().decode())

        app.add_route(docs_url, _render_custom_swagger_html, include_in_schema=False)

    if args.expose_graphql_endpoint:
        with ImportExtensions(required=True):
            from dataclasses import asdict

            import strawberry
            from docarray import DocumentArray
            from docarray.document.strawberry_type import (
                JSONScalar,
                StrawberryDocument,
                StrawberryDocumentInput,
            )
            from strawberry.fastapi import GraphQLRouter

            async def get_docs_from_endpoint(data, target_executor, parameters, exec_endpoint):
                req_generator_input = {
                    "data": [asdict(d) for d in data],
                    "target_executor": target_executor,
                    "parameters": parameters,
                    "exec_endpoint": exec_endpoint,
                    "data_type": DataInputType.DICT,
                }

                if req_generator_input["data"] is not None and "docs" in req_generator_input["data"]:
                    req_generator_input["data"] = req_generator_input["data"]["docs"]
                try:
                    response = await _get_singleton_result(request_generator(**req_generator_input))
                except InternalNetworkError as err:
                    logger.error(f"Error while getting responses from deployments: {err.details()}")
                    raise err  # will be handled by Strawberry
                return DocumentArray.from_dict(response["data"]).to_strawberry_type()

            @strawberry.type
            class Mutation:
                @strawberry.mutation
                async def docs(
                    self,
                    data: Optional[List[StrawberryDocumentInput]] = None,
                    target_executor: Optional[str] = None,
                    parameters: Optional[JSONScalar] = None,
                    exec_endpoint: str = "/search",
                ) -> List[StrawberryDocument]:
                    return await get_docs_from_endpoint(data, target_executor, parameters, exec_endpoint)

            @strawberry.type
            class Query:
                @strawberry.field
                async def docs(
                    self,
                    data: Optional[List[StrawberryDocumentInput]] = None,
                    target_executor: Optional[str] = None,
                    parameters: Optional[JSONScalar] = None,
                    exec_endpoint: str = "/search",
                ) -> List[StrawberryDocument]:
                    return await get_docs_from_endpoint(data, target_executor, parameters, exec_endpoint)

            schema = strawberry.Schema(query=Query, mutation=Mutation)
            app.include_router(GraphQLRouter(schema), prefix="/graphql")

    async def _get_singleton_result(request_iterator) -> Dict:
        """
        Streams results from AsyncPrefetchCall as a dict

        :param request_iterator: request iterator, with length of 1
        :return: the first result from the request iterator
        """
        async for k in streamer.stream(request_iterator=request_iterator):
            request_dict = k.to_dict()
            return request_dict

    return app
