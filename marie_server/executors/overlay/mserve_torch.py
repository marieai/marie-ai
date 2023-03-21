from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, HTTPException
from marie import Client
from marie.logging.predefined import default_logger
from marie_server.rest_extension import (
    parse_response_to_payload,
    parse_payload_to_docs,
    handle_request,
)

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI

overlay_flow_is_ready = False


def extend_rest_interface_overlay(app: FastAPI, client: Client) -> None:
    """
    Extends HTTP Rest endpoint to provide compatibility with existing REST endpoints
    :param client:
    :param app:
    :return:
    """

    async def __process(client: Client, input_docs, parameters):
        payload = {}
        async for resp in client.post(
            "/overlay/segment",
            input_docs,
            request_size=-1,
            parameters=parameters,
            return_responses=True,
        ):
            payload = parse_response_to_payload(resp)
        return payload

    @app.post("/api/overlay", tags=["overlay", "rest-api"])
    async def overlay_post(request: Request):
        """
        Handle API Overlay endpoint
        :param request:
        :return:
        """
        global overlay_flow_is_ready
        print(f"{overlay_flow_is_ready=}")
        if not overlay_flow_is_ready and not await client.is_flow_ready():
            raise HTTPException(status_code=503, detail="Flow is not yet ready")
        overlay_flow_is_ready = True

        return await handle_request("overlay", request, client, __process)

    @app.get("/api/overlay/status", tags=["overlay", "rest-api"])
    async def overlay_status():
        default_logger.info("Executing overlay_status")

        return {"status": "OK"}
