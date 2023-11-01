import asyncio

from fastapi import FastAPI, Request
from fastapi import HTTPException, Depends

from marie import Client
from marie.logging.predefined import default_logger as logger
from marie_server.auth.auth_bearer import TokenBearer
from marie_server.rest_extension import (
    handle_request,
    process_document_request,
)

extract_flow_is_ready = False


def extend_rest_interface_extract(
    app: FastAPI, client: Client, queue: asyncio.Queue
) -> None:
    """
    Extends HTTP Rest endpoint to provide compatibility with existing REST endpoints
    :param client: Marie Client
    :param app: FastAPI app
    :param queue: asyncio.Queue to handle backpressure,
    :return:
    """

    @app.get("/api/document/extract", tags=["document", "rest-api"])
    async def text_extract_get(request: Request):
        logger.info("Executing text_extract_get")
        return {"message": "reply"}

    @app.post(
        "/api/document/extract",
        tags=["document", "rest-api"],
        dependencies=[Depends(TokenBearer())],
    )
    async def text_extract_post(request: Request, token: str = Depends(TokenBearer())):
        """
        Handle API Extract endpoint
        :param request:
        :param token: API Key token
        :return:
        """

        logger.debug(f"text_extract_post : {token}")

        global extract_flow_is_ready
        if not extract_flow_is_ready and not await client.is_flow_ready():
            raise HTTPException(status_code=503, detail="Flow is not yet ready")
        extract_flow_is_ready = True

        return await handle_request(
            token,
            "extract",
            request,
            client,
            process_document_request,
            "/document/extract",
            queue,
            validate_payload_callback=None,
        )

    @app.get("/api/document/status", tags=["text", "rest-api"])
    async def text_status():
        """
        Handle API Status endpoint
        :return:
        """
        logger.info("Executing text_status")
        return {"status": "OK"}
