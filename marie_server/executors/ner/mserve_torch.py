from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Depends, HTTPException

from marie import Client
from marie.logging.predefined import default_logger as logger
from marie_server.auth.auth_bearer import TokenBearer
from marie_server.rest_extension import (
    handle_request,
    process_document_request,
)

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI

ner_flow_is_ready = False


def extend_rest_interface_ner(app: FastAPI, client: Client) -> None:
    """
    Extends HTTP Rest endpoint to provide compatibility with existing REST endpoints
    :param client:
    :param app:
    :return:
    """

    @app.post(
        '/api/ner/extract',
        tags=['ner', 'rest-api'],
        dependencies=[Depends(TokenBearer())],
    )
    async def text_ner_post(request: Request, token: str = Depends(TokenBearer())):
        """
        Handle API Extract endpoint
        :param request:
        :param token: API Key token
        :return:
        """

        logger.info(f"text_extract_post : {token}")

        global ner_flow_is_ready
        if not ner_flow_is_ready and not await client.is_flow_ready():
            raise HTTPException(status_code=503, detail="Flow is not yet ready")
        ner_flow_is_ready = True
        return await handle_request(
            token, "ner", request, client, process_document_request, "/ner/extract"
        )

    @app.get("/api/ner/status", tags=["ner", "rest-api"])
    async def ner_status():
        """
        Handle API Status endpoint
        :return:
        """
        logger.info("Executing ner_status")
        return {"status": "OK"}
