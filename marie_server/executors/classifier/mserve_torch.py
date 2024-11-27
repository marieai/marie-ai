import asyncio
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request

from marie import Client
from marie.auth.auth_bearer import TokenBearer
from marie.logging_core.predefined import default_logger as logger
from marie_server.rest_extension import handle_request, process_document_request

classifier_flow_is_ready = False


def validate_payload(payload: dict) -> (bool, Optional[str]):
    """
    Validate payload for the classify endpoint

    :param payload: The payload to validate
    :return: True if the payload is valid, False otherwise, and an error message if the payload is invalid
    """
    if not payload:
        return False, "Payload is empty"

    if "uri" not in payload:
        return False, "Payload is missing uri field"

    if "doc_id" not in payload:
        return False, "Payload is missing doc_id field"

    if "doc_type" not in payload:
        return False, "Payload is missing doc_type field"

    return True, None


def extend_rest_interface_classifier(
    app: FastAPI, client: Client, queue: asyncio.Queue
) -> None:
    """
    Extends HTTP Rest endpoint to provide compatibility with existing REST endpoints
    :param client: Marie Client
    :param app: FastAPI app
    :param queue: asyncio.Queue to handle backpressure,
    :return:
    """

    @app.get("/api/document/classify", tags=["classify", "rest-api"])
    async def text_classify_get(request: Request):
        logger.info("Executing text_classify_get")
        return {"message": "reply"}

    @app.post(
        "/api/document/classify",
        tags=["classify", "rest-api"],
        dependencies=[Depends(TokenBearer())],
    )
    async def text_classify_post(request: Request, token: str = Depends(TokenBearer())):
        """
        Handle API Classify endpoint
        :param request:
        :param token: API Key token
        :return:
        """

        logger.info(f"text_classify_post : {token}")

        global classifier_flow_is_ready
        if not classifier_flow_is_ready and not await client.is_flow_ready():
            raise HTTPException(status_code=503, detail="Flow is not yet ready")
        classifier_flow_is_ready = True

        return await handle_request(
            token,
            "classify",
            request,
            client,
            process_document_request,
            "/document/classify",
            queue,
            validate_payload_callback=validate_payload,
            # validate_payload_callback=None,
        )
