from typing import TYPE_CHECKING, Optional

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

classifier_flow_is_ready = False


def validate_payload(payload: dict) -> (bool, Optional[str]):
    """
    Validate payload
    :param payload: The payload to validate
    :return:
    """
    if not payload:
        return False, "Payload is empty"

    if "model" not in payload:
        return False, "Payload is missing model field"

    return True, None


def extend_rest_interface_classifier(app: FastAPI, client: Client) -> None:
    """
    Extends HTTP Rest endpoint to provide compatibility with existing REST endpoints
    :param client:
    :param app:
    :return:
    """

    @app.post(
        "/api/document/classify",
        tags=["classify", "rest-api"],
        dependencies=[Depends(TokenBearer())],
    )
    async def text_ner_post(request: Request, token: str = Depends(TokenBearer())):
        """
        Handle API Classify endpoint
        :param request:
        :param token: API Key token
        :return:
        """

        logger.info(f"text_extract_post : {token}")

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
            validate_payload,
        )
