import asyncio
import time
import uuid
from functools import partial
from fastapi import FastAPI, Request, HTTPException
from typing import TYPE_CHECKING, Any, Optional
import traceback
import sys
from fastapi import Request
from marie import Client
from marie.api import extract_payload
from marie.api import value_from_payload_or_args
from marie.logging.predefined import default_logger
from marie.types.request.data import DataRequest
from marie.utils.docs import docs_from_file, docs_from_file_specific
from marie.utils.types import strtobool

from marie.messaging import (
    mark_as_complete,
    mark_as_started,
    mark_as_failed,
)
from marie.utils.utils import ensure_exists

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI, Request


def extend_rest_interface(app: "FastAPI") -> "FastAPI":
    """Register executors REST endpoints that do not depend on DocumentArray
    :param app:
    :return:
    """

    from .executors.extract.mserve_torch import (
        extend_rest_interface_extract,
    )
    from .executors.ner.mserve_torch import (
        extend_rest_interface_ner,
    )
    from .executors.overlay.mserve_torch import (
        extend_rest_interface_overlay,
    )

    client = Client(
        host="0.0.0.0", port=52000, protocol="grpc", request_size=1, asyncio=True
    )

    extend_rest_interface_extract(app, client)
    extend_rest_interface_ner(app, client)
    extend_rest_interface_overlay(app, client)

    return app


def generate_job_id() -> str:
    return str(uuid.uuid4())


def parse_response_to_payload(
    resp: DataRequest, expect_return_value: Optional[bool] = True
):
    """
    We get raw response `marie.types.request.data.DataRequest` and we will extract the returned payload (Dictionary object)
    If the executor is not returning any value, we will return empty dictionary object. This is perfectly valid response
    as we are not expecting any value from the executor.

    :param expect_return_value:  if True, we expect that the response will contain `__results__` key
    :param resp: response from the executor
    :return:  payload
    """

    if "__results__" in resp.parameters:
        results = resp.parameters["__results__"]
        payload = list(results.values())[0]
        return payload

    if expect_return_value:
        # raise ValueError("Response does not contain __results__ key")
        return {
            "status": "FAILED",
            "message": "are you calling valid endpoint, __results__ missing in params",
        }

    return {}


async def parse_payload_to_docs(payload: Any, clear_payload: Optional[bool] = True):
    return parse_payload_to_docs_sync(payload, clear_payload)


def parse_payload_to_docs_sync(payload: Any, clear_payload: Optional[bool] = True):
    """
    Parse payload request, extract file and return list of Document objects

    :param payload:
    :param clear_payload:
    :return:
    """
    # every request should contain queue_id if not present it will default to '0000-0000-0000-0000'
    queue_id = value_from_payload_or_args(
        payload, "queue_id", default="0000-0000-0000-0000"
    )
    tmp_file, checksum, file_type = extract_payload(payload, queue_id)
    pages = []

    try:
        pages_parameter = value_from_payload_or_args(payload, "pages", default="")
        if len(pages_parameter) > 0:
            pages = [int(page) for page in pages_parameter.split(",")]
    except:
        pass

    input_docs = docs_from_file_specific(tmp_file, pages)
    # input_docs = docs_from_file(tmp_file)

    if clear_payload:
        key = "data"
        if "data" in payload:
            key = "data"
        elif "srcData" in payload:
            key = "srcData"
        elif "srcBase64" in payload:
            key = "srcBase64"
        elif "srcFile" in payload:
            key = "srcFile"
        elif "uri" in payload:
            key = "uri"
        del payload[key]

    doc_id = value_from_payload_or_args(payload, "doc_id", default=checksum)
    doc_type = value_from_payload_or_args(payload, "doc_type", default="")

    parameters = {
        "queue_id": queue_id,
        "ref_id": doc_id,
        "ref_type": doc_type,
    }
    return parameters, input_docs


async def handle_request(
    api_tag: str, request: Request, client: Client, handler: callable
):
    try:
        payload = await request.json()

        # write payload to file
        ensure_exists("/tmp/payloads")
        with open(f"/tmp/payloads/{api_tag}.json", "w") as f:
            f.write(str(payload))

        job_id = generate_job_id()
        default_logger.info(f"handle_request[{api_tag}] : {job_id}")
        sync = strtobool(value_from_payload_or_args(payload, "sync", default=False))

        future = [
            asyncio.ensure_future(
                process_request(api_tag, job_id, payload, partial(handler, client))
            )
        ]

        # run the task synchronously for debugging purposes
        if sync:
            results = await asyncio.gather(*future, return_exceptions=True)
            if isinstance(results[0], Exception):
                raise results[0]
            return results[0]

        return {"jobid": job_id, "status": "ok"}
    except Exception as e:
        default_logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


async def process_request(api_tag: str, job_id: str, payload: Any, handler: callable):
    """
    When request is processed, it will be marked as  `STARTED` and then `COMPLETED`.
    If there is an error, it will be marked as `FAILED` with the error message and then `COMPLETED`
    to indicate that the request is finished.

    Args:
        api_tag:
        job_id:
        payload:
        handler:

    Returns:
    """

    status = "OK"
    job_tag = ""
    results = None

    try:
        default_logger.info(f"Starting request: {job_id}")
        parameters, input_docs = await parse_payload_to_docs(payload)
        job_tag = parameters["ref_type"] if "ref_type" in parameters else ""
        parameters["payload"] = payload  # THIS IS TEMPORARY HERE

        # payload data attribute should be stripped at this time
        await mark_as_started(
            job_id, api_tag, job_tag, status, int(time.time()), payload
        )

        async def run(op, _docs, _param):
            return await op(_docs, _param)

        results = await run(handler, input_docs, parameters)
        return results
    except BaseException as e:
        default_logger.error(f"processing error : {e}", exc_info=False)
        status = "FAILED"

        # get the traceback and clear the frames to avoid memory leak
        _, val, tb = sys.exc_info = sys.exc_info()
        traceback.clear_frames(tb)

        filename = tb.tb_frame.f_code.co_filename
        name = tb.tb_frame.f_code.co_name
        line_no = tb.tb_lineno

        exc = {
            "type": type(e).__name__,
            "message": str(e),
            "filename": filename.split("/")[-1],
            "name": name,
            "line_no": line_no,
        }
        results = str(e)
        await mark_as_failed(job_id, api_tag, job_tag, status, int(time.time()), exc)
    finally:
        await mark_as_complete(
            job_id, api_tag, job_tag, status, int(time.time()), results
        )
