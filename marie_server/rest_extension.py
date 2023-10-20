import asyncio
import gc
import os
import sys
import time
import traceback
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

from docarray import DocumentArray
from fastapi import HTTPException
from fastapi import Request

from marie import Client, Flow
from marie._core.utils import run_background_task
from marie.api import extract_payload
from marie.api import value_from_payload_or_args
from marie.logging.mdc import MDC
from marie.logging.predefined import default_logger as logger
from marie.messaging import (
    mark_as_complete,
    mark_as_started,
    mark_as_failed,
)
from marie.messaging.publisher import mark_as_scheduled
from marie.types.request.data import DataRequest
from marie.utils.docs import docs_from_file
from marie.utils.types import strtobool
from marie.utils.utils import ensure_exists
from marie_server.job.job_manager import generate_job_id

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI, Request


async def coro_scheduler(queue: asyncio.Queue, limit: int = 2):
    pending = set()

    while True:
        while len(pending) < limit:
            item = queue.get()
            # pending.add(run_background_task(item))
            pending.add(asyncio.ensure_future(item))

        if not pending:
            continue

        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        while done:
            yield done.pop()


async def coro_consumer(queue: asyncio.Queue, limit: int = 2):
    async for scheduled_coro in coro_scheduler(queue, limit):
        try:
            print(f"async_consumer: {queue.qsize()} : {queue.empty()}")
            print(
                f"Tasks referenced by asyncio internals: count={len(asyncio.all_tasks())}"
            )

            scheduled = scheduled_coro.result()
            await scheduled

            del scheduled
            del scheduled_coro
        except Exception as e:
            # print traceback
            logger.error(f"Error: {e}", exc_info=True)
            raise e


def extend_rest_interface(flow: Flow, app: "FastAPI") -> "FastAPI":
    """Register executors REST endpoints that do not depend on DocumentArray
    :param flow: Marie Flow
    :param app: FastAPI app
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
    from .executors.classifier.mserve_torch import (
        extend_rest_interface_classifier,
    )

    try:
        import gc

        # gc.set_debug(gc.DEBUG_LEAK)
        # gc.set_debug(gc.DEBUG_SAVEALL)
        def f(phase, info):
            if phase == "start":
                print("starting garbage collection....")
            else:
                print(
                    "Finished garbage collection.... \n{}".format(
                        "".join(["{}: {}\n".format(*tup) for tup in info.items()])
                    )
                )

                print(
                    "Unreachable objects: \n{}".format(
                        "\n".join([str(garb) for garb in gc.garbage])
                    )
                )
                print()

        # gc.callbacks.append(f)
    except Exception as e:
        raise e

    # gc.disable()
    client = None
    #
    # client = Client(
    #     host="0.0.0.0", port=52000, protocol="grpc", request_size=1, asyncio=True
    # )

    limit = 1  # multiprocessing.cpu_count()
    backpressure_queue = asyncio.Queue()
    # run_background_task(coroutine=coro_consumer(queue=backpressure_queue, limit=limit))
    receive_task = asyncio.create_task(
        coro_consumer(queue=backpressure_queue, limit=limit)
    )
    extend_rest_interface_extract(app, client, queue=backpressure_queue)

    extend_rest_interface_ner(app, client)
    extend_rest_interface_overlay(app, client)
    extend_rest_interface_classifier(app, client)

    return app


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

    input_docs = docs_from_file(tmp_file, pages)
    # this is a hack to remove the data attribute from the payload and for backward compatibility
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
        elif "srcUrl" in payload:
            key = "srcUrl"
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
    api_key: str,
    api_tag: str,
    request: Request,
    client: Client,
    handler: callable,
    endpoint: str,
    queue: asyncio.Queue,
    validate_payload_callback: Optional[callable] = None,
):
    """
    Handle request from REST API
    :param api_key:  API Key
    :param api_tag:  API Tag (e.g. extract, ner, overlay)
    :param request:  FastAPI request object
    :param client:  Marie Client
    :param handler:  Handler function
    :param endpoint: Endpoint URL to call on the client
    :param validate_payload_callback: Callback function to validate payload
    :param queue:  asyncio.Queue to handle backpressure
    :return:
    """

    silence_exceptions = strtobool(os.environ.get("MARIE_SILENCE_EXCEPTIONS", False))

    try:
        job_id = generate_job_id()
        MDC.put("request_id", job_id)
        payload = await request.json()

        if validate_payload_callback:
            status, msg = validate_payload_callback(payload)
            if not status:
                return {"jobid": job_id, "status": "failed", "message": msg}

        logger.info(f"handle_request[{api_tag}] : {job_id}")
        sync = strtobool(value_from_payload_or_args(payload, "sync", default=False))

        if True:
            coroutine = process_request(
                api_key,
                api_tag,
                job_id,
                payload,
                # partial(handler, client, endpoint=endpoint),
                handler,
                client,
                endpoint,
            )

        # handle backpressure using asyncio.Queue
        # This is a temporary solution to handle backpressure
        # TODO :  replace this with a job_distributor class
        if True:
            if queue:
                try:
                    queue.put_nowait(coroutine)
                except asyncio.QueueFull:
                    return {
                        "jobid": job_id,
                        "status": "failed",
                        "message": "limit reached",
                    }
            else:
                raise ValueError("queue is not defined in handle_request")

        if False:
            # task = run_background_task(coroutine=coroutine)
            #  = [task]
            future = [
                asyncio.ensure_future(
                    process_request(
                        api_key,
                        api_tag,
                        job_id,
                        payload,
                        # partial(handler, client, endpoint=endpoint),
                        handler,
                        client,
                        endpoint,
                    )
                )
            ]

            if sync:
                results = await asyncio.gather(*future, return_exceptions=True)
                if isinstance(results[0], Exception):
                    raise results[0]
                return results[0]
            else:
                # schedule the job and return immediately
                pass

        return {"jobid": job_id, "status": "ok"}
    except Exception as e:
        # print traceback
        logger.error(f"Error: {e}", exc_info=True)
        code = 500
        detail = "Internal Server Error"

        if not silence_exceptions:
            if isinstance(e, HTTPException):
                code = e.status_code
                detail = e.detail
            else:
                detail = e.__str__()

        return {"status": "error", "error": {"code": code, "message": detail}}
        # raise HTTPException(status_code=500, detail="Internal Server Error")


async def process_document_request(
    client: Client, input_docs: DocumentArray, parameters: dict, endpoint: str
):
    """
    Process document request
    :param client:
    :param input_docs:
    :param parameters:
    :param endpoint:
    :return:
    """
    try:
        payload = {}

        print("trace # payload ZZ")
        clientzz = Client(
            host="0.0.0.0", port=52000, protocol="grpc", request_size=1, asyncio=True
        )

        async for resp in clientzz.post(
            protocol="grpc",
            on=endpoint,
            inputs=input_docs,
            request_size=-1,
            parameters=parameters,
            return_responses=True,
            continue_on_error=True,
        ):
            payload = parse_response_to_payload(resp, expect_return_value=False)
            del resp
            break  # we only need the first response

        return payload
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise e


async def process_request(
    api_key: str,
    api_tag: str,
    job_id: str,
    payload: Any,
    handler: callable,
    client: Client,
    endpoint: str,
):
    """
    When request is processed, it will be marked as  `STARTED` and then `COMPLETED` or `FAILED`.
    If there is an error, it will be marked as `FAILED` with the error message supplied from the caller.

    Job Lifecycle:
        SCHEDULED -> STARTED -> COMPLETED
        SCHEDULED -> STARTED -> FAILED

    :param api_key: API Key
    :param api_tag: API Tag (e.g. extract, ner, overlay)
    :param job_id:  Job ID
    :param payload:  Payload
    :param handler:  Handler function
    :return:
    """

    status = "OK"
    job_tag = ""

    try:
        logger.info(f"Starting request: {job_id}")
        parameters, input_docs = await parse_payload_to_docs(payload)
        job_tag = parameters["ref_type"] if "ref_type" in parameters else ""
        parameters["job_id"] = job_id
        # payload data attribute should be stripped at this time
        parameters["payload"] = payload  # THIS IS TEMPORARY HERE

        # Currently we are scheduling the job before we start processing the request to avoid out of order jobs
        # When we start processing the request, we will mark the job as `STARTED` in the worker node
        await mark_as_scheduled(
            api_key, job_id, api_tag, job_tag, status, int(time.time()), payload
        )

        await mark_as_started(
            api_key, job_id, api_tag, job_tag, status, int(time.time()), payload
        )

        # async def run(op, _docs, _param):
        #     return await op(_docs, _param)
        # results = await run(handler, input_docs, parameters)
        results = await handler(client, input_docs, parameters, endpoint)

        # client: Client, input_docs, parameters: dict, endpoint: str
        await mark_as_complete(
            api_key, job_id, api_tag, job_tag, status, int(time.time()), results
        )

        del payload
        del parameters
        del input_docs

        return results
    except BaseException as e:
        logger.error(f"processing error : {e}", exc_info=True)
        status = "FAILED"

        # get the traceback and clear the frames to avoid memory leak
        _, val, tb = sys.exc_info()
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
        await mark_as_failed(
            api_key, job_id, api_tag, job_tag, status, int(time.time()), exc
        )
