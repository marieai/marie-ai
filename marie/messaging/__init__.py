from typing import Dict, Any

from marie.logging.predefined import default_logger
import asyncio

from marie.messaging.publisher import MessagePublisher
from marie.messaging.toast_registry import Toast
from marie.utils.json import to_json


def event_builder(
    job_id: str, api: str, job_tag: str, status: str, timestamp: int, payload: Any
) -> Dict[str, Any]:
    msg = {
        "jobid": job_id,
        "api": api,
        "status": status,
        "jobtag": job_tag,
        "timestamp": timestamp,
        "payload": to_json(payload),
    }

    return msg


async def mark_request_as_started(
    job_id: str, api: str, job_tag: str, status: str, timestamp: int, payload: Any
):
    """
    Mark request as started

    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param api: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return:
    """

    default_logger.info(f"Executing mark_request_as_started : {job_id} : {timestamp}")
    await Toast.notify(
        f"{api}.started",
        event_builder(job_id, api, job_tag, status, timestamp, payload),
    )


async def mark_request_as_failed(
    job_id: str, api: str, job_tag: str, status: str, timestamp: int, payload: Any
):
    """
    Mark request as failed

    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param api: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return:
    """

    default_logger.info(f"Executing mark_request_as_failed : {job_id} : {timestamp}")
    await Toast.notify(
        f"{api}.failed",
        event_builder(job_id, api, job_tag, status, timestamp, payload),
    )


async def mark_request_as_complete(
    job_id: str, api: str, job_tag: str, status: str, timestamp: int, payload: Any
):
    """
    Mark request as complete

    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param api: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return:
    """
    default_logger.info(f"Executing mark_request_as_complete : {job_id} : {timestamp}")
    await Toast.notify(
        f"{api}.complete",
        event_builder(job_id, api, job_tag, status, timestamp, payload),
    )
