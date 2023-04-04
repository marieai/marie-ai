from typing import Dict, Any

from marie.logging.predefined import default_logger
from marie.messaging import Toast
from marie.utils.json import to_json


def event_builder(
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
) -> Dict[str, Any]:
    msg = {
        "jobid": job_id,
        "event": event_name,
        "status": status,
        "jobtag": job_tag,
        "timestamp": timestamp,
        "payload": to_json(payload),
    }
    return msg


async def mark_as_scheduled(
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
):
    """
    Mark request as scheduled for processing, this will be called by when the request is received by the server

    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param event_name: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return:
    """

    default_logger.debug(f"Executing mark_as_scheduled : {job_id} : {timestamp}")
    event = f"{event_name}.scheduled"
    await Toast.notify(
        event,
        event_builder(job_id, event, job_tag, status, timestamp, payload),
    )


async def mark_as_started(
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
):
    """
    Mark request as stared, this will be called when the request has been started by worker process.

    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param event_name: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return:
    """

    default_logger.debug(f"Executing mark_request_as_started : {job_id} : {timestamp}")
    event = f"{event_name}.started"
    await Toast.notify(
        event,
        event_builder(job_id, event, job_tag, status, timestamp, payload),
    )


async def mark_as_failed(
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
):
    """
    Mark request as failed

    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param event_name: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return:
    """

    default_logger.debug(f"Executing mark_request_as_failed : {job_id} : {timestamp}")
    event = f"{event_name}.failed"
    await Toast.notify(
        event,
        event_builder(job_id, event, job_tag, status, timestamp, payload),
    )


async def mark_as_complete(
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
):
    """
    Mark request as completed

    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param event_name: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return:
    """
    default_logger.debug(f"Executing mark_request_as_complete : {job_id} : {timestamp}")
    event = f"{event_name}.completed"
    await Toast.notify(
        event,
        event_builder(job_id, event, job_tag, status, timestamp, payload),
    )
