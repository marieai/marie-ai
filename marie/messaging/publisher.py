from typing import Any

from marie.logging.predefined import default_logger as logger
from marie.messaging import Toast
from marie.messaging.events import EventMessage
from marie.utils.json import to_json


def event_builder(
    api_key: str,
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
) -> EventMessage:
    """Build the event message"""
    return EventMessage(
        **{
            "api_key": api_key,
            "jobid": job_id,
            "event": event_name,
            "status": status,
            "jobtag": job_tag,
            "timestamp": timestamp,
            "payload": to_json(payload),
        }
    )


async def mark_as_scheduled(
    api_key: str,
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
):
    """
    Mark request as scheduled for processing, this will be called by when the request is received by the server

    :param api_key: The API key that is used to authenticate the request.
    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param event_name: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return:
    """

    logger.debug(f"Executing mark_as_scheduled : {job_id} : {timestamp}")
    event = f"{event_name}.scheduled"
    await Toast.notify(
        event,
        event_builder(api_key, job_id, event, job_tag, status, timestamp, payload),
    )


async def mark_as_started(
    api_key: str,
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
):
    """
    Mark request as stared, this will be called when the request has been started by worker process.

    :param api_key: The API key that is used to authenticate the request.
    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param event_name: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return:
    """

    logger.debug(f"Executing mark_request_as_started : {job_id} : {timestamp}")
    event = f"{event_name}.started"
    await Toast.notify(
        event,
        event_builder(api_key, job_id, event, job_tag, status, timestamp, payload),
    )


async def mark_as_failed(
    api_key: str,
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
):
    """
    Mark request as failed

    :param api_key: The API key that is used to authenticate the request.
    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param event_name: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return:
    """

    logger.debug(f"Executing mark_request_as_failed : {job_id} : {timestamp}")
    event = f"{event_name}.failed"
    await Toast.notify(
        event,
        event_builder(api_key, job_id, event, job_tag, status, timestamp, payload),
    )


async def mark_as_complete(
    api_key: str,
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
):
    """
    Mark request as completed

    :param api_key: The API key that is used to authenticate the request.
    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param event_name: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return:
    """
    logger.debug(f"Executing mark_request_as_complete : {job_id} : {timestamp}")
    event = f"{event_name}.completed"
    await Toast.notify(
        event,
        event_builder(api_key, job_id, event, job_tag, status, timestamp, payload),
    )
