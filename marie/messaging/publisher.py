from typing import Any, Literal, Optional

from marie.logging_core.predefined import default_logger as logger
from marie.messaging import Toast
from marie.messaging.events import EventMessage, MarieEventType
from marie.utils.json import to_json

TOAST_DISABLED = False
STATUS_SCHEDULED = "scheduled"
STATUS_STARTED = "started"
STATUS_FAILED = "failed"
STATUS_COMPLETED = "completed"


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


async def _mark_job_status(
    api_key: str,
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
    status_suffix: str,
    disabled_return_value: bool = True,
) -> bool:
    """
    Common implementation for marking job status

    :param api_key: The API key that is used to authenticate the request
    :param job_id: The unique identifier that is assigned to the job
    :param event_name: The operation used to analyze the input document
    :param job_tag: The user-specified identifier for the job
    :param status: The status of the job (Succeeded, Failed, Error)
    :param timestamp: The Unix timestamp in milliseconds
    :param payload: Additional data for the notification
    :param status_suffix: The status suffix to append to event_name
    :param disabled_return_value: Value to return when TOAST_DISABLED is True
    :return: True if the event was sent successfully, False otherwise
    """
    if TOAST_DISABLED:
        logger.info(
            f"Executing mark_as_{status_suffix} DISABLED : {job_id} : {timestamp}"
        )
        return disabled_return_value

    logger.debug(f"Executing mark_request_as_{status_suffix} : {job_id} : {timestamp}")
    event = f"{event_name}.{status_suffix}"
    try:
        await Toast.notify(
            event,
            event_builder(api_key, job_id, event, job_tag, status, timestamp, payload),
        )
        return True
    except Exception as e:
        logger.error(f"Error sending event: {e}")
        return False


async def mark_as_scheduled(
    api_key: str,
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
) -> bool:
    """
    Mark request as scheduled for processing, this will be called by when the request is received by the server

    :param api_key: The API key that is used to authenticate the request.
    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param event_name: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return: True if the event was sent successfully, False otherwise
    """
    return await _mark_job_status(
        api_key,
        job_id,
        event_name,
        job_tag,
        status,
        timestamp,
        payload,
        status_suffix=STATUS_SCHEDULED,
        disabled_return_value=True,
    )


async def mark_as_started(
    api_key: str,
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
) -> bool:
    """
    Mark request as stared, this will be called when the request has been started by worker process.

    :param api_key: The API key that is used to authenticate the request.
    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param event_name: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return: True if the event was sent successfully, False otherwise
    """
    return await _mark_job_status(
        api_key,
        job_id,
        event_name,
        job_tag,
        status,
        timestamp,
        payload,
        status_suffix=STATUS_STARTED,
        disabled_return_value=False,
    )


async def mark_as_failed(
    api_key: str,
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
) -> bool:
    """
    Mark request as failed

    :param api_key: The API key that is used to authenticate the request.
    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param event_name: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return: True if the event was sent successfully, False otherwise
    """
    return await _mark_job_status(
        api_key,
        job_id,
        event_name,
        job_tag,
        status,
        timestamp,
        payload,
        status_suffix=STATUS_FAILED,
        disabled_return_value=True,
    )


async def mark_as_complete(
    api_key: str,
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
) -> bool:
    """
    Mark request as completed

    :param api_key: The API key that is used to authenticate the request.
    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param event_name: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return: True if the event was sent successfully, False otherwise
    """
    return await _mark_job_status(
        api_key,
        job_id,
        event_name,
        job_tag,
        status,
        timestamp,
        payload,
        status_suffix=STATUS_COMPLETED,
        disabled_return_value=False,
    )
