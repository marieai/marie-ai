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


async def mark_as_started(
    job_id: str,
    event_name: str,
    job_tag: str,
    status: str,
    timestamp: int,
    payload: Any,
):
    """
    Mark request as started

    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param event_name: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :param payload:
    :return:
    """

    default_logger.debug(f"Executing mark_request_as_started : {job_id} : {timestamp}")
    await Toast.notify(
        f"{event_name}.started",
        event_builder(
            job_id, f"{event_name}.started", job_tag, status, timestamp, payload
        ),
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
    await Toast.notify(
        f"{event_name}.failed",
        event_builder(job_id, event_name, job_tag, status, timestamp, payload),
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
    await Toast.notify(
        f"{event_name}.completed",
        event_builder(
            job_id, f"{event_name}.completed", job_tag, status, timestamp, payload
        ),
    )
