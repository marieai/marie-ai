from typing import Dict

from marie.logging.predefined import default_logger
import asyncio

from marie.messaging.publisher import MessagePublisher


async def mark_request_as_complete(
    job_id: str, api: str, job_tag: str, status: str, timestamp: int
):
    """
    Mark request as complete

    :param job_id:  The unique identifier that is assigned to the job.
    :param status: The status of the job. Valid values are Succeeded, Failed, or Error.
    :param api: The operation used to analyze the input document, such as Extract or Overlay.
    :param job_tag: The user-specified identifier for the job(ex: ref_type)
    :param timestamp: The Unix timestamp that indicates when the job finished, returned in milliseconds.
    :return:
    """
    default_logger.info(f"Executing mark_request_as_complete : {job_id} : {timestamp}")

    def build_doc_location(job_id: str) -> Dict:
        return {"S3ObjectName": "String", "S3Bucket": "String"}

    msg = {
        "jobid": job_id,
        "api": api,
        "status": status,
        "jobtag": job_tag,
        "timestamp": timestamp,
        "document": build_doc_location(job_id),
    }

    print(msg)

    task = asyncio.ensure_future(MessagePublisher.publish(msg))
    sync = False

    if sync:
        results = await asyncio.gather(task)
        return results[0]
