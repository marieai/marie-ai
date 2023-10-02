from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventMessage:
    """
    Represents an event message that can be sent to a message broker.

    Attributes:
        api_key (str): The API key used to authenticate the message.
        job_id (str): The ID of the job associated with the message.
        event (str): The name of the event associated with the message.
        job_tag (str): A tag associated with the job.
        status (str): The status of the job.
        timestamp (int): The timestamp of the message.
        payload (Any): The payload of the message.
    """

    api_key: str
    job_id: str
    event: str
    job_tag: str
    status: str
    timestamp: int
    payload: Any
