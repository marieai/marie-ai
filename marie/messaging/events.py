from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, NamedTuple, Optional

from marie import check
from marie._core.definitions.metadata import MetadataValue, normalize_metadata
from marie.utils.error import SerializableErrorInfo


class MarieEventType(str, Enum):
    """The types of events that may be yielded by op and job execution."""

    # Resource initialization for execution has started/succeeded/failed.
    RESOURCE_INIT_STARTED = "RESOURCE_INIT_STARTED"
    RESOURCE_INIT_SUCCESS = "RESOURCE_INIT_SUCCESS"
    RESOURCE_INIT_FAILURE = "RESOURCE_INIT_FAILURE"

    STEP_OUTPUT = "STEP_OUTPUT"
    STEP_INPUT = "STEP_INPUT"
    STEP_FAILURE = "STEP_FAILURE"
    STEP_START = "STEP_START"
    STEP_SUCCESS = "STEP_SUCCESS"
    STEP_SKIPPED = "STEP_SKIPPED"

    ASSET_MATERIALIZATION = "ASSET_MATERIALIZATION"
    ASSET_OBSERVATION = "ASSET_OBSERVATION"

    RUN_ENQUEUED = "RUN_ENQUEUED"
    RUN_DEQUEUED = "RUN_DEQUEUED"
    RUN_STARTING = "RUN_STARTING"
    RUN_START = "RUN_START"
    RUN_SUCCESS = "RUN_SUCCESS"
    RUN_FAILURE = "RUN_FAILURE"
    RUN_CANCELING = "RUN_CANCELING"
    RUN_CANCELED = "RUN_CANCELED"

    ENGINE_EVENT = "ENGINE_EVENT"


FAILURE_EVENTS = {
    MarieEventType.RUN_FAILURE,
    MarieEventType.STEP_FAILURE,
    MarieEventType.RUN_CANCELED,
}

MARKER_EVENTS = {
    MarieEventType.ENGINE_EVENT,
    MarieEventType.RESOURCE_INIT_STARTED,
    MarieEventType.RESOURCE_INIT_SUCCESS,
    MarieEventType.RESOURCE_INIT_FAILURE,
}

ASSET_EVENTS = {
    MarieEventType.ASSET_MATERIALIZATION,
    MarieEventType.ASSET_OBSERVATION,
}


@dataclass(frozen=True)
class EventMessage:
    """
    Represents an event message that can be sent to a message broker.

    Attributes:
        api_key (str): The API key used to authenticate the message.
        jobid (str): The ID of the job associated with the message.
        event (str): The name of the event associated with the message.
        jobtag (str): A tag associated with the job.
        status (str): The status of the job.
        timestamp (int): The timestamp of the message.
        payload (Any): The payload of the message.
    """

    api_key: str
    jobid: str
    event: str
    jobtag: str
    status: str
    timestamp: int
    payload: Any


# @dataclass(frozen=True)


@dataclass(frozen=False)
class EngineEventData:
    metadata: Mapping[str, MetadataValue]
    error: Optional[SerializableErrorInfo]
    marker_start: Optional[str]
    marker_end: Optional[str]

    def __init__(
        self,
        metadata: Optional[Mapping[str, MetadataValue]] = None,
        error: Optional[SerializableErrorInfo] = None,
        marker_start: Optional[str] = None,
        marker_end: Optional[str] = None,
    ):
        self.metadata = normalize_metadata(
            check.opt_mapping_param(metadata, "metadata", key_type=str)
        )
        self.error = check.opt_inst_param(error, "error", SerializableErrorInfo)
        self.marker_start = check.opt_str_param(marker_start, "marker_start")
        self.marker_end = check.opt_str_param(marker_end, "marker_end")

    @staticmethod
    def engine_error(error: SerializableErrorInfo) -> "EngineEventData":
        return EngineEventData(metadata={}, error=error)


class MarieEvent:
    @staticmethod
    def engine_event(
        message: str,
        event_specific_data: Optional["EngineEventData"] = None,
    ) -> "MarieEvent":
        print("MarieEvent.engine_event")
        print(message)
        print(event_specific_data)
        return None
        # return MarieEvent.from_job(
        #     MarieEventType.ENGINE_EVENT,
        #     message,
        #     event_specific_data=event_specific_data,
        # )
