"""
Base Exporter - Protocol definition for LLM tracking exporters.

Exporters are responsible for sending events to their destination
(console, RabbitMQ, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Protocol, runtime_checkable

from marie.llm_tracking.types import Observation, QueueMessage, RawEvent, Score, Trace


@runtime_checkable
class BaseExporter(Protocol):
    """
    Protocol for LLM tracking event exporters.

    Exporters must implement methods to export traces, observations,
    and scores, as well as lifecycle management.
    """

    def start(self) -> None:
        """Initialize the exporter (open connections, etc.)."""
        ...

    def stop(self) -> None:
        """Shutdown the exporter (close connections, flush, etc.)."""
        ...

    def export_trace(self, trace: Trace) -> None:
        """
        Export a single trace.

        Args:
            trace: Trace to export
        """
        ...

    def export_traces(self, traces: List[Trace]) -> None:
        """
        Export multiple traces (batched).

        Args:
            traces: List of traces to export
        """
        ...

    def export_observation(self, observation: Observation) -> None:
        """
        Export a single observation.

        Args:
            observation: Observation to export
        """
        ...

    def export_observations(self, observations: List[Observation]) -> None:
        """
        Export multiple observations (batched).

        Args:
            observations: List of observations to export
        """
        ...

    def export_score(self, score: Score) -> None:
        """
        Export a single score.

        Args:
            score: Score to export
        """
        ...

    def export_scores(self, scores: List[Score]) -> None:
        """
        Export multiple scores (batched).

        Args:
            scores: List of scores to export
        """
        ...

    def flush(self) -> None:
        """Flush any buffered events."""
        ...


class AbstractExporter(ABC):
    """
    Abstract base class for exporters.

    Provides default implementations for batch methods that call
    single-item methods. Subclasses can override for efficiency.
    """

    def start(self) -> None:
        """Initialize the exporter. Override if needed."""
        pass

    def stop(self) -> None:
        """Shutdown the exporter. Override if needed."""
        pass

    @abstractmethod
    def export_trace(self, trace: Trace) -> None:
        """Export a single trace."""
        pass

    def export_traces(self, traces: List[Trace]) -> None:
        """Export multiple traces. Default: call export_trace for each."""
        for trace in traces:
            self.export_trace(trace)

    @abstractmethod
    def export_observation(self, observation: Observation) -> None:
        """Export a single observation."""
        pass

    def export_observations(self, observations: List[Observation]) -> None:
        """Export multiple observations. Default: call export_observation for each."""
        for observation in observations:
            self.export_observation(observation)

    @abstractmethod
    def export_score(self, score: Score) -> None:
        """Export a single score."""
        pass

    def export_scores(self, scores: List[Score]) -> None:
        """Export multiple scores. Default: call export_score for each."""
        for score in scores:
            self.export_score(score)

    def flush(self) -> None:
        """Flush any buffered events. Override if needed."""
        pass
