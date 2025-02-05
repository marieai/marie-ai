from marie.core.instrumentation.events.base import BaseEvent
from marie.core.base.response.schema import RESPONSE_TYPE
from marie.core.schema import QueryType


class QueryStartEvent(BaseEvent):
    """QueryStartEvent.

    Args:
        query (QueryType): Query as a string or query bundle.
    """

    query: QueryType

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "QueryStartEvent"


class QueryEndEvent(BaseEvent):
    """QueryEndEvent.

    Args:
        query (QueryType): Query as a string or query bundle.
        response (RESPONSE_TYPE): Response.
    """

    query: QueryType
    response: RESPONSE_TYPE

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "QueryEndEvent"
