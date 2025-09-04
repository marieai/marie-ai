from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable,
)

from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import MatchSection

# Type alias for region processor functions
RegionProcessorFn = Callable[
    [ExecutionContext, MatchSection, Dict, List[Dict]], List[Dict]
]


@runtime_checkable
class RegionProcessorProto(Protocol):
    """Protocol for region processor classes."""

    def process(
        self,
        context: ExecutionContext,
        parent_section: MatchSection,
        region_parser_config: Dict,
        regions_config: List[Dict],
    ) -> List[Dict]:
        """Process regions and return parsed region data."""
        ...


TRegionProcessor = TypeVar(
    "TRegionProcessor",
    RegionProcessorFn,
    RegionProcessorProto,
    Type[RegionProcessorProto],
)
