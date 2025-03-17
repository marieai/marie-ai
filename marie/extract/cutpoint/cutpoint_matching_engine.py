import logging
from collections import defaultdict
from typing import List, Optional

from marie.extract.continuation.base import ContinuationStrategy
from marie.extract.models.definition import ExecutionContext, Layer, SelectorSet
from marie.extract.models.match import MatchSection, ScanResult
from marie.extract.processor.selector_matcher import SelectorMatcher

LOGGER = logging.getLogger(__name__)


class CutpointMatchingEngine:
    handlers = []

    def __init__(self):
        # self.handlers.append(BlobCutpointMatchingHandler())
        # self.handlers.append(ImageAnchorCutpointMatchingHandler())
        pass

    def find_cutpoints(
        self,
        context: ExecutionContext,
        selector_sets: List[SelectorSet],
        parent: MatchSection,
        selector_hits: Optional[List[ScanResult]] = None,
    ) -> List[ScanResult]:
        cutpoints = []
        selector_matcher = SelectorMatcher(context, parent)

        return cutpoints
