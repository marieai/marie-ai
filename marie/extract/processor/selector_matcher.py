import logging
from typing import List, Optional

from marie.extract.models.base import Selector
from marie.extract.models.definition import ExecutionContext, Layer, SelectorSet
from marie.extract.models.match import MatchSection, ScanResult, ScoredMatchResult
from marie.extract.processor.page_span import PageSpan

LOGGER = logging.getLogger(__name__)


class SelectorMatcher:
    OVERLAP_DEFAULT_CUTOFF = 0.40

    def __init__(self, context: ExecutionContext, parent: MatchSection):
        self.context = context
        self.parent = parent

    def visit(self, selector: Selector) -> List[ScanResult]:
        start = self.parent.start
        stop = self.parent.stop

        page_span = PageSpan.create(self.context, start, stop)
        spanned_pages = (
            self.parent.span if self.parent.span else page_span.spanned_pages
        )

        results = []
        futures = []
        result = ScoredMatchResult()

        return results
