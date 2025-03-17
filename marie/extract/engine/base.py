import logging
from collections import defaultdict
from typing import DefaultDict, Dict, List

from marie.extract.engine.processing_visitor import ProcessingVisitor
from marie.extract.models.definition import ExecutionContext, Layer

# from marie.extract.event import LifecycleEvent, LifecycleEventType, LifecycleManager
from marie.extract.models.match import (
    MatchFieldRow,
    MatchSection,
    MatchSectionVisitor,
    SubzeroResult,
)

LOGGER = logging.getLogger(__name__)


class BaseProcessingVisitor(ProcessingVisitor):
    """
    Base class for ProcessingVisitor
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def pre_visit(self, context: "ExecutionContext", parent: SubzeroResult) -> None:
        # LifecycleManager.fire(
        #     LifecycleEvent(context, LifecycleEventType.PROCESSOR_STARTED)
        # )
        LOGGER.debug(f"\n ----------  preVisit {self.__class__.__name__} ---------- \n")

    def post_visit(self, context: "ExecutionContext", parent: SubzeroResult) -> None:
        # LifecycleManager.fire(
        #     LifecycleEvent(context, LifecycleEventType.PROCESSOR_ENDED)
        # )
        LOGGER.debug(
            f"\n ----------  postVisit {self.__class__.__name__} ---------- \n"
        )

    def is_enabled(self) -> bool:
        return self.enabled

    def collect_match_field_rows(self, match: SubzeroResult) -> List[MatchFieldRow]:
        if match is None:
            return []

        collected_rows = []

        def visit_sections(section: MatchSection):
            matched_rows = section.get_matched_field_rows()
            collected_rows.extend(matched_rows)
            for row in matched_rows:
                collected_rows.extend(row.get_children())
            for subsection in section.get_sections():
                visit_sections(subsection)

        match.visit(lambda section: visit_sections(section))

    def get_sections_by_layer(
        self, result: SubzeroResult
    ) -> DefaultDict[Layer, List[MatchSection]]:
        sections_by_layer = defaultdict(list)

        class Visitor(MatchSectionVisitor):
            def visit(self, result: MatchSection):
                LOGGER.debug("-------- SECTION -------------")

                layer = result.get_owner_layer()
                if layer:
                    sections_by_layer[layer].append(result)

                sections = result.get_sections()
                if sections:
                    for section in sections:
                        self.visit(section)

        result.visit(Visitor())
        return sections_by_layer
