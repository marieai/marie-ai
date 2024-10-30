import logging
from typing import List

from marie.subzero.engine.base import BaseProcessingVisitor
from marie.subzero.models.definition import ExecutionContext
from marie.subzero.models.match import (
    MatchField,
    MatchFieldRow,
    MatchSection,
    MatchSectionVisitor,
    SubzeroResult,
)

LOGGER = logging.getLogger(__name__)


# from marie.logging_core.predefined import default_logger as logger


class PrintVisitor(BaseProcessingVisitor):
    """
    Simple Print visitor that dumps results from given parent
    """

    def __init__(self, enabled: bool):
        super().__init__(enabled)

    def visit(self, context: ExecutionContext, parent: SubzeroResult) -> None:
        LOGGER.info("Printing Results")

        class Visitor(MatchSectionVisitor):
            def __init__(self):
                self.depth = 0

            def visit(self, result: MatchSection):
                pad = "  " * (self.depth + 1)

                LOGGER.info(f"{pad} : {self.depth} :: {result}")

                if False:
                    matched_non_repeating_fields = (
                        result.get_matched_non_repeating_fields()
                    )
                    if matched_non_repeating_fields:
                        LOGGER.debug(
                            f"{pad}   Total Fields : {len(matched_non_repeating_fields)}"
                        )
                        for mf in matched_non_repeating_fields:
                            LOGGER.debug(f"      {mf}")

                    matched_field_rows = result.get_matched_field_rows()
                    if matched_field_rows:
                        LOGGER.debug(
                            f"{pad}   Total matched rows : {len(matched_field_rows)}"
                        )
                        for mfr in matched_field_rows:
                            LOGGER.debug(f"{pad}          {mfr}")

                self.depth += 1
                for section in result.sections:
                    self.visit(section)
                self.depth -= 1

        parent.visit(Visitor())
