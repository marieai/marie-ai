from collections import deque
from typing import List

from marie.extract.engine.base import BaseProcessingVisitor
from marie.extract.models.base import Selector
from marie.extract.models.definition import ExecutionContext, Layer
from marie.extract.models.match import MatchSection, SubzeroResult


class MockProcessingVisitor(BaseProcessingVisitor):
    """
    Extract field values from the matched sections.

    """

    def __init__(self, enabled: bool):
        super().__init__(enabled)

    def visit(self, context: ExecutionContext, parent: MatchSection) -> None:

        print("----------------------------------------")
        print("Processing MockProcessingVisitor")

        queue = deque([parent])
        while queue:
            current = queue.popleft()
            if current is None:
                continue
            self.process_section(context, parent, current)
            queue.extend(current.sections)

        print("Finished processing MockProcessingVisitor")
        print("----------------------------------------")

    def process_section(
        self, context: ExecutionContext, parent: MatchSection, section: MatchSection
    ) -> None:
        pass
