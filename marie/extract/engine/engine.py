import logging
from typing import List

from marie.extract.engine.cutpoint_visitor import CutpointProcessingVisitor
from marie.extract.engine.print_visitor import PrintVisitor
from marie.extract.engine.processing_visitor import ProcessingVisitor
from marie.extract.engine.template_validator_visitor import TemplateValidatorVisitor
from marie.extract.models.definition import ExecutionContext
from marie.extract.models.match import SubzeroResult

LOGGER = logging.getLogger(__name__)


class SubzeroEngine:
    """Subzero Engine"""

    def __init__(self):
        self.cleanup_on_shutdown = True
        self.visitors: List[ProcessingVisitor] = []
        self.bootstrap()

    def bootstrap(self):
        """
        Initializes the bootstrap process for the engine.
        """
        # Order of these visitors is important
        self.visitors.append(TemplateValidatorVisitor(True))
        self.visitors.append(CutpointProcessingVisitor())
        # self.visitors.append(MatchSectionExtractionProcessingVisitor(True))
        self.visitors.append(PrintVisitor(True))

    def match(self, context: ExecutionContext) -> SubzeroResult:
        """
        :param context: The execution context that provides necessary information and state for the function to perform its matching logic.
        :return: A SubzeroResult object containing the outcome of the match execution.
        """
        return self.__execute_match(context)

    def __execute_match(self, context: ExecutionContext) -> SubzeroResult:
        self.__validate(context)
        # LifecycleManager.fire(
        #     LifecycleEvent(self.context, LifecycleEventType.TEMPLATE_PROCESSING_STARTED)
        # )

        # Our root node that will be used to aggregate results
        root = SubzeroResult("ROOT")
        for visitor in self.visitors:
            if visitor.is_enabled():
                visitor.pre_visit(context, root)
                visitor.visit(context, root)
                visitor.post_visit(context, root)

        # LifecycleManager.fire(
        #     LifecycleEvent(
        #         self.context, LifecycleEventType.TEMPLATE_PROCESSING_FINISHED
        #     )
        # )

        return root

    @staticmethod
    def __validate(context: ExecutionContext):
        """
        :param context: The execution context containing the template and other metadata.
        :raise ValueError: If the context or the context template is null.
        """
        if context is None or context.template is None:
            raise ValueError(f"Context and Template can't be null :\n {context}")
