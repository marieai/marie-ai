import logging
from typing import List

from marie.extract.engine.cutpoint_visitor import CutpointProcessingVisitor
from marie.extract.engine.match_section_extract_visitor import (
    MatchSectionExtractionProcessingVisitor,
)
from marie.extract.engine.match_section_region_processor_visitor import (
    MatchSectionRegionProcessorVisitor,
)
from marie.extract.engine.processing_visitor import ProcessingVisitor
from marie.extract.engine.template_validator_visitor import TemplateValidatorVisitor
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import SubzeroResult
from marie.extract.registry import component_registry

LOGGER = logging.getLogger(__name__)


class DocumentExtractEngine:
    """Subzero Engine"""

    def __init__(self, processing_visitors: List[str] = None):
        """
        Initializes the class instance, preparing it for use and optionally bootstrapping
        processing visitors. This includes setting up initial configurations and initializing
        visitors if a list is provided during instantiation.

        Parameters:
            processing_visitors: Optional[List[str]]
                An ORDERED list of REGISTERED processing visitors. Defaults to None..
        """
        self.cleanup_on_shutdown = True
        self.visitors: List[ProcessingVisitor] = []
        self.bootstrap(processing_visitors)

    def bootstrap(self, visitor_names: List[str] = None):
        """
        Bootstraps a series of processing visitors to be applied in a specific order. Visitors include
        the core set (defined internally) and a custom list from the user. The method ensures that
        essential visitors are added, and the provided custom visitors are validated  through the
        component registry.

        Parameters:
            visitor_names: Optional[List[str]]
                A list of visitor class names to add to the processing pipeline. If not provided, the method
                will use the default core visitors.
        Raises:
            ValueError: If a visitor specified in the `visitors` list is not registered in the component registry.
        """
        # Order of these visitors is important
        # TODO : This should be configurable so the client can add change/visitors
        self.visitors.append(TemplateValidatorVisitor(True))
        self.visitors.append(CutpointProcessingVisitor())
        self.visitors.append(MatchSectionRegionProcessorVisitor(True))
        self.visitors.append(MatchSectionExtractionProcessingVisitor(True))
        # self.visitors.append(MatchSectionRenderingVisitor(True))
        # self.visitors.append(PrintVisitor(True))

        if visitor_names is None:
            LOGGER.warning(
                "No additional visitors specified. Using core visitors only."
            )
            return

        LOGGER.info(f"Bootstrapping visitors: {visitor_names}")
        for name in visitor_names:
            visitor_cls = component_registry.get_processing_visitor(name)
            if visitor_cls is None:
                LOGGER.error(f"Visitor '{name}' not found in registry.")
                raise ValueError(f"Visitor '{name}' not found in registry.")
            self.visitors.append(visitor_cls(enabled=True))

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
