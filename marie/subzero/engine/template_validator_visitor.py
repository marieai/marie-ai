from collections import deque
from typing import List

from marie.subzero.engine.base import BaseProcessingVisitor
from marie.subzero.models.base import Selector
from marie.subzero.models.definition import ExecutionContext, Layer
from marie.subzero.models.results import SubzeroResult


class TemplateValidatorVisitor(BaseProcessingVisitor):
    """
    Validate the template to make sure that it is valid for processing.
    """

    STRICT_VALIDATION = True

    def __init__(self, enabled: bool):
        super().__init__(enabled)

    def visit(self, context: ExecutionContext, parent: SubzeroResult) -> None:
        template = context.template
        if template is None or template.layers is None:
            raise ValueError("Template or Template layers cannot be None")

        stack = deque(template.layers)

        while stack:
            layer = stack.pop()
            if layer:
                layers = layer.layers
                self.validate(context, layer)
                stack.extend(layers)

    def validate(self, context: ExecutionContext, layer: Layer) -> None:
        sets = [[layer.start_selector_set], layer.stop_selector_sets]

        for set_list in sets:
            for ss in set_list:
                ss.selectors = self.validate_selectors(context, ss.selectors)

    def validate_selectors(
        self, context: ExecutionContext, selectors: List[Selector]
    ) -> List[Selector]:
        return selectors
        # fixed = []

        # for s in selectors:
        #     if isinstance(s, ImageSelector):
        #         anchor = s.anchor
        #         if anchor is None:
        #             LifecycleManager.fire(
        #                 LifecycleEvent(
        #                     context,
        #                     LifecycleEventType.TEMPLATE_VALIDATION_WARNING,
        #                     f"Anchor for image selector {s.identifier} is null",
        #                 )
        #             )
        #             continue
        #
        #     search_perimeter = s.search_perimeter
        #
        #     if search_perimeter is None:
        #         LifecycleManager.fire(
        #             LifecycleEvent(
        #                 context,
        #                 LifecycleEventType.TEMPLATE_VALIDATION_WARNING,
        #                 f"Perimeter for selector {s.identifier} is null, using defaults",
        #             )
        #         )
        #         s.search_perimeter = Perimeter(0, ApplicationConstants.MAX_PAGE_WIDTH)
        #     else:
        #         if search_perimeter.x_min < 0:
        #             search_perimeter.x_min = 0
        #         if search_perimeter.x_max > ApplicationConstants.MAX_PAGE_WIDTH:
        #             search_perimeter.x_max = ApplicationConstants.MAX_PAGE_WIDTH
        #
        #     fixed.append(s)
        # return fixed
