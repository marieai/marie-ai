from typing import List

from marie.extract.models.base import Rectangle, Selector, TextSelector
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import MatchSection, ResultType, ScanResult
from marie.extract.structures.concrete_annotations import TypedAnnotation


class SelectorMatcher:
    OVERLAP_DEFAULT_CUTOFF = 0.40

    def __init__(self, context: ExecutionContext, parent: MatchSection):
        assert context.document is not None, "Document cannot be None"
        assert context.template is not None, "Template cannot be None"
        self.context = context
        self.parent = parent

    def visit(self, selector: Selector) -> List[ScanResult]:
        print(f"Checking selector: {selector}")
        doc = self.context.document
        # start = self.parent.start
        # stop = self.parent.stop

        # we are only interested in text selectors for now
        if not isinstance(selector, TextSelector):
            raise ValueError(
                f"Selector type must be 'TextSelector', but got '{selector}'"
            )

        if selector.strategy != "ANNOTATION":
            raise ValueError(
                f"Selector type must be 'ANNOTATOR', but got '{selector.type}'"
            )

        # page_span = PageSpan.create(self.context, start, stop)
        # spanned_pages = (
        #     self.parent.span if self.parent.span else page_span.spanned_pages
        # )

        results = []

        for page_id in range(doc.page_count):
            lines = doc.lines_for_page(page_id)
            for meta_line in lines:
                if meta_line.annotations is None or len(meta_line.annotations) == 0:
                    continue
                for annotation in meta_line.annotations:
                    if isinstance(annotation, TypedAnnotation):
                        # TODO : Check if the annotation is within the selector's span
                        if (
                            True
                            or annotation.annotation_type == selector.annotation_type
                        ):
                            if annotation.name == selector.text:
                                sr = ScanResult(
                                    page=page_id,
                                    type=ResultType.ANNOTATION,
                                    area=Rectangle.create_empty(),
                                    confidence=(
                                        annotation.confidence
                                        if hasattr(annotation, 'confidence')
                                        else 1.0
                                    ),
                                    x_offset=0,
                                    y_offset=0,
                                    selection_type="POSITIVE",
                                    line=meta_line,
                                )

                                results.append(sr)
                    else:
                        raise ValueError(f"Unknown annotation type : {annotation}")
        return results
