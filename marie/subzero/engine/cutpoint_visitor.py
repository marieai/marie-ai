import logging
from typing import List, Optional

from marie.subzero.continuation.default_continuation import DefaultContinuationStrategy
from marie.subzero.cutpoint.cutpoint_matching_engine import CutpointMatchingEngine
from marie.subzero.engine.base import BaseProcessingVisitor
from marie.subzero.engine.candidate_validator import CandidateValidator
from marie.subzero.models.definition import ExecutionContext, Layer, SelectionType
from marie.subzero.models.results import MatchSection, MatchSectionType, SubzeroResult
from marie.subzero.processor.page_span import PageSpan

LOGGER = logging.getLogger(__name__)


class CutpointProcessingVisitor(BaseProcessingVisitor):
    def __init__(self):
        super().__init__()
        self.matcher = CutpointMatchingEngine()
        self.candidate_validator = CandidateValidator()
        self.continuation = DefaultContinuationStrategy()

    def is_enabled(self) -> bool:
        return True

    def visit(self, context: ExecutionContext, parent: SubzeroResult) -> None:
        template = context.get_template()
        parent.set_pages(context.get_pages())
        layers = template.get_layers()

        if layers:
            for layer in layers:
                self.process_layer(context, layer, parent, None)

    def process_layer(
        self,
        context: ExecutionContext,
        layer: Layer,
        parent: MatchSection,
        parent_layer: Optional[Layer],
    ) -> None:
        assert context is not None
        assert layer is not None
        assert parent is not None

        start_selectors = [layer.start_selector_set]
        stop_selectors = layer.get_stop_selector_sets()

        start_selector_hits = []
        stop_selector_hits = []

        start_candidates = self.matcher.find_cutpoints(
            context, start_selectors, parent, start_selector_hits
        )
        stop_candidates = self.matcher.find_cutpoints(
            context, stop_selectors, parent, stop_selector_hits
        )

        matched_sections = self.candidate_validator.fix_mismatched_sections(
            start_candidates, stop_candidates, parent, layer
        )

        if matched_sections and not matched_sections:
            context.get_selector_hits().remove_all(start_selector_hits)
            context.get_selector_hits().remove_all(stop_selector_hits)

        self.populate_values(layer, matched_sections)
        self.prepare_initial_page_spans(context, matched_sections)

        if layer.get_selection_type() == SelectionType.NEGATION:
            self.process_negative_layer(context, layer, matched_sections)
            return

        self.continuation.find_continuation(
            context, layer, matched_sections, parent_layer
        )

        for ms in matched_sections:
            ms.owner_layer_identifier = layer.get_identifier()
            ms.owner_layer = layer
            ms.parent = parent
            ms.start_candidates = start_selector_hits
            ms.stop_candidates = stop_selector_hits

        child_layers = layer.layers
        if child_layers and matched_sections:
            for found_match_section in matched_sections:
                for child_layer in child_layers:
                    self.process_layer(context, child_layer, found_match_section, layer)

        if matched_sections:
            for sec in matched_sections:
                sec.set_type(MatchSectionType.CONTENT)
                sec.set_owner_layer(layer)
                sec.set_owner_layer_identifier(layer.get_identifier())
                parent.add_section(sec)

    def prepare_initial_page_spans(
        self, context: ExecutionContext, matched_sections: List[MatchSection]
    ) -> None:
        if not matched_sections:
            return

        for section in matched_sections:
            start = section.get_start()
            stop = section.get_stop()
            page_span = PageSpan.create(context, start, stop)
            spans = page_span.spanned_pages
            section.set_span(spans)

    def populate_values(
        self, layer: Layer, matched_sections: List[MatchSection]
    ) -> None:
        if not layer or not matched_sections:
            return

        row_extraction_strategy = layer.get_row_extraction_strategy()
        for ms in matched_sections:
            ms.set_row_extraction_strategy(row_extraction_strategy)

    def process_negative_layer(
        self,
        context: ExecutionContext,
        layer: Layer,
        matched_sections: List[MatchSection],
    ) -> None:
        if not matched_sections:
            return

        tree = context.get_rtree()
        for section in matched_sections:
            spans = section.get_span()
            if spans:
                span = spans[0]
                tree.delete(
                    span.get_page(),
                    0,
                    span.get_y(),
                    4000,
                    # ApplicationConstants.MAX_PAGE_WIDTH,
                    span.get_h(),
                )
