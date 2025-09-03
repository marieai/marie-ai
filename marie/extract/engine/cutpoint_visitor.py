from typing import List, Optional

from marie.extract.continuation.default_continuation import DefaultContinuationStrategy
from marie.extract.cutpoint.cutpoint_matching_engine import CutpointMatchingEngine
from marie.extract.engine.base import BaseProcessingVisitor
from marie.extract.engine.candidate_validator import CandidateValidator
from marie.extract.models.base import CutpointStrategy
from marie.extract.models.definition import Layer, SelectionType
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import (
    MatchSection,
    MatchSectionType,
    ScanResult,
    SubzeroResult,
)
from marie.extract.results.span_util import pagespan_from_start_stop


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
        parent.set_pages(context.pages)
        layers = template.layers

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

        matched_sections: List[MatchSection]
        cutpoint_strategy = layer.cutpoint_strategy

        if cutpoint_strategy == CutpointStrategy.ANNOTATION:
            selectors = layer.start_selector_sets
            selector_hits = []
            # Use the matcher to find candidates (annotations)
            candidates = self.matcher.find_cutpoints(
                context, selectors, parent, selector_hits, cutpoint_strategy
            )
            if not candidates:
                matched_sections = []
            else:
                matched_sections = []
                # we expect pairs of [start, stop] candidates from find_cutpoints
                for i in range(0, len(candidates), 2):
                    start_cand = candidates[i]
                    stop_cand = candidates[i + 1]
                    section = MatchSection(start=start_cand, stop=stop_cand)
                    matched_sections.append(section)
        else:
            start_selectors = layer.start_selector_sets
            stop_selectors = layer.stop_selector_sets

            start_selector_hits = []
            stop_selector_hits = []

            start_candidates = self.matcher.find_cutpoints(
                context, start_selectors, parent, start_selector_hits, cutpoint_strategy
            )
            stop_candidates = self.matcher.find_cutpoints(
                context, stop_selectors, parent, stop_selector_hits, cutpoint_strategy
            )

            matched_sections = self.candidate_validator.fix_mismatched_sections(
                context, start_candidates, stop_candidates, parent, layer
            )

        self.populate_values(layer, matched_sections)
        self.prepare_initial_page_spans(context, matched_sections)

        if layer.selection_type == SelectionType.NEGATION:
            self.process_negative_layer(context, layer, matched_sections)
            return

        self.continuation.find_continuation(
            context, layer, matched_sections, parent_layer
        )

        for ms in matched_sections:
            # ms.owner_layer_identifier = layer.get_identifier()
            ms.owner_layer = layer
            ms.parent = parent
            # ms.start_candidates = start_selector_hits
            # ms.stop_candidates = stop_selector_hits

        child_layers = layer.layers
        if child_layers and matched_sections:
            for found_match_section in matched_sections:
                for child_layer in child_layers:
                    self.process_layer(context, child_layer, found_match_section, layer)

        if matched_sections:
            for sec in matched_sections:
                sec.type = MatchSectionType.CONTENT
                sec.owner_layer = layer
                # sec.set_owner_layer_identifier(layer.get_identifier())
                parent.add_section(sec)

    def prepare_initial_page_spans(
        self, context: ExecutionContext, matched_sections: List[MatchSection]
    ) -> None:
        if not matched_sections:
            return

        for section in matched_sections:
            page_span = pagespan_from_start_stop(context, section.start, section.stop)
            section.span = page_span.spanned_pages

    def populate_values(
        self, layer: Layer, matched_sections: List[MatchSection]
    ) -> None:
        if not layer or not matched_sections:
            return

        row_extraction_strategy = layer.row_extraction_strategy
        for ms in matched_sections:
            ms.row_extraction_strategy = row_extraction_strategy

    def process_negative_layer(
        self,
        context: ExecutionContext,
        layer: Layer,
        matched_sections: List[MatchSection],
    ) -> None:
        if not matched_sections:
            pass

        if True:
            raise NotImplemented
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
