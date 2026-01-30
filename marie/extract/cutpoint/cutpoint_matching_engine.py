from abc import ABC
from collections import defaultdict
from typing import List, Optional

from marie.extract.models.base import CutpointStrategy, Rectangle, Selector
from marie.extract.models.definition import SelectorSet
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import (
    MatchSection,
    ResultType,
    ScanResult,
    ScoredMatchResult,
)
from marie.extract.processor.selector_matcher import SelectorMatcher


class CutpointMatchingEngine:
    handlers = []

    def __init__(self):
        pass

    def find_cutpoints(
        self,
        context: ExecutionContext,
        selector_sets: List[SelectorSet],
        parent: MatchSection,
        selector_hits: Optional[List[ScanResult]] = None,
        cutpoint_strategy: Optional[CutpointStrategy] = None,
    ) -> List[ScanResult]:
        # if cutpoint_strategy == CutpointStrategy.ANNOTATION:
        print("\n===== Start of Cutpoint Matching =====")
        print("Selector Sets:", selector_sets)
        print("Parent:", parent)

        cutpoints = []
        selector_matcher = SelectorMatcher(context, parent)
        handler = AnnotationHandler()
        matcher = AnnotationMatchingEngine()

        for selector_set in selector_sets:
            selectors = selector_set.selectors
            candidates = []

            for selector in selectors:
                results = selector_matcher.visit(selector)
                if results:
                    possibles = handler.find_selector_candidates(
                        selectors, selector, results
                    )
                    if possibles:
                        candidates.extend(possibles)
            if not candidates:
                continue

            # Group candidates by page
            grouped_by_page = defaultdict(list)
            for candidate_group in candidates:
                for candidate in candidate_group:
                    page_id = candidate.line.metadata.page_id
                    grouped_by_page[page_id].append(candidate)

            # Ensure grouped_by_page keys are in ascending order
            grouped_by_page = dict(sorted(grouped_by_page.items()))

            # Print grouped candidates
            print("----------------------------")
            for pageIndex, candidates_on_page in grouped_by_page.items():
                print(f"Page {pageIndex}: {len(candidates_on_page)} candidates")
                cutpoints_by_page = matcher.match(candidates_on_page, selectors)
                print(cutpoints_by_page)

                for smr in cutpoints_by_page:
                    if not smr.items:
                        continue

                    if cutpoint_strategy == CutpointStrategy.ANNOTATION:
                        # smr.items are sorted, find min and max
                        start_sr = smr.items[0]
                        stop_sr = smr.items[-1]

                        # Create cutpoint for start
                        cutpoint_start = ScanResult()
                        cutpoint_start.line = start_sr.line
                        cutpoint_start.page = start_sr.line.metadata.page_id
                        cutpoint_start.type = ResultType.CUTPOINT
                        cutpoint_start.area = Rectangle.create_empty()
                        cutpoints.append(cutpoint_start)

                        # Create cutpoint for stop
                        cutpoint_stop = ScanResult()
                        cutpoint_stop.line = stop_sr.line
                        cutpoint_stop.page = stop_sr.line.metadata.page_id
                        cutpoint_stop.type = ResultType.CUTPOINT
                        cutpoint_stop.area = Rectangle.create_empty()
                        cutpoints.append(cutpoint_stop)
                    else:
                        # TODO : implement findCutpointYOffset
                        sr: ScanResult = smr.items[
                            0
                        ]  # items are already sorted so we pick the first one as the start
                        assert sr.line is not None
                        assert sr.line.metadata is not None

                        cutpoint = ScanResult()
                        cutpoint.line = sr.line
                        cutpoint.page = cutpoint.line.metadata.page_id
                        cutpoint.type = ResultType.CUTPOINT
                        cutpoint.area = (
                            Rectangle.create_empty()
                        )  # TODO : implement findCutpointArea, this could be derived from lines

                        cutpoints.append(cutpoint)

            print("----------------------------")
        print("Cutpoints found:", len(cutpoints))
        print("===== End of Cutpoint Matching =====\n")
        return cutpoints


class AnnotationHandler:
    def __init__(self):
        pass

    def find_selector_candidates(
        self, selectors: List[Selector], selector: Selector, results: List[ScanResult]
    ) -> List[List[ScanResult]]:
        """
        Groups neighboring annotations together based on their proximity.

        Args:
            selectors: A list of Selector objects from which the selector was derived.(SelectorSet)
            selector: Selector object to process.
            results: A list of ScanResult objects to group.

        Returns:
            A list of lists, where each sublist contains grouped neighboring annotations.
        """
        grouped_results = []
        current_group = []
        # Ensure results are sorted by page and then by line
        results = sorted(
            results, key=lambda r: (r.line.metadata.page_id, r.line.metadata.line_id)
        )

        for result in results:
            if not current_group:
                current_group.append(result)
            else:
                last_in_group = current_group[-1]
                if self._are_neighbors(last_in_group, result):
                    current_group.append(result)
                else:
                    grouped_results.append(current_group)
                    current_group = [result]

        if current_group:
            grouped_results.append(current_group)

        return grouped_results

    def _are_neighbors(self, first: ScanResult, second: ScanResult) -> bool:
        """
        Determines if two ScanResult objects are neighbors.

        Args:
            first: The first ScanResult object.
            second: The second ScanResult object.

        Returns:
            True if the two are neighbors, False otherwise.
        """
        # Define neighborhood criteria based on positions
        MAX_LINE_DISTANCE = 4  # Example threshold for line distance

        return (
            first.line.metadata.page_id == second.line.metadata.page_id
            and abs(first.line.metadata.line_id - second.line.metadata.line_id)
            <= MAX_LINE_DISTANCE
        )


class MatchingEngine(ABC):
    def __init__(self):
        pass

    def match(
        self, candidates_by_page: List[ScanResult], selectors: List[Selector]
    ) -> List[ScanResult]:
        """
        Matches the candidates with the selectors.

        Args:
            candidates_by_page: A list of ScanResult objects to match.
            selectors: A list of Selector objects to match against.

        Returns:
            A list of matched ScanResult objects.
        """
        ...


class AnnotationMatchingEngine(MatchingEngine):
    def __init__(self):
        super().__init__()

    def match(
        self, candidates_by_page: List[ScanResult], selectors: List[Selector]
    ) -> List[ScoredMatchResult]:
        """
        Groups all annotations within 4 lines of each other and organizes them into matched ScanResult groups.

        Args:
            candidates_by_page: A list of ScanResult objects to group.
            selectors: A list of Selector objects (not directly used in grouping here).

        Returns:
            A list of grouped ScanResult objects.
        """
        grouped_results = []
        current_group = []

        # Sort candidates first by page, then by line
        candidates_by_page = sorted(
            candidates_by_page,
            key=lambda r: (r.line.metadata.page_id, r.line.metadata.line_id),
        )

        for candidate in candidates_by_page:
            if not current_group:
                current_group.append(candidate)
            else:
                last_in_group = current_group[-1]
                if self._are_neighbors(last_in_group, candidate):
                    current_group.append(candidate)
                else:
                    grouped_results.append(current_group)
                    current_group = [candidate]

        if current_group:
            grouped_results.append(current_group)

        # TODO: THIS SHOULD BE CONFIGURABLE and will accept a callback to make determination if the group is valid
        # TODO: Example would be comparing to a patter, regex, embeddings
        min_selectors_required = len(selectors) // 2
        grouped_results = [
            group for group in grouped_results if len(group) >= min_selectors_required
        ]

        # grouped_output = [candidate for group in grouped_results for candidate in group]
        scored_result = []
        for grouped_result in grouped_results:
            # Sort each group by the line_id to ensure the first item is at the beginning
            # This is important for the cutpoint matching
            grouped_result = sorted(
                grouped_result, key=lambda x: x.line.metadata.line_id
            )

            scored = ScoredMatchResult(
                score=1.0,
                items=grouped_result,
                candidates=[],  # TODO: Remove
            )
            scored_result.append(scored)

        return scored_result

    def _are_neighbors(self, first: ScanResult, second: ScanResult) -> bool:
        """
        Determines if two ScanResult objects are neighbors within 4 lines of each other.

        Args:
            first: The first ScanResult object.
            second: The second ScanResult object.

        Returns:
            True if the two objects are neighbors, False otherwise.
        """
        MAX_LINE_DISTANCE = 4  # TODO : Will need to be configurable

        return (
            first.line.metadata.page_id == second.line.metadata.page_id
            and abs(first.line.metadata.line_id - second.line.metadata.line_id)
            <= MAX_LINE_DISTANCE
        )
