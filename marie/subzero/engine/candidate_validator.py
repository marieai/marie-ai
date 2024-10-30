import logging
from collections import deque
from typing import List, Optional

from marie.subzero.models.definition import CutpointStrategy, Layer
from marie.subzero.models.results import (
    Location,
    LocationType,
    MatchSection,
    ScanResult,
    TypedScanResult,
)

LOGGER = logging.getLogger(__name__)


class CandidateValidator:
    def fix_mismatched_sections(
        self,
        start_candidates: List[ScanResult],
        stop_candidates: List[ScanResult],
        parent: MatchSection,
        layer: Optional[Layer],
    ) -> List[MatchSection]:
        if not start_candidates and not stop_candidates:
            return []

        starts = TypedScanResult.wrap(start_candidates, LocationType.START)
        stops = TypedScanResult.wrap(stop_candidates, LocationType.STOP)

        locations = deque(starts + stops)
        # locations = sorted(
        #     locations,
        #     key=ComparatorFactory.SCANRESULT_Y_COMPARATOR.get_comparator(
        #         Ordering.ASCENDING
        #     ),
        # )

        current = None
        last = None
        last_type = None

        return []

        if False:
            cutpoint_strategy = CutpointStrategy.START_ON_STOP
            if layer:
                cutpoint_strategy = layer.get_cutpoint_strategy()

            merged_sections = []
            while locations:
                current = locations.popleft()
                current_type = current.location_type
                LOGGER.debug(f"CStatus: {current_type}")

                if current_type == LocationType.START:
                    if (
                        cutpoint_strategy == CutpointStrategy.START_ON_STOP
                        and last_type == LocationType.START
                    ):
                        ca = current.get_area()
                        current_y_offset = current.get_y_offset()
                        stop_y = (
                            ca.y - current_y_offset if current_y_offset >= 0 else ca.y
                        )

                        section = self.extract_match_section(
                            last,
                            current,
                            last.page,
                            last.get_area().y,
                            current.page,
                            stop_y,
                            last.get_x_offset(),
                            last.get_y_offset(),
                        )
                        merged_sections.add(section)

                if current_type == LocationType.STOP:
                    if last_type == LocationType.START:
                        section = self.extract_match_section(
                            last,
                            current,
                            last.page,
                            last.get_area().y,
                            current.page,
                            current.get_area().y,
                            last.get_x_offset(),
                            last.get_y_offset(),
                        )
                        merged_sections.add(section)
                    else:
                        LOGGER.debug(f"Rejecting cutpoint: {current}")

                if not locations and current_type == LocationType.START:
                    stop = parent.get_stop()
                    cutpoint = self.extract_match_section(
                        current,
                        None,
                        current.page,
                        current.get_area().y,
                        stop.page,
                        stop.y,
                        current.get_x_offset(),
                        current.get_y_offset(),
                    )
                    merged_sections.add(cutpoint)

                last = current
                last_type = last.location_type

        return merged_sections

    def extract_match_section(
        self,
        start: TypedScanResult,
        stop: Optional[TypedScanResult],
        start_page: int,
        start_y: float,
        stop_page: int,
        stop_y: float,
        x_offset: int,
        y_offset: int,
    ) -> MatchSection:
        section = MatchSection("Cut Section")
        section.set_x_offset(x_offset)
        section.set_y_offset(y_offset)

        if start_page == stop_page and (stop_y - start_y) < 0:
            start_y, stop_y = stop_y, start_y

        section.set_start(Location(start_page, int(start_y)))
        section.set_stop(Location(stop_page, int(stop_y)))

        if start:
            section.set_start_selector_set_owner_identifier(
                start.get_owner_identifier()
            )
        if stop:
            section.set_stop_selector_set_owner_identifier(stop.get_owner_identifier())

        return section
