import logging
from collections import deque
from typing import List, Optional

from marie.extract.adaptive_dfa import AdaptiveDFA, State
from marie.extract.models.definition import Layer
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import (
    LocationType,
    MatchSection,
    ResultType,
    ScanResult,
    TypedScanResult,
)
from marie.extract.models.models import LineModel
from marie.extract.structures import UnstructuredDocument
from marie.extract.structures.line_metadata import LineMetadata
from marie.extract.structures.line_with_meta import LineWithMeta

LOGGER = logging.getLogger(__name__)


def _create_end_of_context(document: UnstructuredDocument) -> TypedScanResult:
    """
    Creates an end-of-context stop cutpoint to ensure all document lines are included.

    This function extracts the metadata from the last line of the given document
    to calculate the location parameters for an end-of-context marker. It returns
    a result specifying a stop location.

    Args:
        document (UnstructuredDocument): The document from the context.

    Returns:
        TypedScanResult: Encapsulates the location type and metadata indicating
        the end-of-context marker.
    """
    last_line = document.lines[-1]
    last_line_meta = last_line.metadata
    page = last_line_meta.page_id
    line = last_line_meta.line_id + 1

    return TypedScanResult(
        location_type=LocationType.STOP,
        type=ResultType.CUTPOINT,
        page=page,
        line=LineWithMeta(
            str(line),
            LineMetadata(
                page_id=page,
                line_id=line,
                model=LineModel(
                    line=line,
                    wordids=[],
                    text="",
                    bbox=[0, 0, 0, 0],
                    confidence=1.0,
                ),
            ),
        ),
    )


class CandidateValidator:

    def fix_mismatched_sections(
        self,
        context: ExecutionContext,
        start_candidates: List[ScanResult],
        stop_candidates: List[ScanResult],
        parent: MatchSection,
        layer: Optional[Layer],
    ) -> List[MatchSection]:
        if not start_candidates and not stop_candidates:
            return []

        assert layer is not None, "Layer cannot be None"
        assert context.document is not None, "Document cannot be None"
        document = context.document

        starts = TypedScanResult.wrap(start_candidates, LocationType.START)
        stops = TypedScanResult.wrap(stop_candidates, LocationType.STOP)

        locations = deque(starts + stops)
        locations = sorted(
            locations, key=lambda x: (x.line.metadata.page_id, x.line.metadata.line_id)
        )

        print('--' * 50)
        print(f"Total locations: {len(locations)}")
        for loc in locations:
            print(
                f"Page {loc.page}, Line {loc.line.metadata.line_id}, Type: {loc.location_type}"
            )
            print(loc)

        # build our adaptive dfa, THIS NEED TO BE CONFIGURED via the config per layer/layout
        first_line = document.lines[0]
        begin = State(
            "BEGIN", ScanResult(page=first_line.metadata.page_id, line=first_line)
        )
        start = State("START")
        continuation = State("CONTINUATION")
        stop = State("STOP")

        end = State("END", _create_end_of_context(document))

        dfa = AdaptiveDFA(initial_state=begin)

        # Add them to DFA
        for state in (begin, start, stop, end, continuation):
            dfa.add_state(state)

        # Setup ROOT transitions
        dfa.add_transition(begin, start, "BEGIN_PROCESS")
        dfa.add_transition(begin, stop, "BEGIN_PROCESS")

        # we can go from start to stop or start to start or start to end
        dfa.add_transition(start, stop, "VALID")
        dfa.add_transition(start, start, "VALID")
        dfa.add_transition(start, end, "VALID")

        # dfa.add_transition(start, continuation, "VALID")
        # dfa.add_transition(continuation, end, "VALID")
        # dfa.add_transition(continuation, start, "VALID")

        # We can't go from stop to start or stop
        dfa.add_transition(stop, stop, "INVALID")
        dfa.add_transition(stop, start, "INVALID")

        dfa.add_transition(stop, end, "FINALIZE")

        # create the dfa states to process the locations
        states = []  # do not add the BEGIN as it is the initial state
        for loc in locations:
            print(
                f"Page {loc.page}, Line {loc.line.metadata.line_id}, Type: {loc.location_type}"
            )
            label = loc.location_type.name
            # state = State(label, f"STATE-{loc.page}-{loc.line.metadata.line_id}")
            state = State(label, loc)
            states.append(state)
        states.append(end)

        dfa.process_transitions(*states)
        dfa.generate_state_diagram()
        dfa.print_transition_history()

        print("\n✅ Valid Transitions:")
        valid_transitions = []
        for t in dfa.get_all_transitions():
            if t.label == "VALID":
                print(f"Step {t.step}: {t.from_state.name} -> {t.to_state.name}")
                valid_transitions.append(t)

        print('Converting to valid transitions to match sections')
        # we will create a match section for each valid transition
        match_sections = []
        for tran in valid_transitions:
            from_state = tran.from_state
            to_state = tran.to_state
            print(
                f"Valid Transition: {from_state.name} -> {to_state.name} : > {from_state.payload} -> {to_state.payload}"
            )
            section = self.extract_match_section(from_state.payload, to_state.payload)
            section.owner_layer = layer

            match_sections.append(section)

        return match_sections

    def extract_match_section(
        self,
        start: TypedScanResult,
        stop: TypedScanResult,
    ) -> MatchSection:
        assert isinstance(start, TypedScanResult)
        assert isinstance(stop, TypedScanResult)

        def filtered_vars(tsr: TypedScanResult):
            return {k: v for k, v in vars(tsr).items() if k != 'location_type'}

        section = MatchSection(label="Cut Section")
        section.start = ScanResult(**filtered_vars(start))
        section.stop = ScanResult(**filtered_vars(stop))

        return section
