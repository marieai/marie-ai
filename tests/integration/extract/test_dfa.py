import pytest

from marie import Flow
from marie.extract.adaptive_dfa import AdaptiveDFA, State


def test_basic_dfa():
    dfa = AdaptiveDFA()

    # Create State instances
    begin = State("BEGIN")
    start = State("START")
    stop = State("STOP")
    end = State("END")

    # Add states
    dfa.add_state(begin)
    dfa.add_state(start)
    dfa.add_state(stop)
    dfa.add_state(end)

    # Add transitions
    dfa.add_transition(begin, start, "BEGIN_PROCESS")
    dfa.add_transition(begin, stop, "BEGIN_PROCESS")
    dfa.add_transition(start, stop, "VALID")
    dfa.add_transition(stop, stop, "INVALID")
    dfa.add_transition(stop, start, "INVALID")
    dfa.add_transition(stop, end, "FINALIZE")
    dfa.add_transition(start, end, "INVALID")

    # Process transitions
    dfa.process_transitions(start, stop, start, stop, start, stop, end)

    dfa.generate_state_diagram()
    dfa.print_transition_history()


def test_basic_dfa_001():
    dfa = AdaptiveDFA()

    # Create State instances
    begin = State("BEGIN")
    start = State("START")
    stop = State("STOP")
    end = State("END")

    # Add them to DFA
    for state in (begin, start, stop, end):
        dfa.add_state(state)

    # Add transitions
    dfa.add_transition(begin, start, "BEGIN_PROCESS")
    dfa.add_transition(begin, stop, "BEGIN_PROCESS")

    dfa.add_transition(start, stop, "VALID")
    dfa.add_transition(start, start, "VALID")

    dfa.add_transition(stop, stop, "INVALID")
    dfa.add_transition(stop, start, "INVALID")

    dfa.add_transition(stop, end, "FINALIZE")
    dfa.add_transition(start, end, "FINALIZE")

    # Process transitions
    dfa.process_transitions(start, start, start, end)

    dfa.generate_state_diagram()
    dfa.print_transition_history()
