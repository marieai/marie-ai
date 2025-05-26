import json
import os
from dataclasses import dataclass
from typing import List

from graphviz import Digraph


@dataclass
class State:
    def __init__(self, name, payload=None):
        self.name = name
        self.payload = payload

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        """State equality based on name only.
        As the name is unique, we can use it to identify the state. while the payload will change
        """
        return isinstance(other, State) and self.name == other.name

    def __repr__(self):
        return f"State(name={self.name}, payload={self.payload})"


@dataclass
class TransitionRecord:
    from_state: State
    to_state: State
    label: str
    step: int


class AdaptiveDFA:
    def __init__(self, initial_state: State = None):
        self.states = set()
        self.transitions = {}
        self.current_state = State("BEGIN") if initial_state is None else initial_state
        self.transition_history: List[TransitionRecord] = []
        self._step_counter = 0

    def add_state(self, state: State):
        self.states.add(state)

    def add_transition(self, from_state: State, to_state: State, label: str):
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        self.transitions[from_state][to_state] = label

    def process_transitions(self, *states: State):
        for next_state in states:
            label = self.transitions.get(self.current_state, {}).get(next_state)

            if label:
                self._step_counter += 1
                record = TransitionRecord(
                    from_state=self.current_state,
                    to_state=next_state,
                    label=label,
                    step=self._step_counter,
                )
                self.transition_history.append(record)
                self.current_state = next_state
            else:
                print(
                    f"❌ Error: No valid transition from '{self.current_state.name}' to '{next_state.name}', stopping."
                )
                break

    def reset(self):
        self.current_state = State("BEGIN")
        self.transition_history.clear()
        self._step_counter = 0
        print("🔄 DFA has been reset to the BEGIN state.")

    def generate_state_diagram(self, output_file="dfa_diagram"):
        dot = Digraph()
        for state in self.states:
            shape = "doublecircle" if state.name == "END" else "circle"
            dot.node(state.name, shape=shape)

        for from_state, transitions in self.transitions.items():
            for to_state, label in transitions.items():
                dot.edge(from_state.name, to_state.name, label=label)

        output_file = os.path.expanduser("~/dfa_diagram.png")
        dot.render(output_file, format="png", cleanup=True)
        print(f"📌 State diagram saved as {output_file}.png")

    def print_transition_history(self):
        print("📜 Transition History:")
        for record in self.transition_history:
            print(
                f"[{record.step}] ➡️ {record.from_state.name} -> {record.to_state.name} on '{record.label}'"
            )
        print(f"🏁 Final State: {self.current_state.name}")

    def get_all_transitions(self) -> List[TransitionRecord]:
        return self.transition_history

    def to_json(self) -> str:
        """Convert the DFA graph to a JSON representation."""
        dfa_representation = {
            "states": [state.name for state in self.states],
            "transitions": [
                {
                    "from_state": from_state.name,
                    "to_state": to_state.name,
                    "label": label,
                }
                for from_state, transitions in self.transitions.items()
                for to_state, label in transitions.items()
            ],
            "initial_state": self.current_state.name,
        }
        return json.dumps(dfa_representation, indent=4)

    def from_json(json_str: str) -> 'AdaptiveDFA':
        """Load the DFA graph from a JSON representation."""
        data = json.loads(json_str)
        dfa = AdaptiveDFA(State(data["initial_state"]))
        for state_name in data["states"]:
            dfa.add_state(State(state_name))
        for transition in data["transitions"]:
            from_state = State(transition["from_state"])
            to_state = State(transition["to_state"])
            label = transition["label"]
            dfa.add_transition(from_state, to_state, label)
        return dfa


if __name__ == "__main__":
    dfa = AdaptiveDFA()

    # Define states without payloads for simplicity
    begin = State("BEGIN")
    start = State("START")
    stop = State("STOP")
    end = State("END")

    # Add them to DFA
    for state in (begin, start, stop, end):
        dfa.add_state(state)

    # Define transitions
    dfa.add_transition(begin, start, "BEGIN_PROCESS")
    dfa.add_transition(begin, stop, "BEGIN_PROCESS")
    dfa.add_transition(start, stop, "VALID")
    dfa.add_transition(stop, stop, "INVALID")
    dfa.add_transition(stop, start, "INVALID")
    dfa.add_transition(stop, end, "FINALIZE")
    dfa.add_transition(start, end, "INVALID")

    # Run transitions
    dfa.process_transitions(start, stop, start, stop, end)
    dfa.generate_state_diagram()
    dfa.print_transition_history()

    # Example: custom processing on transitions
    print("\n🧠 Custom Processing: Transitions that were 'INVALID':")
    for t in dfa.get_all_transitions():
        if t.label == "INVALID":
            print(f"Step {t.step}: {t.from_state.name} -> {t.to_state.name}")

    # output all the valid transitions
    print("\n✅ Valid Transitions:")
    for t in dfa.get_all_transitions():
        if t.label == "VALID":
            print(f"Step {t.step}: {t.from_state.name} -> {t.to_state.name}")
