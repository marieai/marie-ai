from graphviz import Digraph


class AdaptiveDFA:
    def __init__(self):
        self.states = set()
        self.transitions = {}
        self.current_state = "BEGIN"
        self.transition_history = []

    def add_state(self, state):
        """Adds a state to the DFA."""
        self.states.add(state)

    def add_transition(self, from_state, to_state, label):
        """Adds a transition from one state to another with a given label."""
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        self.transitions[from_state][to_state] = label

    def process_transitions(self, *states):
        """Processes a sequence of state transitions dynamically."""
        for next_state in states:
            valid_transition = self.transitions.get(self.current_state, {}).get(
                next_state
            )

            if valid_transition:
                self.transition_history.append(
                    (self.current_state, valid_transition, next_state)
                )
                self.current_state = next_state  # ✅ Move to next state
            else:
                print(
                    f"❌ Error: No valid transition from '{self.current_state}' to '{next_state}', stopping."
                )
                break

    def reset(self):
        """Resets the DFA to the BEGIN state and clears the transition history."""
        self.current_state = "BEGIN"
        self.transition_history = []
        print("🔄 DFA has been reset to the BEGIN state.")

    def generate_state_diagram(self, output_file="dfa_diagram"):
        """Generates a graphical representation of the DFA."""
        dot = Digraph()
        for state in self.states:
            shape = "doublecircle" if state == "END" else "circle"
            dot.node(state, shape=shape)

        for from_state, transitions in self.transitions.items():
            for to_state, label in transitions.items():
                dot.edge(from_state, to_state, label=label)

        dot.render(output_file, format="png", cleanup=True)
        print(f"📌 State diagram saved as {output_file}.png")

    def print_transition_history(self):
        """Prints the sequence of transitions taken."""
        print("📜 Transition History:")
        for from_state, label, to_state in self.transition_history:
            print(f"➡️ {from_state} -> {to_state} on '{label}'")
        print(f"🏁 Final State: {self.current_state}")


if __name__ == "__main__":
    dfa = AdaptiveDFA()

    # **Add States**
    dfa.add_state("BEGIN")
    dfa.add_state("START")
    dfa.add_state("STOP")
    dfa.add_state("END")

    #  **Add Transitions**
    dfa.add_transition("BEGIN", "START", "BEGIN_PROCESS")  # ✅ BEGIN → START
    dfa.add_transition("BEGIN", "STOP", "BEGIN_PROCESS")  # ✅ BEGIN → START
    dfa.add_transition("START", "STOP", "VALID")  # ✅ START → STOP
    dfa.add_transition("STOP", "STOP", "INVALID")  # ✅ STOP → STOP
    dfa.add_transition("STOP", "START", "INVALID")  # ✅ STOP → START
    dfa.add_transition("STOP", "END", "FINALIZE")  # ✅ STOP → END
    dfa.add_transition("START", "END", "INVALID")  # ✅ START → END

    # dfa.process_transitions("STOP", "STOP", "STOP", "STOP", "START", "STOP", "END")
    dfa.process_transitions("START", "STOP", "START", "STOP", "START", "STOP", "END")
    dfa.generate_state_diagram()
    dfa.print_transition_history()
