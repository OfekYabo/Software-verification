import networkx as nx
import matplotlib.pyplot as plt
from typing import Set, Dict, Tuple, Union, Optional

State = Union[str, Tuple]  # A state can be a string or a tuple (location, environment)
Action = str  # Actions are represented as strings
Transition = Tuple[State, Action, State]  # (source_state, action, target_state)
LabelingMap = Dict[State, Set[str]]  # Maps states to atomic propositions


class TransitionSystem:
    """
    A Transition System (TS) representation.

    Attributes:
        S (Set[State]): The set of all states (strings or tuples).
        Act (Set[Action]): The set of all possible actions.
        Transitions (Set[Transition]): The set of transitions, each represented as (state_origin, action, state_target).
        I (Set[State]): The set of initial states.
        AP (Set[str]): The set of atomic propositions.
        _L (LabelingMap): A dictionary mapping states to their respective atomic propositions.
    """

    def __init__(
        self,
        states: Optional[Set[State]] = None,
        actions: Optional[Set[Action]] = None,
        transitions: Optional[Set[Transition]] = None,
        initial_states: Optional[Set[State]] = None,
        atomic_props: Optional[Set[str]] = None,
        labeling_map: Optional[LabelingMap] = None,
    ) -> None:
        """
        Initializes the Transition System.

        :param states: A set of states (each a string or a tuple). Defaults to an empty set.
        :param actions: A set of actions. Defaults to an empty set.
        :param transitions: A set of transitions, each as (state_origin, action, state_target). Defaults to an empty set.
        :param initial_states: A set of initial states. Defaults to an empty set.
        :param atomic_props: A set of atomic propositions. Defaults to an empty set.
        :param labeling_map: A dictionary mapping states to sets of atomic propositions. Defaults to an empty dictionary.
        """
        self.S: Set[State] = set(states) if states is not None else set()
        self.Act: Set[Action] = set(actions) if actions is not None else set()
        self.Transitions: Set[Transition] = set(transitions) if transitions is not None else set()
        self.I: Set[State] = set(initial_states) if initial_states is not None else set()
        self.AP: Set[str] = set(atomic_props) if atomic_props is not None else set()
        self._L: LabelingMap = dict(labeling_map) if labeling_map is not None else {}

    def add_state(self, *states: State) -> "TransitionSystem":
        """
        Adds one or more states to the transition system.

        :param states: One or more states (strings or tuples) to be added.
        :return: The TransitionSystem instance (for method chaining).
        """
        self.S.update(states)
        return self

    def add_action(self, *actions: Action) -> "TransitionSystem":
        """
        Adds one or more actions to the transition system.

        :param actions: One or more actions (strings) to be added.
        :return: The TransitionSystem instance (for method chaining).
        """
        self.Act.update(actions)
        return self

    def add_transition(self, *transitions: Transition) -> "TransitionSystem":
        """
        Adds one or more transitions to the transition system.
        Ensures that all involved states and actions exist before adding the transitions.

        Each transition must be provided as a tuple of the form `(state_from, action, state_to)`, where:
        - `state_from` is the source state.
        - `action` is the action performed.
        - `state_to` is the resulting state.

        :param transitions: One or more transitions, each as a tuple `(state_from, action, state_to)`.
        :raises ValueError:
            - If a transition is not a tuple of length 3.
            - If `state_from` or `state_to` does not exist in `self.S`.
            - If `action` is not in `self.Act`.
        :return: The `TransitionSystem` instance (for method chaining).
        """
        for transition in transitions:
            if len(transition) != 3:
                raise ValueError("Each transition must be a tuple of length 3.")
            state_from, action, state_to = transition
            if state_from not in self.S:
                raise ValueError(f"State {state_from} is not in the transition system.")
            if state_to not in self.S:
                raise ValueError(f"State {state_to} is not in the transition system.")
            if action not in self.Act:
                raise ValueError(f"Action {action} is not in the transition system.")
            self.Transitions.add(transition)
        return self

    def add_initial_state(self, *states: State) -> "TransitionSystem":
        """
        Adds one or more states to the set of initial states.

        :param states: One or more states to be marked as initial.
        :raises ValueError: If any state does not exist in the system.
        :return: The TransitionSystem instance (for method chaining).
        """
        for state in states:
            if state not in self.S:
                raise ValueError(f"Initial state {state} must be in the transition system.")
        self.I.update(states)
        return self

    def add_atomic_proposition(self, *props: str) -> "TransitionSystem":
        """
        Adds one or more atomic propositions to the transition system.

        :param props: One or more atomic propositions (strings) to be added.
        :return: The TransitionSystem instance (for method chaining).
        """
        self.AP.update(props)
        return self

    def add_label(self, state: State, *labels: str) -> "TransitionSystem":
        """
        Adds one or more atomic propositions to a given state.

        :param state: The state to label.
        :param labels: One or more atomic propositions to be assigned to the state.
        :raises ValueError: If the state is not in the system or if any label is not a valid atomic proposition.
        :return: The TransitionSystem instance (for method chaining).
        """
        if state not in self.S:
            raise ValueError(f"Cannot set labels for {state}. State is not in the transition system.")
        if state not in self._L:
            self._L[state] = set()
        invalid = set(labels) - self.AP
        if invalid:
            raise ValueError(
                f"Cannot assign labels {invalid}. They are not in the set of atomic propositions (AP)."
            )
        self._L[state].update(labels)
        return self

    def L(self, state: State) -> Set[str]:
        """
        Retrieves the set of atomic propositions that hold in a given state.

        :param state: The state whose atomic propositions are being retrieved.
        :raises ValueError: If the state is not in the transition system.
        :return: A set of atomic propositions associated with the given state.
        """
        if state not in self.S:
            raise ValueError(f"State {state} is not in the transition system.")
        return self._L.get(state, set())

    def pre(self, S: Union[State, Set[State]], action: Optional[Action] = None) -> Set[State]:
        """
        Computes the set of predecessor states from which a given state or set of states can be reached.

        :param S: A single state (string/tuple) or a collection of states.
        :param action: (Optional) If provided, filters only the transitions that use this action.
        :return: A set of predecessor states.
        """
        if isinstance(S, str):
            S = {S}
        predecessors = set()
        for state_from, action_taken, state_to in self.Transitions:
            if action is None or action_taken == action:
                if state_to in S:
                    predecessors.add(state_from)
        return predecessors

    def post(self, S: Union[State, Set[State]], action: Optional[Action] = None) -> Set[State]:
        """
        Computes the set of successor states reachable from a given state or a collection of states.

        :param S: A single state or a collection of states.
        :param action: (Optional) Filters transitions by this action.
        :return: A set of successor states.
        """
        if not isinstance(S, set):
            S = {S}
        successors = set()
        for state_from, action_taken, state_to in self.Transitions:
            if action is None or action_taken == action:
                if state_from in S:
                    successors.add(state_to)
                    print(f"Added {state_to} to successors from {state_from} using action {action_taken}")
        return successors

    def reach(self) -> Set[State]:
        """
        Computes the set of all reachable states from the initial states.

        :return: A set of reachable states.
        """
        # like BFS 
        reachable = set(self.I)
        new_states = set(self.I)
        while new_states:
            new_states = self.post(new_states)
            reachable.update(new_states)
        return reachable

    def is_action_deterministic(self) -> bool:
        """
        Checks whether the transition system is action-deterministic.

        A transition system is action-deterministic if:
        - It has at most one initial state.
        - For each state and action, there is at most one successor state.

        :return: True if the transition system is action-deterministic, False otherwise.
        """
        if len(self.I) > 1:
            return False
        for state in self.S:
            for action in self.Act:
                if len(self.post(state, action)) > 1:
                    return False
        return True

    def is_label_deterministic(self) -> bool:
        """
        Checks whether the transition system is label-deterministic.

        A transition system is label-deterministic if:
        - It has at most one initial state.
        - For each state, the number of reachable successor states is equal to the number of unique label sets
          of these successor states.

        :return: True if the transition system is label-deterministic, False otherwise.
        """
        if len(self.I) > 1:
            return False
        for state in self.S:
            labels = {frozenset(self.L(s)) for s in self.post(state)}
            if len(labels) != len(self.post(state)):
                return False
        return True

    def __repr__(self) -> str:
        """
        Returns a string representation of the Transition System.

        :return: A formatted string representation of the TS.
        """
        return (
            f"TransitionSystem(\n"
            f"  States: {self.S}\n"
            f"  Actions: {self.Act}\n"
            f"  Transitions: {len(self.Transitions)}\n"
            f"  Initial States: {self.I}\n"
            f"  Atomic Propositions: {self.AP}\n"
            f"  Labels: {self._L}\n"
            f")"
        )


    def plot(self, title: str = "Transition System", figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plots the Transition System as a directed graph.

        :param title: Title of the plot.
        :param figsize: Figure size for the plot.
        """
        G = nx.DiGraph()

        # Add nodes (states)
        for state in self.S:
            label = f"{state}\n{' '.join(self.L(state))}" if self.L(state) else str(state)
            print(label)
            G.add_node(state, label=label, color="blue" if state in self.I else "yellow")

        # Add edges (transitions)
        for state_from, action, state_to in self.Transitions:
            G.add_edge(state_from, state_to, label=action)

        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G)  # Positioning algorithm for layout

        # Draw nodes
        node_colors = [G.nodes[n]["color"] for n in G.nodes]
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, "label"), node_color=node_colors, edgecolors="black", node_size=2000, font_size=10)

        # Draw edge labels (actions)
        edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

        plt.title(title)
        # plt.show()
