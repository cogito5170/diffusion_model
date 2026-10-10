"""Deterministic workflow states and the allowed transitions between them.

Creative Brief -> Concept Generation -> Art Direction -> Schema Validation -> Prompt
Compilation -> Candidate Generation -> Technical QA -> Critique -> Selection -> Bounded
Iteration -> Human Approval -> Final Report.
"""

from __future__ import annotations

from enum import Enum


class State(str, Enum):
    CREATED = "CREATED"                          # brief validated and frozen
    BRIEF_ACCEPTED = "BRIEF_ACCEPTED"            # original-intent hash recorded
    RESEARCHED = "RESEARCHED"                    # concept-researcher output ingested
    CONCEPTS_READY = "CONCEPTS_READY"            # concept-generator output ingested
    ART_DIRECTION_DRAFTED = "ART_DIRECTION_DRAFTED"  # art-director draft ingested
    SPEC_VALIDATED = "SPEC_VALIDATED"            # merged art direction passed schema + semantic checks
    PROMPTS_COMPILED = "PROMPTS_COMPILED"        # generation requests for the current batch exist
    GENERATED = "GENERATED"                      # every request in the batch has a recorded result
    QA_DONE = "QA_DONE"                          # technical QA recorded for succeeded results
    CRITIQUED = "CRITIQUED"                      # critiques ingested for QA-passed candidates
    SELECTED = "SELECTED"                        # ranking + curation recorded
    AWAITING_HUMAN_APPROVAL = "AWAITING_HUMAN_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"                        # human rejected all candidates
    DONE = "DONE"                                # final report written
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL = frozenset({State.DONE, State.FAILED, State.CANCELLED})

# Normal forward edges. FAILED / CANCELLED are reachable from every non-terminal state.
_EDGES: dict[State, frozenset[State]] = {
    State.CREATED: frozenset({State.BRIEF_ACCEPTED}),
    State.BRIEF_ACCEPTED: frozenset({State.RESEARCHED}),
    State.RESEARCHED: frozenset({State.CONCEPTS_READY}),
    State.CONCEPTS_READY: frozenset({State.ART_DIRECTION_DRAFTED}),
    State.ART_DIRECTION_DRAFTED: frozenset({State.SPEC_VALIDATED}),
    State.SPEC_VALIDATED: frozenset({State.PROMPTS_COMPILED}),
    State.PROMPTS_COMPILED: frozenset({State.GENERATED}),
    State.GENERATED: frozenset({State.QA_DONE}),
    State.QA_DONE: frozenset({State.CRITIQUED}),
    State.CRITIQUED: frozenset({State.SELECTED}),
    # SELECTED -> SPEC_VALIDATED is the bounded-iteration edge (new art-direction version).
    State.SELECTED: frozenset({State.SPEC_VALIDATED, State.AWAITING_HUMAN_APPROVAL, State.APPROVED}),
    State.AWAITING_HUMAN_APPROVAL: frozenset({State.APPROVED, State.REJECTED}),
    State.APPROVED: frozenset({State.DONE}),
    State.REJECTED: frozenset({State.DONE}),
}


class IllegalTransition(RuntimeError):
    pass


def allowed_targets(current: State) -> frozenset[State]:
    if current in TERMINAL:
        return frozenset()
    return _EDGES.get(current, frozenset()) | {State.FAILED, State.CANCELLED}


def check_transition(current: State, target: State) -> None:
    if target not in allowed_targets(current):
        raise IllegalTransition(f"{current.value} -> {target.value} is not allowed")
