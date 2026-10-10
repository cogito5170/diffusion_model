import unittest

from atelier.state_machine import TERMINAL, IllegalTransition, State, allowed_targets, check_transition


class StateMachineTests(unittest.TestCase):
    def test_forward_path(self):
        path = [State.CREATED, State.BRIEF_ACCEPTED, State.RESEARCHED, State.CONCEPTS_READY, State.ART_DIRECTION_DRAFTED,
                State.SPEC_VALIDATED, State.PROMPTS_COMPILED, State.GENERATED, State.QA_DONE, State.CRITIQUED,
                State.SELECTED, State.AWAITING_HUMAN_APPROVAL, State.APPROVED, State.DONE]
        for a, b in zip(path, path[1:]):
            check_transition(a, b)

    def test_iteration_edge(self):
        check_transition(State.SELECTED, State.SPEC_VALIDATED)

    def test_cannot_skip_steps(self):
        for a, b in ((State.CREATED, State.PROMPTS_COMPILED), (State.SPEC_VALIDATED, State.GENERATED),
                     (State.GENERATED, State.CRITIQUED), (State.CRITIQUED, State.APPROVED),
                     (State.AWAITING_HUMAN_APPROVAL, State.DONE)):
            with self.assertRaises(IllegalTransition):
                check_transition(a, b)

    def test_terminal_states_are_final(self):
        for t in TERMINAL:
            self.assertEqual(allowed_targets(t), frozenset())

    def test_fail_and_cancel_reachable_everywhere(self):
        for s in State:
            if s not in TERMINAL:
                self.assertIn(State.FAILED, allowed_targets(s))
                self.assertIn(State.CANCELLED, allowed_targets(s))


if __name__ == "__main__":
    unittest.main()
