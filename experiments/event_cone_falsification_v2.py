#!/usr/bin/env python3
from __future__ import annotations

import random
from typing import List, Sequence, Set, Tuple

import event_cone_falsification as base


def build_attempt_trace(
    initial: Set[base.Atom],
    actions: Sequence[base.base.GroundAction],
    length: int,
    rng: random.Random,
) -> Tuple[List[base.Event], List[int]]:
    events: List[base.Event] = []
    pivots: List[int] = []
    state = set(initial)

    while len(events) < length:
        applicable_indices = [i for i, action in enumerate(actions) if action.applicable(state)]
        if not applicable_indices:
            raise RuntimeError("continuous trace reached a state with no applicable grounded action")

        pivot_action_index = rng.choice(applicable_indices)
        pivot_index = len(events)
        state = base.append_event(events, pivot_action_index, actions, state)
        if events[-1].success:
            pivots.append(pivot_index)

        if len(events) >= length:
            break

        shuffled = list(applicable_indices)
        rng.shuffle(shuffled)
        anti_index = None
        for candidate_index in shuffled[: min(len(shuffled), 256)]:
            if not actions[candidate_index].applicable(state):
                anti_index = candidate_index
                break
        if anti_index is None:
            for candidate_index in applicable_indices:
                if not actions[candidate_index].applicable(state):
                    anti_index = candidate_index
                    break
        if anti_index is None:
            failed = [i for i, action in enumerate(actions) if not action.applicable(state)]
            anti_index = rng.choice(failed) if failed else rng.randrange(len(actions))
        state = base.append_event(events, anti_index, actions, state)

        if len(events) >= length:
            break

        state = base.append_event(events, rng.randrange(len(actions)), actions, state)

    for i in range(len(events) - 1):
        if events[i].after != events[i + 1].before:
            raise AssertionError(f"non-contiguous trace at {i}->{i+1}")
    return events, pivots


base.build_attempt_trace = build_attempt_trace

if __name__ == "__main__":
    raise SystemExit(base.main())
