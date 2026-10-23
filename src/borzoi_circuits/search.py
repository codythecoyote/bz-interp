"""Exact hierarchical localization that preserves interacting state coalitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class SpatialSearchResult:
    positions: tuple[int, ...]
    intervals: tuple[tuple[int, int], ...]
    induced_effect: float
    target_effect: float
    recovery: float
    evaluations: int
    budget_exhausted: bool = False


def _positions(intervals: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(sorted({position for start, end in intervals for position in range(start, end)}))


def _recovery(effect: float, target: float, noise_floor: float) -> float:
    return 1.0 - abs(effect - target) / (abs(target) + noise_floor)


def hierarchical_state_search(
    length: int,
    effect: Callable[[tuple[int, ...]], float],
    target_effect: float,
    *,
    recovery_threshold: float = 0.95,
    minimum_span: int = 1,
    noise_floor: float = 1e-8,
    maximum_evaluations: int | None = None,
) -> SpatialSearchResult:
    """Backward-eliminate spans at successively finer resolutions.

    Every keep/remove decision evaluates the complete retained union. Thus two
    individually weak spans survive when their joint removal would violate the
    target, unlike independent-attribution thresholding.
    """

    if length < 1 or minimum_span < 1 or minimum_span > length:
        raise ValueError("invalid search geometry")
    if not 0 < recovery_threshold <= 1:
        raise ValueError("recovery_threshold must lie in (0, 1]")
    if maximum_evaluations is not None and maximum_evaluations < 1:
        raise ValueError("maximum_evaluations must be positive")
    evaluations = 0

    class EvaluationBudgetExceeded(RuntimeError):
        pass

    def evaluate(intervals: Sequence[tuple[int, int]]) -> float:
        nonlocal evaluations
        if maximum_evaluations is not None and evaluations >= maximum_evaluations:
            raise EvaluationBudgetExceeded
        evaluations += 1
        return float(effect(_positions(intervals)))

    active = [(0, length)]
    full_effect = evaluate(active)
    if _recovery(full_effect, target_effect, noise_floor) < recovery_threshold:
        raise ValueError("full state does not reach the requested recovery threshold")
    active_effect = full_effect
    budget_exhausted = False
    try:
        while any(end - start > minimum_span for start, end in active):
            children: list[tuple[int, int]] = []
            for start, end in active:
                if end - start <= minimum_span:
                    children.append((start, end))
                else:
                    midpoint = (start + end) // 2
                    children.extend(((start, midpoint), (midpoint, end)))
            retained = list(children)
            retained_effect = active_effect
            while len(retained) > 1:
                options = []
                for index in range(len(retained)):
                    proposed = retained[:index] + retained[index + 1 :]
                    proposed_effect = evaluate(proposed)
                    options.append(
                        (
                            _recovery(proposed_effect, target_effect, noise_floor),
                            -(retained[index][1] - retained[index][0]),
                            index,
                            proposed_effect,
                        )
                    )
                viable = [option for option in options if option[0] >= recovery_threshold]
                if not viable:
                    break
                _, _, remove_index, retained_effect = max(viable)
                retained.pop(remove_index)
            if retained == active:
                break
            active = retained
            active_effect = retained_effect
    except EvaluationBudgetExceeded:
        budget_exhausted = True
    final_effect = active_effect
    if not budget_exhausted:
        final_effect = evaluate(active)
    return SpatialSearchResult(
        positions=_positions(active),
        intervals=tuple(active),
        induced_effect=final_effect,
        target_effect=float(target_effect),
        recovery=_recovery(final_effect, target_effect, noise_floor),
        evaluations=evaluations,
        budget_exhausted=budget_exhausted,
    )
