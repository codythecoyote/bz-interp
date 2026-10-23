"""Spatially aligned full-channel frontier interventions."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .interventions import patch_positions


def map_spatial_positions(
    positions: Sequence[int], source_length: int, target_length: int
) -> tuple[int, ...]:
    """Map bins between resolutions using their shared genomic extent."""

    if source_length < 1 or target_length < 1:
        raise ValueError("spatial lengths must be positive")
    values = np.asarray(tuple(positions), dtype=np.int64)
    if values.size == 0:
        return ()
    if values.min() < 0 or values.max() >= source_length:
        raise IndexError("source position lies outside the spatial axis")
    mapped = np.floor(values * target_length / source_length).astype(np.int64)
    mapped = np.clip(mapped, 0, target_length - 1)
    return tuple(int(value) for value in np.unique(mapped))


def patch_frontier_positions(
    recipient: Sequence[np.ndarray],
    donor: Sequence[np.ndarray],
    positions: Sequence[int],
    *,
    spatial_length: int | None = None,
) -> tuple[list[np.ndarray], tuple[tuple[int, ...], ...]]:
    """Patch aligned full-channel positions in every frontier tensor."""

    if len(recipient) != len(donor) or not recipient:
        raise ValueError("recipient and donor frontiers must have equal nonzero length")
    recipient_arrays = [np.asarray(value) for value in recipient]
    donor_arrays = [np.asarray(value) for value in donor]
    for left, right in zip(recipient_arrays, donor_arrays):
        if left.shape != right.shape or left.ndim < 3:
            raise ValueError("frontier tensors must have matching spatial-channel shapes")
    base_length = spatial_length or max(value.shape[-2] for value in recipient_arrays)
    mapped = tuple(
        map_spatial_positions(positions, base_length, value.shape[-2])
        for value in recipient_arrays
    )
    patched = [
        patch_positions(left, right, indices)
        for left, right, indices in zip(recipient_arrays, donor_arrays, mapped)
    ]
    return patched, mapped


def bidirectional_recovery(
    sufficiency_effect: float,
    necessity_effect: float,
    target_effect: float,
    noise_floor: float,
) -> float:
    """Return conservative recovery: one minus the worse directional error."""

    denominator = abs(float(target_effect)) + float(noise_floor)
    if denominator <= 0:
        raise ValueError("target magnitude plus noise floor must be positive")
    maximum_error = max(
        abs(float(sufficiency_effect) - float(target_effect)),
        abs(float(necessity_effect) - float(target_effect)),
    )
    return float(1.0 - maximum_error / denominator)


def coverage_delta_consistency(
    observed: float,
    expected: float,
    noise_floor: float,
    *,
    normalized_tolerance: float = 0.01,
) -> tuple[bool, float, bool, bool]:
    """Check platform replication without imposing an absolute float threshold."""

    if noise_floor <= 0 or normalized_tolerance < 0:
        raise ValueError("noise floor must be positive and tolerance nonnegative")
    normalized_error = abs(float(observed) - float(expected)) / (
        abs(float(expected)) + float(noise_floor)
    )
    sign_match = bool(np.sign(observed) == np.sign(expected))
    above_noise = bool(abs(float(observed)) > float(noise_floor))
    passed = normalized_error <= normalized_tolerance and sign_match and above_noise
    return passed, float(normalized_error), sign_match, above_noise


def spatial_intervals_to_genomic(
    intervals: Sequence[tuple[int, int]],
    *,
    window_start: int,
    input_length: int,
    spatial_length: int,
) -> tuple[tuple[int, int], ...]:
    """Convert half-open model-bin intervals to half-open genomic intervals."""

    if input_length < 1 or spatial_length < 1:
        raise ValueError("input and spatial lengths must be positive")
    converted = []
    for start, end in intervals:
        if not 0 <= start < end <= spatial_length:
            raise ValueError("invalid spatial interval")
        genomic_start = int(window_start + np.floor(start * input_length / spatial_length))
        genomic_end = int(window_start + np.ceil(end * input_length / spatial_length))
        converted.append((genomic_start, genomic_end))
    return tuple(converted)
