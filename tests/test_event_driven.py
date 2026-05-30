"""Unit tests for event-driven masking and threshold evaluation.

Uses a minimal :class:`StubModel` so tests do not train an MLP or download data.
"""

import numpy as np

from experiments.event_driven import (
    apply_event_mask,
    build_event_inputs,
    compute_activity,
    compute_event_mask,
    evaluate_thresholds,
)


class StubModel:
    """Tiny classifier for testing :func:`evaluate_thresholds` without sklearn training."""

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        # Predict class 1 when the row sum exceeds a fixed cutoff.
        return np.where(inputs.sum(axis=1) > 0.4, 1, 0)


def test_compute_event_mask_uses_strict_threshold():
    """Features with delta <= threshold must be masked out (strict inequality)."""
    current = np.array([[0.1, 0.4, 0.7]], dtype=np.float32)
    previous = np.array([[0.1, 0.2, 0.9]], dtype=np.float32)

    mask = compute_event_mask(current, previous, threshold=0.2)

    # Deltas: 0.0, 0.2, 0.2 — none strictly exceed 0.2.
    np.testing.assert_array_equal(mask, np.array([[0.0, 0.0, 0.0]], dtype=np.float32))


def test_apply_event_mask_zeroes_inactive_features():
    """Masked output keeps current values only where the mask is 1."""
    current = np.array([[0.3, 0.5, 0.9]], dtype=np.float32)
    previous = np.array([[0.1, 0.45, 0.1]], dtype=np.float32)

    masked = apply_event_mask(current, previous, threshold=0.15)

    np.testing.assert_array_equal(masked, np.array([[0.3, 0.0, 0.9]], dtype=np.float32))


def test_build_event_inputs_uses_previous_sample_as_reference():
    """Row i is masked against row i-1; row 0 uses zeros as previous."""
    inputs = np.array(
        [
            [0.0, 0.5],
            [0.0, 0.55],
            [0.4, 0.55],
        ],
        dtype=np.float32,
    )

    event_inputs = build_event_inputs(inputs, threshold=0.1)

    np.testing.assert_array_equal(
        event_inputs,
        np.array(
            [
                [0.0, 0.5],
                [0.0, 0.0],
                [0.4, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_compute_activity_returns_fraction_of_nonzero_values():
    """Activity is the fraction of non-zero entries in the array."""
    values = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)

    assert compute_activity(values) == 0.5


def test_evaluate_thresholds_returns_baseline_and_threshold_metrics():
    """Sweep returns baseline row plus one row per threshold."""
    model = StubModel()
    test_x = np.array(
        [
            [0.3, 0.3],
            [0.3, 0.31],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    test_y = np.array([1, 0, 0])

    results = evaluate_thresholds(model, test_x, test_y, [0.0, 0.1])

    assert len(results) == 3
    assert results[0]["threshold"] is None
    assert results[0]["regime"] == "dense"
    assert results[1]["threshold"] == 0.0
    assert results[2]["regime"] == "critical"
    assert all(0.0 <= row["activity"] <= 1.0 for row in results)
    assert all(0.0 <= row["sparsity"] <= 1.0 for row in results)
    # Thresholded inputs should not be denser than the baseline.
    assert results[1]["activity"] <= results[0]["activity"]
