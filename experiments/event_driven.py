"""Event-driven masking, metrics, and plotting helpers.

Core idea (neuromorphic-inspired input gating)
----------------------------------------------
Only propagate features whose change since the previous observation exceeds
a threshold::

    delta   = |x_t - x_{t-1}|
    mask    = delta > threshold
    x_event = x_t * mask

At inference, masked inputs are fed to the same dense MLP trained on full
images. Sparsity and activity are computed on the **input tensor**, not on
hidden activations.
"""

from __future__ import annotations

from csv import DictWriter
from pathlib import Path
from time import perf_counter
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

# Regime labels are heuristic buckets for reporting, not fitted critical points.
_REGIME_DENSE_THRESHOLD = 0.01
_REGIME_CRITICAL_THRESHOLD = 0.1


def compute_event_mask(
    current_input: np.ndarray,
    previous_input: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return a binary mask for features whose delta exceeds the threshold.

    Parameters
    ----------
    current_input, previous_input:
        Arrays of the same shape (e.g. one flattened image or a batch).
    threshold:
        Minimum absolute change required to mark a feature as "active".

    Returns
    -------
    np.ndarray
        Float32 array of 0.0/1.0 with the same shape as the inputs.
    """
    delta = np.abs(current_input - previous_input)
    return (delta > threshold).astype(np.float32)


def apply_event_mask(
    current_input: np.ndarray,
    previous_input: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Keep only the current values that triggered an event.

    Inactive features (no change above threshold) are zeroed out.
    """
    return current_input * compute_event_mask(current_input, previous_input, threshold)


def compute_activity(values: np.ndarray) -> float:
    """Compute the fraction of non-zero values in ``values``.

    Used as ``activity = nonzero / total``; sparsity is ``1 - activity``.
    """
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(values) / values.size)


def build_event_inputs(inputs: np.ndarray, threshold: float) -> np.ndarray:
    """Apply event-driven masking relative to the previous sample in the batch.

    For sample index ``i``, the reference frame is ``inputs[i - 1]``.
    The first sample uses an all-zero previous frame (so any non-zero pixel
    can fire on the first row).

    Parameters
    ----------
    inputs:
        Test (or train) feature matrix, shape ``(n_samples, n_features)``.
    threshold:
        Event threshold passed to :func:`apply_event_mask`.

    Returns
    -------
    np.ndarray
        Masked inputs of the same shape as ``inputs``.
    """
    previous_inputs = np.zeros_like(inputs)
    previous_inputs[1:] = inputs[:-1]
    return apply_event_mask(inputs, previous_inputs, threshold)


def measure_inference_time(model, inputs: np.ndarray) -> tuple[np.ndarray, float]:
    """Run inference and return predictions with elapsed wall-clock time.

    Parameters
    ----------
    model:
        Any object with a ``predict(inputs)`` method (e.g. ``MLPClassifier``).
    inputs:
        Feature matrix passed to ``predict``.

    Returns
    -------
    predictions, inference_time
        Class predictions and duration in seconds (``perf_counter``).
    """
    start_time = perf_counter()
    predictions = model.predict(inputs)
    inference_time = perf_counter() - start_time
    return predictions, inference_time


def _classify_regime(threshold: float) -> str:
    """Map a threshold value to a dense / critical / sparse regime label."""
    if threshold <= _REGIME_DENSE_THRESHOLD:
        return "dense"
    if threshold <= _REGIME_CRITICAL_THRESHOLD:
        return "critical"
    return "sparse"


def evaluate_thresholds(
    model,
    test_x: np.ndarray,
    test_y: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    """Evaluate baseline and thresholded event-driven inputs.

    The first result row is the **baseline**: full ``test_x``, no masking,
    with ``threshold`` set to ``None``. Subsequent rows use
    :func:`build_event_inputs` for each value in ``thresholds``.

    Parameters
    ----------
    model:
        Trained classifier with ``predict``.
    test_x, test_y:
        Held-out features and labels.
    thresholds:
        List of event thresholds to sweep (e.g. ``[0.0, 0.01, 0.05, 0.1, 0.2]``).

    Returns
    -------
    list[dict]
        Each dict contains ``regime``, ``threshold``, ``accuracy``,
        ``activity``, ``sparsity``, and ``inference_time``.
    """
    baseline_predictions, baseline_time = measure_inference_time(model, test_x)
    baseline_activity = compute_activity(test_x)

    results = [
        {
            "regime": "dense",
            "threshold": None,
            "accuracy": float(accuracy_score(test_y, baseline_predictions)),
            "activity": baseline_activity,
            "sparsity": 1.0 - baseline_activity,
            "inference_time": baseline_time,
        }
    ]

    for threshold in thresholds:
        event_x = build_event_inputs(test_x, threshold)
        predictions, inference_time = measure_inference_time(model, event_x)
        activity = compute_activity(event_x)

        results.append(
            {
                "regime": _classify_regime(threshold),
                "threshold": threshold,
                "accuracy": float(accuracy_score(test_y, predictions)),
                "activity": activity,
                "sparsity": 1.0 - activity,
                "inference_time": inference_time,
            }
        )

    return results


def save_results(results: list[dict], output_dir: Path) -> None:
    """Persist experiment metrics as JSON and CSV.

    Creates ``output_dir`` if it does not exist. Files written:

    - ``metrics.json`` — pretty-printed list of result dicts
    - ``metrics.csv`` — same fields in tabular form
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(
            handle,
            fieldnames=["regime", "threshold", "accuracy", "activity", "sparsity", "inference_time"],
        )
        writer.writeheader()
        writer.writerows(results)


def plot_results(results: list[dict], output_dir: Path) -> None:
    """Create accuracy / sparsity / activity tradeoff plots.

    Skips the baseline row (``threshold is None``) for threshold-axis plots.
    Writes four PNG files under ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_results = [row for row in results if row["threshold"] is not None]
    thresholds = [row["threshold"] for row in threshold_results]
    accuracies = [row["accuracy"] for row in threshold_results]
    sparsities = [row["sparsity"] for row in threshold_results]
    activities = [row["activity"] for row in threshold_results]

    _plot_series(thresholds, accuracies, "Threshold", "Accuracy", output_dir / "accuracy_vs_threshold.png")
    _plot_series(thresholds, sparsities, "Threshold", "Sparsity", output_dir / "sparsity_vs_threshold.png")
    _plot_xy(
        sparsities,
        accuracies,
        "Sparsity",
        "Accuracy",
        output_dir / "accuracy_vs_sparsity.png",
    )
    _plot_xy(
        activities,
        accuracies,
        "Activity",
        "Accuracy",
        output_dir / "accuracy_vs_activity.png",
    )


def _plot_series(
    x_values: list[float],
    y_values: list[float],
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Save a simple line plot (threshold sweeps)."""
    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_xy(
    x_values: list[float],
    y_values: list[float],
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Save a scatter + line plot (tradeoff curves)."""
    plt.figure(figsize=(6, 4))
    plt.scatter(x_values, y_values)
    plt.plot(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
