"""Event-driven masking, metrics, and plotting helpers."""

from __future__ import annotations

from csv import DictWriter
from pathlib import Path
from time import perf_counter
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


def compute_event_mask(
    current_input: np.ndarray,
    previous_input: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return a binary mask for features whose delta exceeds the threshold."""
    delta = np.abs(current_input - previous_input)
    return (delta > threshold).astype(np.float32)


def apply_event_mask(
    current_input: np.ndarray,
    previous_input: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Keep only the current values that triggered an event."""
    return current_input * compute_event_mask(current_input, previous_input, threshold)


def compute_activity(values: np.ndarray) -> float:
    """Compute the fraction of non-zero values."""
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(values) / values.size)


def build_event_inputs(inputs: np.ndarray, threshold: float) -> np.ndarray:
    """Apply event-driven masking relative to the previous sample."""
    previous_inputs = np.zeros_like(inputs)
    previous_inputs[1:] = inputs[:-1]
    return apply_event_mask(inputs, previous_inputs, threshold)


def measure_inference_time(model, inputs: np.ndarray) -> tuple[np.ndarray, float]:
    """Run inference and return predictions with elapsed wall-clock time."""
    start_time = perf_counter()
    predictions = model.predict(inputs)
    inference_time = perf_counter() - start_time
    return predictions, inference_time


def evaluate_thresholds(model, test_x: np.ndarray, test_y: np.ndarray, thresholds: list[float]):
    """Evaluate baseline and thresholded event-driven inputs."""
    baseline_predictions, baseline_time = measure_inference_time(model, test_x)
    results = [
        {
            "regime": "dense",
            "threshold": None,
            "accuracy": float(accuracy_score(test_y, baseline_predictions)),
            "activity": compute_activity(test_x),
            "sparsity": 1.0 - compute_activity(test_x),
            "inference_time": baseline_time,
        }
    ]

    for threshold in thresholds:
        event_x = build_event_inputs(test_x, threshold)
        predictions, inference_time = measure_inference_time(model, event_x)
        activity = compute_activity(event_x)
        if threshold <= 0.01:
            regime = "dense"
        elif threshold <= 0.1:
            regime = "critical"
        else:
            regime = "sparse"

        results.append(
            {
                "regime": regime,
                "threshold": threshold,
                "accuracy": float(accuracy_score(test_y, predictions)),
                "activity": activity,
                "sparsity": 1.0 - activity,
                "inference_time": inference_time,
            }
        )

    return results


def save_results(results: list[dict], output_dir: Path) -> None:
    """Persist experiment metrics as JSON and CSV."""
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
    """Create the requested accuracy/activity/sparsity tradeoff plots."""
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


def _plot_series(x_values: list[float], y_values: list[float], x_label: str, y_label: str, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_xy(x_values: list[float], y_values: list[float], x_label: str, y_label: str, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(x_values, y_values)
    plt.plot(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
