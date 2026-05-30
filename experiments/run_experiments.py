"""CLI entry point for the event-driven neural dynamics experiments.

End-to-end pipeline
-------------------
1. Load dataset (MNIST or offline digits)
2. Train baseline MLP on full training pixels
3. Evaluate baseline + event-threshold variants on the test split
4. Write ``results/metrics.{json,csv}`` and plots under ``plots/``

Run from the repository root::

    python -m experiments.run_experiments --dataset mnist
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.data import load_dataset
from experiments.event_driven import evaluate_thresholds, plot_results, save_results
from models.mlp import build_mlp

# Default sweep from the project brief; override with ``--thresholds``.
DEFAULT_THRESHOLDS = [
    0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25,
    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]


def parse_args() -> argparse.Namespace:
    """Build the argument parser for ``run_experiments``."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "digits"],
        default="mnist",
        help="mnist: OpenML download; digits: offline sklearn smoke test.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Use only the first N samples (debug / fast runs).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for the test split.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Maximum MLP training iterations (epochs cap).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for split and MLP.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=DEFAULT_THRESHOLDS,
        help="Event thresholds to evaluate after training.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for metrics artifacts.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for plot artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the full train → evaluate → save → plot workflow."""
    args = parse_args()

    dataset = load_dataset(
        args.dataset,
        sample_limit=args.sample_limit,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = build_mlp(max_iter=args.max_iter, random_state=args.random_state)
    model.fit(dataset.train_x, dataset.train_y)

    results = evaluate_thresholds(model, dataset.test_x, dataset.test_y, args.thresholds)
    save_results(results, args.results_dir)
    plot_results(results, args.plots_dir)

    print(f"Dataset: {dataset.name}")
    print(f"Train samples: {len(dataset.train_x)} | Test samples: {len(dataset.test_x)}")
    for row in results:
        threshold = "baseline" if row["threshold"] is None else row["threshold"]
        print(
            f"threshold={threshold} regime={row['regime']} "
            f"accuracy={row['accuracy']:.4f} activity={row['activity']:.4f} "
            f"sparsity={row['sparsity']:.4f} inference_time={row['inference_time']:.6f}s"
        )


if __name__ == "__main__":
    main()
