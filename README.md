# event_driven_neural_system

This project explores event-driven neural computation inspired by biological neural systems and neuromorphic principles. The goal is to build a sparse, threshold-based neural inference pipeline and evaluate tradeoffs between accuracy and computational efficiency.

## Project structure

```text
event_driven_neural_system/
├── experiments/
│   ├── data.py
│   ├── event_driven.py
│   └── run_experiments.py
├── models/
│   └── mlp.py
├── plots/
├── results/
├── tests/
└── README.md
```

## Method

The event-driven preprocessing path follows the thresholding rule from the project brief:

```text
delta = |x_t - x_{t-1}|
mask = delta > threshold
x_event = x_t * mask
```

The baseline model is an MLP with hidden layers sized to the requested `784 → 256 → 128 → 10` MNIST configuration. The implementation uses `scikit-learn`'s `MLPClassifier`, which infers the `784` input dimension directly from the training data.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Running the experiments

### Full MNIST run

```bash
python -m experiments.run_experiments --dataset mnist
```

This trains the baseline MLP, evaluates the event-driven path across thresholds `0.0, 0.01, 0.05, 0.1, 0.2`, and writes:

- `results/metrics.json`
- `results/metrics.csv`
- `plots/accuracy_vs_threshold.png`
- `plots/sparsity_vs_threshold.png`
- `plots/accuracy_vs_sparsity.png`
- `plots/accuracy_vs_activity.png`

### Offline smoke test

The sandbox used for development cannot currently reach OpenML, so an offline verification path is also available:

```bash
python -m experiments.run_experiments --dataset digits --sample-limit 500 --max-iter 5
```

This uses the built-in `sklearn` digits dataset to validate the end-to-end training, event masking, metric export, and plotting pipeline without network access.

## Metrics and phase-transition interpretation

Each run records:

- accuracy
- activity (`nonzero_elements / total_elements`)
- sparsity (`1 - activity`)
- inference time as a compute proxy

The reported regimes align with the intended interpretation:

- **Dense**: high activity / high accuracy
- **Critical**: intermediate thresholds with the best tradeoff
- **Sparse**: strong thresholding with degraded accuracy

## Medium article outline

1. **Problem**: dense computation wastes work on unchanged inputs.
2. **Insight**: biological systems respond primarily to meaningful change.
3. **Method**: threshold-based event masking before inference.
4. **Results**: accuracy/activity/sparsity tradeoffs across thresholds.
5. **Conclusion**: efficient AI can emerge from event-driven design.

## Resume bullet

Designed an event-driven neural network with sparse, threshold-based computation inspired by neural dynamics; analyzed tradeoffs between sparsity and accuracy using benchmarking and ablation studies.
