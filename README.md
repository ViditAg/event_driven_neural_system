# Event-Driven Neural Dynamics for Efficient Edge Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Explore **event-driven neural computation** inspired by biological and neuromorphic systems. This repository trains a baseline MNIST classifier, applies **threshold-based input sparsity** (compute only where input changes), and measures tradeoffs between **accuracy**, **activity/sparsity**, and **inference time**.

> **Scope:** Input-level event masking + a dense scikit-learn MLP. Hidden layers still run dense inference; sparsity is measured on masked pixels. See [Limitations](#limitations) for natural extensions.

## License

This project is released under the **[MIT License](LICENSE)** (Copyright © 2026 Vidit Agrawal). You may use, modify, and distribute the code with attribution and a copy of the license.

## Requirements

- **Python** 3.10+ (3.11+ recommended)
- Dependencies listed in [`requirements.txt`](requirements.txt): NumPy, scikit-learn, Matplotlib, pytest

## Quick start

Clone the repository, create a virtual environment (optional but recommended), install dependencies, and run the experiment pipeline from the **repository root**:

```bash
git clone https://github.com/ViditAg/event_driven_neural_system.git
cd event_driven_neural_system

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

python -m pip install -r requirements.txt
```

### Digits experiment (offline, recommended)

Uses sklearn’s built-in 8×8 digits dataset—no network required:

```bash
python -m experiments.run_experiments --dataset digits --max-iter 50
```

Writes to `results/digits/` and `plots/digits/` (gitignored). Plots use 0–1 axes and a 15-point threshold sweep by default.

Quick smoke test:

```bash
python -m experiments.run_experiments --dataset digits --sample-limit 500 --max-iter 5
```

### MNIST experiment

Downloads MNIST via OpenML when reachable (`api.openml.org`):

```bash
python -m experiments.run_experiments --dataset mnist --max-iter 50
```

Writes to `results/mnist/` and `plots/mnist/`. If OpenML fails on your network, use digits locally and run full MNIST on a machine with reliable access or a future local `data/mnist/` loader.

### Per-dataset output layout

| Dataset | Metrics | Plots |
|---------|---------|-------|
| `digits` | `results/digits/metrics.{json,csv}` | `plots/digits/*.png` |
| `mnist` | `results/mnist/metrics.{json,csv}` | `plots/mnist/*.png` |

Override the base directories with `--results-dir` and `--plots-dir`; the CLI appends the dataset name automatically unless the path already ends with it (e.g. `results/digits`).

### Run tests

```bash
pytest
```

## Method

Event-driven preprocessing (per feature, relative to the previous sample in the batch):

```text
delta   = |x_t - x_{t-1}|
mask    = delta > threshold
x_event = x_t * mask
```

The classifier is an MLP **784 → 256 → 128 → 10** (MNIST), implemented with `sklearn.neural_network.MLPClassifier`.

## Outputs

Each run produces four figures (y-axis 0–1; tradeoff plots also use x-axis 0–1) plus metrics:

| Artifact | Description |
|----------|-------------|
| `results/<dataset>/metrics.json` | Full results list (baseline + each threshold) |
| `results/<dataset>/metrics.csv` | Same data in tabular form |
| `plots/<dataset>/accuracy_vs_threshold.png` | Accuracy vs event threshold |
| `plots/<dataset>/sparsity_vs_threshold.png` | Sparsity vs threshold |
| `plots/<dataset>/accuracy_vs_sparsity.png` | Tradeoff curve (key plot) |
| `plots/<dataset>/accuracy_vs_activity.png` | Phase-style view: accuracy vs activity |

### Metrics

| Metric | Definition |
|--------|------------|
| **accuracy** | Classification accuracy on the test split |
| **activity** | `nonzero_elements / total_elements` on (masked) inputs |
| **sparsity** | `1 - activity` |
| **inference_time** | Wall-clock seconds for `model.predict` (compute proxy) |
| **regime** | Heuristic label: `dense`, `critical`, or `sparse` |

### Regimes (interpretation)

Heuristic labels from threshold buckets in code: **dense** (≤ 0.05), **critical** (≤ 0.2), **sparse** (> 0.2).

- **Dense** — high activity, typically highest accuracy (baseline or low threshold)
- **Critical** — intermediate thresholds; best sparsity vs accuracy tradeoff
- **Sparse** — strong thresholding; low activity, often lower accuracy

## CLI reference

```bash
python -m experiments.run_experiments --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `mnist` | `mnist` (OpenML) or `digits` (offline) |
| `--sample-limit` | none | Cap samples (debug / smoke tests) |
| `--test-size` | `0.2` | Holdout fraction |
| `--max-iter` | `20` | MLP training epochs cap |
| `--random-state` | `42` | Reproducibility seed |
| `--thresholds` | `0.0 … 1.0` (15 values) | Event thresholds to sweep |
| `--results-dir` | `results` | Metrics base dir → `results/<dataset>/` |
| `--plots-dir` | `plots` | Figures base dir → `plots/<dataset>/` |

Custom threshold sweep:

```bash
python -m experiments.run_experiments --dataset digits \
  --thresholds 0.0 0.2 0.4 0.6 0.8 1.0
```

## Repository layout

```text
event_driven_neural_system/
├── models/                 # Classifier definitions
├── experiments/            # Data loading, event logic, CLI
├── tests/                  # Unit tests (pytest)
├── results/                # Generated metrics (gitignored)
├── plots/                  # Generated figures (gitignored)
├── requirements.txt
├── LICENSE                 # MIT
└── README.md
```

## Module reference

### `models/`

| File | Role |
|------|------|
| [`mlp.py`](models/mlp.py) | Builds the baseline `MLPClassifier` (256→128 hidden, ReLU, Adam). Shared by dense and event-driven evaluation—only **inputs** change at inference time. |

### `experiments/`

| File | Role |
|------|------|
| [`data.py`](experiments/data.py) | Loads MNIST (OpenML) or sklearn digits; normalizes features to `[0, 1]`; stratified train/test split; returns a `DatasetBundle`. |
| [`event_driven.py`](experiments/event_driven.py) | Event mask (`delta > threshold`), batch-wise “previous sample” reference, activity/sparsity metrics, threshold sweep, JSON/CSV export, and Matplotlib plots. |
| [`run_experiments.py`](experiments/run_experiments.py) | **CLI entry point:** load data → train MLP → evaluate thresholds → save results → plot → print summary. |

### `tests/`

| File | Role |
|------|------|
| [`test_event_driven.py`](tests/test_event_driven.py) | Unit tests for masking, activity, and `evaluate_thresholds` with a tiny stub model. |
| [`conftest.py`](tests/conftest.py) | Adds project root to `sys.path` so `pytest` resolves `experiments` and `models` imports. |

## Development notes

- Run commands from the **repo root** so `python -m experiments.run_experiments` resolves packages correctly.
- `results/*` and `plots/*` are gitignored; commit figures manually if you want them in the repo (e.g. under `docs/`).
- **MNIST / OpenML:** requires working HTTPS to `api.openml.org`. Clear a broken cache with `rm -rf ~/scikit_learn_data/openml` if downloads fail mid-way.
- **Roadmap:** local MNIST files under `data/mnist/` (not in repo) for offline full-scale runs on a workstation.

## Limitations

- Sparsity is **input-only**; the MLP does not skip multiply-adds for zero pixels.
- “Previous frame” is the **prior test sample**, not a temporal video stream.
- Regime labels are **threshold buckets**, not a fitted critical point.
- Inference time may not drop much with sparsity on CPU + sklearn.

These are intentional simplifications for a reproducible research prototype.

## Citation & attribution

If you use this code in a publication or portfolio, a link to this repository is appreciated. The MIT license applies; see [LICENSE](LICENSE).

## Related writing

Outline for a longer article on dense vs event-driven edge inference:

1. Problem — dense computation on static inputs  
2. Insight — biological systems respond to change  
3. Method — threshold masking before inference  
4. Results — accuracy / activity / sparsity tradeoffs  
5. Conclusion — event-driven design for efficient AI  
