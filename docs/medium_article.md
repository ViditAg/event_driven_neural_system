# Event-Driven Neural Dynamics for Efficient Edge Inference

**Threshold-based input sparsity before inference — and what MNIST teaches us about the accuracy–activity tradeoff.**

*Subtitle for Medium:* A sparse, threshold-based inference experiment inspired by neural dynamics (MNIST + ablation on digits).

**Repo:** [github.com/ViditAg/event_driven_neural_system](https://github.com/ViditAg/event_driven_neural_system)

---

> **Publishing on Mac:** You already have MNIST plots (`plots/mnist/*.png`). Regenerate digits locally:
> `python -m experiments.run_experiments --dataset digits --max-iter 50`
> then upload `plots/digits/accuracy_vs_sparsity.png` as the comparison figure.

---

## 1. Problem — dense computation wastes work on unchanged inputs

Standard neural networks run the same forward pass on every input, every time. For a static camera frame or a sensor reading that barely moved, most pixels are redundant. Yet a dense MLP still multiplies through all 784 MNIST features.

That mismatch matters at the edge: battery, latency, and thermals all punish unnecessary work. The question is not whether sparsity is desirable, but **where** to introduce it without rebuilding the entire stack.

This project starts at the input: gate features before they reach a conventional classifier.

---

## 2. Insight — biological systems respond primarily to meaningful change

Retina, cochlea, and cortex do not re-encode the full world on every tick. They emphasize **change** — spikes and events when something crosses a threshold. Neuromorphic chips (Loihi, SpiNNaker, and others) borrow that idea for hardware.

I did not implement a spiking network here. The insight is simpler: **if change is what matters, maybe we only need to forward what changed** into an otherwise ordinary model.

That motivates a minimal rule, not a new architecture.

---

## 3. Method — threshold-based event masking before inference

For each feature, compare the current input to the previous observation. Keep the pixel only if the absolute delta exceeds a threshold; otherwise zero it.

```
delta   = |x_t - x_{t-1}|
mask    = delta > threshold
x_event = x_t * mask
```

**Model:** scikit-learn `MLPClassifier`, 784 → 256 → 128 → 10 (MNIST), ReLU, Adam. Trained on **full** images; only **inference** uses masked inputs.

**Benchmarking:** sweep 15 thresholds from 0.0 to 1.0. At each point log accuracy, activity, sparsity, and wall-clock inference time.

**Ablation / regimes:** label each threshold bucket heuristically (not fitted):

| Regime | Threshold | Expected behavior |
|--------|-----------|-------------------|
| **Dense** | ≤ 0.05 | High activity, accuracy near baseline |
| **Critical** | ≤ 0.20 | Best sparsity vs accuracy tradeoff (target band) |
| **Sparse** | > 0.20 | Strong gating, accuracy degrades |

**Activity metric** (from the roadmap):

```
activity = nonzero_elements / total_elements
sparsity = 1 - activity
```

**Reference frame:** for test sample *i*, the previous frame is sample *i − 1* in the batch (zeros for *i = 0*). This is a deliberate simplification — not a video stream — and it shapes how much extra sparsity masking can extract.

**Data:** MNIST from local IDX files (`data/mnist/`, ~53 MB, not in git). Digits (sklearn 8×8) as a fast offline ablation.

---

## 4. Results — accuracy, activity, and sparsity across thresholds

### MNIST (primary experiment)

**Baseline (no masking):** 98.3% accuracy · activity ≈ 0.19 · sparsity ≈ 0.81

MNIST digits already sit on a black background — only ~19% of pixels are non-zero before any masking. Event gating therefore moves activity in a **narrow band** (roughly 0.19 → 0.05), not from dense to empty.

| Threshold | Regime | Accuracy | Activity |
|-----------|--------|----------|----------|
| 0.05 | dense | 93.9% | 0.16 |
| 0.10 | critical | 92.1% | 0.15 |
| 0.20 | critical | 88.3% | 0.13 |
| 0.50 | sparse | 76.4% | 0.10 |
| 0.90 | sparse | 58.6% | 0.05 |
| 1.00 | sparse | 8.9% (chance) | 0.00 |

**Figures to upload (you have these on Mac):**

1. **`plots/mnist/accuracy_vs_threshold.png`** — accuracy vs threshold (main MNIST story)
2. **`plots/mnist/accuracy_vs_activity.png`** — accuracy vs activity (phase-style view from roadmap)
3. **`plots/mnist/accuracy_vs_sparsity.png`** — tradeoff curve
4. **`plots/mnist/sparsity_vs_threshold.png`** — how sparsity rises with threshold

**Interpretation:** MNIST does **not** show a sharp physics-style phase transition. Accuracy falls **gradually** from 98% toward 76% across the middle of the sweep. That is still a useful result: a dense MLP trained on full digits **tolerates** aggressive input gating better than you might expect — a robustness curve, not a cliff.

Why no sharp knee on MNIST?

- Inputs are already sparse; masking has less room to move the activity axis.
- Consecutive test digits are unrelated (3 → 7 → 1), so pixel deltas stay large; low thresholds still pass most ink.
- The MLP was never trained for sparse inputs; graceful degradation is expected.

The **critical regime** in the table is still where a practitioner might operate (threshold ~0.1–0.2): ~88–92% accuracy with modestly lower activity than baseline.

### Digits (ablation — regenerate on Mac)

Smaller 8×8 images, denser inputs (~51% activity at baseline). Here the tradeoff is **visually clearer**.

**Baseline:** 96.9% accuracy · activity ≈ 0.51

| Threshold | Accuracy | Sparsity |
|-----------|----------|----------|
| 0.10 | 85.8% | 59% |
| 0.20 | 75.3% | 69% |
| 0.50 | 49.4% | 86% |

**Figure (generate on Mac):** `plots/digits/accuracy_vs_sparsity.png` — best “hero” plot for the accuracy–sparsity story.

Use digits to **support** the method; lead the narrative with **MNIST** since that is your full-scale run.

### Benchmarking negative result (honest ablation)

**Inference time did not drop** with higher sparsity (~0.17 s on MNIST regardless of threshold). scikit-learn performs dense matmuls; zero input pixels do not skip hidden-layer work.

**Lesson:** input sparsity is necessary but not sufficient for compute savings. End-to-end sparse execution or neuromorphic hardware is the next step.

---

## 5. Conclusion — efficient AI can emerge from event-driven design

Event-driven design is not only a hardware story. Even a **threshold gate before a standard MLP** exposes a controllable tradeoff between how much of the input you propagate and how well the classifier still performs.

**What this project demonstrates**

- A reproducible pipeline: train dense → mask at inference → sweep thresholds → plot regimes.
- **MNIST:** high baseline accuracy, gradual degradation under masking — strong robustness, weak “phase transition” visually.
- **Digits ablation:** clearer accuracy–sparsity knee on denser inputs.
- **Benchmarking:** activity and sparsity move as designed; inference time does not (stack limitation).

**What efficient edge AI still needs**

- Temporal streams where consecutive frames correlate.
- Sparse kernels or hardware that consumes events natively.
- Training or fine-tuning with masked inputs so the model expects change.

Efficient AI can emerge from event-driven design — first at the **data interface**, then in the **compute path**. This repo maps the first layer.

---

## Resume alignment (optional — do not paste verbatim unless you want)

*Designed an event-driven neural inference pipeline with sparse, threshold-based input gating inspired by neural dynamics; benchmarked accuracy, activity, and sparsity across 15 thresholds on MNIST, with ablation on digits, and analyzed dense / critical / sparse regimes.*

---

## Reproduce

```bash
git clone https://github.com/ViditAg/event_driven_neural_system.git
cd event_driven_neural_system
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest

# MNIST (after download)
bash scripts/download_mnist.sh
python -m experiments.run_experiments --dataset mnist --max-iter 50

# Digits ablation (offline)
python -m experiments.run_experiments --dataset digits --max-iter 50
```

Artifacts: `results/<dataset>/metrics.{json,csv}` · `plots/<dataset>/*.png`

---

## Medium publishing checklist (Mac final push)

**Copy & structure**

- [ ] Use the five H2 sections: Problem → Insight → Method → Results → Conclusion
- [ ] Paste repo link in intro and footer
- [ ] Tags: `Machine Learning`, `Edge Computing`, `Neuromorphic Computing`, `Python`, `Sparsity`

**Figures (MNIST — from email / branch)**

- [ ] `accuracy_vs_threshold.png` (lead)
- [ ] `accuracy_vs_activity.png` (regime / phase-style)
- [ ] `accuracy_vs_sparsity.png`
- [ ] `sparsity_vs_threshold.png` (optional)

**Figures (digits — generate on Mac)**

- [ ] Run digits experiment → `accuracy_vs_sparsity.png` (ablation inset or second half)

**Git**

- [ ] Push `docs/medium_article.md`, roadmap, and any post-testing assets to your article branch
- [ ] Do not commit `data/mnist/` binaries or `results/` / `plots/` unless you intentionally add a `docs/` figure bundle

**Tone**

- [ ] Claim tradeoffs and regimes; do **not** overclaim a sharp MNIST phase transition or wall-clock speedup on sklearn

---

*MIT License · [Event-Driven Neural Dynamics for Efficient Edge Inference](https://github.com/ViditAg/event_driven_neural_system)*
