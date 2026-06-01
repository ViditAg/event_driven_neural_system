# Event-Driven Neural Dynamics for Efficient Edge Inference

## Overview
This project explores event-driven neural computation inspired by biological neural systems and neuromorphic principles. The goal is to build a sparse, threshold-based neural inference pipeline and evaluate tradeoffs between accuracy and computational efficiency.

---

## Core Idea
Traditional neural networks compute on every input regardless of change. Event-driven systems compute only when input changes exceed a threshold.

Key equation:

    delta = |x_t - x_{t-1}|
    mask = delta > threshold
    x_event = x_t * mask

---

## Phase 1: Baseline Model (2–3 hrs)
- Dataset: MNIST
- Model: MLP (784 → 256 → 128 → 10)
- Train to ~97% accuracy
- Log accuracy + inference time

---

## Phase 2: Event-Driven Mechanism (3–4 hrs)
- Compute input delta
- Apply threshold mask
- Propagate sparse input
- Evaluate performance across thresholds: [0.0, 0.01, 0.05, 0.1, 0.2]

---

## Phase 3: Experiments (3–4 hrs)
Metrics:
- Accuracy
- Sparsity (% active neurons)
- Compute proxy (inference time or ops)

Plots:
- Accuracy vs Threshold
- Sparsity vs Threshold
- Accuracy vs Sparsity (key plot)

---

## Phase 4: Phase Transition Insight (2 hrs)
Interpret system behavior as three regimes:

- Dense: high activity, high accuracy
- Critical: optimal tradeoff
- Sparse: low activity, degraded accuracy

Add metric:

    activity = nonzero_elements / total_elements

Plot:
- Accuracy vs Activity

---

## Phase 5: Medium Article (3–4 hrs)

Structure:
1. Problem: Dense computation is inefficient
2. Insight: Brain uses event-driven computation
3. Method: Threshold-based updates
4. Results: Tradeoffs between sparsity and accuracy
5. Conclusion: Efficient AI via event-driven design

---

## Key Resume Bullet
Designed an event-driven neural network with sparse, threshold-based computation inspired by neural dynamics; analyzed tradeoffs between sparsity and accuracy using benchmarking and ablation studies.

---

## GitHub Structure

    event-driven-nn/
      models/
      experiments/
      results/
      plots/
      README.md

---

## Key Insight

Event-driven neural systems exhibit phase transition behavior, with a critical regime enabling efficient computation without significant loss in accuracy.
