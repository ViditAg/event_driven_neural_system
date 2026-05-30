"""Experiment pipeline for the event-driven neural system.

Submodules
----------
data
    Dataset loading and train/test splits.
event_driven
    Threshold masking, metrics, persistence, and plotting.
run_experiments
    CLI: train baseline MLP and sweep event thresholds.

Typical usage::

    python -m experiments.run_experiments --dataset mnist
"""
