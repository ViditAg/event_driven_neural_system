"""Baseline MLP configuration for dense and event-driven inference.

The event-driven path in :mod:`experiments.event_driven` only changes
**inputs** at inference time; the same ``MLPClassifier`` instance is used
for both baseline (full pixels) and threshold-masked pixels.
"""

from sklearn.neural_network import MLPClassifier

# Architecture target for MNIST: 784 inputs → 256 → 128 → 10 classes.
# Input size is inferred from training data, not set explicitly here.
HIDDEN_LAYER_SIZES = (256, 128)


def build_mlp(max_iter: int = 20, random_state: int = 42) -> MLPClassifier:
    """Create the baseline MLP used across all experiments.

    Parameters
    ----------
    max_iter:
        Maximum training epochs (passed to ``MLPClassifier.max_iter``).
        Increase (e.g. 50) if MNIST accuracy is below your target (~97%).
    random_state:
        Seed for weight initialization and data shuffling inside sklearn.

    Returns
    -------
    MLPClassifier
        Untrained classifier with ReLU activations, Adam solver, and
        early stopping on a validation fraction held out during ``fit``.
    """
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        # Stop if validation score does not improve for n_iter_no_change passes.
        early_stopping=True,
        n_iter_no_change=3,
        random_state=random_state,
    )
