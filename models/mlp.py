"""Baseline MLP configuration for dense and event-driven inference."""

from sklearn.neural_network import MLPClassifier


def build_mlp(max_iter: int = 20, random_state: int = 42) -> MLPClassifier:
    """Create the baseline MLP used across all experiments."""
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=3,
        random_state=random_state,
    )
