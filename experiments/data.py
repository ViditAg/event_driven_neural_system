"""Dataset loading utilities for dense and event-driven experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetBundle:
    """Container for train/test splits and metadata."""

    name: str
    train_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    test_y: np.ndarray
    image_shape: tuple[int, int]


def _normalize_features(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32)
    max_value = float(features.max()) if features.size else 0.0
    if max_value > 1.0:
        features /= max_value
    return features


def load_dataset(
    name: str = "mnist",
    *,
    sample_limit: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBundle:
    """Load MNIST or an offline digits fallback for smoke testing."""
    if name == "mnist":
        features, labels = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False,
            parser="auto",
        )
        image_shape = (28, 28)
    elif name == "digits":
        digits = load_digits()
        features, labels = digits.data, digits.target
        image_shape = (8, 8)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    features = _normalize_features(np.asarray(features))
    labels = np.asarray(labels).astype(int)

    if sample_limit is not None:
        features = features[:sample_limit]
        labels = labels[:sample_limit]

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    return DatasetBundle(
        name=name,
        train_x=train_x,
        test_x=test_x,
        train_y=train_y,
        test_y=test_y,
        image_shape=image_shape,
    )
