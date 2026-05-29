"""Dataset loading utilities for dense and event-driven experiments.

Supports full MNIST via OpenML and a small offline ``digits`` dataset for
CI or air-gapped smoke tests. All pixel features are scaled to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetBundle:
    """Container for train/test splits and metadata.

    Attributes
    ----------
    name:
        Dataset identifier (``"mnist"`` or ``"digits"``).
    train_x, test_x:
        Feature matrices, shape ``(n_samples, n_features)``, float32 in [0, 1].
    train_y, test_y:
        Integer class labels.
    image_shape:
        Spatial dimensions ``(height, width)`` for visualization only;
        features are stored flattened (e.g. 28×28 → 784).
    """

    name: str
    train_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    test_y: np.ndarray
    image_shape: tuple[int, int]


def _normalize_features(features: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1] if they appear to be in [0, 255]."""
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
    """Load MNIST or an offline digits fallback for smoke testing.

    Parameters
    ----------
    name:
        ``"mnist"`` fetches ``mnist_784`` from OpenML (requires network on
        first download). ``"digits"`` uses sklearn's 8×8 handwritten digits.
    sample_limit:
        If set, only the first ``sample_limit`` rows are used (useful for
        fast debugging). Applied **before** the train/test split.
    test_size:
        Fraction of data held out for testing (stratified by label).
    random_state:
        Seed passed to :func:`sklearn.model_selection.train_test_split`.

    Returns
    -------
    DatasetBundle
        Ready for ``model.fit(train_x, train_y)`` and event-driven eval on
        ``test_x``.

    Raises
    ------
    ValueError
        If ``name`` is not ``"mnist"`` or ``"digits"``.
    """
    if name == "mnist":
        # OpenML version 1: 70k samples, 784 features per image.
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
