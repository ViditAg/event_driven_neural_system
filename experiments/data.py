"""Dataset loading utilities for dense and event-driven experiments.

Supports MNIST from local IDX files and a small offline ``digits`` dataset for
CI or air-gapped smoke tests. All pixel features are scaled to [0, 1].
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Gitignored MNIST IDX files (see scripts/download_mnist.sh).
DEFAULT_MNIST_DIR = Path(__file__).resolve().parents[1] / "data" / "mnist"

LOCAL_MNIST_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)

IDX_IMAGE_MAGIC = 2051
IDX_LABEL_MAGIC = 2049


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


def _read_idx_images(path: Path) -> np.ndarray:
    """Parse an IDX image file into ``(n_samples, n_pixels)`` uint8 array."""
    with path.open("rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != IDX_IMAGE_MAGIC:
            raise ValueError(f"Expected image magic {IDX_IMAGE_MAGIC} in {path}, got {magic}")
        buffer = handle.read()
    expected = count * rows * cols
    if len(buffer) != expected:
        raise ValueError(f"{path}: expected {expected} image bytes, got {len(buffer)}")
    return np.frombuffer(buffer, dtype=np.uint8).reshape(count, rows * cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    """Parse an IDX label file into a 1-D integer array."""
    with path.open("rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != IDX_LABEL_MAGIC:
            raise ValueError(f"Expected label magic {IDX_LABEL_MAGIC} in {path}, got {magic}")
        buffer = handle.read()
    if len(buffer) != count:
        raise ValueError(f"{path}: expected {count} label bytes, got {len(buffer)}")
    return np.frombuffer(buffer, dtype=np.uint8).astype(int)


def local_mnist_available(mnist_dir: Path | None = None) -> bool:
    """Return True when all four decompressed IDX files exist under ``mnist_dir``."""
    root = DEFAULT_MNIST_DIR if mnist_dir is None else mnist_dir
    return all((root / name).is_file() for name in LOCAL_MNIST_FILES)


def _load_mnist_from_local(mnist_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the official 60k/10k MNIST train and test splits from IDX files."""
    if not local_mnist_available(mnist_dir):
        missing = [name for name in LOCAL_MNIST_FILES if not (mnist_dir / name).is_file()]
        raise FileNotFoundError(
            f"Local MNIST files missing under {mnist_dir}: {', '.join(missing)}. "
            "Run: bash scripts/download_mnist.sh"
        )

    train_x = _read_idx_images(mnist_dir / "train-images-idx3-ubyte")
    train_y = _read_idx_labels(mnist_dir / "train-labels-idx1-ubyte")
    test_x = _read_idx_images(mnist_dir / "t10k-images-idx3-ubyte")
    test_y = _read_idx_labels(mnist_dir / "t10k-labels-idx1-ubyte")
    return train_x, train_y, test_x, test_y


def _apply_sample_limit_to_split(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    sample_limit: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Truncate an existing train/test split for fast debugging runs."""
    train_count = int(sample_limit * (1.0 - test_size))
    test_count = sample_limit - train_count
    if train_count <= 0 or test_count <= 0:
        raise ValueError(
            f"sample_limit={sample_limit} with test_size={test_size} leaves no train or test samples"
        )
    return (
        train_x[:train_count],
        train_y[:train_count],
        test_x[:test_count],
        test_y[:test_count],
    )


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
        ``"mnist"`` loads from local IDX files in ``data/mnist/``.
        ``"digits"`` uses sklearn's 8×8 handwritten digits.
    sample_limit:
        If set, only the first ``sample_limit`` rows are used (useful for
        fast debugging). For MNIST this truncates the official 60k/10k split
        proportionally; for digits it is applied before splitting.
    test_size:
        Fraction of data held out for testing when using digits. Ignored for
        MNIST except when ``sample_limit`` is set (then controls the train/test
        ratio of the truncated subset).
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
    FileNotFoundError
        If MNIST is requested but ``data/mnist/`` is missing required files.
    """
    if name == "mnist":
        image_shape = (28, 28)
        train_x, train_y, test_x, test_y = _load_mnist_from_local(DEFAULT_MNIST_DIR)
        train_x = _normalize_features(train_x)
        test_x = _normalize_features(test_x)
        if sample_limit is not None:
            train_x, train_y, test_x, test_y = _apply_sample_limit_to_split(
                train_x,
                train_y,
                test_x,
                test_y,
                sample_limit=sample_limit,
                test_size=test_size,
            )
    elif name == "digits":
        digits = load_digits()
        features = _normalize_features(np.asarray(digits.data))
        labels = np.asarray(digits.target).astype(int)
        image_shape = (8, 8)
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
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return DatasetBundle(
        name=name,
        train_x=train_x,
        test_x=test_x,
        train_y=train_y,
        test_y=test_y,
        image_shape=image_shape,
    )
