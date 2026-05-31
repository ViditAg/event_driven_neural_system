"""Unit tests for local MNIST IDX parsing and availability checks."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from experiments.data import (
    _read_idx_images,
    _read_idx_labels,
    local_mnist_available,
)


def _write_idx_images(path: Path, images: np.ndarray, rows: int, cols: int) -> None:
    count = images.shape[0]
    header = struct.pack(">IIII", 2051, count, rows, cols)
    path.write_bytes(header + images.astype(np.uint8).tobytes())


def _write_idx_labels(path: Path, labels: np.ndarray) -> None:
    count = labels.shape[0]
    header = struct.pack(">II", 2049, count)
    path.write_bytes(header + labels.astype(np.uint8).tobytes())


def test_read_idx_images_and_labels(tmp_path: Path):
    images = np.arange(2 * 2 * 2, dtype=np.uint8).reshape(2, 4)
    labels = np.array([3, 7], dtype=np.uint8)
    image_path = tmp_path / "tiny-images-idx3-ubyte"
    label_path = tmp_path / "tiny-labels-idx1-ubyte"

    _write_idx_images(image_path, images, rows=2, cols=2)
    _write_idx_labels(label_path, labels)

    parsed_images = _read_idx_images(image_path)
    parsed_labels = _read_idx_labels(label_path)

    np.testing.assert_array_equal(parsed_images, images)
    np.testing.assert_array_equal(parsed_labels, labels)


def test_read_idx_images_rejects_bad_magic(tmp_path: Path):
    path = tmp_path / "bad-images-idx3-ubyte"
    path.write_bytes(struct.pack(">IIII", 9999, 1, 2, 2) + b"\x00" * 4)

    with pytest.raises(ValueError, match="Expected image magic"):
        _read_idx_images(path)


def test_local_mnist_available_requires_all_four_files(tmp_path: Path):
    assert not local_mnist_available(tmp_path)

    (tmp_path / "train-images-idx3-ubyte").write_bytes(b"x")
    assert not local_mnist_available(tmp_path)

    for name in (
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ):
        (tmp_path / name).write_bytes(b"x")

    assert local_mnist_available(tmp_path)
