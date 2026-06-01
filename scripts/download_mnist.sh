#!/usr/bin/env bash
# Download official MNIST IDX files into data/mnist/ (gitignored).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MNIST_DIR="${REPO_ROOT}/data/mnist"
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

FILES=(
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz
  t10k-labels-idx1-ubyte.gz
)

mkdir -p "${MNIST_DIR}"
cd "${MNIST_DIR}"

for f in "${FILES[@]}"; do
  echo "Downloading ${f}..."
  curl -LO "${BASE_URL}/${f}"
done

echo "Decompressing..."
gunzip -f *.gz

echo "Done. Files in ${MNIST_DIR}:"

ls -lh "${MNIST_DIR}"/*-ubyte 2>/dev/null || ls -lh "${MNIST_DIR}"
