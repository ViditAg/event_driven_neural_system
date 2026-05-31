# Local MNIST data (not committed)

This directory holds the official MNIST IDX files (~53 MB decompressed). It is
listed in `.gitignore` so the dataset is never pushed to GitHub.

## Download

From the repository root:

```bash
bash scripts/download_mnist.sh
```

Expected files after a successful run:

| File | Role |
|------|------|
| `train-images-idx3-ubyte` | 60,000 training images (28×28) |
| `train-labels-idx1-ubyte` | 60,000 training labels |
| `t10k-images-idx3-ubyte` | 10,000 test images |
| `t10k-labels-idx1-ubyte` | 10,000 test labels |

## Run experiments

```bash
python -m experiments.run_experiments --dataset mnist --max-iter 50
```
