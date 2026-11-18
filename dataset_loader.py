"""
dataset_loader.py
-----------------
Loads the Fashion-MNIST dataset from Keras and provides
helper utilities like class names and dataset statistics.

The dataset is automatically downloaded on first run and
cached locally by Keras (usually in ~/.keras/datasets/).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Class label mapping (official Zalando Research names)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "T-shirt/top",   # 0
    "Trouser",       # 1
    "Pullover",      # 2
    "Dress",         # 3
    "Coat",          # 4
    "Sandal",        # 5
    "Shirt",         # 6
    "Sneaker",       # 7
    "Bag",           # 8
    "Ankle boot",    # 9
]


def load_fashion_mnist():
    """
    Load Fashion-MNIST from Keras.

    Returns
    -------
    (x_train, y_train) : training images (60 000 × 28 × 28, uint8) and labels
    (x_test,  y_test)  : testing  images (10 000 × 28 × 28, uint8) and labels
    """
    # Import here so the rest of the module works even without TensorFlow
    # (e.g. during unit-testing mock runs)
    try:
        from tensorflow.keras.datasets import fashion_mnist
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to load Fashion-MNIST. "
            "Install it with:  pip install tensorflow"
        ) from exc

    print("[INFO] Loading Fashion-MNIST dataset …")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print("[INFO] Dataset loaded successfully.")
    return (x_train, y_train), (x_test, y_test)


def print_dataset_info(x_train, y_train, x_test, y_test):
    """Print a human-readable summary of the dataset."""
    total = len(x_train) + len(x_test)
    print("\n" + "=" * 55)
    print("  FASHION-MNIST DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total images        : {total:,}")
    print(f"  Training images     : {len(x_train):,}")
    print(f"  Testing images      : {len(x_test):,}")
    print(f"  Image dimensions    : {x_train.shape[1]} × {x_train.shape[2]} pixels")
    print(f"  Features per image  : {x_train.shape[1] * x_train.shape[2]:,}")
    print(f"  Number of classes   : {len(CLASS_NAMES)}")
    print(f"  Pixel value range   : {x_train.min()} – {x_train.max()}")
    print("=" * 55)
    print("  Class labels:")
    for idx, name in enumerate(CLASS_NAMES):
        count = int(np.sum(y_train == idx))
        print(f"    {idx}  {name:<15}  ({count:,} training samples)")
    print("=" * 55 + "\n")


def get_class_name(label: int) -> str:
    """Return the string class name for a numeric label (0–9)."""
    if 0 <= label <= 9:
        return CLASS_NAMES[label]
    raise ValueError(f"Label must be 0–9, got {label}")
