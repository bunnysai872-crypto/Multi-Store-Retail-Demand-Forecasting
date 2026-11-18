"""
preprocessing.py
----------------
Converts raw Fashion-MNIST images into a 2-D feature matrix
ready for PCA.

Steps
-----
1. Cast to float32.
2. Normalise pixels from [0, 255] → [0, 1].
3. Flatten each 28×28 image into a 784-element vector.
"""

import numpy as np


def preprocess(images: np.ndarray) -> np.ndarray:
    """
    Preprocess a batch of raw Fashion-MNIST images.

    Parameters
    ----------
    images : np.ndarray, shape (N, 28, 28), dtype uint8
        Raw pixel arrays straight from Keras.

    Returns
    -------
    np.ndarray, shape (N, 784), dtype float32
        Normalised, flattened feature matrix.
    """
    # Step 1: cast to float32 so division doesn't overflow
    images = images.astype(np.float32)

    # Step 2: normalise — brings pixel values into [0, 1]
    images = images / 255.0

    # Step 3: flatten 28×28 → 784
    n_samples = images.shape[0]
    images = images.reshape(n_samples, -1)   # -1 infers 784

    return images


def preprocess_train_test(x_train_raw: np.ndarray, x_test_raw: np.ndarray):
    """
    Convenience wrapper that preprocesses both splits and prints statistics.

    Returns
    -------
    x_train : np.ndarray, shape (60000, 784)
    x_test  : np.ndarray, shape (10000, 784)
    """
    print("[INFO] Preprocessing training data …")
    x_train = preprocess(x_train_raw)

    print("[INFO] Preprocessing test data …")
    x_test = preprocess(x_test_raw)

    print("\n" + "=" * 55)
    print("  PREPROCESSING SUMMARY")
    print("=" * 55)
    print(f"  Original shape (train) : {x_train_raw.shape}")
    print(f"  Original shape (test)  : {x_test_raw.shape}")
    print(f"  Processed shape (train): {x_train.shape}")
    print(f"  Processed shape (test) : {x_test.shape}")
    print(f"  Dtype                  : {x_train.dtype}")
    print(f"  Pixel range after norm : [{x_train.min():.2f}, {x_train.max():.2f}]")
    print("=" * 55 + "\n")

    return x_train, x_test
