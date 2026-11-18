"""
reconstruction.py
-----------------
Dedicated module for image reconstruction using PCA inverse_transform.

This module provides clean wrappers that:
  1. Reconstruct images from different component counts.
  2. Calculate reconstruction error (MSE).
  3. Provide results in a format ready for display / saving.
"""

import numpy as np
from sklearn.decomposition import PCA


def reconstruct_images(pca: PCA,
                       x: np.ndarray,
                       n_components_list: list) -> dict:
    """
    Reconstruct images using multiple component counts.

    Parameters
    ----------
    pca               : fitted sklearn PCA (must have >= max(n_components_list) components)
    x                 : (N, 784) normalised input images
    n_components_list : e.g. [10, 50, 100]

    Returns
    -------
    dict  {n_components: np.ndarray of shape (N, 784)}
        Pixel values are clipped to [0, 1].
    """
    results = {}
    max_comp = pca.n_components_

    for k in n_components_list:
        k_actual = min(k, max_comp)

        # --- Manual truncated inverse-transform ---
        # 1. Project to k-dimensional latent space
        x_centered = x - pca.mean_
        x_latent   = x_centered @ pca.components_[:k_actual].T  # (N, k)

        # 2. Reconstruct back to pixel space
        x_rec = x_latent @ pca.components_[:k_actual]           # (N, 784)
        x_rec += pca.mean_                                        # add mean back

        # 3. Clip so pixel values stay in valid range
        x_rec = np.clip(x_rec, 0.0, 1.0)

        results[k] = x_rec.astype(np.float32)
        print(f"[INFO] Reconstructed with {k_actual} components. "
              f"MSE = {mse(x, x_rec):.6f}")

    return results


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error between original and reconstructed images."""
    return float(np.mean((original.astype(np.float64) -
                          reconstructed.astype(np.float64)) ** 2))


def compute_all_errors(x_original: np.ndarray,
                       reconstructions: dict) -> dict:
    """
    Compute MSE for every reconstructed configuration.

    Returns
    -------
    dict  {n_components: mse_value}
    """
    errors = {}
    for k, x_rec in reconstructions.items():
        errors[k] = mse(x_original, x_rec)
    return errors
