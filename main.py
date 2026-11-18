"""
main.py
-------
Entry point for the Fashion-MNIST PCA Machine Learning experiments.

Run with:
    python main.py

What it does
------------
1. Loads the Fashion-MNIST dataset.
2. Preprocesses (normalise + flatten).
3. Fits PCA on training data (or loads a saved model).
4. Runs experiments with 2, 3, 10, 50, 100 components.
5. Generates and saves all visualisations.
6. Prints a results summary table.
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from dataset_loader  import load_fashion_mnist, print_dataset_info, CLASS_NAMES
from preprocessing   import preprocess_train_test
from pca_model       import (load_or_fit_pca, transform, variance_summary,
                              run_experiments, COMPONENT_CONFIGS)
from reconstruction  import reconstruct_images, compute_all_errors
from visualization   import (plot_original_images, plot_pca_2d, plot_pca_3d,
                              plot_explained_variance,
                              plot_dimensionality_comparison,
                              plot_reconstruction)


def print_banner():
    print("\n" + "=" * 65)
    print("   FASHION-MNIST PCA — DIMENSIONALITY REDUCTION PROJECT")
    print("=" * 65 + "\n")


def print_results_table(experiments: list, errors: dict):
    """Pretty-print the PCA experiment results."""
    print("\n" + "=" * 70)
    print("  PCA EXPERIMENT RESULTS")
    print("=" * 70)
    header = f"{'Components':>12}  {'Explained Var':>14}  {'Reduction %':>12}  {'Recon. MSE':>12}"
    print(header)
    print("-" * 70)
    for exp in experiments:
        k   = exp["n_actual"]
        var = exp["variance"] * 100
        red = exp["reduction_pct"]
        mse = errors.get(k, float("nan"))
        print(f"{k:>12}  {var:>13.2f}%  {red:>11.2f}%  {mse:>12.6f}")
    print("=" * 70 + "\n")


def main():
    print_banner()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    (x_train_raw, y_train), (x_test_raw, y_test) = load_fashion_mnist()
    print_dataset_info(x_train_raw, y_train, x_test_raw, y_test)

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    x_train, x_test = preprocess_train_test(x_train_raw, x_test_raw)

    # ------------------------------------------------------------------
    # 3. Fit / load PCA  (fit on training data only!)
    # ------------------------------------------------------------------
    # Fit 200 components so the 95% and 99% variance thresholds resolve correctly
    # (Fashion-MNIST needs ~180 components for 99% variance)
    pca = load_or_fit_pca(x_train, n_components=200)
    vstats = variance_summary(pca)

    print(f"\n[INFO] Variance thresholds:")
    print(f"       90% variance → {vstats['n_for_90']} components")
    print(f"       95% variance → {vstats['n_for_95']} components")
    print(f"       99% variance → {vstats['n_for_99']} components")

    # ------------------------------------------------------------------
    # 4. Run multi-component experiments
    # ------------------------------------------------------------------
    print("\n[INFO] Running experiments …")
    experiments = run_experiments(pca, x_train, COMPONENT_CONFIGS)

    # ------------------------------------------------------------------
    # 5. Reconstruct images
    # ------------------------------------------------------------------
    print("\n[INFO] Reconstructing images …")
    # Use a small subset of test images for reconstruction (faster)
    x_test_small  = x_test[:200]
    reconstructions = reconstruct_images(pca, x_test_small,
                                         n_components_list=[2, 3, 10, 50, 100])
    errors = compute_all_errors(x_test_small, reconstructions)

    # ------------------------------------------------------------------
    # 6. Visualisations
    # ------------------------------------------------------------------
    print("\n[INFO] Generating visualisations …")

    # A – Original images
    plot_original_images(x_train_raw, y_train)

    # B – 2D scatter
    x_2d = transform(pca, x_train, 2)
    plot_pca_2d(x_2d, y_train)

    # C – 3D scatter (interactive HTML)
    x_3d = transform(pca, x_train, 3)
    plot_pca_3d(x_3d, y_train)

    # D – Explained variance
    plot_explained_variance(
        vstats["individual_variance"],
        vstats["cumulative_variance"],
        vstats["n_for_90"], vstats["n_for_95"], vstats["n_for_99"],
    )

    # E – Dimensionality comparison
    plot_dimensionality_comparison(experiments)

    # F – Image reconstruction
    plot_reconstruction(x_test_small, reconstructions)

    # ------------------------------------------------------------------
    # 7. Results summary
    # ------------------------------------------------------------------
    print_results_table(experiments, errors)

    print("[DONE] All outputs saved in the  outputs/  directory.")
    print("       Run  streamlit run app.py  to launch the dashboard.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Exiting …")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
