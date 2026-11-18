"""
visualization.py
----------------
All static (Matplotlib / Seaborn) and interactive (Plotly)
visualizations for the Fashion-MNIST PCA project.

Outputs are saved to the  outputs/  directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from dataset_loader import CLASS_NAMES

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 10 distinct colours – one per class
PALETTE = list(px.colors.qualitative.Plotly)[:10]


# ===========================================================================
# A. Original Dataset Images
# ===========================================================================

def plot_original_images(x_raw: np.ndarray, y: np.ndarray,
                         n_per_class: int = 3,
                         save: bool = True) -> str:
    """
    Display a grid of sample images (n_per_class per clothing category).

    Returns the saved file path.
    """
    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class * 2, n_classes * 2))
    fig.suptitle("Fashion-MNIST — Sample Images by Category",
                 fontsize=14, fontweight="bold", y=1.01)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        # Pick the first n_per_class samples from this class
        indices = np.where(y == cls_idx)[0][:n_per_class]
        for col, img_idx in enumerate(indices):
            ax = axes[cls_idx][col]
            ax.imshow(x_raw[img_idx], cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls_name, fontsize=8, rotation=0,
                              labelpad=55, va="center")

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "original_images.png")
    if save:
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"[INFO] Saved → {path}")
    plt.close(fig)
    return path


# ===========================================================================
# B. PCA 2D Scatter
# ===========================================================================

def plot_pca_2d(x_2d: np.ndarray, y: np.ndarray,
                sample_size: int = 5000,
                save: bool = True) -> str:
    """
    2-D PCA scatter coloured by clothing category.

    Parameters
    ----------
    x_2d        : (N, 2) projected coordinates
    y           : integer class labels (N,)
    sample_size : cap the number of points to keep the plot fast
    """
    if len(x_2d) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_2d), sample_size, replace=False)
        x_2d, y = x_2d[idx], y[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = (y == cls_idx)
        ax.scatter(x_2d[mask, 0], x_2d[mask, 1],
                   label=cls_name, alpha=0.5, s=8,
                   color=PALETTE[cls_idx % len(PALETTE)])

    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.set_title("PCA 2D Visualisation — Fashion-MNIST",
                 fontsize=14, fontweight="bold")
    ax.legend(markerscale=3, fontsize=9,
              loc="upper right", framealpha=0.7)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, "pca_2d.png")
    if save:
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"[INFO] Saved → {path}")
    plt.close(fig)
    return path


# ===========================================================================
# C. PCA 3D Scatter (Plotly – interactive HTML)
# ===========================================================================

def plot_pca_3d(x_3d: np.ndarray, y: np.ndarray,
                sample_size: int = 4000,
                save: bool = True) -> str:
    """
    Interactive Plotly 3-D scatter.  Saves as an HTML file.
    """
    if len(x_3d) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_3d), sample_size, replace=False)
        x_3d, y = x_3d[idx], y[idx]

    # Build a list of class-name strings for hover labels
    class_labels = [CLASS_NAMES[int(label)] for label in y]

    fig = px.scatter_3d(
        x=x_3d[:, 0], y=x_3d[:, 1], z=x_3d[:, 2],
        color=class_labels,
        color_discrete_sequence=PALETTE,
        labels={"x": "PC 1", "y": "PC 2", "z": "PC 3",
                "color": "Category"},
        title="PCA 3D Visualisation — Fashion-MNIST",
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(
        legend_title_text="Clothing Category",
        scene=dict(
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            zaxis_title="PC 3",
        ),
    )

    path = os.path.join(OUTPUTS_DIR, "pca_3d.html")
    if save:
        fig.write_html(path)
        print(f"[INFO] Saved → {path}")
    return path


# ===========================================================================
# D. Explained Variance Graph
# ===========================================================================

def plot_explained_variance(individual: np.ndarray,
                            cumulative: np.ndarray,
                            n_for_90: int, n_for_95: int, n_for_99: int,
                            save: bool = True) -> str:
    """
    Dual-panel plot: individual variance (scree) + cumulative variance.
    """
    n = len(individual)
    x = np.arange(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Scree plot (individual) ---
    ax1.bar(x, individual * 100, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Principal Component", fontsize=11)
    ax1.set_ylabel("Explained Variance (%)", fontsize=11)
    ax1.set_title("Scree Plot — Individual Variance", fontsize=12,
                  fontweight="bold")
    ax1.set_xlim(0, n + 1)
    ax1.grid(axis="y", alpha=0.3)

    # --- Cumulative variance ---
    ax2.plot(x, cumulative * 100, "o-", color="darkorange",
             markersize=3, linewidth=1.5, label="Cumulative variance")

    thresholds = [(90, n_for_90, "blue"), (95, n_for_95, "green"),
                  (99, n_for_99, "red")]
    for pct, n_comp, colour in thresholds:
        ax2.axhline(pct, color=colour, linestyle="--", linewidth=1,
                    alpha=0.7, label=f"{pct}% threshold ({n_comp} components)")
        ax2.axvline(n_comp, color=colour, linestyle=":", linewidth=1,
                    alpha=0.5)

    ax2.set_xlabel("Number of Components", fontsize=11)
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=11)
    ax2.set_title("Cumulative Explained Variance", fontsize=12,
                  fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.7)
    ax2.set_xlim(0, n + 1)
    ax2.set_ylim(0, 102)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "explained_variance.png")
    if save:
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"[INFO] Saved → {path}")
    plt.close(fig)
    return path


# ===========================================================================
# E. Dimensionality Comparison Bar Chart
# ===========================================================================

def plot_dimensionality_comparison(experiments: list,
                                   save: bool = True) -> str:
    """
    Horizontal bar chart comparing dimensionality across configurations.

    Parameters
    ----------
    experiments : list of dicts from pca_model.run_experiments()
    """
    original_dims = 784
    labels  = ["Original (784)"] + [f"{e['n_components']} PCs" for e in experiments]
    dims    = [original_dims]     + [e["n_components"]          for e in experiments]
    pcts    = [0.0]               + [e["reduction_pct"]         for e in experiments]
    colours = ["#4c72b0"] + ["#dd8452"] * len(experiments)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, dims, color=colours, edgecolor="white", height=0.5)

    for bar, dim, pct in zip(bars[1:], dims[1:], pcts[1:]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"−{pct:.1f}%  ({dim} dims)",
                va="center", fontsize=9, color="#dd8452")

    ax.text(bars[0].get_width() + 5,
            bars[0].get_y() + bars[0].get_height() / 2,
            f"{original_dims} dims",
            va="center", fontsize=9, color="#4c72b0")

    ax.set_xlabel("Number of Dimensions", fontsize=11)
    ax.set_title("Dimensionality Comparison — PCA Reduction",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, original_dims * 1.3)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, "comparison.png")
    if save:
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"[INFO] Saved → {path}")
    plt.close(fig)
    return path


# ===========================================================================
# F. Image Reconstruction Comparison
# ===========================================================================

def plot_reconstruction(x_original: np.ndarray,
                        reconstructions: dict,
                        n_images: int = 5,
                        save: bool = True) -> str:
    """
    Side-by-side reconstruction quality comparison.

    Parameters
    ----------
    x_original    : (N, 784) normalised original images
    reconstructions : dict  {n_components: (N, 784) reconstructed}
    n_images      : number of example images to show per row
    """
    configs     = list(reconstructions.keys())
    n_rows      = 1 + len(configs)   # 1 original row + 1 per config
    n_cols      = n_images

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2, n_rows * 2.2))
    fig.suptitle("Image Reconstruction Quality", fontsize=14,
                 fontweight="bold", y=1.01)

    # Choose the same n_images indices for every row
    sample_idx = list(range(n_images))

    # Row 0 – original
    for col, img_idx in enumerate(sample_idx):
        ax = axes[0][col]
        ax.imshow(x_original[img_idx].reshape(28, 28), cmap="gray")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("Original", fontsize=9,
                          rotation=0, labelpad=55, va="center")

    # Rows 1+ – reconstructions
    for row, k in enumerate(configs, start=1):
        x_rec = reconstructions[k]
        for col, img_idx in enumerate(sample_idx):
            ax = axes[row][col]
            img = np.clip(x_rec[img_idx].reshape(28, 28), 0, 1)
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"{k} PCs", fontsize=9,
                              rotation=0, labelpad=55, va="center")

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "reconstruction.png")
    if save:
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"[INFO] Saved → {path}")
    plt.close(fig)
    return path


# ===========================================================================
# Utility: Mean Squared Error
# ===========================================================================

def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return the mean squared error between two image arrays."""
    return float(np.mean((original - reconstructed) ** 2))
