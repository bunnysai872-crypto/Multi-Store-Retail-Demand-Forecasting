"""
pca_model.py
------------
Fits, saves, loads, and applies PCA to Fashion-MNIST feature matrices.

Design decisions
----------------
* PCA is ONLY fitted on training data (x_train).
* The fitted model is saved to models/pca_model.pkl so subsequent
  runs can skip the expensive fitting step.
* Multiple component counts (2, 3, 10, 50, 100, N) are supported.
"""

import os
import pickle
import numpy as np
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "pca_model.pkl")

# Component counts used in experiments
COMPONENT_CONFIGS = [2, 3, 10, 50, 100]


# ---------------------------------------------------------------------------
# Fit / save / load
# ---------------------------------------------------------------------------

def fit_pca(x_train: np.ndarray, n_components: int = 100,
            random_state: int = 42) -> PCA:
    """
    Fit a PCA model on training data.

    Parameters
    ----------
    x_train      : preprocessed training feature matrix (N, 784)
    n_components : number of principal components to keep
    random_state : for reproducibility (uses 'randomized' svd_solver)

    Returns
    -------
    Fitted sklearn PCA object
    """
    print(f"[INFO] Fitting PCA with {n_components} components …")

    # 'randomized' solver is much faster for large matrices
    pca = PCA(n_components=n_components,
              svd_solver="randomized",
              random_state=random_state)
    pca.fit(x_train)

    print(f"[INFO] PCA fitting complete.")
    print(f"       Variance explained by {n_components} components: "
          f"{pca.explained_variance_ratio_.sum() * 100:.2f}%")
    return pca


def save_pca(pca: PCA, path: str = MODEL_PATH) -> None:
    """Persist a fitted PCA model to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pca, f)
    print(f"[INFO] PCA model saved → {path}")


def load_pca(path: str = MODEL_PATH) -> PCA:
    """
    Load a previously saved PCA model.

    Raises FileNotFoundError if the model file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved PCA model found at '{path}'. "
            "Run main.py first to fit and save the model."
        )
    with open(path, "rb") as f:
        pca = pickle.load(f)
    print(f"[INFO] PCA model loaded ← {path}")
    return pca


def load_or_fit_pca(x_train: np.ndarray,
                    n_components: int = 100,
                    force_refit: bool = False) -> PCA:
    """
    Load a saved PCA model if available; otherwise fit a new one.

    Parameters
    ----------
    x_train      : preprocessed training data
    n_components : components for a new fit (ignored when loading)
    force_refit  : if True, always refit even if a saved model exists
    """
    if not force_refit and os.path.exists(MODEL_PATH):
        try:
            pca = load_pca()
            # Sanity-check: make sure the loaded model matches the request
            if pca.n_components_ >= n_components:
                return pca
            print(f"[WARN] Saved model has {pca.n_components_} components "
                  f"but {n_components} were requested. Refitting …")
        except Exception as exc:
            print(f"[WARN] Could not load saved model ({exc}). Refitting …")

    pca = fit_pca(x_train, n_components=n_components)
    save_pca(pca)
    return pca


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def transform(pca: PCA, x: np.ndarray, n_components: int) -> np.ndarray:
    """
    Project x into the first n_components principal components.

    If the fitted PCA has fewer components than requested, we use
    as many as are available.
    """
    max_comp = pca.n_components_
    k = min(n_components, max_comp)
    return pca.transform(x)[:, :k]


def reconstruct(pca: PCA, x: np.ndarray, n_components: int) -> np.ndarray:
    """
    Reconstruct images from their n_components-dimensional representations.

    1. Project to k-dimensional space.
    2. Inverse-transform back to 784 dimensions.
    """
    k = min(n_components, pca.n_components_)

    # Build a temporary PCA object that uses only the first k components
    pca_k = _truncate_pca(pca, k)
    x_proj = x @ pca_k.components_.T          # (N, k)
    x_rec  = x_proj @ pca_k.components_       # (N, 784)
    x_rec += pca_k.mean_                      # add back mean
    return x_rec


def _truncate_pca(pca: PCA, k: int) -> PCA:
    """Return a shallow copy of pca restricted to the first k components."""
    p = PCA.__new__(PCA)
    p.__dict__.update(pca.__dict__)
    p.components_             = pca.components_[:k]
    p.explained_variance_     = pca.explained_variance_[:k]
    p.explained_variance_ratio_ = pca.explained_variance_ratio_[:k]
    p.n_components_           = k
    return p


# ---------------------------------------------------------------------------
# Variance statistics
# ---------------------------------------------------------------------------

def variance_summary(pca: PCA) -> dict:
    """
    Return a dictionary of variance statistics.

    Keys
    ----
    individual_variance : array of per-component explained variance ratios
    cumulative_variance : array of cumulative explained variance
    total_components    : total components in the fitted model
    n_for_90, n_for_95, n_for_99 : components needed for 90/95/99 % variance
    """
    ev  = pca.explained_variance_ratio_
    cum = np.cumsum(ev)

    def _n_for(threshold):
        idx = np.searchsorted(cum, threshold)
        return int(idx + 1)   # +1 because searchsorted is 0-indexed

    return {
        "individual_variance": ev,
        "cumulative_variance": cum,
        "total_components":    len(ev),
        "n_for_90":            _n_for(0.90),
        "n_for_95":            _n_for(0.95),
        "n_for_99":            _n_for(0.99),
    }


def run_experiments(pca: PCA, x_train: np.ndarray,
                    configs: list = None) -> list:
    """
    Run PCA experiments for multiple component counts.

    Returns a list of dicts, one per configuration:
        {
          "n_components"    : int,
          "variance"        : float (cumulative, 0–1),
          "reduction_pct"   : float (% reduction from 784 dims),
        }
    """
    if configs is None:
        configs = COMPONENT_CONFIGS

    original_dims = x_train.shape[1]   # 784
    cum = np.cumsum(pca.explained_variance_ratio_)
    results = []

    for k in configs:
        k_actual = min(k, pca.n_components_)
        variance = float(cum[k_actual - 1])
        reduction = (1 - k_actual / original_dims) * 100

        results.append({
            "n_components":   k,          # keep user-requested k for display
            "n_actual":       k_actual,   # actual components used
            "variance":       variance,
            "reduction_pct":  reduction,
        })

    return results
