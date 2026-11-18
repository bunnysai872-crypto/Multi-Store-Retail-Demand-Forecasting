"""
app.py
------
FashionVision – PCA Explorer
A professional Streamlit dashboard for the Fashion-MNIST PCA project.

Run with:
    streamlit run app.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FashionVision – PCA Explorer",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when launched from a different cwd
sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader  import CLASS_NAMES
from preprocessing   import preprocess
from pca_model       import (load_or_fit_pca, transform,
                              variance_summary, run_experiments,
                              COMPONENT_CONFIGS)
from reconstruction  import reconstruct_images, compute_all_errors

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
PALETTE     = px.colors.qualitative.Plotly[:10]

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: #1a1f2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #4f8ef7;
    }

    /* Section headers */
    h1, h2, h3 { color: #1a1f2e; }

    /* Page title pill */
    .page-title {
        background: linear-gradient(90deg, #4f8ef7, #9b59b6);
        color: white !important;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: inline-block;
    }

    /* Info box */
    .info-box {
        background: #eef3ff;
        border-left: 4px solid #4f8ef7;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Cached data-loading functions
# ===========================================================================

@st.cache_data(show_spinner="Loading Fashion-MNIST dataset …")
def load_data():
    from tensorflow.keras.datasets import fashion_mnist
    (x_train_raw, y_train), (x_test_raw, y_test) = fashion_mnist.load_data()
    return x_train_raw, y_train, x_test_raw, y_test


@st.cache_data(show_spinner="Preprocessing images …")
def get_preprocessed(x_train_raw, x_test_raw):
    x_train = preprocess(x_train_raw)
    x_test  = preprocess(x_test_raw)
    return x_train, x_test


@st.cache_resource(show_spinner="Fitting PCA model (this may take a minute) …")
def get_pca(x_train):
    return load_or_fit_pca(x_train, n_components=200)


@st.cache_data(show_spinner="Running PCA experiments …")
def get_experiments(_pca, x_train):
    return run_experiments(_pca, x_train, COMPONENT_CONFIGS)


@st.cache_data(show_spinner="Computing variance statistics …")
def get_variance_stats(_pca):
    return variance_summary(_pca)


@st.cache_data(show_spinner="Projecting data …")
def get_projections(_pca, x_train):
    x_2d = transform(_pca, x_train, 2)
    x_3d = transform(_pca, x_train, 3)
    return x_2d, x_3d


@st.cache_data(show_spinner="Reconstructing images …")
def get_reconstructions(_pca, x_test):
    sample = x_test[:300]
    recs = reconstruct_images(_pca, sample, [2, 3, 10, 50, 100])
    errs = compute_all_errors(sample, recs)
    return sample, recs, errs


# ===========================================================================
# Sidebar navigation
# ===========================================================================

PAGES = [
    "🏠  Dashboard Overview",
    "🖼️  Dataset Explorer",
    "⚙️  Data Preprocessing",
    "📊  PCA Analysis",
    "🔵  2D Visualisation",
    "🌐  3D Visualisation",
    "📈  Explained Variance",
    "🔄  Image Reconstruction",
    "✅  Results & Conclusion",
]

with st.sidebar:
    st.markdown("## 👗 FashionVision")
    st.markdown("### PCA Explorer")
    st.markdown("---")
    page = st.radio("Navigate to", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Fashion-MNIST PCA Project**")
    st.markdown("Dimensionality Reduction with PCA on 70,000 clothing images.")
    st.markdown("---")
    st.caption("Dataset: Zalando Research / Keras")


# ===========================================================================
# Load all data once
# ===========================================================================

try:
    x_train_raw, y_train, x_test_raw, y_test = load_data()
    x_train, x_test                           = get_preprocessed(x_train_raw, x_test_raw)
    pca                                        = get_pca(x_train)
    experiments                                = get_experiments(pca, x_train)
    vstats                                     = get_variance_stats(pca)
    x_2d, x_3d                                = get_projections(pca, x_train)
    x_test_sample, reconstructions, rec_errors = get_reconstructions(pca, x_test)
    data_loaded = True
except Exception as e:
    st.error(f"**Data loading error:** {e}")
    st.info("Make sure TensorFlow is installed: `pip install tensorflow`")
    data_loaded = False
    st.stop()


# ===========================================================================
# Helper utilities
# ===========================================================================

def metric_row(cols_data: list):
    """Render a row of Streamlit metric boxes."""
    cols = st.columns(len(cols_data))
    for col, (label, value, delta) in zip(cols, cols_data):
        col.metric(label, value, delta)


def section_title(icon: str, title: str):
    st.markdown(f'<div class="page-title">{icon} {title}</div>',
                unsafe_allow_html=True)


def info_box(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


# ===========================================================================
# Page 1 – Dashboard Overview
# ===========================================================================

if page == PAGES[0]:
    section_title("🏠", "Dashboard Overview")

    st.markdown("""
    Welcome to **FashionVision – PCA Explorer**, an interactive machine learning
    dashboard that demonstrates how **Principal Component Analysis (PCA)** can
    reduce 784-dimensional clothing image data into just 2 or 3 dimensions while
    preserving most of the meaningful information.
    """)

    # Key metrics
    st.markdown("### 📊 Dataset at a Glance")
    metric_row([
        ("Total Images",    "70,000",   None),
        ("Training Images", "60,000",   None),
        ("Testing Images",  "10,000",   None),
        ("Original Features","784",     None),
        ("Clothing Categories","10",    None),
    ])

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Project Objective")
        st.markdown("""
        Reduce **784 pixel features** per image to a much smaller number of
        **principal components** while:
        - Preserving maximum variance (information).
        - Enabling 2D and 3D visualisation of high-dimensional data.
        - Demonstrating image reconstruction quality vs component count.
        """)

    with col2:
        st.markdown("### 🔬 What is PCA?")
        st.markdown("""
        **Principal Component Analysis (PCA)** is an unsupervised linear
        technique that:
        1. Finds the directions (components) of maximum variance.
        2. Projects data onto those directions.
        3. Discards low-variance directions, reducing dimensionality.
        4. Allows reconstruction back to the original space.
        """)

    st.markdown("---")
    st.markdown("### 🗺️ Project Workflow")

    workflow_cols = st.columns(5)
    steps = [
        ("1️⃣", "Load Dataset",     "70K Fashion-MNIST images from Keras"),
        ("2️⃣", "Preprocess",       "Normalise & flatten to 784 features"),
        ("3️⃣", "Fit PCA",          "Learn principal components on train set"),
        ("4️⃣", "Transform",        "Project images to 2D / 3D / N-D space"),
        ("5️⃣", "Visualise",        "Interactive plots & reconstruction"),
    ]
    for col, (num, title, desc) in zip(workflow_cols, steps):
        with col:
            st.markdown(f"**{num} {title}**")
            st.caption(desc)


# ===========================================================================
# Page 2 – Dataset Explorer
# ===========================================================================

elif page == PAGES[1]:
    section_title("🖼️", "Dataset Explorer")

    # Class distribution
    st.markdown("### 📦 Class Distribution")
    counts = [int(np.sum(y_train == i)) for i in range(10)]
    df_dist = pd.DataFrame({"Category": CLASS_NAMES, "Count": counts})
    fig_bar = px.bar(df_dist, x="Category", y="Count",
                     color="Category", color_discrete_sequence=PALETTE,
                     title="Training Set Class Distribution",
                     text="Count")
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧵 Sample Images by Category")

    # Show 5 images per class
    n_per_class = 5
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        indices = np.where(y_train == cls_idx)[0][:n_per_class]
        cols    = st.columns(n_per_class + 1)
        with cols[0]:
            st.markdown(f"**{cls_name}**")
        for i, img_idx in enumerate(indices):
            with cols[i + 1]:
                fig, ax = plt.subplots(figsize=(1.5, 1.5))
                ax.imshow(x_train_raw[img_idx], cmap="gray")
                ax.axis("off")
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

    st.markdown("---")
    st.markdown("### 📋 Dataset Info Table")
    info_df = pd.DataFrame({
        "Property":   ["Total images", "Training", "Testing",
                       "Image size", "Features", "Classes", "Pixel range"],
        "Value":      ["70,000", "60,000", "10,000",
                       "28 × 28", "784", "10", "0 – 255"],
    })
    st.table(info_df)


# ===========================================================================
# Page 3 – Data Preprocessing
# ===========================================================================

elif page == PAGES[2]:
    section_title("⚙️", "Data Preprocessing")

    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Shape (train)", "60000 × 28 × 28")
    col2.metric("After Flatten",     "60000 × 784")
    col3.metric("Pixel Range",       "0.0 → 1.0 (normalised)")

    st.markdown("---")

    st.markdown("### Step-by-Step Preprocessing Pipeline")

    steps_data = [
        ("Step 1", "Cast to float32",
         "Convert integer pixel values (uint8) to float32 to allow\n"
         "mathematical operations without integer overflow."),
        ("Step 2", "Normalise pixels",
         "Divide by 255 to scale all pixel values from [0, 255] → [0, 1].\n"
         "This prevents features with large numeric ranges from dominating PCA."),
        ("Step 3", "Flatten images",
         "Reshape each 28×28 image into a 1-D vector of 784 features.\n"
         "PCA requires a 2-D matrix of shape (samples, features)."),
        ("Step 4", "Split maintenance",
         "PCA is fitted ONLY on training data.\n"
         "The test set is transformed using the same fitted PCA object\n"
         "to prevent data leakage."),
    ]

    for step, title, desc in steps_data:
        with st.expander(f"**{step}: {title}**", expanded=True):
            st.code(desc)

    st.markdown("---")
    st.markdown("### 🔍 Before vs After")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before Preprocessing**")
        st.markdown("- Shape: `(60000, 28, 28)`")
        st.markdown("- dtype: `uint8`")
        st.markdown("- Pixel range: `0 – 255`")

    with c2:
        st.markdown("**After Preprocessing**")
        st.markdown("- Shape: `(60000, 784)`")
        st.markdown("- dtype: `float32`")
        st.markdown("- Pixel range: `0.0 – 1.0`")

    # Show one image before & after (histogram)
    st.markdown("---")
    st.markdown("### 📊 Pixel Value Distribution (sample image)")
    sample_img = x_train_raw[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].hist(sample_img.flatten(), bins=50, color="#4f8ef7", edgecolor="white")
    axes[0].set_title("Before Normalisation (uint8)")
    axes[0].set_xlabel("Pixel value")
    axes[0].set_ylabel("Count")

    sample_norm = sample_img.astype(np.float32) / 255.0
    axes[1].hist(sample_norm.flatten(), bins=50, color="#9b59b6", edgecolor="white")
    axes[1].set_title("After Normalisation (float32)")
    axes[1].set_xlabel("Pixel value")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ===========================================================================
# Page 4 – PCA Analysis
# ===========================================================================

elif page == PAGES[3]:
    section_title("📊", "PCA Analysis")

    st.markdown("### ⚙️ Select Number of Components")
    n_comp = st.slider("Principal Components", min_value=2,
                       max_value=100, value=50, step=1)

    cum_var = np.cumsum(vstats["individual_variance"])
    actual_var = float(cum_var[n_comp - 1]) * 100
    reduction  = (1 - n_comp / 784) * 100

    metric_row([
        ("Original Dimensions",   "784",                None),
        ("Reduced Dimensions",    str(n_comp),          f"−{reduction:.1f}%"),
        ("Variance Explained",    f"{actual_var:.2f}%", None),
        ("Info Lost",             f"{100-actual_var:.2f}%", None),
    ])

    st.markdown("---")
    st.markdown("### 📋 Experiment Summary Table")

    rows = []
    for exp in experiments:
        k   = exp["n_actual"]
        var = exp["variance"] * 100
        red = exp["reduction_pct"]
        mse = rec_errors.get(k, "N/A")
        rows.append({
            "Components":        k,
            "Reduced Dims":      k,
            "Explained Var (%)": f"{var:.2f}",
            "Reduction (%)":     f"{red:.2f}",
            "Recon. MSE":        f"{mse:.6f}" if isinstance(mse, float) else mse,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📖 PCA Equation")
    st.markdown(r"""
    Given data matrix **X** (N × 784), PCA finds orthogonal directions that
    maximise variance:

    $$Z = X \cdot W_k$$

    where $W_k$ are the top-$k$ eigenvectors of the covariance matrix
    $C = \frac{1}{N} X^T X$.

    Reconstruction is the inverse projection:

    $$\hat{X} = Z \cdot W_k^T + \bar{x}$$
    """)


# ===========================================================================
# Page 5 – 2D Visualisation
# ===========================================================================

elif page == PAGES[4]:
    section_title("🔵", "PCA 2D Visualisation")

    st.markdown("Each point represents one clothing image projected onto "
                "the first 2 principal components.")

    # Sample for performance
    sample_size = st.slider("Sample size", 1000, 15000, 5000, 500)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_2d), min(sample_size, len(x_2d)), replace=False)

    x2d_s = x_2d[idx]
    y_s   = y_train[idx]
    class_labels = [CLASS_NAMES[int(l)] for l in y_s]

    fig = px.scatter(
        x=x2d_s[:, 0], y=x2d_s[:, 1],
        color=class_labels,
        color_discrete_sequence=PALETTE,
        labels={"x": "Principal Component 1",
                "y": "Principal Component 2",
                "color": "Category"},
        title=f"PCA 2D — Fashion-MNIST ({sample_size:,} samples)",
        opacity=0.6,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=580, legend_title_text="Category")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    info_box(
        "💡 <strong>Interpretation:</strong> Clusters of similar colours indicate "
        "that PCA has successfully separated clothing categories even though "
        "it had no access to the class labels during fitting."
    )


# ===========================================================================
# Page 6 – 3D Visualisation
# ===========================================================================

elif page == PAGES[5]:
    section_title("🌐", "PCA 3D Visualisation")

    st.markdown("Rotate, zoom, and pan the 3D scatter plot to explore "
                "how the 10 clothing categories separate in 3-component PCA space.")

    sample_size_3d = st.slider("Sample size", 500, 8000, 3000, 500)
    rng = np.random.default_rng(42)
    idx3 = rng.choice(len(x_3d), min(sample_size_3d, len(x_3d)), replace=False)

    x3d_s = x_3d[idx3]
    y3_s  = y_train[idx3]
    class_labels3 = [CLASS_NAMES[int(l)] for l in y3_s]

    fig3d = px.scatter_3d(
        x=x3d_s[:, 0], y=x3d_s[:, 1], z=x3d_s[:, 2],
        color=class_labels3,
        color_discrete_sequence=PALETTE,
        labels={"x": "PC 1", "y": "PC 2", "z": "PC 3",
                "color": "Category"},
        title=f"PCA 3D — Fashion-MNIST ({sample_size_3d:,} samples)",
        opacity=0.7,
    )
    fig3d.update_traces(marker=dict(size=2.5))
    fig3d.update_layout(height=620)
    st.plotly_chart(fig3d, use_container_width=True)


# ===========================================================================
# Page 7 – Explained Variance
# ===========================================================================

elif page == PAGES[6]:
    section_title("📈", "Explained Variance")

    ev  = vstats["individual_variance"]
    cum = vstats["cumulative_variance"]
    n90 = vstats["n_for_90"]
    n95 = vstats["n_for_95"]
    n99 = vstats["n_for_99"]

    metric_row([
        ("90% Variance", f"{n90} components", None),
        ("95% Variance", f"{n95} components", None),
        ("99% Variance", f"{n99} components", None),
    ])

    st.markdown("---")
    st.markdown("### 📉 Scree Plot — Individual Component Variance")

    n_show = st.slider("Show first N components", 10, 100, 50, 5)
    x_vals = list(range(1, n_show + 1))

    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(
        x=x_vals, y=(ev[:n_show] * 100).tolist(),
        name="Individual variance",
        marker_color="#4f8ef7",
    ))
    fig_scree.update_layout(
        title="Scree Plot — Individual Explained Variance",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance (%)",
        height=400,
    )
    st.plotly_chart(fig_scree, use_container_width=True)

    st.markdown("### 📈 Cumulative Explained Variance")

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=x_vals, y=(cum[:n_show] * 100).tolist(),
        mode="lines+markers", name="Cumulative variance",
        marker=dict(size=4), line=dict(color="darkorange", width=2),
    ))
    for pct, n_comp, colour in [(90, n90, "blue"), (95, n95, "green"),
                                 (99, n99, "red")]:
        fig_cum.add_hline(y=pct, line_dash="dash", line_color=colour,
                          annotation_text=f"{pct}% ({n_comp} comps)",
                          annotation_position="right")

    fig_cum.update_layout(
        title="Cumulative Explained Variance",
        xaxis_title="Number of Components",
        yaxis_title="Cumulative Variance (%)",
        yaxis_range=[0, 102],
        height=420,
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    st.markdown("---")
    info_box(
        f"💡 <strong>Recommendation:</strong> Use <strong>{n90} components</strong> "
        f"to capture 90% of the information, reducing 784 → {n90} dimensions "
        f"({(1-n90/784)*100:.1f}% size reduction)."
    )


# ===========================================================================
# Page 8 – Image Reconstruction
# ===========================================================================

elif page == PAGES[7]:
    section_title("🔄", "Image Reconstruction")

    st.markdown("See how image quality changes as we reconstruct from "
                "different numbers of principal components.")

    metric_row([
        ("10 PCs MSE",  f"{rec_errors.get(10, 0):.6f}", None),
        ("50 PCs MSE",  f"{rec_errors.get(50, 0):.6f}", None),
        ("100 PCs MSE", f"{rec_errors.get(100, 0):.6f}", None),
    ])

    st.markdown("---")
    st.markdown("### 🖼️ Side-by-Side Comparison")

    # Pick how many images to display
    n_show = st.slider("Number of example images", 3, 10, 5)
    img_idx = st.slider("Starting image index", 0, 290, 0)

    # Row labels
    row_labels = ["Original", "10 PCs", "50 PCs", "100 PCs"]
    rows_data  = [
        x_test_sample,
        reconstructions[10],
        reconstructions[50],
        reconstructions[100],
    ]

    for label, row_imgs in zip(row_labels, rows_data):
        st.markdown(f"**{label}**")
        cols = st.columns(n_show)
        for i, col in enumerate(cols):
            with col:
                img = row_imgs[img_idx + i].reshape(28, 28)
                img = np.clip(img, 0, 1)
                fig, ax = plt.subplots(figsize=(1.5, 1.5))
                ax.imshow(img, cmap="gray")
                ax.axis("off")
                col.pyplot(fig, use_container_width=False)
                plt.close(fig)

    st.markdown("---")
    st.markdown("### 📊 Reconstruction Error (MSE) vs Components")
    comp_list = sorted(rec_errors.keys())
    mse_vals  = [rec_errors[k] for k in comp_list]

    fig_mse = px.bar(
        x=[str(k) for k in comp_list], y=mse_vals,
        labels={"x": "PCA Components", "y": "MSE"},
        title="Reconstruction MSE vs Number of Components",
        color=mse_vals, color_continuous_scale="RdYlGn_r",
    )
    fig_mse.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_mse, use_container_width=True)

    info_box(
        "💡 Lower MSE = better reconstruction. "
        "More components → closer to the original image but higher dimensionality."
    )


# ===========================================================================
# Page 9 – Results & Conclusion
# ===========================================================================

elif page == PAGES[8]:
    section_title("✅", "Results & Conclusion")

    st.markdown("### 🏆 Final PCA Experiment Results")

    rows = []
    for exp in experiments:
        k   = exp["n_actual"]
        var = exp["variance"] * 100
        red = exp["reduction_pct"]
        mse = rec_errors.get(k, "N/A")
        rows.append({
            "Components":        k,
            "Reduced Dims":      k,
            "Explained Var (%)": f"{var:.2f}",
            "Reduction (%)":     f"{red:.2f}",
            "Recon. MSE":        f"{mse:.6f}" if isinstance(mse, float) else mse,
        })

    df_res = pd.DataFrame(rows)
    st.dataframe(df_res, use_container_width=True)

    st.markdown("---")

    # Dimensionality comparison
    st.markdown("### 📉 Dimensionality Comparison Chart")
    labels = ["Original (784)"] + [f"{e['n_actual']} PCs" for e in experiments]
    dims   = [784] + [e["n_actual"] for e in experiments]
    pcts   = [0.0] + [e["reduction_pct"] for e in experiments]
    vars_  = [100.0] + [e["variance"]*100 for e in experiments]

    df_comp = pd.DataFrame({
        "Config":        labels,
        "Dimensions":    dims,
        "Variance (%)":  vars_,
        "Reduction (%)": pcts,
    })
    fig_comp = px.bar(df_comp, x="Config", y="Dimensions",
                      color="Variance (%)", color_continuous_scale="Blues",
                      text="Reduction (%)",
                      title="Dimensionality vs Variance Retained")
    fig_comp.update_traces(texttemplate="−%{text:.1f}%", textposition="outside")
    fig_comp.update_layout(height=430)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Advantages of PCA")
        st.markdown("""
        - Reduces storage and computation requirements significantly.
        - Removes noise by discarding low-variance components.
        - Enables 2D/3D visualisation of high-dimensional data.
        - Speeds up downstream machine learning algorithms.
        - Works without class labels (unsupervised).
        """)

    with col2:
        st.markdown("### ⚠️ Limitations of PCA")
        st.markdown("""
        - Assumes linear relationships between features.
        - Components are not directly interpretable as features.
        - Some information (variance) is always lost.
        - Sensitive to feature scaling (must normalise first).
        - Slower on very large datasets.
        """)

    st.markdown("---")
    st.markdown("### 🔭 Future Enhancements")
    st.markdown("""
    - **t-SNE / UMAP**: Non-linear dimensionality reduction for better cluster separation.
    - **Kernel PCA**: Extend PCA to non-linear manifolds.
    - **Autoencoder**: Deep learning-based dimensionality reduction.
    - **Classification pipeline**: Use PCA features as input to a classifier.
    - **Incremental PCA**: Handle datasets too large to fit in memory.
    """)

    st.markdown("---")
    st.success(
        "✅ **Conclusion:** PCA successfully reduces Fashion-MNIST from 784 to "
        f"{vstats['n_for_95']} dimensions while retaining 95% of the variance. "
        "The 2D and 3D projections clearly show that PCA separates most clothing "
        "categories even without using class labels during fitting."
    )
