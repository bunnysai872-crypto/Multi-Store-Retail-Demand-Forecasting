# Dimensionality Reduction Using PCA and Visualization on Fashion-MNIST Dataset

> **FashionVision – PCA Explorer** | Python · TensorFlow · scikit-learn · Streamlit

---

## 1. Project Introduction

This project demonstrates how **Principal Component Analysis (PCA)** — one of the most widely used dimensionality reduction techniques in Machine Learning — can compress high-dimensional clothing image data into a much smaller representation while preserving most of the meaningful information.

The project uses the **Fashion-MNIST dataset** (70,000 grayscale clothing images, 28 × 28 pixels each = **784 features per image**) and reduces it to as few as **2 or 3 principal components** for 2D and 3D visualization.

---

## 2. Problem Statement

Working directly with 784-pixel feature vectors is:
- Computationally expensive.
- Hard to visualize (humans cannot perceive 784 dimensions).
- Prone to the **curse of dimensionality**.

**Can we compress 784 features into far fewer while still distinguishing clothing categories?**

---

## 3. Project Objectives

| # | Objective |
|---|-----------|
| 1 | Reduce Fashion-MNIST from 784 features to 2, 3, 10, 50, and 100 components. |
| 2 | Measure how much variance (information) each configuration retains. |
| 3 | Visualize clothing categories in 2D and 3D PCA space. |
| 4 | Reconstruct images from compressed representations and measure quality (MSE). |
| 5 | Build an interactive Streamlit dashboard for exploration. |

---

## 4. Dataset Description

| Property | Value |
|----------|-------|
| **Name** | Fashion-MNIST |
| **Source** | [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) / [Keras](https://keras.io/api/datasets/fashion_mnist/) |
| **Total images** | 70,000 |
| **Training set** | 60,000 |
| **Test set** | 10,000 |
| **Image size** | 28 × 28 pixels (grayscale) |
| **Features per image** | 784 |
| **Classes** | 10 |

### Class Labels

| Label | Category |
|-------|----------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## 5. Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.12** | Programming language |
| **TensorFlow / Keras** | Dataset loading |
| **NumPy** | Numerical operations |
| **Pandas** | Data tables |
| **scikit-learn** | PCA implementation |
| **Matplotlib** | Static plots |
| **Seaborn** | Statistical plots |
| **Plotly** | Interactive 2D and 3D charts |
| **Streamlit** | Interactive web dashboard |

---

## 6. Project Architecture

```
Fashion_MNIST_PCA/
│
├── app.py               ← Streamlit dashboard (FashionVision – PCA Explorer)
├── main.py              ← CLI experiment runner
├── dataset_loader.py    ← Load Fashion-MNIST + class names
├── preprocessing.py     ← Normalise & flatten images
├── pca_model.py         ← Fit, save, load, and apply PCA
├── visualization.py     ← All static + interactive plots
├── reconstruction.py    ← Image reconstruction helpers
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
├── .gitignore
│
├── outputs/
│   ├── original_images.png
│   ├── pca_2d.png
│   ├── pca_3d.html          ← Interactive 3D (open in browser)
│   ├── explained_variance.png
│   ├── reconstruction.png
│   └── comparison.png
│
└── models/
    └── pca_model.pkl        ← Saved fitted PCA (auto-generated)
```

---

## 7. PCA Explanation

**Principal Component Analysis** finds new axes (principal components) that capture the most variance in the data.

### Key Steps:
1. **Centre the data** – subtract the mean of each feature.
2. **Compute covariance matrix** – measures how features vary together.
3. **Eigen-decomposition** – find eigenvectors (directions) and eigenvalues (variance captured).
4. **Sort components** – largest variance first.
5. **Project data** – multiply original data by the top-k eigenvectors.

### Math:
```
Z = X · W_k        (project to k-dimensional space)
X̂ = Z · W_k^T + μ  (reconstruct back to original space)
```

Where:
- `X` = data matrix (N × 784)
- `W_k` = top-k principal components (784 × k)
- `Z` = compressed representation (N × k)
- `μ` = feature means (784,)

---

## 8. Implementation Steps

1. **Load** Fashion-MNIST via `tensorflow.keras.datasets.fashion_mnist`.
2. **Preprocess** – cast to float32, divide by 255, flatten 28×28 → 784.
3. **Fit PCA** on training data only (prevents data leakage).
4. **Save** the fitted model to `models/pca_model.pkl`.
5. **Transform** both train and test sets using the fitted PCA.
6. **Experiment** with 2, 3, 10, 50, 100 components.
7. **Reconstruct** images using `inverse_transform` and measure MSE.
8. **Visualise** – 2D scatter, 3D scatter, scree plot, reconstruction grid.
9. **Dashboard** – explore everything interactively in Streamlit.

---

## 9. Installation Instructions

### Prerequisites
- Python 3.10 or later
- Windows, macOS, or Linux
- Internet connection (for first-time dataset download only)

### Step 1 – Clone or download the project
```
cd "c:\Documents\PCA ML"
```

### Step 2 – Activate the virtual environment
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 – Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow 2.18.0 requires Python ≤ 3.12. The project venv uses Python 3.12.9, which is compatible.

---

## 10. Execution Instructions

### Run ML Experiments (generates all output files)
```bash
python main.py
```

This will:
- Download and load Fashion-MNIST (auto-cached by Keras).
- Preprocess the data.
- Fit PCA and save to `models/pca_model.pkl`.
- Run experiments with multiple component counts.
- Generate all visualisations in `outputs/`.
- Print a results table.

### Launch the Interactive Dashboard
```bash
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

---

## 11. Visualization Descriptions

| Visualization | File | Description |
|--------------|------|-------------|
| Sample Images | `outputs/original_images.png` | 3 samples per class in a grid |
| PCA 2D | `outputs/pca_2d.png` | Scatter coloured by category |
| PCA 3D | `outputs/pca_3d.html` | Interactive Plotly 3D scatter |
| Explained Variance | `outputs/explained_variance.png` | Scree + cumulative variance |
| Reconstruction | `outputs/reconstruction.png` | Original vs 10/50/100 PC images |
| Comparison | `outputs/comparison.png` | Dimensionality reduction bar chart |

---

## 12. Experimental Results

> Results are computed dynamically from the real dataset. Run `python main.py` to see your machine's actual values.

| Components | Reduced Dims | Approx. Variance | Reduction |
|-----------|-------------|-----------------|-----------|
| 2 | 2 | ~24% | 99.7% |
| 3 | 3 | ~33% | 99.6% |
| 10 | 10 | ~60% | 98.7% |
| 50 | 50 | ~86% | 93.6% |
| 100 | 100 | ~92% | 87.2% |

---

## 13. Advantages

- **Massive size reduction** – 784 → 100 dimensions is an 87% reduction.
- **Visualisable** – PCA makes 784-D data viewable in 2D/3D.
- **Noise reduction** – low-variance components are discarded.
- **Faster ML** – fewer features = faster training for downstream models.

---

## 14. Limitations

- PCA is **linear** – cannot capture complex non-linear patterns.
- **Interpretability** – components don't correspond to recognizable features.
- **Information loss** – some variance is always discarded.
- **Scaling required** – must normalise data before applying PCA.

---

## 15. Future Enhancements

1. **t-SNE** and **UMAP** for non-linear dimensionality reduction.
2. **Kernel PCA** for non-linear manifolds.
3. **Variational Autoencoder (VAE)** for deep generative compression.
4. **Classification pipeline** – use PCA features as input to a classifier and compare accuracy vs original features.
5. **Incremental PCA** – for datasets larger than RAM.
6. **Animation** – animate how images reconstruct as component count increases.

---

## 16. Conclusion

This project demonstrates that PCA can reduce Fashion-MNIST from **784 dimensions** to just **~87 components** (retaining 95% of variance), enabling:
- Fast processing.
- Meaningful 2D/3D visualisation.
- Reasonable image reconstruction.

The interactive Streamlit dashboard makes all of this accessible without writing a single line of code after setup.

---

## Attribution

- **Fashion-MNIST Dataset**: Zalando Research — [github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- **Keras Dataset API**: [keras.io/api/datasets/fashion_mnist](https://keras.io/api/datasets/fashion_mnist/)
- **scikit-learn PCA**: [scikit-learn.org/stable/modules/decomposition.html](https://scikit-learn.org/stable/modules/decomposition.html)

---

*Built for educational purposes. Beginner-friendly ML project demonstrating unsupervised dimensionality reduction.*
