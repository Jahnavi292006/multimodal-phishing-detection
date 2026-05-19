# Phish360: Multimodal Phishing Detection System

> A 9-model, 3-modality Late-Fusion stacking architecture for high-accuracy phishing URL detection — combining URL heuristics, NLP text semantics, and computer vision into a single unified system.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Pipeline Phases](#pipeline-phases)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Dependencies](#dependencies)

---

## Overview

**Phish360** is an end-to-end multimodal phishing detection system built for the MLPR (Machine Learning and Pattern Recognition) capstone project. The system detects phishing websites by simultaneously analyzing three independent signals:

| Modality | Signal Source | Feature Type |
|---|---|---|
| **URL** | Raw URL string | 54 handcrafted lexical heuristics |
| **Text** | Page HTML body text | Dense semantic embeddings (MPNet) |
| **Visual** | Website screenshot | CNN image embeddings (ResNet50) |

Three expert models are trained per modality (9 base models total). The best-performing model from each modality is then passed to a **Late-Fusion Meta-Classifier** that makes the final prediction. The full pipeline uses a **strict 80/20 lockbox split** with 5-fold stratified cross-validation to prevent data leakage.

---

## Architecture

```
                    +-----------------------------------------------+
                    |               INPUT: Website Data             |
                    |      URL string  |  HTML text  |  Screenshot  |
                    +-----------------------------------------------+
                           |                |               |
                           v                v               v
               +-----------------+  +--------------+  +---------------+
               |  MODALITY 1     |  |  MODALITY 2  |  |  MODALITY 3   |
               |  URL Experts    |  | Text Experts |  | Vision Experts|
               |                 |  |              |  |               |
               |  - XGBoost      |  |  - SVM (RBF) |  |  - LogReg     |
               |  - RandomForest |  |  - MLP       |  |  - LinearSVM  |
               |  - LightGBM     |  |  - LightGBM  |  |  - LightGBM   |
               +-----------------+  +--------------+  +---------------+
                           |                |               |
                           v                v               v
                      Best (F1)        Best (F1)        Best (F1)
                           |                |               |
                           +----------------+---------------+
                                            |
                                            v
                              +---------------------------+
                              |      META-CLASSIFIER      |
                              |    (Late-Fusion Stack)    |
                              |                           |
                              |  Candidates:              |
                              |  - LogisticRegression     |
                              |  - XGBoost Meta           |
                              |  - MLP Meta               |
                              +---------------------------+
                                            |
                                            v
                                  +-------------------+
                                  |   FINAL VERDICT   |
                                  |  Phishing / Legit |
                                  +-------------------+
```

---

## Dataset

**Phish360** — a multimodal phishing dataset stored as Parquet files.

| File | Contents |
|---|---|
| `Phish360_phish.parquet` | Confirmed phishing website records |
| `Phish360_legit.parquet` | Legitimate website records |

Each record contains:
- `URL` — the raw URL string
- `BeautifulSoup_text` — extracted HTML body text
- `image_path` — relative path to the website screenshot
- `Class` — label (`phish` or `legit`)

**Balancing:** The dataset is strictly balanced at 50/50 using random subsampling of the larger class, seeded at `random_state=42` for reproducibility.

**Train/Test Split:** 80% training set (used for cross-validation) / 20% lockbox test set — stratified, never touched until final evaluation.

---

## Pipeline Phases

### Phase 1 — Data Loading & Balancing
- Loads phishing and legitimate Parquet files
- Enforces strict 50/50 class balance
- Applies fail-fast image path validation: if >5% of images are missing, the pipeline halts with a detailed error before wasting compute
- Visual proof: renders 4 random sample images to confirm data accessibility

### Phase 2 — Feature Extraction

**URL Features (54 handcrafted heuristics):**
- Length features (URL, domain, TLD, subdomain, path, query)
- Structural flags (HTTPS, IP-as-domain, obfuscation, credentials in URL, hidden redirects)
- Character ratio features (letter ratio, digit ratio, special char ratio)
- Entropy features (URL entropy, domain entropy, path entropy)
- Character count features (hyphens, dots, slashes, `@`, `%`, `#`, `~`, `_`, `=`, `?`, `&`)
- Phishing keyword flags (`login`, `secure`, `account`, `verify`, `banking`, etc.)

**Text Features (MPNet Embeddings):**
- Model: `paraphrase-multilingual-mpnet-base-v2` via `sentence-transformers`
- Input: `BeautifulSoup_text` (HTML-extracted page body)
- Output: Dense 768-dimensional semantic embedding vector per page

**Visual Features (ResNet50 Embeddings):**
- Model: `ResNet50` pretrained on ImageNet (final classification head removed)
- Input: Website screenshot resized to 224×224, normalized to ImageNet stats
- Output: 2048-dimensional feature vector per image

### Phase 3 — Base Model Training (5-Fold CV)
- Each of the 9 base models is trained using **Stratified K-Fold (k=5)** on the training set
- Out-Of-Fold (OOF) predictions are recorded for each model
- Metrics computed per fold and averaged

### Phase 4 — Late-Fusion Meta-Classifier
- The best-performing model from each modality (selected by F1-score on OOF predictions) feeds its OOF predictions into a 3-feature meta-training matrix
- Three meta-classifier candidates are evaluated on the same 5-fold CV
- The winning meta-classifier is retrained on the full 3-column meta-training stack

### Phase 5 — Lockbox Evaluation
- The 20% held-out test set is processed through the full pipeline
- Base model test embeddings are extracted, stacked, and passed to the meta-classifier
- Final metrics are computed and all 11 visualizations are generated

---

## Models

### Modality 1: URL Expert Models

| Model | Key Hyperparameters |
|---|---|
| `XGBClassifier` | `n_estimators=150`, `max_depth=4`, `learning_rate=0.05` |
| `RandomForestClassifier` | `n_estimators=150`, `max_depth=6` |
| `LGBMClassifier` | `n_estimators=150`, `max_depth=4` |

### Modality 2: Text Expert Models

| Model | Key Hyperparameters |
|---|---|
| `SVC (RBF)` | `C=1.0`, `probability=True` |
| `MLPClassifier` | `hidden_layer_sizes=(128,)`, `alpha=0.01`, `early_stopping=True` |
| `LGBMClassifier` | `n_estimators=100`, `max_depth=3` |

### Modality 3: Vision Expert Models

| Model | Key Hyperparameters |
|---|---|
| `LogisticRegression` | `max_iter=500`, `C=0.1` |
| `SVC (Linear)` | `kernel='linear'`, `C=0.1`, `probability=True` |
| `LGBMClassifier` | `n_estimators=100`, `max_depth=3` |

### Meta-Classifier Candidates

| Model | Key Hyperparameters |
|---|---|
| `LogisticRegression` | `C=1.0` |
| `XGBClassifier` (Meta) | `n_estimators=50`, `max_depth=3` |
| `MLPClassifier` (Meta) | `hidden_layer_sizes=(16,)` |

---

## Evaluation Metrics

The following metrics are tracked for every model across all phases:

| Metric | Description |
|---|---|
| **Accuracy** | Proportion of correct predictions |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1-Score** | Harmonic mean of precision and recall |
| **MCC** | Matthews Correlation Coefficient — robust to class imbalance |
| **Spearman ρ** | Rank correlation between predicted probability and true label |
| **Pearson r** | Linear correlation between predicted probability and true label |
| **PR-AUC** | Area under the Precision-Recall curve |

All per-fold metrics are averaged across 5 folds and saved as structured CSV files.

---

## Visualizations

The pipeline generates 11 publication-ready plots, all saved as 300 DPI PNG files:

| Graph | Description |
|---|---|
| `Graph_1_Dataset_EDA.png` | Class balance proof + URL length distribution by class |
| `Graph_2_Base_Models_Showdown.png` | Accuracy and F1 comparison across all 9 base models |
| `Graph_3_Meta_Candidates.png` | Meta-classifier candidate comparison (Accuracy, F1, PR-AUC) |
| `Graph_4_Confusion_Matrix_Evolution.png` | 2×2 confusion matrix grid — URL winner, Text winner, Image winner, Final system |
| `Graph_5_ROC_Overlay.png` | ROC curves for all 4 final models overlaid with AUC annotations |
| `Graph_6_URL_Feature_Importance.png` | Top 15 most important URL heuristic features (tree-based models only) |
| `Graph_7_Modality_Weights.png` | Meta-classifier's relative weight assigned to each modality |
| `Graph_8_Model_Correlation.png` | Lower-triangle heatmap of OOF prediction correlations across all 9 models (diversity proof) |
| `Graph_9_Confidence_Density.png` | KDE plot of predicted probability distributions for phishing vs. legit samples |
| `Graph_10_Modality_Violins.png` | Split violin plots showing prediction spread and decisiveness per modality |
| `Graph_11_URL_Feature_Correlation.png` | Correlation heatmap of the top 20 highest-variance URL features |

---

## Project Structure

```
phish360/
│
├── Phish360/ # Dataset root (local, not tracked)
│ ├── Phish360_phish.parquet # Phishing website records
│ ├── Phish360_legit.parquet # Legitimate website records
│ ├── images/ # Website screenshots
│ │ └── *.png / *.jpg
│ │
│ ├── Final_Balanced_Phish360_with_Features.parquet # Saved after URL extraction
│ ├── Metrics_URL_Modality.csv
│ ├── Metrics_Text_Modality.csv
│ ├── Metrics_Image_Modality.csv
│ ├── Metrics_Meta_Classifier_Candidates.csv
│ ├── Metrics_FINAL_SYSTEM_EVALUATION.csv
│ │
│ ├── Graph_1_Dataset_EDA.png
│ ├── Graph_2_Base_Models_Showdown.png
│ ├── Graph_3_Meta_Candidates.png
│ ├── Graph_4_Confusion_Matrix_Evolution.png
│ ├── Graph_5_ROC_Overlay.png
│ ├── Graph_6_URL_Feature_Importance.png
│ ├── Graph_7_Modality_Weights.png
│ ├── Graph_8_Model_Correlation.png
│ ├── Graph_9_Confidence_Density.png
│ ├── Graph_10_Modality_Violins.png
│ └── Graph_11_URL_Feature_Correlation.png
│
└── Final_Phish360_Master_Pipeline_v4.ipynb # Main pipeline notebook
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (falls back to CPU automatically)
- ~8 GB RAM minimum; ~16 GB recommended for full image embedding extraction

### Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xgboost lightgbm scikit-learn
pip install sentence-transformers
pip install pandas numpy scipy matplotlib seaborn tqdm
pip install tldextract pillow
pip install pyarrow # for Parquet file support
```

Or install all at once:

```bash
pip install torch torchvision xgboost lightgbm scikit-learn sentence-transformers \
 pandas numpy scipy matplotlib seaborn tqdm tldextract pillow pyarrow
```

### Configure Data Path

Open the notebook and update the `BASE_DIR` variable in **Phase 1** to point to your local dataset:

```python
# Windows
BASE_DIR = r'C:\Users\YOUR_USERNAME\Downloads\phish360\Phish360'

# Linux / macOS
BASE_DIR = '/home/your_username/data/phish360/Phish360'
```

---

## Running the Pipeline

Open `Final_Phish360_Master_Pipeline_v4.ipynb` in Jupyter Notebook or JupyterLab and **run all cells in order**. The pipeline enforces a strict fail-fast protocol — it will halt with a descriptive error if data is missing, rather than silently training on zero-padded inputs.

Recommended execution order:

| Step | Cell | Description |
|---|---|---|
| 1 | Imports | Library imports and device detection |
| 2 | Phase 1 | Data loading, balancing, train/test split |
| 3 | Checkpoint | Image path validation + visual proof |
| 4 | Phase 2 | 54-feature URL extraction, save to Parquet |
| 5 | URL Training | 3 URL models, 5-fold CV |
| 6 | Text Training | MPNet encoding + 3 text models, 5-fold CV |
| 7 | Vision Training | ResNet50 encoding + 3 vision models, 5-fold CV |
| 8 | Meta Training | Best-of-3 stacking, meta-classifier selection |
| 9 | Fix Cell | Extract test set text + image embeddings |
| 10 | Lockbox Eval | Unlock test set, compute final metrics |
| 11 | Graphs 1–7 | Core performance visualizations |
| 12 | Graphs 8–11 | Advanced academic visualizations |

> **Do not skip the Fix Cell (Step 9).** It extracts test set embeddings that are required before the lockbox evaluation can run.

---

## Outputs

After a complete run, the following files are written to `BASE_DIR`:

| File | Description |
|---|---|
| `Final_Balanced_Phish360_with_Features.parquet` | Full dataset with all 54 URL features appended |
| `Metrics_URL_Modality.csv` | 5-fold CV metrics for XGBoost, RandomForest, LightGBM on URLs |
| `Metrics_Text_Modality.csv` | 5-fold CV metrics for SVM, MLP, LightGBM on text embeddings |
| `Metrics_Image_Modality.csv` | 5-fold CV metrics for LogReg, LinearSVM, LightGBM on image embeddings |
| `Metrics_Meta_Classifier_Candidates.csv` | 5-fold CV metrics for the 3 meta-classifier candidates |
| `Metrics_FINAL_SYSTEM_EVALUATION.csv` | Lockbox test set final evaluation metrics |
| `Graph_1` – `Graph_11` (PNG) | All 11 visualizations at 300 DPI |

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | ResNet50 image embedding extraction, GPU acceleration |
| `sentence-transformers` | MPNet text embedding extraction |
| `xgboost` | URL and meta-classifier gradient boosting |
| `lightgbm` | Fast gradient boosting across all 3 modalities |
| `scikit-learn` | SVM, MLP, LogReg, RandomForest, CV utilities, metrics |
| `pandas` / `numpy` | Data manipulation and array operations |
| `scipy` | Spearman and Pearson correlation metrics |
| `matplotlib` / `seaborn` | All visualizations |
| `tldextract` | Robust URL parsing and TLD extraction |
| `Pillow` | Image loading and preprocessing |
| `tqdm` | Progress bars for long extraction loops |
| `pyarrow` | Parquet file I/O |

---

## Notes

- **Reproducibility:** All random seeds are fixed at `42` throughout the pipeline.
- **Device:** The pipeline auto-detects CUDA and falls back to CPU. GPU is strongly recommended for the text (MPNet) and vision (ResNet50) embedding steps.
- **Fail-Fast Protocol:** The pipeline will intentionally crash if more than 5% of image paths are missing, preventing silent training on zero vectors.
- **No Data Leakage:** The 20% test set indices (`idx_test`) are locked before any feature extraction and never used during model training or meta-classifier selection.

---

*Built for MLPR Capstone — Plaksha University*
