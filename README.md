# Multi-Signal Hybrid Restaurant Recommendation System

A content-based recommendation system addressing the cold-start problem in restaurant discovery. This project evolved from a simple Linear Weighted Combination to a Deep Neural Network Learning-to-Rank architecture, achieving **R² = 0.852** and **NDCG@10 = 0.974** on held-out test data.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📊 Key Results

### Model Performance (Test Set: n=1,627)

| Model | R² | RMSE | MAE | NDCG@10 |
|-------|-----|------|-----|---------|
| OLS (Baseline) | 0.8516 | 0.2509 | 0.1996 | 0.9573 |
| XGBoost (Primary) | 0.8507 | 0.2516 | 0.1968 | 0.9301 |
| **DNN** | **0.8529** | **0.2497** | **0.1968** | **0.9741** |

**Key Finding**: Three architecturally distinct models converge to identical R² (~0.85), validating the **Data-Centric AI hypothesis**: when features effectively capture the signal, model complexity provides minimal marginal benefit.

### Ablation Study

| Configuration | R² | Δ R² | Interpretation |
|---------------|-----|------|----------------|
| Full Model | 0.8507 | — | Baseline |
| w/o Sentiment | 0.3555 | **-0.4953** | Sentiment explains 49.5% of variance |
| w/o X_sim | 0.8498 | -0.0009 | Validates two-phase design |
| w/o Cuisine | 0.8451 | -0.0056 | Marginal contribution |

### MMR Diversity Optimization

| Method | NDCG@20 | ILD@20 | Δ NDCG | Δ ILD |
|--------|---------|--------|--------|-------|
| Greedy (λ=1.0) | 0.934 | 0.306 | — | — |
| **MMR (λ=0.7)** | **0.940** | **0.394** | **+0.6%** | **+29%** |

**Pareto Improvement**: At λ=0.7, both relevance AND diversity improve simultaneously.

---

## 🔄 Project Evolution

### Original Proposal
```
R_Linear = w₁·X_Sent + w₂·X_sim + w₃·X_pop
```
- 3 features
- Manual weight tuning
- Linear combination only

### Current Implementation
```
R_DNN = f_DNN(X; θ)  where X ∈ ℝ³⁶
```
- 36 engineered features
- Weights learned via gradient descent
- Non-linear interactions captured

**Justification**: The original three-feature model lacked capacity to capture sentiment dominance pattern and non-linear feature interactions discovered during EDA.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│  01_yelp_etl.py       →  Data extraction & filtering               │
│  02_spatial_etl.py    →  K-Means spatial clustering                │
│  03_sentiment.py      →  VADER three-moment extraction             │
│  04_sbert.py          →  Hierarchical chunking + SBERT             │
│  05_similarity_pop.py →  X_sim and X_pop computation               │
│  06_feature_builder.py→  36-feature assembly                       │
│  07_model_training.py →  OLS, XGBoost, DNN training                │
│  08_ablation.py       →  Feature importance analysis               │
│  09_mmr.py            →  Diversity evaluation                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
restaurant-recommender/
├── src/                        # Pipeline scripts (01-09)
│   ├── config.py               # Central configuration
│   ├── 01_yelp_etl.py          # Data extraction
│   ├── 02_spatial_etl.py       # Spatial clustering
│   ├── 03_sentiment.py         # VADER sentiment
│   ├── 04_sbert.py             # SBERT embeddings
│   ├── 05_similarity_pop.py    # Similarity computation
│   ├── 06_feature_builder.py   # Feature assembly
│   ├── 07_model_training.py    # Model training
│   ├── 08_ablation.py          # Ablation study
│   └── 09_mmr.py               # MMR evaluation
│
├── data/processed/             # Feature datasets
│   ├── features_all.parquet
│   └── restaurant_embeddings.parquet
│
├── models/                     # Trained checkpoints
│   ├── ols_model.joblib
│   ├── xgb_model.joblib
│   ├── dnn_model.keras
│   └── scaler.joblib
│
├── results/                    # Evaluation outputs
├── figures/                    # Visualizations
│
├── requirements.txt
├── run_models.py               # Quick-start (Local)
└── run_Models.ipynb            # Quick-start (Colab)
```

---

## 🚀 Quick Start

### Option 1: Run with Pre-processed Data (Recommended)

```bash
# Clone repository
git clone https://github.com/YourUsername/restaurant-recommender.git
cd restaurant-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run model training & evaluation
python run_models.py
```

### Option 2: Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YourUsername/restaurant-recommender/blob/main/run_Models.ipynb)

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/capstone
!pip install -q xgboost sentence-transformers vaderSentiment
!python run_models.py
```

---

## 🔬 Feature Engineering

### 36-Feature Summary

| Category | Features | Count | Description |
|----------|----------|-------|-------------|
| **Semantic** | X_sim | 1 | Cosine similarity (two-phase design) |
| **Sentiment** | X_sent_mean, X_sent_std, X_sent_skew | 3 | Three-moment distribution |
| **Popularity** | X_pop | 1 | log(review_count + 1) |
| **Amenities** | TakeOut, Delivery, OutdoorSeating, etc. | 8 | Binary attributes |
| **Cuisine** | American, Mexican, Italian, etc. | 22 | One-hot encoded |
| **Spatial** | neighborhood_type | 1 | K-Means cluster |
| **Total** | | **36** | |

### Two-Phase X_sim Design

| Phase | Query | Purpose |
|-------|-------|---------|
| **Training** | Fixed proxy: *"delicious food excellent service nice atmosphere good value"* | Quality signal |
| **Inference** | User's actual query | Semantic relevance |

This design explains minimal training ablation impact (ΔR² = -0.0009) alongside substantial inference utility for semantic search.

---

## 📊 Dataset

| Metric | Value |
|--------|-------|
| **Total Restaurants** | 10,841 |
| **Total Reviews** | 2,390,705 |
| **Geographic Coverage** | PA, FL, TN, AZ, LA |
| **Minimum Reviews** | 50 per restaurant |
| **Train / Val / Test Split** | 70% / 15% / 15% |
| **Test Set Size** | n = 1,627 |
| **Reviews > 512 tokens** | 78,043 (2.6%) |

**Source**: [Yelp Open Dataset](https://www.yelp.com/dataset) (2024)

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.26
scikit-learn>=1.5
xgboost>=2.1
tensorflow>=2.16
sentence-transformers>=3.0
vaderSentiment>=3.3
scipy>=1.11
matplotlib>=3.8
tqdm>=4.66
joblib>=1.3
pyarrow>=14.0
```

<details>
<summary>Apple Silicon (M1/M2/M3) Installation</summary>

```bash
pip install tensorflow-macos tensorflow-metal
```
</details>

---

## 📚 References

| # | Citation |
|---|----------|
| [1] | Hutto & Gilbert (2014). "VADER: A parsimonious rule-based model for sentiment analysis." *ICWSM* |
| [2] | Reimers & Gurevych (2019). "Sentence-BERT: Sentence embeddings using siamese BERT-networks." *EMNLP* |
| [3] | Chen & Guestrin (2016). "XGBoost: A scalable tree boosting system." *KDD* |
| [4] | Abadi et al. (2016). "TensorFlow: A system for large-scale machine learning." *OSDI* |
| [5] | Carbonell & Goldstein (1998). "MMR diversity-based reranking." *SIGIR* |
| [6] | Liu (2009). "Learning to rank for information retrieval." *FnTIR* |
| [7] | Yelp Open Dataset (2024). https://www.yelp.com/dataset |
| [8] | Ng (2021). "Data-Centric AI." DeepLearning.AI |

---

## 🙏 Acknowledgments

### AI Tool Usage Disclosure

The author acknowledges the use of AI-assisted tools during development:

- **Claude (Anthropic)** and **Gemini (Google)**: Used for code debugging, structural suggestions, and accuracy verification
- All AI-generated suggestions were reviewed, validated, and modified by the author
- The research design, methodology, analysis, and conclusions represent the original work of the author

This usage complies with academic integrity policies regarding AI tools as assistive technology.

### Additional Acknowledgments

- Yelp for providing the Open Dataset
- Sentence-Transformers team for the SBERT implementation
- NVIDIA for CUDA/GPU acceleration support

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
