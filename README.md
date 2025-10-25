# SBERT-Enhanced Deep Neural Learning-to-Rank Model for Restaurant Recommendations

## Project Overview

This repository contains the complete implementation and analysis for the **CS 4824: Machine Learning Capstone Project**.  
The work develops a **hybrid recommender system** addressing the *cold-start problem* by integrating textual semantics, sentiment, and popularity-based features into a unified **Deep Neural Network (DNN) Learning-to-Rank (LTR)** framework.

The core contribution lies in replacing a fixed linear scoring mechanism with a **non-linear DNN ranker** that adaptively learns feature interactions from data.  

This enables more accurate and context-aware recommendations compared to the **Ordinary Least Squares (OLS)** baseline model.  
A subsequent **Maximal Marginal Relevance (MMR)** stage is applied to promote diversity among highly ranked items, ensuring both precision and novelty in final recommendations.

### Methodology
 
* **Input Features ($\mathbf{X}$):**  
  * $\mathbf{X}_{\text{sim}}$ : SBERT-derived cosine similarity representing semantic closeness between restaurant reviews.  
  * $\mathbf{X}_{\text{sent}}$ : VADER sentiment polarity aggregated at the business level.  
  * $\mathbf{X}_{\text{pop}}$ : Composite popularity metric combining average star rating and log-transformed review count to stabilize scale variance.

* **Ranking Models:**  
  A **dual-model framework** is implemented for empirical comparison:
  * **Baseline:** Ordinary Least Squares (OLS) regression estimating explicit weights for the linear score:
    
     $$\mathbf{R}_{\text{Linear}} = w_1\mathbf{X}_{\text{sim}} + w_2\mathbf{X}_{\text{sent}} + w_3\mathbf{X}_{\text{pop}}$$

  * **Non-Linear Ranker:** A **pointwise Deep Neural Network (DNN)** trained on a regression objective ($\mathbf{Y}$), optimizing via stochastic gradient descent to learn non-linear interactions among features.

* **Target Label ($\mathbf{Y}$):**  
  Constructed as a **weighted average rating**, where individual review ratings are weighted by review volume to mitigate sampling bias.

* **Diversity Enhancement:**  
  Applies **Maximal Marginal Relevance (MMR)** as a post-ranking stage to balance relevance and novelty, promoting a diverse set of recommended restaurants.

## Getting Started

### Prerequisites

The computational components of this project—particularly **SBERT embedding generation** and **Deep Neural Network (DNN) training**—are optimized for GPU acceleration to ensure efficient large-scale experimentation.

* **Operating System:** macOS (Apple Silicon M1/M2/M3) or Linux. GPU acceleration on macOS leverages the **Metal Performance Shaders** backend for TensorFlow.  
* **Recommended Environment:** [**Google Colab Pro**](https://colab.research.google.com/signup) for reproducible, GPU-accelerated training and hyperparameter tuning.  
* **Environment Manager:** Miniforge or the built-in Python `venv` environment.  
* **Python Version:** Python 3.10 or higher (tested with Python 3.11).  
* **Hardware Requirements:** Minimum 8 GB RAM; 16 GB+ recommended for full SBERT embedding generation.

### Installation and Environment Setup

You can set up the environment locally (recommended for Apple Silicon) or within a Google Colab runtime.

```bash
# 1. Clone the repository
git clone https://github.com/YingWana/Restaurant-Recommender.git
cd Restaurant-Recommender

# 2. (macOS M1/M2/M3) — Install TensorFlow triplet manually first
pip install numpy==1.26.4 tensorflow-macos==2.16.2 tensorflow-metal==1.2.0

# 3. Install TensorFlow Ranking (ignore CPU-only deps)
pip install tensorflow-ranking==0.5.1 --no-deps

# 4. Install remaining dependencies
pip install -r requirements.txtnstall sentence-transformers nltk requests
```
**Note:** If training on Google Colab's T4/A100 GPU, use

```bash
pip install tensorflow==2.17.0 tensorflow-ranking==0.5.1
pip install -r requirements.txt
```

### Data and Preprocessing

The project is based on the high-volume [**Yelp Academic Dataset**](https://business.yelp.com/data/resources/open-dataset/), comprising approximately **6.99 million reviews** and **150,346 businesses** across **11 metropolitan areas** in North America.  

This corpus provides a reliable and singular foundation for reproducible feature development and large-scale recommender benchmarking.

* **Data Corpus:** Contains **3,066** filtered Florida-based restaurants and cafes that remain open and **588,377** aggregated reviews.
* **Processing Pipeline:** The preprocessing sequence is executed once to construct the feature matrix:
    * **Text Aggregation:** All reviews are grouped by `business_id` to form the restaurant-level corpus $\mathbf{C}_R$.
    * **Embedding Generation:** SBERT is applied to $\mathbf{C}_R$ to produce dense semantic vectors $\mathbf{v}_R$.
    * **Feature Scaling:** Each feature in $\mathbf{X}$ (similarity, sentiment, popularity) is standardized using `StandardScaler` for stable training across models.

### Usage and Execution

The project execution is divided into three distinct phases that correspond to the methodological stages of the study.

1.  **Baseline Model Derivation (In Progress - Week 6)**
Computes the linear baseline weights using **Ordinary Least Squares (OLS)** regression to produce the benchmark ranking function $\mathbf{R}_{\text{Linear}}$.

```bash
# Script: src/baseline_ols.py
# Action: Calculates optimal w1, w2, w3 via OLS Regression against Y_Relevance target.
python src/baseline_ols.py
```

2. **DNN Training and Tuning (Starting Week 6)**
Implements the **Deep Neural Network Learning-to-Rank (DNN LTR)** model, training on the standardized feature matrix ($\mathbf{X}$) using stochastic gradient descent.
Hyperparameter tuning employs Bayesian Optimization for efficient exploration.

```bash
# Script: src/train_dnn_ltr.py
# Purpose: Define and train the DNN/MLP ranker on (X, Y)
# Execution: Recommended on Google Colab GPU for accelerated training
python src/train_dnn_ltr.py
```

3. **Final Evaluation and Reranking (Weeks 7-8)**
Loads both ranking models, applies the Maximal Marginal Relevance (MMR) diversity filter, and performs comparative evaluation across ranking metrics.

```bash
# Script: src/evaluate_mmr.py
# Action: Compares R_Linear vs. R_DNN performance on NDCG@k and ILD metrics.
# Example: Apply MMR with a relevance bias (lambda=0.7)
python src/evaluate_mmr.py --lambda 0.7
```
## Project Structure

## Restaurant-Recommender/

```text
.
├── **data/**
│   │
│   └── **processed/**
│       ├── yelp_restaurant_florida_reviews.csv   # Unified, Cleaned, Filtered Data 
│       └── feature_matrix_X_Y.csv                # Final DNN Training Matrix 
│
├── **notebooks/**
│   ├── yelp_clean.ipynb             # Data cleaning and Exploratory Data Analysis (EDA)
│   └── model_analysis.ipynb         # Visualization of SHAP/MMR/Feature 
│
├── **src/**
│   ├── extract_yelp_data.py         # Script to extract and filter Yelp data.
│   ├── features.py**                # Utility script for calculating Y, Sent, Pop features.
│   ├── baseline_ols.py              # Compute linear baseline weights via OLS.
│   ├── train_dnn_ltr.py             # Train Deep Neural Network Learning-to-Rank model.
│   ├── evaluate_mmr.py              # Apply MMR re-ranking and comparative evaluation.
│
├── **results/**
│   ├── metrics_ndcg_ild.csv         # Evaluation metrics (NDCG@k, ILD).
│   └── plots/                       # Visualization outputs (ranking curves, loss plots, etc.).
│
├── **models/**
│   ├── dnn_ranker.h5                # Trained DNN model weights
│   └── linear_baseline_weights.json # Stores final w1, w2, w3 coefficients
│
├── **vectors/**
│   └── sbert_vectors_vR.npy         # Stores all pre-calculated SBERT vectors (vR)
│
├──.gitignore                        # Files and directories excluded from the repository.
├── requirements.txt                 # Stores final w1, w2, w3 coefficients.
└── README.md
```

## Project Status

| Component | Status | Target Completion |
| :--- | :--- | :--- |
| **Data & Features** | **Completed** | Week 5 |
| **Linear Baseline** | **In Progress** | Week 6 |
| **DNN LTR Model** | **Not Started** (Architecture Defined) | Week 6 |
| **MMR Implementation** | **Not Started** (Logic Defined) | Week 7 |
| **Final Evaluation** | **Not Started** | Week 8 |

## Appendix
- Yelp, Inc. *Yelp Open Dataset (2025).* [Available online](https://business.yelp.com/data/resources/open-dataset/).  



