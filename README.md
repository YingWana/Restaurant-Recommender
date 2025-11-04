# A Deep Learning-to-Rank Framework for Restaurant Recommendations Using SBERT and Yelp Data

## Project Overview

This repository contains the full implementation and analysis for the **CS 4824: Machine Learning Capstone Project**.  

The work develops a **hybrid recommender system** addressing the *cold-start problem* by integrating textual **semantics, sentiment, and popularity-based features** within a unified **Deep Neural Network (DNN) Learning-to-Rank (LTR)** framework.

The model replaces a fixed linear scoring function with a **non-linear DNN ranker** that learns feature interactions directly from data. 

This enables more accurate and context-aware recommendations compared to the **Ordinary Least Squares (OLS)** baseline model.  
A final **Maximal Marginal Relevance (MMR)**  introduces diversity among the top-ranked results, balancing precision with novelty in the final recommendation list.

## Methodology
 
* **Input Features ($\mathbf{X}$):**  
  * $\mathbf{X}_{\text{sim}}$ : SBERT,derived cosine similarity representing semantic closeness between restaurant reviews (Reimers and Gurevych, 2022).  
  * $\mathbf{X}_{\text{sent}}$ : VADER sentiment polarity aggregated at the business level.  
  * $\mathbf{X}_{\text{pop}}$ : Composite popularity metric combining average star rating and log-transformed review count to stabilize scale variance.

* **Ranking Models:**  
  A **dual-model framework** is implemented for empirical comparison:
  * **Baseline:** Ordinary Least Squares (OLS) regression estimating explicit weights for the linear score:
    ```
    R_linear = w1 * X_sim + w2 * X_sent + w3 * X_pop
    ```
    The linear scoring formulation follows established hybrid recommender design principles that combine content, sentiment, and popularity signals (Burke, 2002; Lops et al., 2011).

  * **Non-Linear Ranker:** A **pointwise Deep Neural Network (DNN)** A pointwise Deep Neural Network (DNN) trained on a regression objective (`Y`) using stochastic gradient descent, enabling the model to learn non-linear feature interactions (Liu 2009).

* **Target Label ($\mathbf{Y}$):**  
  Constructed as a weighted average rating, where each business’s aggregated rating is weighted by its total review count to mitigate sampling bias.  

  This design reflects prior work connecting textual sentiment and user ratings in recommendation contexts (McAuley and Leskovec 2013).

* **Diversity Enhancement:**  
    A Maximal Marginal Relevance (MMR) stage is applied post-ranking to balance relevance and novelty, promoting a more diverse and contextually representative set of top-ranked recommendations (Carbonell and Goldstein 1998).
  
## Data and Preprocessing

The project is based on the high-volume [**Yelp Academic Dataset**](https://business.yelp.com/data/resources/open-dataset/), comprising approximately **6.99 million reviews** and **150,346 businesses** across **11 metropolitan areas** in North America.  

For experimental consistency, a geographically focused subset was extracted:

* **Data Corpus:** Contains **3,066** filtered Florida-based restaurants and cafes that remain open and **588,377** aggregated reviews (filtered to include only businesses with ≥50 reviews).
* **Processing Pipeline:** The preprocessing workflow is executed once to construct the feature matrix:
    * **Text Aggregation:** All reviews are grouped by `business_id` to form the restaurant-level corpus $\mathbf{C}_R$.
    * **Embedding Generation:** SBERT is applied to $\mathbf{C}_R$ to produce dense semantic vectors $\mathbf{v}_R$.
    * **Feature Scaling:** Each feature in $\mathbf{X}$ (similarity, sentiment, popularity) is standardized using `StandardScaler` for stable training across models.
      
## Environment, Reproducibility, and Project Structure

### Prerequisites

The computational components of this project, particularly **SBERT embedding generation** and **Deep Neural Network (DNN) training** are optimized for GPU acceleration to ensure efficient large,scale experimentation.

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
### Project Structure

```text
.
├── data/
│   │
│   └── processed/
│       ├── yelp_restaurant_florida_reviews.csv   # Unified, Cleaned, Filtered Data 
│       └── feature_matrix_X_Y.csv                # Final DNN Training Matrix 
│
├── notebooks/
│   ├── yelp_clean.ipynb             # Data cleaning and Exploratory Data Analysis (EDA)
│   └── model_analysis.ipynb         # Visualization of SHAP/MMR/Feature 
│
├── src/
│   ├── extract_yelp_data.py         # Script to extract and filter Yelp data.
│   ├── features.py**                # Utility script for calculating Y, Sent, Pop features.
│   ├── baseline_ols.py              # Compute linear baseline weights via OLS.
│   ├── train_dnn_ltr.py             # Train Deep Neural Network Learning-to-Rank model.
│   ├── evaluate_mmr.py              # Apply MMR re-ranking and comparative evaluation.
│
├── results/
│   ├── metrics_ndcg_ild.csv         # Evaluation metrics (NDCG@k, ILD).
│   └── plots/                       # Visualization outputs (ranking curves, loss plots, etc.).
│
├── models/
│   ├── dnn_ranker.h5                # Trained DNN model weights
│   └── linear_baseline_weights.json # Stores final w1, w2, w3 coefficients
│
├── vectors/
│   └── sbert_vectors_vR.npy         # Stores all pre-calculated SBERT vectors (vR)
│
├──.gitignore                        # Files and directories excluded from the repository.
├── requirements.txt                 # Stores final w1, w2, w3 coefficients.
└── README.md
```

## Usage and Execution

Project execution is divided into three distinct phases that correspond to the methodological stages of the study.

1.  **Baseline Model Derivation**
Computes the linear baseline weights using **Ordinary Least Squares (OLS)** regression to produce the benchmark ranking function $\mathbf{R}_{\text{Linear}}$.

```bash
# Script: src/baseline_ols.py
# Action: Calculates optimal w1, w2, w3 via OLS Regression against Y_Relevance target.
python src/baseline_ols.py
```

2. **DNN Training and Tuning**
Implements the **Deep Neural Network Learning-to-Rank (DNN LTR)** model, training on the standardized feature matrix ($\mathbf{X}$) using stochastic gradient descent.
Hyperparameter tuning employs Bayesian Optimization for efficient exploration.

**Note:** The network design and training workflow follow practical guidelines for neural recommender systems proposed by Reisinger and Rossi (2022). 

```bash
# Script: src/train_dnn_ltr.py
# Purpose: Define and train the DNN/MLP ranker on (X, Y)
# Execution: Recommended on Google Colab GPU for accelerated training
python src/train_dnn_ltr.py
```

3. **Final Evaluation and Reranking**
Loads both ranking models, applies the Maximal Marginal Relevance (MMR) diversity filter, and performs comparative evaluation across ranking metrics.

```bash
# Script: src/evaluate_mmr.py
# Action: Compares R_Linear vs. R_DNN performance on NDCG@k and ILD metrics.
# Example: Apply MMR with a relevance bias (lambda=0.7)
python src/evaluate_mmr.py --lambda 0.7
```

## Project Status

| Component | Status | Target Completion |
| :--- | :--- | :--- |
| **Data & Features** | **Completed** | Week 5 |
| **Linear Baseline** | **In Progress** | Week 6 |
| **DNN LTR Model** | **Not Started** (Architecture Defined) | Week 7 |
| **MMR Implementation** | **Not Started** (Logic Defined) | Week 8 |
| **Final Evaluation** | **Not Started** | Week 9 |

## Appendix
- Yelp, Inc. *Yelp Open Dataset (2025).* [Available online](https://business.yelp.com/data/resources/open-dataset/).

## References
- Burke, Robin. *Hybrid Recommender Systems: Survey and Experiments.* *User Modeling and User-Adapted Interaction*, vol. 12, no. 4, 2002, pp. 331–370. https://doi.org/10.1023/A:1021240730564  

- Lops, Pasquale, Marco de Gemmis, and Giovanni Semeraro. *Content-Based Recommender Systems: State of the Art and Trends.* *Recommender Systems Handbook*, Springer, 2011, pp. 73–105. https://doi.org/10.1007/978-0-387-85820-3_3  

- McAuley, Julian, and Jure Leskovec. *Hidden Factors and Hidden Topics: Understanding Rating Dimensions with Review Text.* *Proceedings of the 7th ACM Conference on Recommender Systems (RecSys ’13)*, 2013, pp. 165–172. https://doi.org/10.1145/2507157.2507163  

- Carbonell, Jaime, and Jade Goldstein. *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* *Proceedings of SIGIR ’98*, 1998, pp. 335–336. https://doi.org/10.1145/290941.291025  

- Liu, Tie-Yan. *Learning to Rank for Information Retrieval.* *Foundations and Trends in Information Retrieval*, vol. 3, no. 3, 2009, pp. 225–331. https://doi.org/10.1561/1500000016  

- Reimers, Nils, and Iryna Gurevych. *Sentence-Transformers: Sentence Embeddings Using Siamese BERT-Networks.* *arXiv preprint arXiv:2210.12437*, 2022. https://arxiv.org/pdf/2210.12437  

- Reisinger, Louis, and Lorenzo Rossi. *Building a Winning Recommendation System – Part 2: Deep Learning for Recommender Systems.* *NVIDIA Developer Blog*, 2022. https://developer.nvidia.com/blog/how-to-build-a-winning-recommendation-system-part-2-deep-learning-for-recommender-systems  



