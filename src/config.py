#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CONFIG.PY - Central Configuration for Restaurant Recommender System
================================================================================
All scripts import paths and hyperparameters from this single source of truth.
This ensures reproducibility and consistent behavior across the pipeline.

Usage:
    from src.config import CONFIG, PATHS
================================================================================
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT ROOT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# Automatically detect project root (works in both local and Colab environments)

def get_project_root() -> Path:
    """Detect project root directory."""
    # Check if running in Google Colab
    try:
        from google.colab import drive
        # Colab environment - use Google Drive path
        return Path('/content/drive/MyDrive/capstone')
    except ImportError:
        # Local environment - find project root
        current = Path(__file__).resolve().parent
        while current != current.parent:
            if (current / 'requirements.txt').exists():
                return current
            current = current.parent
        # Fallback to current working directory
        return Path.cwd()

PROJECT_ROOT = get_project_root()


# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class PATHS:
    """All file paths used in the project."""
    
    # Root directories
    ROOT = PROJECT_ROOT
    DATA_DIR = ROOT / 'data'
    RAW_DIR = DATA_DIR / 'raw'
    PROCESSED_DIR = DATA_DIR / 'processed'
    MODELS_DIR = ROOT / 'models'
    RESULTS_DIR = ROOT / 'results'
    FIGURES_DIR = ROOT / 'figures'
    
    # Raw data files (input)
    RAW_REVIEW = RAW_DIR / 'review.parquet'
    RAW_BUSINESS = RAW_DIR / 'business.parquet'
    
    # Processed feature files
    FEATURES_BUSINESS = PROCESSED_DIR / 'features_business.parquet'
    FEATURES_SENTIMENT = PROCESSED_DIR / 'features_sentiment.parquet'
    FEATURES_SIMILARITY = PROCESSED_DIR / 'features_similarity.parquet'
    FEATURES_POPULARITY = PROCESSED_DIR / 'features_popularity.parquet'
    FEATURES_ALL = PROCESSED_DIR / 'features_all.parquet'
    RESTAURANT_EMBEDDINGS = PROCESSED_DIR / 'restaurant_embeddings.parquet'
    
    # Model checkpoints
    OLS_MODEL = MODELS_DIR / 'ols_model.joblib'
    XGB_MODEL = MODELS_DIR / 'xgb_model.joblib'
    DNN_MODEL = MODELS_DIR / 'dnn_model.keras'
    SCALER = MODELS_DIR / 'scaler.joblib'
    
    # Results
    MODEL_COMPARISON = RESULTS_DIR / 'model_comparison.csv'
    FEATURE_IMPORTANCE = RESULTS_DIR / 'feature_importance.csv'
    ABLATION_RESULTS = RESULTS_DIR / 'ablation_results.csv'
    DNN_ABLATION_RESULTS = RESULTS_DIR / 'dnn_ablation_results.csv'
    MMR_LAMBDA_SWEEP = RESULTS_DIR / 'mmr_lambda_sweep.csv'
    MMR_COMPARISON = RESULTS_DIR / 'mmr_comparison.csv'
    
    @classmethod
    def ensure_dirs(cls):
        """Create all directories if they don't exist."""
        for attr in ['DATA_DIR', 'RAW_DIR', 'PROCESSED_DIR', 'MODELS_DIR', 
                     'RESULTS_DIR', 'FIGURES_DIR']:
            path = getattr(cls, attr)
            path.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class CONFIG:
    """
    Configuration for model training.

    FEATURE DESIGN NOTES:
    ---------------------
    continuous_features breakdown:

    1. X_sim (Semantic Similarity)
       - TRAINING: Computed against fixed quality-proxy query
       - INFERENCE: Dynamically computed against user query
       - Role: Quality proxy at train-time, relevance score at inference

    2. X_sent_mean (Sentiment Mean)
       - VADER compound score averaged across all reviews
       - Primary predictor of rating (r = 0.905 with stars)

    3. X_sent_std (Sentiment Std Dev)
       - Measures review consistency/polarization
       - High std = controversial restaurant

    4. X_sent_skew (Sentiment Skewness)
       - Detects negativity bias (few angry reviews)
       - Negative skew = long tail of negative reviews

    5. X_pop (Log Popularity)
       - log(review_count + 1)
       - Controls for popularity bias
    """

    # ─────────────────────────────────────────────────────────────
    # SBERT SETTINGS
    # ─────────────────────────────────────────────────────────────
    sbert_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    sbert_embedding_dim = 384  # Output dimension for MiniLM

    # Chunking Settings
    max_tokens_per_chunk = 450  # Safe buffer under 512 limit
    min_chunk_tokens = 20      # Ignore chunks smaller than this

    # X_sim Training Query
    # DESIGN NOTE: This fixed query serves as a QUALITY PROXY during training.
    # At inference time, MultiModelEngine replaces this with the actual user query.
    QUALITY_PROXY_QUERY = "delicious food excellent service nice atmosphere good value"

    # ─────────────────────────────────────────────────────────────
    # FEATURES
    # ─────────────────────────────────────────────────────────────
    target_col = 'stars'

    continuous_features = [
        'X_sim',           # Semantic similarity (quality proxy at training)
        'X_sent_mean',     # Sentiment mean (PRIMARY predictor)
        'X_sent_std',      # Sentiment std (consistency signal)
        'X_sent_skew',     # Sentiment skewness (negativity bias)
        'X_pop',           # Log popularity
    ]

    binary_features = [
        'is_TakeOut', 'is_Delivery', 'has_OutdoorSeating', 'has_Alcohol',
        'is_GoodForKids', 'is_GoodForGroups', 'has_HappyHour', 'is_WheelchairAccessible',
    ]

    cuisine_features = [
        'is_American', 'is_Mexican', 'is_Pizza', 'is_Italian',
        'is_Chinese', 'is_Burgers', 'is_Thai', 'is_Japanese',
        'is_Sushi', 'is_Indian', 'is_Seafood', 'is_Mediterranean',
        'is_Sandwiches', 'is_Vietnamese', 'is_Steakhouses',
        'is_Korean', 'is_Barbeque', 'is_Greek',
        'is_Dessert_Baking', 'is_Coffee_Tea_Drinks',
        'is_Breakfast_and_Brunch', 'is_Other',
    ]

    spatial_feature = 'neighborhood_type'

    # ─────────────────────────────────────────────────────────────
    # DATA SPLIT
    # ─────────────────────────────────────────────────────────────
    test_size = 0.15
    val_size = 0.15
    random_state = 42  # CRITICAL: For reproducibility

    # ─────────────────────────────────────────────────────────────
    # XGBOOST HYPERPARAMETERS
    # ─────────────────────────────────────────────────────────────
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
    }
    xgb_early_stopping_rounds = 20

    # ─────────────────────────────────────────────────────────────
    # DNN HYPERPARAMETERS
    # ─────────────────────────────────────────────────────────────
    dnn_hidden_layers = [128, 64, 32]
    dnn_dropout_rate = 0.3
    dnn_learning_rate = 0.001
    dnn_epochs = 100
    dnn_batch_size = 64
    dnn_early_stopping_patience = 15

    # ─────────────────────────────────────────────────────────────
    # EVALUATION
    # ─────────────────────────────────────────────────────────────
    ndcg_k_values = [5, 10, 20]
    mmr_lambda = 0.7
    mmr_top_k = 20
    mmr_lambda_sweep_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # ─────────────────────────────────────────────────────────────
    # SENTIMENT SETTINGS
    # ─────────────────────────────────────────────────────────────
    min_reviews_for_skew = 3  # Minimum reviews needed to calculate skewness

    @classmethod
    def get_all_features(cls):
        """Return list of all feature column names."""
        return (cls.continuous_features + cls.binary_features +
                cls.cuisine_features + [cls.spatial_feature])


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

# Create directories on import
PATHS.ensure_dirs()

print(f"✅ Configuration loaded")
print(f"   Project root: {PATHS.ROOT}")
print(f"   Total features: {len(CONFIG.get_all_features())}")
