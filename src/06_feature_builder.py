#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
06_FEATURE_BUILDER.PY - Feature Assembly and Final Matrix Construction
================================================================================
This script combines all engineered features into the final training matrix.

Combines:
    - X_sim (semantic similarity)
    - X_sent_mean, X_sent_std, X_sent_skew (sentiment moments)
    - X_pop (popularity)
    - Business attributes (price, amenities)
    - Cuisine categories (22 super-groups)
    - Spatial features (neighborhood_type)

Input:
    - data/processed/features_similarity.parquet
    - data/processed/features_sentiment.parquet
    - data/processed/features_business.parquet

Output:
    - data/processed/features_all.parquet

Usage:
    python src/06_feature_builder.py
================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CONFIG, PATHS


def load_all_features():
    """Load all feature files."""
    print("\n" + "="*60)
    print("LOADING FEATURE FILES")
    print("="*60)
    
    # Load similarity features
    print(f"\n1. Loading X_sim from: {PATHS.FEATURES_SIMILARITY}")
    X_sim = pd.read_parquet(PATHS.FEATURES_SIMILARITY)
    print(f"   Shape: {X_sim.shape}")
    
    # Load sentiment features
    print(f"\n2. Loading sentiment from: {PATHS.FEATURES_SENTIMENT}")
    X_sent = pd.read_parquet(PATHS.FEATURES_SENTIMENT)
    print(f"   Shape: {X_sent.shape}")
    
    # Load business features
    print(f"\n3. Loading business from: {PATHS.FEATURES_BUSINESS}")
    business = pd.read_parquet(PATHS.FEATURES_BUSINESS)
    print(f"   Shape: {business.shape}")
    
    return X_sim, X_sent, business


def merge_features(X_sim: pd.DataFrame, 
                   X_sent: pd.DataFrame, 
                   business: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all features into a single DataFrame.
    
    Also computes X_pop from n_reviews.
    """
    print("\n" + "="*60)
    print("MERGING FEATURES")
    print("="*60)
    
    # Compute X_pop from sentiment data (which has n_reviews)
    print("\n1. Computing X_pop from review counts...")
    X_sent['X_pop'] = np.log(X_sent['n_reviews'] + 1)
    
    # Merge X_sent with X_sim
    print("\n2. Merging X_sim with sentiment features...")
    df = pd.merge(X_sim, X_sent, on='business_id')
    print(f"   Shape after merge: {df.shape}")
    
    # Merge business features
    print("\n3. Merging business features...")
    df = pd.merge(df, business, on='business_id')
    print(f"   Shape after merge: {df.shape}")
    
    return df


def validate_features(df: pd.DataFrame):
    """Validate the merged feature DataFrame."""
    print("\n" + "="*60)
    print("FEATURE VALIDATION")
    print("="*60)
    
    # Check for required features
    all_features = CONFIG.get_all_features()
    missing = [f for f in all_features if f not in df.columns]
    
    if missing:
        print(f"\n⚠️ Missing features: {missing}")
    else:
        print("\n✅ All required features present")
    
    # Check for target column
    if CONFIG.target_col not in df.columns:
        print(f"\n❌ Target column '{CONFIG.target_col}' not found!")
    else:
        print(f"\n✅ Target column '{CONFIG.target_col}' present")
        print(f"   Range: [{df[CONFIG.target_col].min()}, {df[CONFIG.target_col].max()}]")
        print(f"   Mean: {df[CONFIG.target_col].mean():.2f}")
    
    # Check for NaN values
    print("\n" + "-"*40)
    print("NaN CHECK")
    print("-"*40)
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    
    if len(nan_cols) > 0:
        print("Columns with NaN values:")
        for col, count in nan_cols.items():
            print(f"   {col}: {count} ({100*count/len(df):.1f}%)")
    else:
        print("✅ No NaN values found")
    
    # Correlation check
    print("\n" + "-"*40)
    print("CORRELATION WITH TARGET")
    print("-"*40)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if CONFIG.target_col in numeric_cols:
        corr_matrix = df[numeric_cols].corr()
        target_corr = corr_matrix[CONFIG.target_col].sort_values(ascending=False)
        
        print("\nTop correlations with stars:")
        for feat, corr in target_corr.head(10).items():
            if feat != CONFIG.target_col:
                print(f"   {feat:25s}: {corr:+.3f}")
    
    return df


def main():
    """Main feature building pipeline."""
    print("\n" + "="*60)
    print("FEATURE BUILDER PIPELINE")
    print("="*60)
    
    # Ensure directories exist
    PATHS.ensure_dirs()
    
    # Load all features
    X_sim, X_sent, business = load_all_features()
    
    # Merge features
    df = merge_features(X_sim, X_sent, business)
    
    # Validate
    df = validate_features(df)
    
    # Display summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(df.describe())
    
    # Save
    print("\n" + "="*60)
    print("SAVING OUTPUT")
    print("="*60)
    
    df.to_parquet(PATHS.FEATURES_ALL, index=False)
    print(f"✅ Saved: {PATHS.FEATURES_ALL}")
    
    # Final stats
    print("\n" + "-"*40)
    print("FINAL DATASET")
    print("-"*40)
    print(f"Total samples: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Feature columns: {len(CONFIG.get_all_features())}")
    
    print("\n✅ FEATURE BUILDING COMPLETE!")
    
    return df


if __name__ == "__main__":
    main()
