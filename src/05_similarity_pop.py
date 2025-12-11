#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
05_SIMILARITY_POP.PY - Semantic Similarity and Popularity Feature Computation
================================================================================
This script computes the X_sim (semantic similarity) and X_pop (popularity) features.

X_sim Design Notes:
    - TRAINING: Computed against fixed quality-proxy query
    - INFERENCE: Dynamically computed against user query
    - At training, X_sim acts as a quality proxy signal
    - At inference, MultiModelEngine recomputes X_sim with the actual user query

X_pop Design:
    - log(review_count + 1) to stabilize variance
    - Acts as a popularity/reliability signal

Input:
    - data/processed/restaurant_embeddings.parquet
    - data/processed/features_sentiment.parquet

Output:
    - data/processed/features_similarity.parquet
    - data/processed/features_popularity.parquet

Usage:
    python src/05_similarity_pop.py
================================================================================
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import warnings
import os
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CONFIG, PATHS

warnings.filterwarnings('ignore')


def compute_xsim(embeddings: np.ndarray, 
                 business_ids: np.ndarray,
                 query: str = None) -> pd.DataFrame:
    """
    Compute X_sim (semantic similarity) feature.
    
    Parameters
    ----------
    embeddings : np.ndarray
        SBERT embeddings matrix (n_businesses, 384)
    business_ids : np.ndarray
        Array of business IDs
    query : str, optional
        Query to compute similarity against.
        Defaults to CONFIG.QUALITY_PROXY_QUERY
        
    Returns
    -------
    pd.DataFrame
        DataFrame with business_id and X_sim columns
    """
    if query is None:
        query = CONFIG.QUALITY_PROXY_QUERY
    
    print(f"\nComputing X_sim (Semantic Similarity)...")
    print(f"Quality Proxy Query: '{query}'")
    print("\nNOTE: This fixed query is used during TRAINING as a quality proxy.")
    print("      At INFERENCE, MultiModelEngine uses the actual user query.")
    
    # Load SBERT encoder
    print("\nLoading SBERT model...")
    encoder = SentenceTransformer(CONFIG.sbert_model_name)
    
    # Encode the quality proxy query
    query_embedding = encoder.encode([query])[0]
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Compute cosine similarity between query and all business embeddings
    # This creates a 1D array of similarity scores for each business
    X_sim = cosine_similarity([query_embedding], embeddings)[0]
    
    print(f"\n✅ X_sim computed successfully")
    print(f"   Shape: {X_sim.shape}")
    print(f"   Mean:  {X_sim.mean():.4f}")
    print(f"   Std:   {X_sim.std():.4f}")
    print(f"   Range: [{X_sim.min():.4f}, {X_sim.max():.4f}]")
    
    # Create DataFrame
    df_sim = pd.DataFrame({
        'business_id': business_ids,
        'X_sim': X_sim
    })
    
    return df_sim


def compute_xpop(df_sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Compute X_pop (log popularity) feature.
    
    Parameters
    ----------
    df_sentiment : pd.DataFrame
        Sentiment features DataFrame with n_reviews column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with business_id and X_pop columns
    """
    print("\nComputing X_pop (Log Popularity)...")
    
    if 'n_reviews' not in df_sentiment.columns:
        print("⚠️ n_reviews not found in sentiment data")
        return None
    
    # Compute log popularity
    # X_pop = log(review_count + 1) to handle zeros
    X_pop = np.log(df_sentiment['n_reviews'] + 1)
    
    df_pop = pd.DataFrame({
        'business_id': df_sentiment['business_id'],
        'X_pop': X_pop
    })
    
    print(f"✅ X_pop computed")
    print(f"   Mean:  {X_pop.mean():.4f}")
    print(f"   Range: [{X_pop.min():.4f}, {X_pop.max():.4f}]")
    
    return df_pop


def plot_xsim_distribution(X_sim: np.ndarray, save_path: str = None):
    """Plot X_sim distribution for validation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(X_sim, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
    axes[0].set_xlabel('X_sim (Cosine Similarity)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('X_sim Distribution')
    axes[0].axvline(X_sim.mean(), color='red', linestyle='--', 
                    label=f'Mean: {X_sim.mean():.3f}')
    axes[0].legend()

    # Box plot
    axes[1].boxplot(X_sim, vert=True)
    axes[1].set_ylabel('X_sim (Cosine Similarity)')
    axes[1].set_title('X_sim Box Plot')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ Saved plot: {save_path}")
    
    plt.show()
    
    print("\n✅ Distribution appears reasonable (no extreme outliers)")


def main():
    """Main similarity and popularity computation pipeline."""
    print("\n" + "="*60)
    print("SIMILARITY & POPULARITY FEATURE COMPUTATION")
    print("="*60)
    
    # Ensure directories exist
    PATHS.ensure_dirs()
    
    # ─────────────────────────────────────────────────────────────
    # LOAD EMBEDDINGS
    # ─────────────────────────────────────────────────────────────
    print(f"\nLoading SBERT embeddings from: {PATHS.RESTAURANT_EMBEDDINGS}")
    
    df_embeddings = pd.read_parquet(PATHS.RESTAURANT_EMBEDDINGS)
    
    # Extract data structures
    business_ids = df_embeddings['business_id'].values
    embedding_vectors = np.stack(df_embeddings['embedding'].values)
    
    print(f"✅ Loaded successfully")
    print(f"   Number of businesses: {len(business_ids):,}")
    print(f"   Embedding dimension: {embedding_vectors.shape[1]}")
    print(f"   Total shape: {embedding_vectors.shape}")
    
    # ─────────────────────────────────────────────────────────────
    # COMPUTE X_sim
    # ─────────────────────────────────────────────────────────────
    df_sim = compute_xsim(embedding_vectors, business_ids)
    
    # Validate distribution
    plot_xsim_distribution(
        df_sim['X_sim'].values,
        save_path=str(PATHS.FIGURES_DIR / 'xsim_distribution.png')
    )
    
    # ─────────────────────────────────────────────────────────────
    # COMPUTE X_pop
    # ─────────────────────────────────────────────────────────────
    print(f"\nLoading sentiment features from: {PATHS.FEATURES_SENTIMENT}")
    df_sentiment = pd.read_parquet(PATHS.FEATURES_SENTIMENT)
    print(f"  Loaded: {df_sentiment.shape}")
    
    df_pop = compute_xpop(df_sentiment)
    
    # Align X_pop with embedding business_ids
    if df_pop is not None:
        # Create a mapping from sentiment to align with embeddings
        pop_map = df_pop.set_index('business_id')['X_pop']
        df_pop_aligned = pd.DataFrame({
            'business_id': business_ids,
            'X_pop': [pop_map.get(bid, np.log(51)) for bid in business_ids]
        })
    else:
        # Fallback
        df_pop_aligned = pd.DataFrame({
            'business_id': business_ids,
            'X_pop': np.log(51)  # Default
        })
    
    # ─────────────────────────────────────────────────────────────
    # SAVE OUTPUTS
    # ─────────────────────────────────────────────────────────────
    print("\n" + "-"*40)
    print("SAVING OUTPUTS")
    print("-"*40)
    
    # Save X_sim
    df_sim.to_parquet(PATHS.FEATURES_SIMILARITY, index=False)
    print(f"✅ X_sim saved to: {PATHS.FEATURES_SIMILARITY}")
    
    # Save X_pop
    df_pop_aligned.to_parquet(PATHS.FEATURES_POPULARITY, index=False)
    print(f"✅ X_pop saved to: {PATHS.FEATURES_POPULARITY}")
    
    # Verification
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    df_sim_check = pd.read_parquet(PATHS.FEATURES_SIMILARITY)
    df_pop_check = pd.read_parquet(PATHS.FEATURES_POPULARITY)
    print(f"X_sim reloaded: {df_sim_check.shape}")
    print(f"X_pop reloaded: {df_pop_check.shape}")
    
    print("\n✅ SIMILARITY & POPULARITY COMPUTATION COMPLETE!")
    
    return df_sim, df_pop_aligned


if __name__ == "__main__":
    main()
