#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
03_SENTIMENT.PY - Sentiment Feature Extraction (X_sent Split)
================================================================================
This script extracts THREE statistical moments from sentiment distribution:
    - X_sent_mean: Central tendency
    - X_sent_std:  Dispersion (consistency vs polarization)
    - X_sent_skew: Asymmetry

UPDATE: Imputes NaN std/skew with 0.0 (Statistical Truth) instead of Median.

Input:
    - data/raw/review.parquet

Output:
    - data/processed/features_sentiment.parquet

Usage:
    python src/03_sentiment.py
================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CONFIG, PATHS

# NLTK VADER setup
import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)  # Offline safety
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """VADER-based sentiment analyzer with distributional features."""

    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()

    def get_sentiment(self, text: str) -> float:
        """Get VADER compound score for text. Returns [-1, +1]."""
        if pd.isna(text) or not str(text).strip():
            return 0.0
        try:
            return self.sid.polarity_scores(str(text))['compound']
        except Exception:
            return 0.0

    def _safe_skew(self, x: pd.Series) -> float:
        """Calculate skewness with edge case handling."""
        if len(x) < CONFIG.min_reviews_for_skew:
            return 0.0
        try:
            sk = stats.skew(x, nan_policy='omit')
            return float(sk) if np.isfinite(sk) else 0.0
        except Exception:
            return 0.0

    def calculate_features(
        self,
        df_reviews: pd.DataFrame,
        show_progress: bool = True) -> pd.DataFrame:
        """
        Calculate sentiment features per restaurant.

        Returns DataFrame with: business_id, X_sent_mean, X_sent_std, X_sent_skew
        """
        print("\n[Sentiment] Calculating features...")

        df = df_reviews[['business_id', 'text']].copy()

        # Calculate per-review sentiment
        if show_progress:
            tqdm.pandas(desc="  VADER")
            df['_sent'] = df['text'].progress_apply(self.get_sentiment)
        else:
            df['_sent'] = df['text'].apply(self.get_sentiment)

        # Aggregate to restaurant level
        print("  Aggregating distributional stats...")
        df_agg = df.groupby('business_id').agg({
            '_sent': ['mean', 'std', self._safe_skew, 'count']
        }).reset_index()

        # Flatten MultiIndex columns
        df_agg.columns = ['business_id', 'X_sent_mean', 'X_sent_std', 'X_sent_skew', 'n_reviews']

        # ---------------------------------------------------------
        # THE FIX: Statistical Truth Imputation
        # ---------------------------------------------------------
        # If N=1, std is undefined (mathematically 0 variance for one point).
        # If N<3, skew is undefined.
        # We fill with 0.0 to represent "No Observed Variance/Asymmetry".

        df_agg['X_sent_std'] = df_agg['X_sent_std'].fillna(0.0)
        df_agg['X_sent_skew'] = df_agg['X_sent_skew'].fillna(0.0)

        print(f"  ✓ Calculated for {len(df_agg):,} restaurants")
        print(f"    X_sent_mean: [{df_agg['X_sent_mean'].min():.3f}, {df_agg['X_sent_mean'].max():.3f}]")
        print(f"    X_sent_std:  [{df_agg['X_sent_std'].min():.3f}, {df_agg['X_sent_std'].max():.3f}]")
        print(f"    X_sent_skew: [{df_agg['X_sent_skew'].min():.3f}, {df_agg['X_sent_skew'].max():.3f}]")

        return df_agg


def calculate_sentiment_features(df_reviews: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    """Convenience function for sentiment calculation."""
    analyzer = SentimentAnalyzer()
    return analyzer.calculate_features(df_reviews, show_progress)


def main():
    """Main sentiment extraction pipeline."""
    print("\n" + "="*60)
    print("SENTIMENT FEATURE EXTRACTION")
    print("="*60)
    
    # Ensure directories exist
    PATHS.ensure_dirs()
    
    # Load reviews
    print(f"\nLoading reviews from: {PATHS.RAW_REVIEW}")
    df_raw = pd.read_parquet(PATHS.RAW_REVIEW)
    print(f"  Shape: {df_raw.shape}")
    print(f"  NaN counts:\n{df_raw.isna().sum()}")
    
    # Calculate sentiment features
    df_sentiment = calculate_sentiment_features(df_raw)
    
    # Display statistics
    print("\n" + "-"*40)
    print("FEATURE STATISTICS")
    print("-"*40)
    print(df_sentiment.describe())
    
    # Save output
    print("\n" + "-"*40)
    print("SAVING OUTPUT")
    print("-"*40)
    df_sentiment.to_parquet(PATHS.FEATURES_SENTIMENT, index=False)
    print(f"✅ Saved: {PATHS.FEATURES_SENTIMENT}")
    
    print("\n✅ SENTIMENT EXTRACTION COMPLETE!")
    
    return df_sentiment


if __name__ == "__main__":
    main()
