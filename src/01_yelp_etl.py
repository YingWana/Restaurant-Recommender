#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
01_YELP_ETL.PY - Yelp Data Extraction, Transformation, and Loading
================================================================================
This script extracts and filters restaurant data from the Yelp Academic Dataset.

Pipeline Steps:
    1. Load raw Yelp JSON files (review.json, business.json)
    2. Filter to restaurants/cafes in target states
    3. Apply quality thresholds (minimum reviews, open status)
    4. Merge reviews with filtered businesses
    5. Save as Parquet files for efficient downstream processing

Input:
    - yelp_academic_dataset_review.json
    - yelp_academic_dataset_business.json

Output:
    - data/raw/review.parquet
    - data/raw/business.parquet

Usage:
    python src/01_yelp_etl.py --review_path <path> --business_path <path>
================================================================================
"""

import pandas as pd
import re
import argparse
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CONFIG, PATHS


def load_reviews(review_path: str, cols_to_keep: list = None) -> pd.DataFrame:
    """
    Load and filter review data from JSON.
    
    Uses chunked reading to handle large files efficiently.
    """
    if cols_to_keep is None:
        cols_to_keep = ["review_id", "user_id", "business_id", "stars", "text"]
    
    print(f"Loading reviews from: {review_path}")
    
    chunks = pd.read_json(review_path, lines=True, chunksize=100000)
    filtered_reviews = []
    
    for chunk in chunks:
        chunk = chunk[cols_to_keep]
        filtered_reviews.append(chunk)
    
    reviews_df = pd.concat(filtered_reviews, ignore_index=True)
    print(f"  Loaded {len(reviews_df):,} reviews")
    
    return reviews_df


def load_business(business_path: str) -> pd.DataFrame:
    """Load business data from JSON."""
    print(f"Loading businesses from: {business_path}")
    business_df = pd.read_json(business_path, lines=True)
    print(f"  Loaded {len(business_df):,} businesses")
    return business_df


def filter_business(df: pd.DataFrame, min_reviews: int = 50) -> pd.DataFrame:
    """
    Filters business dataset by state, category pattern, review count, and open status.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw business DataFrame
    min_reviews : int
        Minimum number of reviews required
        
    Returns
    -------
    pd.DataFrame
        Filtered business DataFrame
    """
    # Target states
    states = ['PA', 'FL', 'TN', 'AZ', 'LA']
    
    print(f"Filtering businesses...")
    print(f"  States: {states}")
    print(f"  Min reviews: {min_reviews}")
    
    # Pattern to match restaurants, cafes, coffee shops, food establishments
    pattern = r'\b(?:Restaurants?|Caf(?:e|és)|Coffee|Food)\b'
    
    mask = (
        (df['state'].isin(states)) &
        (df['categories'].fillna('').str.contains(pattern, case=False, regex=True)) &
        (df['review_count'] >= min_reviews) &
        (df['is_open'] == 1)
    )
    
    filtered_df = df[mask]
    print(f"  Filtered to {len(filtered_df):,} restaurants")
    
    return filtered_df


def merge_reviews_with_business(reviews_df: pd.DataFrame, 
                                 business_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge reviews with filtered business data.
    
    Only keeps reviews for businesses in the filtered set.
    """
    print("Merging reviews with business data...")
    
    filtered_reviews_df = reviews_df.merge(
        business_df[['business_id', 'name']],
        on='business_id',
        how='inner'
    )
    
    print(f"  Merged: {len(filtered_reviews_df):,} reviews")
    return filtered_reviews_df


def main(review_path: str = None, business_path: str = None, 
         min_reviews: int = 50):
    """
    Main ETL pipeline.
    
    Parameters
    ----------
    review_path : str
        Path to yelp_academic_dataset_review.json
    business_path : str
        Path to yelp_academic_dataset_business.json
    min_reviews : int
        Minimum reviews per business
    """
    print("\n" + "="*60)
    print("YELP DATA ETL PIPELINE")
    print("="*60)
    
    # Ensure output directories exist
    PATHS.ensure_dirs()
    
    # Load data
    reviews_df = load_reviews(review_path)
    business_df = load_business(business_path)
    
    # Check for missing values
    print("\nReview NaN counts:")
    print(reviews_df.isna().sum())
    print("\nBusiness NaN counts:")
    print(business_df.isna().sum())
    
    # Filter businesses
    filtered_business_df = filter_business(business_df, min_reviews)
    
    # Merge reviews
    filtered_reviews_df = merge_reviews_with_business(reviews_df, filtered_business_df)
    
    # Display stats
    print("\n" + "-"*40)
    print("FINAL DATASET STATISTICS")
    print("-"*40)
    print(f"Businesses: {len(filtered_business_df):,}")
    print(f"Reviews: {len(filtered_reviews_df):,}")
    print(f"Avg reviews per business: {len(filtered_reviews_df)/len(filtered_business_df):.1f}")
    
    # Save outputs
    print("\nSaving outputs...")
    
    filtered_reviews_df.to_parquet(PATHS.RAW_REVIEW, index=False)
    print(f"  ✅ Saved: {PATHS.RAW_REVIEW}")
    
    filtered_business_df.to_parquet(PATHS.RAW_BUSINESS, index=False)
    print(f"  ✅ Saved: {PATHS.RAW_BUSINESS}")
    
    print("\n✅ ETL COMPLETE!")
    
    return filtered_reviews_df, filtered_business_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yelp Data ETL Pipeline")
    parser.add_argument('--review_path', type=str, required=True,
                        help='Path to yelp_academic_dataset_review.json')
    parser.add_argument('--business_path', type=str, required=True,
                        help='Path to yelp_academic_dataset_business.json')
    parser.add_argument('--min_reviews', type=int, default=50,
                        help='Minimum reviews per business (default: 50)')
    
    args = parser.parse_args()
    
    main(
        review_path=args.review_path,
        business_path=args.business_path,
        min_reviews=args.min_reviews
    )
