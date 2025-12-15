#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
02_SPATIAL_ETL.PY - Spatial Clustering and Business Feature Engineering
================================================================================
This script performs spatial clustering and extracts business features.

Pipeline Steps:
    1. Load filtered business data
    2. Apply K-Means clustering per (State, City) for neighborhood classification
    3. Extract business attributes (price, amenities)
    4. Engineer food category features (22 cuisine super-groups)
    5. Impute missing values
    6. Save processed features

Input:
    - data/raw/business.parquet

Output:
    - data/processed/features_business.parquet

Usage:
    python src/02_spatial_etl.py
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CONFIG, PATHS


def run_spatial_clustering(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    Applies K-Means clustering per (State, City) pair and re-maps labels based on density.

    Returns:
        df (pd.DataFrame): DataFrame with new column 'neighborhood_type' (0-4)
    """
    # Create the placeholder column initialized with -1
    df['neighborhood_type'] = -1

    # 1. Group by BOTH State and City
    location_counts = df.groupby(['state', 'city']).size()

    # Filter to locations with enough data (> 50 restaurants)
    valid_locations = location_counts[location_counts > 50].index.tolist()

    print(f"Running Spatial Clustering for {len(valid_locations)} unique State-City pairs...")

    # Loop through the unique tuples
    for state, city in valid_locations:

        # 2. Select rows that match BOTH the state and the city
        loc_mask = (df['state'] == state) & (df['city'] == city)

        # Ensure we actually have data
        if loc_mask.sum() < n_clusters:
            continue

        coords = df.loc[loc_mask, ['latitude', 'longitude']]

        # 3. Fit K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=CONFIG.random_state, n_init=10)
        raw_labels = kmeans.fit_predict(coords)

        # 4. Density Sorting
        unique, counts = np.unique(raw_labels, return_counts=True)
        density_dict = dict(zip(unique, counts))

        # Sort: Rank 0 = Highest Density (Downtown)
        sorted_labels = sorted(density_dict, key=density_dict.get, reverse=True)
        rank_map = {original: rank for rank, original in enumerate(sorted_labels)}

        # 5. Apply Mapping
        aligned_labels = np.array([rank_map[l] for l in raw_labels])

        # Assign back to main DataFrame using the specific mask
        df.loc[loc_mask, 'neighborhood_type'] = aligned_labels

    return df


def engineer_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract business attributes from the attributes column.
    
    Creates binary features for amenities and price level.
    """
    print("Extracting Attributes...")

    # 1. EXTRACT PRICE
    def get_price(attr):
        if attr is None or not isinstance(attr, dict):
            return np.nan
        return attr.get('RestaurantsPriceRange2', np.nan)

    df['price'] = df['attributes'].apply(get_price)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # 2. HELPER: Parse messy Booleans ('True', 'False', None, 'None')
    def parse_bool(attr_dict, key):
        if attr_dict is None or not isinstance(attr_dict, dict):
            return 0
        val = str(attr_dict.get(key, 'False')).lower().strip()
        if val == 'true':
            return 1
        return 0

    # 3. HELPER: Parse Alcohol (Categorical -> Binary)
    def parse_alcohol(attr_dict):
        if attr_dict is None or not isinstance(attr_dict, dict):
            return 0
        val = str(attr_dict.get('Alcohol', 'none')).lower()
        # Check if 'none' is IN the string
        if 'none' in val or val == 'none' or val == 'nan':
            return 0
        return 1

    # 4. APPLY EXTRACTIONS
    df['is_TakeOut'] = df['attributes'].apply(lambda x: parse_bool(x, 'RestaurantsTakeOut'))
    df['is_Delivery'] = df['attributes'].apply(lambda x: parse_bool(x, 'RestaurantsDelivery'))
    df['has_OutdoorSeating'] = df['attributes'].apply(lambda x: parse_bool(x, 'OutdoorSeating'))
    df['has_Alcohol'] = df['attributes'].apply(parse_alcohol)

    # 2. NEW ADDITIONS (The "Dealbreakers")
    df['is_GoodForKids'] = df['attributes'].apply(lambda x: parse_bool(x, 'GoodForKids'))
    df['is_GoodForGroups'] = df['attributes'].apply(lambda x: parse_bool(x, 'RestaurantsGoodForGroups'))
    df['has_HappyHour'] = df['attributes'].apply(lambda x: parse_bool(x, 'HappyHour'))
    df['is_WheelchairAccessible'] = df['attributes'].apply(lambda x: parse_bool(x, 'WheelchairAccessible'))

    print(f"  -> Features Created")
    return df


def engineer_food_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer food category features with 22 super-groups.
    
    Consolidates 100+ Yelp categories into manageable cuisine types.
    """
    print("--- Starting Food Category Engineering (with Super-Groups) ---")

    # 1. SETUP
    STANDARD_CUISINES = [
        "Mexican", "Pizza", "Italian", "Chinese", "Burgers", "Thai",
        "Japanese", "Sushi", "Indian", "Seafood", "Mediterranean",
        "Sandwiches", "Breakfast & Brunch", "Vietnamese", "Steakhouses",
        "Korean", "Barbeque", "Greek"
    ]

    df['categories'] = df['categories'].fillna('')
    created_cols = []

    # 2. SPECIAL LOGIC: The "American" Merge
    print("  -> Merging 'American (New)' and 'American (Traditional)'...")

    # r'...' makes this a raw string so \( is treated as a literal parenthesis
    american_pattern = r'American \(Traditional\)|American \(New\)'

    american_mask = df['categories'].str.contains(american_pattern, case=False, regex=True)
    df['is_American'] = american_mask.astype(int)
    created_cols.append('is_American')

    # 3. SPECIAL LOGIC: Sweets
    print("  -> Merging Sweets, Bakeries & Desserts...")
    sweets_pattern = r"Bakeries|Desserts|Donuts|Cupcakes|Bagels|Ice Cream|Frozen Yogurt"
    df['is_Dessert_Baking'] = df['categories'].str.contains(sweets_pattern, case=False, regex=True).astype(int)
    created_cols.append('is_Dessert_Baking')

    # 4. SPECIAL LOGIC: Drinks
    print("  -> Merging Coffee, Tea, Cafes & Drinks...")
    drinks_pattern = r"Coffee|Tea|Bubble Tea|Juice|Smoothies|Cafes"
    df['is_Coffee_Tea_Drinks'] = df['categories'].str.contains(drinks_pattern, case=False, regex=True).astype(int)
    created_cols.append('is_Coffee_Tea_Drinks')

    # 5. STANDARD LOOP
    print(f"  -> Processing {len(STANDARD_CUISINES)} standard cuisines...")
    for cuisine in STANDARD_CUISINES:
        clean_name = cuisine.replace(' ', '_').replace('&', 'and').replace('(', '').replace(')', '')
        col_name = f"is_{clean_name}"

        df[col_name] = df['categories'].str.contains(cuisine, case=False, regex=False).astype(int)
        created_cols.append(col_name)

    # 6. RESIDUAL LOGIC
    print("  -> Calculating 'is_Other'...")
    row_sums = df[created_cols].sum(axis=1)
    df['is_Other'] = (row_sums == 0).astype(int)

    total_other = df['is_Other'].sum()
    print(f"Finished. Found {total_other} 'Other' restaurants ({total_other/len(df)*100:.1f}% of data).")

    return df


def impute_missing_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaN in 'price' with Median per neighborhood.
    """
    print(f"Original Missing Price Count: {df['price'].isna().sum()}")

    # 1. GROUP BY SPATIAL CLUSTER
    # Calculate the median price for each of your 6 neighborhood types (0-5)
    # transform('median') broadcasts that value back to the original rows
    cluster_medians = df.groupby('neighborhood_type')['price'].transform('median')

    # 2. FILL NA
    # Only fills the missing rows
    df['price'] = df['price'].fillna(cluster_medians)

    # 3. FALLBACK
    # If a cluster is totally empty, fill with global median
    global_median = df['price'].median()
    df['price'] = df['price'].fillna(global_median)

    print(f"Final Missing Price Count: {df['price'].isna().sum()}")
    return df


def finalize_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop raw/unused columns and finalize feature set.
    """
    print(f"Data size before cleanup: {df.shape}")

    # DROP RAW/JUNK COLUMNS
    cols_to_drop = [
        'address', 'postal_code', 'attributes', 'categories', 'hours', 'is_open'
    ]

    # Drop only if they exist (errors='ignore')
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # VERIFY REMAINING COLUMNS
    print(f"Final Column Count: {len(df.columns)}")
    print(f"Columns kept: {list(df.columns)}")
    print(f"Data size after cleanup: {df.shape}")

    return df


def main():
    """Main spatial ETL pipeline."""
    print("\n" + "="*60)
    print("SPATIAL ETL PIPELINE")
    print("="*60)
    
    # Ensure directories exist
    PATHS.ensure_dirs()
    
    # Load business file
    print(f"\nLoading business data from: {PATHS.RAW_BUSINESS}")
    df_raw = pd.read_parquet(PATHS.RAW_BUSINESS)
    print(f"  Shape: {df_raw.shape}")
    print(f"  NaN counts:\n{df_raw.isna().sum()}")
    
    # Step 1: Spatial clustering
    print("\n" + "-"*40)
    print("STEP 1: SPATIAL CLUSTERING")
    print("-"*40)
    df = run_spatial_clustering(df_raw.copy())
    
    # Re-map -1 (Unknown) to 5
    df['neighborhood_type'] = df['neighborhood_type'].replace(-1, 5)
    print(f"Neighborhood type distribution:\n{df['neighborhood_type'].value_counts().sort_index()}")
    
    # Mapping neighborhood names
    neighborhood_name = {
        0: "Downtown",
        1: "Urban / Commercial",
        2: "Residential / Mixed",
        3: "Suburban",
        4: "Rural",
        5: "Small Town"
    }
    df['neighborhood_name'] = df['neighborhood_type'].map(neighborhood_name)
    
    # Step 2: Extract attributes
    print("\n" + "-"*40)
    print("STEP 2: ATTRIBUTE EXTRACTION")
    print("-"*40)
    df = engineer_attributes(df)
    
    # Step 3: Food categories
    print("\n" + "-"*40)
    print("STEP 3: FOOD CATEGORY ENGINEERING")
    print("-"*40)
    df = engineer_food_categories(df)
    
    # Step 4: Handle missing values
    print("\n" + "-"*40)
    print("STEP 4: MISSING VALUE IMPUTATION")
    print("-"*40)
    df = impute_missing_price(df)
    print(f"Total NaN remaining:\n{df.isna().sum()}")
    
    # Step 5: Finalize
    print("\n" + "-"*40)
    print("STEP 5: FINALIZE FEATURES")
    print("-"*40)
    df = finalize_business_features(df)
    
    # Print feature counts
    food_cols = [c for c in df.columns if c.startswith('is_') and c not in CONFIG.binary_features]
    service_cols = CONFIG.binary_features
    
    print("\nFood category counts:")
    if food_cols:
        count_food = df[[c for c in food_cols if c in df.columns]].sum().sort_values(ascending=False)
        print(count_food.head(10))
    
    print("\nService feature counts:")
    if service_cols:
        count_service = df[[c for c in service_cols if c in df.columns]].sum().sort_values(ascending=False)
        print(count_service)
    
    # Save
    print("\n" + "-"*40)
    print("SAVING OUTPUT")
    print("-"*40)
    df.to_parquet(PATHS.FEATURES_BUSINESS, index=False)
    print(f"✅ Saved: {PATHS.FEATURES_BUSINESS}")
    
    print("\n✅ SPATIAL ETL COMPLETE!")
    
    return df


if __name__ == "__main__":
    main()
