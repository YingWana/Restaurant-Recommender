#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
09_MMR.PY - Maximal Marginal Relevance Diversity Evaluation
================================================================================
This script evaluates the relevance-diversity trade-off using MMR re-ranking.

MMR Equation:
    MMR(item) = λ × Relevance - (1-λ) × MaxSim(item, Selected)

Key Finding:
    λ = 0.7 achieves Pareto improvement: +29% diversity with no relevance loss

Input:
    - data/processed/features_all.parquet
    - data/processed/restaurant_embeddings.parquet
    - models/xgb_model.joblib

Output:
    - results/mmr_lambda_sweep.csv
    - results/mmr_comparison.csv
    - figures/mmr_lambda_sweep.png
    - figures/mmr_comparison.png

Usage:
    python src/09_mmr.py
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import os
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CONFIG, PATHS


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Compute DCG@k."""
    relevance = np.asarray(relevance)[:k]
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return np.sum(relevance / discounts)


def ndcg_at_k(y_true: np.ndarray, ranking: List[int], k: int) -> float:
    """Compute NDCG@k for a given ranking."""
    relevance_ranked = y_true[ranking[:k]]
    ideal_relevance = np.sort(y_true)[::-1][:k]

    dcg = dcg_at_k(relevance_ranked, k)
    idcg = dcg_at_k(ideal_relevance, k)

    return dcg / idcg if idcg > 0 else 0.0


def calculate_ild(similarity_matrix: np.ndarray, ranking: List[int], k: int = None) -> float:
    """
    Compute Intra-List Diversity (ILD).
    ILD = 1 - average pairwise similarity
    
    Higher ILD = more diverse recommendations.
    """
    if k is None:
        k = len(ranking)

    top_k_indices = ranking[:k]

    if len(top_k_indices) < 2:
        return 0.0

    sub_matrix = similarity_matrix[np.ix_(top_k_indices, top_k_indices)]
    n = len(top_k_indices)
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_sims = sub_matrix[upper_tri_indices]

    return 1 - np.mean(pairwise_sims)


# ==============================================================================
# MMR CORE ALGORITHM
# ==============================================================================

def mmr_rerank(
    relevance_scores: np.ndarray,
    similarity_matrix: np.ndarray,
    lambda_param: float = 0.7,
    top_k: int = 20
) -> List[int]:
    """
    Maximal Marginal Relevance re-ranking.

    MMR(item) = λ × Relevance - (1-λ) × MaxSim(item, Selected)
    
    Parameters
    ----------
    relevance_scores : np.ndarray
        Predicted relevance/quality scores for each item
    similarity_matrix : np.ndarray
        Pairwise similarity matrix (n × n)
    lambda_param : float
        Trade-off parameter. Higher = more relevance focus.
        0.0 = pure diversity, 1.0 = pure relevance (greedy)
    top_k : int
        Number of items to return
        
    Returns
    -------
    List[int]
        Indices of selected items in MMR order
    """
    n = len(relevance_scores)

    if n == 0:
        return []

    # Normalize relevance to [0, 1]
    rel_min, rel_max = relevance_scores.min(), relevance_scores.max()
    if rel_max > rel_min:
        rel_norm = (relevance_scores - rel_min) / (rel_max - rel_min)
    else:
        rel_norm = np.ones(n)

    selected = []
    remaining = set(range(n))

    for _ in range(min(top_k, n)):
        if not remaining:
            break

        if not selected:
            # First item: highest relevance
            best_idx = max(remaining, key=lambda i: rel_norm[i])
        else:
            # Subsequent items: MMR score
            best_score = -np.inf
            best_idx = next(iter(remaining))

            for idx in remaining:
                max_sim = max(similarity_matrix[idx, s] for s in selected)
                mmr_score = lambda_param * rel_norm[idx] - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def greedy_rerank(relevance_scores: np.ndarray, top_k: int = 20) -> List[int]:
    """Greedy re-ranking (pure relevance, λ=1.0)."""
    return np.argsort(relevance_scores)[::-1][:top_k].tolist()


# ==============================================================================
# LAMBDA SWEEP
# ==============================================================================

def lambda_sweep(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    similarity_matrix: np.ndarray,
    lambda_values: List[float] = None,
    top_k: int = 20
) -> pd.DataFrame:
    """Sweep over λ values to find optimal trade-off."""

    if lambda_values is None:
        lambda_values = CONFIG.mmr_lambda_sweep_values

    results = []

    for lam in lambda_values:
        ranking = mmr_rerank(predictions, similarity_matrix, lam, top_k)

        ndcg = ndcg_at_k(ground_truth, ranking, top_k)
        ild = calculate_ild(similarity_matrix, ranking, top_k)
        combined = 0.6 * ndcg + 0.4 * ild  # Weighted combination

        results.append({
            'lambda': lam,
            'NDCG': ndcg,
            'ILD': ild,
            'Combined': combined
        })

    return pd.DataFrame(results)


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_lambda_sweep(sweep_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot λ sweep results."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # NDCG vs Lambda
    axes[0].plot(sweep_df['lambda'], sweep_df['NDCG'], 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('λ (Lambda)')
    axes[0].set_ylabel('NDCG@k')
    axes[0].set_title('Relevance vs λ')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='λ=0.7')
    axes[0].legend()

    # ILD vs Lambda
    axes[1].plot(sweep_df['lambda'], sweep_df['ILD'], 'g-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('λ (Lambda)')
    axes[1].set_ylabel('ILD')
    axes[1].set_title('Diversity vs λ')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='λ=0.7')
    axes[1].legend()

    # Trade-off curve (Pareto frontier)
    scatter = axes[2].scatter(sweep_df['ILD'], sweep_df['NDCG'],
                              c=sweep_df['lambda'], cmap='coolwarm', s=100)
    axes[2].plot(sweep_df['ILD'], sweep_df['NDCG'], 'gray', alpha=0.5)
    for _, row in sweep_df.iterrows():
        axes[2].annotate(f"λ={row['lambda']:.1f}",
                        (row['ILD'], row['NDCG']),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[2].set_xlabel('ILD (Diversity) →')
    axes[2].set_ylabel('NDCG (Relevance) →')
    axes[2].set_title('Relevance-Diversity Trade-off')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[2], label='λ')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")

    plt.show()


def plot_comparison(greedy_metrics: Dict, mmr_metrics: Dict,
                    lambda_val: float, save_path: Optional[str] = None):
    """Plot Greedy vs MMR comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    methods = ['Greedy\n(λ=1.0)', f'MMR\n(λ={lambda_val})']
    colors = ['#e74c3c', '#2ecc71']

    # NDCG
    ndcg_vals = [greedy_metrics['ndcg'], mmr_metrics['ndcg']]
    bars1 = axes[0].bar(methods, ndcg_vals, color=colors)
    axes[0].set_ylabel('NDCG@k')
    axes[0].set_title('Relevance (Higher = Better)')
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars1, ndcg_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.3f}', ha='center', fontsize=12)

    # ILD
    ild_vals = [greedy_metrics['ild'], mmr_metrics['ild']]
    bars2 = axes[1].bar(methods, ild_vals, color=colors)
    axes[1].set_ylabel('ILD')
    axes[1].set_title('Diversity (Higher = Better)')
    axes[1].set_ylim(0, 1)
    for bar, val in zip(bars2, ild_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.3f}', ha='center', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")

    plt.show()


# ==============================================================================
# LOAD EMBEDDINGS
# ==============================================================================

def load_embeddings_from_parquet(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load SBERT embeddings from parquet file."""
    print(f"\nLoading embeddings from: {path}")

    df_emb = pd.read_parquet(path)
    print(f"  Shape: {df_emb.shape}")

    business_ids = df_emb['business_id'].values

    if isinstance(df_emb['embedding'].iloc[0], (list, np.ndarray)):
        embeddings = np.vstack(df_emb['embedding'].values)
    else:
        embeddings = np.vstack(df_emb['embedding'].apply(eval).values)

    print(f"  Embeddings shape: {embeddings.shape}")

    return embeddings, business_ids


# ==============================================================================
# MAIN EVALUATION PIPELINE
# ==============================================================================

def run_mmr_evaluation(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    test_embeddings: np.ndarray,
    test_business_ids: np.ndarray = None,
    df_test: pd.DataFrame = None,
    lambda_param: float = 0.7,
    top_k: int = 20,
    save_results: bool = True
) -> Dict:
    """
    Complete MMR evaluation pipeline.
    """
    print("\n" + "="*60)
    print("MMR EVALUATION PIPELINE")
    print("="*60)

    # 1. Compute similarity matrix
    print("\n1. Computing similarity matrix...")
    similarity_matrix = cosine_similarity(test_embeddings)
    print(f"   Shape: {similarity_matrix.shape}")

    # 2. Lambda sweep
    print("\n2. Running λ sweep...")
    sweep_df = lambda_sweep(y_pred, y_true, similarity_matrix, top_k=top_k)
    print(sweep_df.to_string(index=False))

    optimal_idx = sweep_df['Combined'].idxmax()
    optimal_lambda = sweep_df.loc[optimal_idx, 'lambda']
    print(f"\n   Optimal λ (combined score): {optimal_lambda}")

    # 3. Compare Greedy vs MMR
    print(f"\n3. Comparing Greedy vs MMR (λ={lambda_param})...")

    greedy_ranking = greedy_rerank(y_pred, top_k)
    greedy_ndcg = ndcg_at_k(y_true, greedy_ranking, top_k)
    greedy_ild = calculate_ild(similarity_matrix, greedy_ranking, top_k)

    mmr_ranking = mmr_rerank(y_pred, similarity_matrix, lambda_param, top_k)
    mmr_ndcg = ndcg_at_k(y_true, mmr_ranking, top_k)
    mmr_ild = calculate_ild(similarity_matrix, mmr_ranking, top_k)

    greedy_metrics = {'ndcg': greedy_ndcg, 'ild': greedy_ild, 'ranking': greedy_ranking}
    mmr_metrics = {'ndcg': mmr_ndcg, 'ild': mmr_ild, 'ranking': mmr_ranking}

    print(f"\n   Greedy: NDCG={greedy_ndcg:.4f}, ILD={greedy_ild:.4f}")
    print(f"   MMR:    NDCG={mmr_ndcg:.4f}, ILD={mmr_ild:.4f}")

    # 4. Visualizations
    print("\n4. Generating visualizations...")

    PATHS.ensure_dirs()

    plot_lambda_sweep(
        sweep_df,
        save_path=str(PATHS.FIGURES_DIR / 'mmr_lambda_sweep.png') if save_results else None
    )

    plot_comparison(
        greedy_metrics, mmr_metrics, lambda_param,
        save_path=str(PATHS.FIGURES_DIR / 'mmr_comparison.png') if save_results else None
    )

    # 5. Show top recommendations
    if df_test is not None and test_business_ids is not None:
        print("\n5. Top-10 Recommendations:")
        print("-" * 80)
        print(f"{'Rank':<5} {'GREEDY':<35} {'MMR':<35}")
        print("-" * 80)

        for i in range(min(10, top_k)):
            greedy_idx = greedy_ranking[i]
            mmr_idx = mmr_ranking[i]

            greedy_bid = test_business_ids[greedy_idx]
            mmr_bid = test_business_ids[mmr_idx]

            greedy_row = df_test[df_test['business_id'] == greedy_bid]
            mmr_row = df_test[df_test['business_id'] == mmr_bid]

            greedy_name = greedy_row['name'].values[0][:32] if len(greedy_row) > 0 else "Unknown"
            mmr_name = mmr_row['name'].values[0][:32] if len(mmr_row) > 0 else "Unknown"

            print(f"{i+1:<5} {greedy_name:<35} {mmr_name:<35}")

    # 6. Save results
    if save_results:
        print("\n6. Saving results...")

        sweep_df.to_csv(PATHS.MMR_LAMBDA_SWEEP, index=False)
        print(f"   ✅ Saved: {PATHS.MMR_LAMBDA_SWEEP}")

        comparison_df = pd.DataFrame([
            {'Method': 'Greedy', 'Lambda': 1.0, 'NDCG': greedy_ndcg, 'ILD': greedy_ild},
            {'Method': 'MMR', 'Lambda': lambda_param, 'NDCG': mmr_ndcg, 'ILD': mmr_ild}
        ])
        comparison_df.to_csv(PATHS.MMR_COMPARISON, index=False)
        print(f"   ✅ Saved: {PATHS.MMR_COMPARISON}")

    # Summary
    ndcg_change = mmr_ndcg - greedy_ndcg
    ild_change = mmr_ild - greedy_ild

    print("\n" + "="*60)
    print("MMR EVALUATION COMPLETE!")
    print("="*60)
    print(f"""
Results Summary:
----------------
Greedy → MMR (λ={lambda_param}):
  • NDCG: {greedy_ndcg:.4f} → {mmr_ndcg:.4f} ({ndcg_change:+.4f}, {100*ndcg_change/greedy_ndcg:+.1f}%)
  • ILD:  {greedy_ild:.4f} → {mmr_ild:.4f} ({ild_change:+.4f}, {100*ild_change/max(greedy_ild, 0.001):+.1f}%)

Interpretation:
  • Relevance change: {ndcg_change:+.1%}
  • Diversity gain: {ild_change:+.1%}
  • Trade-off: {'PARETO IMPROVEMENT ✅' if ild_change > 0 and ndcg_change >= -0.01 else 'ACCEPTABLE ⚠️'}
    """)

    return {
        'sweep_df': sweep_df,
        'greedy': greedy_metrics,
        'mmr': mmr_metrics,
        'optimal_lambda': optimal_lambda,
        'similarity_matrix': similarity_matrix
    }


def main():
    """Main MMR evaluation pipeline."""
    print("\n" + "="*60)
    print("MMR DIVERSITY EVALUATION")
    print("="*60)

    PATHS.ensure_dirs()

    # Load features
    print("\n1. Loading feature data...")
    df = pd.read_parquet(PATHS.FEATURES_ALL)
    feature_cols = CONFIG.get_all_features()
    
    for f in feature_cols:
        if f not in df.columns:
            df[f] = 0
    for col in CONFIG.continuous_features:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[feature_cols].copy()
    y = df[CONFIG.target_col].copy()

    # Split to get test set
    _, X_test, _, y_test, _, idx_test = train_test_split(
        X, y, df.index,
        test_size=CONFIG.test_size,
        random_state=CONFIG.random_state
    )
    df_test = df.loc[idx_test].copy()
    print(f"   Test Set Size: {len(df_test)}")

    # Load model and predict
    print("\n2. Loading XGBoost model and predicting...")
    if not PATHS.XGB_MODEL.exists():
        raise FileNotFoundError(f"XGBoost model not found at {PATHS.XGB_MODEL}. Run training first!")
    
    xgb_model = joblib.load(PATHS.XGB_MODEL)
    scaler = joblib.load(PATHS.SCALER)

    continuous_idx = [feature_cols.index(f) for f in CONFIG.continuous_features]
    X_test_scaled = X_test.copy()
    X_test_scaled.iloc[:, continuous_idx] = scaler.transform(X_test.iloc[:, continuous_idx])
    y_pred = xgb_model.predict(X_test_scaled)
    print("   Predictions generated.")

    # Load embeddings
    print("\n3. Loading and aligning SBERT embeddings...")
    all_embeddings, all_business_ids = load_embeddings_from_parquet(str(PATHS.RESTAURANT_EMBEDDINGS))

    id_to_idx = {bid: i for i, bid in enumerate(all_business_ids)}
    test_ids = df_test['business_id'].values

    valid_test_idx = []
    valid_emb_idx = []
    for i, bid in enumerate(test_ids):
        if bid in id_to_idx:
            valid_test_idx.append(i)
            valid_emb_idx.append(id_to_idx[bid])

    final_y_pred = y_pred[valid_test_idx]
    final_y_true = y_test.values[valid_test_idx]
    final_embeddings = all_embeddings[valid_emb_idx]
    final_test_ids = test_ids[valid_test_idx]
    df_test_aligned = df_test.iloc[valid_test_idx]
    print(f"   Aligned Matches: {len(final_y_pred)}")

    # Run MMR evaluation
    results = run_mmr_evaluation(
        y_pred=final_y_pred,
        y_true=final_y_true,
        test_embeddings=final_embeddings,
        test_business_ids=final_test_ids,
        df_test=df_test_aligned,
        lambda_param=CONFIG.mmr_lambda,
        top_k=CONFIG.mmr_top_k,
        save_results=True
    )

    return results


if __name__ == "__main__":
    results = main()
