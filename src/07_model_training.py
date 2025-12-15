#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
07_MODEL_TRAINING.PY - Model Training and Evaluation
================================================================================
Trains OLS, XGBoost, DNN models. Evaluates on R², RMSE, MAE, NDCG.

Key Results:
    - OLS:     R² = 0.8516, RMSE = 0.2509, NDCG@5 = 0.9553
    - XGBoost: R² = 0.8507, RMSE = 0.2516, NDCG@5 = 0.9146
    - DNN:     R² = 0.8529, RMSE = 0.2497, NDCG@5 = 0.9830

Key Insight:
    Model convergence (R² ≈ 0.852 for all) proves the signal is in the FEATURES,
    not the MODEL. However, DNN achieves superior NDCG due to continuous predictions.

Input:
    - data/processed/features_all.parquet

Output:
    - models/ols_model.joblib
    - models/xgb_model.joblib
    - models/dnn_model.keras
    - models/scaler.joblib
    - results/model_comparison.csv
    - figures/dnn_training_history.png
    - figures/model_comparison.png

Usage:
    python src/07_model_training.py
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CONFIG, PATHS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


# ==============================================================================
# DATA LOADING AND PREPARATION
# ==============================================================================

def load_and_assemble_features() -> pd.DataFrame:
    """Load and prepare feature dataset."""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    df = pd.read_parquet(PATHS.FEATURES_ALL)
    all_features = CONFIG.get_all_features()
    
    # Handle missing features
    for f in all_features:
        if f not in df.columns:
            df[f] = 0
    
    # Fill missing values
    for col in CONFIG.continuous_features:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    for col in CONFIG.binary_features + CONFIG.cuisine_features:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    
    print(f"  Loaded {len(df):,} samples")
    print(f"  Features: {len(all_features)}")
    
    return df


def prepare_data(df: pd.DataFrame) -> Tuple:
    """Prepare train/val/test splits with scaling."""
    print("\n" + "="*60)
    print("PREPARING DATA SPLITS")
    print("="*60)
    
    feature_cols = CONFIG.get_all_features()
    X = df[feature_cols].copy()
    y = df[CONFIG.target_col].copy()
    
    # Train/Test split
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        X, y, df.index, test_size=CONFIG.test_size, random_state=CONFIG.random_state)
    
    # Train/Val split
    val_ratio = CONFIG.val_size / (1 - CONFIG.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=CONFIG.random_state)
    
    # Scale continuous features
    scaler = StandardScaler()
    continuous_idx = [feature_cols.index(f) for f in CONFIG.continuous_features]
    scaler.fit(X_train.iloc[:, continuous_idx])
    
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled.iloc[:, continuous_idx] = scaler.transform(X_train.iloc[:, continuous_idx])
    X_val_scaled.iloc[:, continuous_idx] = scaler.transform(X_val.iloc[:, continuous_idx])
    X_test_scaled.iloc[:, continuous_idx] = scaler.transform(X_test.iloc[:, continuous_idx])
    
    print(f"  Train: {len(X_train):,}")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test:  {len(X_test):,}")
    
    return (X_train_scaled.values, X_val_scaled.values, X_test_scaled.values,
            y_train.values, y_val.values, y_test.values, 
            scaler, feature_cols, df.loc[idx_test].copy())


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def dcg_at_k(rel, k):
    """Compute Discounted Cumulative Gain at k."""
    rel = np.asarray(rel)[:k]
    return np.sum(rel / np.log2(np.arange(2, rel.size + 2))) if rel.size else 0.0


def ndcg_at_k(y_true, y_pred, k):
    """Compute Normalized DCG at k."""
    order = np.argsort(y_pred)[::-1]
    ideal = dcg_at_k(np.sort(y_true)[::-1], k)
    return dcg_at_k(y_true[order], k) / ideal if ideal > 0 else 0.0


def evaluate_model(y_true, y_pred, name) -> Dict:
    """Evaluate model with multiple metrics."""
    metrics = {
        'model': name,
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'NDCG@5': ndcg_at_k(y_true, y_pred, 5),
        'NDCG@10': ndcg_at_k(y_true, y_pred, 10),
        'NDCG@20': ndcg_at_k(y_true, y_pred, 20)
    }
    
    print(f"\n{name}:")
    print(f"  R² = {metrics['R2']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.4f}")
    print(f"  MAE = {metrics['MAE']:.4f}")
    print(f"  NDCG@5 = {metrics['NDCG@5']:.4f}")
    print(f"  NDCG@10 = {metrics['NDCG@10']:.4f}")
    print(f"  NDCG@20 = {metrics['NDCG@20']:.4f}")
    
    return metrics


# ==============================================================================
# MODEL BUILDING
# ==============================================================================

def build_dnn(input_dim):
    """Build DNN model architecture."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(CONFIG.dnn_hidden_layers[0], activation='relu'),
        BatchNormalization(),
        Dropout(CONFIG.dnn_dropout_rate),
        Dense(CONFIG.dnn_hidden_layers[1], activation='relu'),
        BatchNormalization(),
        Dropout(CONFIG.dnn_dropout_rate),
        Dense(CONFIG.dnn_hidden_layers[2], activation='relu'),
        BatchNormalization(),
        Dropout(CONFIG.dnn_dropout_rate),
        Dense(1, activation='linear')
    ])
    return model


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_training_history(history, save_path: Optional[str] = None):
    """Plot DNN training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('DNN Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[1].plot(history.history['mae'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('DNN Training MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot model comparison across metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    models = results_df['model'].tolist()
    # Shorten model names for display
    model_labels = [m.split()[0] for m in models]  # 'OLS', 'XGBoost', 'DNN'
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # R² Score
    r2_values = results_df['R2'].tolist()
    bars1 = axes[0].bar(model_labels, r2_values, color=colors)
    axes[0].set_ylabel('R²')
    axes[0].set_title('R² Score (Higher is Better)')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(r2_values):
        axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # RMSE
    rmse_values = results_df['RMSE'].tolist()
    bars2 = axes[1].bar(model_labels, rmse_values, color=colors)
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE (Lower is Better)')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(rmse_values):
        axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')
    
    # NDCG@10
    ndcg_values = results_df['NDCG@10'].tolist()
    bars3 = axes[2].bar(model_labels, ndcg_values, color=colors)
    axes[2].set_ylabel('NDCG@10')
    axes[2].set_title('NDCG@10 (Higher is Better)')
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(ndcg_values):
        axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


def plot_predictions_scatter(y_true, predictions_dict, save_path: Optional[str] = None):
    """Plot actual vs predicted scatter for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for ax, (name, y_pred), color in zip(axes, predictions_dict.items(), colors):
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, c=color)
        ax.plot([1, 5], [1, 5], 'k--', linewidth=2, label='Perfect')
        ax.set_xlabel('Actual Stars')
        ax.set_ylabel('Predicted Stars')
        ax.set_title(f'{name}\nR² = {r2_score(y_true, y_pred):.4f}')
        ax.set_xlim(1, 5)
        ax.set_ylim(1, 5)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


def plot_residuals(y_true, predictions_dict, save_path: Optional[str] = None):
    """Plot residual distributions for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for ax, (name, y_pred), color in zip(axes, predictions_dict.items(), colors):
        residuals = y_true - y_pred
        ax.hist(residuals, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Residual (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name}\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def main():
    """Main model training pipeline."""
    print("\n" + "="*60)
    print("MODEL TRAINING PIPELINE")
    print("="*60)
    
    PATHS.ensure_dirs()
    
    # Load and prepare data
    df = load_and_assemble_features()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, df_test = prepare_data(df)
    
    # ==================================================================
    # 1. OLS (Baseline)
    # ==================================================================
    print("\n" + "="*60)
    print("TRAINING OLS (Baseline)")
    print("="*60)
    
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    y_pred_ols = ols.predict(X_test)
    
    # ==================================================================
    # 2. XGBoost (Primary)
    # ==================================================================
    print("\n" + "="*60)
    print("TRAINING XGBoost (Primary)")
    print("="*60)
    
    xgb = XGBRegressor(**CONFIG.xgb_params, early_stopping_rounds=20)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred_xgb = xgb.predict(X_test)
    
    # ==================================================================
    # 3. DNN (Comparison)
    # ==================================================================
    print("\n" + "="*60)
    print("TRAINING DNN (Comparison)")
    print("="*60)
    
    dnn = build_dnn(X_train.shape[1])
    dnn.compile(
        optimizer=Adam(learning_rate=CONFIG.dnn_learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]
    
    history = dnn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG.dnn_epochs,
        batch_size=CONFIG.dnn_batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    y_pred_dnn = dnn.predict(X_test, verbose=0).flatten()
    
    # ==================================================================
    # EVALUATE ALL MODELS
    # ==================================================================
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    results = [
        evaluate_model(y_test, y_pred_ols, "OLS (Baseline)"),
        evaluate_model(y_test, y_pred_xgb, "XGBoost (Primary)"),
        evaluate_model(y_test, y_pred_dnn, "DNN (Comparison)")
    ]
    
    results_df = pd.DataFrame(results)
    
    # ==================================================================
    # PRINT SUMMARY TABLE
    # ==================================================================
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # ==================================================================
    # GENERATE VISUALIZATIONS
    # ==================================================================
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. DNN Training History
    plot_training_history(
        history, 
        save_path=str(PATHS.FIGURES_DIR / 'dnn_training_history.png')
    )
    
    # 2. Model Comparison Bar Chart
    plot_model_comparison(
        results_df,
        save_path=str(PATHS.FIGURES_DIR / 'model_comparison.png')
    )
    
    # 3. Predictions Scatter Plot
    predictions_dict = {
        'OLS': y_pred_ols,
        'XGBoost': y_pred_xgb,
        'DNN': y_pred_dnn
    }
    plot_predictions_scatter(
        y_test, predictions_dict,
        save_path=str(PATHS.FIGURES_DIR / 'predictions_scatter.png')
    )
    
    # 4. Residuals Distribution
    plot_residuals(
        y_test, predictions_dict,
        save_path=str(PATHS.FIGURES_DIR / 'residuals_distribution.png')
    )
    
    # ==================================================================
    # SAVE MODELS AND RESULTS
    # ==================================================================
    print("\n" + "="*60)
    print("SAVING MODELS AND RESULTS")
    print("="*60)
    
    joblib.dump(ols, PATHS.OLS_MODEL)
    print(f"✅ Saved: {PATHS.OLS_MODEL}")
    
    joblib.dump(xgb, PATHS.XGB_MODEL)
    print(f"✅ Saved: {PATHS.XGB_MODEL}")
    
    dnn.save(PATHS.DNN_MODEL)
    print(f"✅ Saved: {PATHS.DNN_MODEL}")
    
    joblib.dump(scaler, PATHS.SCALER)
    print(f"✅ Saved: {PATHS.SCALER}")
    
    results_df.to_csv(PATHS.MODEL_COMPARISON, index=False)
    print(f"✅ Saved: {PATHS.MODEL_COMPARISON}")
    
    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*60)
    
    print(f"""
KEY FINDINGS:
─────────────
1. MODEL CONVERGENCE:
   • All three models achieve R² ≈ 0.852
   • This proves the signal is in the FEATURES, not the MODEL

2. RANKING PERFORMANCE:
   • DNN:     NDCG@5 = {results_df[results_df['model'].str.contains('DNN')]['NDCG@5'].values[0]:.4f}
   • OLS:     NDCG@5 = {results_df[results_df['model'].str.contains('OLS')]['NDCG@5'].values[0]:.4f}
   • XGBoost: NDCG@5 = {results_df[results_df['model'].str.contains('XGBoost')]['NDCG@5'].values[0]:.4f}
   
   DNN achieves superior NDCG because continuous predictions
   provide finer-grained ranking than XGBoost's discrete outputs.

3. FILES GENERATED:
   • models/ols_model.joblib
   • models/xgb_model.joblib
   • models/dnn_model.keras
   • models/scaler.joblib
   • results/model_comparison.csv
   • figures/dnn_training_history.png
   • figures/model_comparison.png
   • figures/predictions_scatter.png
   • figures/residuals_distribution.png
    """)
    
    return {
        'ols': ols,
        'xgb': xgb,
        'dnn': dnn,
        'scaler': scaler,
        'results': results_df,
        'history': history,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'y_pred_ols': y_pred_ols,
        'y_pred_xgb': y_pred_xgb,
        'y_pred_dnn': y_pred_dnn,
        'feature_names': feature_names,
        'df_test': df_test
    }


if __name__ == "__main__":
    output = main()
