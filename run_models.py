#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
RUN_MODELS.PY - Run Model Training & Evaluation Only
================================================================================
Use this script when you already have processed data in data/processed/:
    - features_all.parquet
    - restaurant_embeddings.parquet

This SKIPS the time-consuming preprocessing steps (SBERT, sentiment, etc.)

Usage:
    python run_models.py              # Run all 3 scripts
    python run_models.py --train      # Only training
    python run_models.py --ablation   # Only ablation
    python run_models.py --mmr        # Only MMR
================================================================================
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent


def check_required_files():
    """Verify all required processed files exist."""
    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    
    required_files = [
        'features_all.parquet',
        'restaurant_embeddings.parquet',
    ]
    
    print("=" * 60)
    print("CHECKING REQUIRED FILES")
    print("=" * 60)
    
    all_exist = True
    for name in required_files:
        path = processed_dir / name
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n❌ ERROR: Missing required files!")
        print("   You need to run the preprocessing scripts first.")
        sys.exit(1)
    
    print("\n✅ All required files found!")
    return True


def run_script(script_name):
    """Run a Python script using subprocess."""
    script_path = PROJECT_ROOT / 'src' / script_name
    
    print("\n" + "=" * 60)
    print(f"RUNNING: {script_name}")
    print("=" * 60)
    
    if not script_path.exists():
        print(f"❌ ERROR: {script_path} not found!")
        return False
    
    # Run the script
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT)
    )
    
    if result.returncode == 0:
        print(f"✅ {script_name} completed successfully!")
        return True
    else:
        print(f"❌ {script_name} failed with return code {result.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run model training & evaluation (skips preprocessing)"
    )
    parser.add_argument('--train', action='store_true', help='Run only model training')
    parser.add_argument('--ablation', action='store_true', help='Run only ablation study')
    parser.add_argument('--mmr', action='store_true', help='Run only MMR evaluation')
    parser.add_argument('--skip-check', action='store_true', help='Skip file existence check')
    
    args = parser.parse_args()
    
    # If no specific flags, run all
    run_all = not (args.train or args.ablation or args.mmr)
    
    # Check required files
    if not args.skip_check:
        check_required_files()
    
    success = True
    
    # Run requested scripts
    if run_all or args.train:
        if not run_script('07_model_training.py'):
            success = False
    
    if run_all or args.ablation:
        if not run_script('08_ablation.py'):
            success = False
    
    if run_all or args.mmr:
        if not run_script('09_mmr.py'):
            success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL COMPLETE!")
    else:
        print("⚠️  Some scripts had errors. Check output above.")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    main()
