#!/usr/bin/env python3
"""
Step 9.1: Prepare ML Modeling Dataset
Loads the Step 7 features dataset, performs cleaning/selection, and saves
model-ready matrices for downstream modeling.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
STEP7_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step7_ml_features"
STEP9_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step9_ml_modeling"
DATA_DIR = STEP9_DIR / "data"
RESULTS_DIR = STEP9_DIR / "results"

for d in [STEP9_DIR, DATA_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_step7_dataset() -> pd.DataFrame:
    input_path = STEP7_DIR / "results/ml_features_dataset.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing Step 7 features at {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded Step 7 dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    # Identify label and groups/ids
    label_col_candidates = ["label_became_core", "became_core"]
    label_col = next((c for c in label_col_candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError("Label column not found (expected label_became_core or became_core)")

    # Drop obvious leakage/ids
    id_like = {
        'project_name', 'contributor_email', 'project_type',
        'first_core_week', 'weeks_to_core', 'time_to_core', 'time_to_event_weeks',
        'total_weeks_observed', 'first_commit_week'
    }
    id_like = [c for c in id_like if c in df.columns]

    # Numeric columns only, excluding label and ids
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if label_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[label_col])
    numeric_df = numeric_df.drop(columns=[c for c in id_like if c in numeric_df.columns], errors='ignore')

    # Remove constant columns
    nunique = numeric_df.nunique(dropna=False)
    keep_cols = nunique[nunique > 1].index.tolist()
    X = numeric_df[keep_cols].copy()

    y = df[label_col].astype(int)
    meta_cols = [c for c in ['project_name', 'project_type', 'contributor_email'] if c in df.columns]
    meta = df[meta_cols].copy()

    print(f"Selected {X.shape[1]} numeric features; positive rate: {y.mean():.3f}")
    return X, y, meta


def save_outputs(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> None:
    X_path = DATA_DIR / "X_features.parquet"
    y_path = DATA_DIR / "y_labels.parquet"
    meta_path = DATA_DIR / "meta.parquet"
    X.to_parquet(X_path, index=False)
    y.to_frame('label').to_parquet(y_path, index=False)
    meta.to_parquet(meta_path, index=False)
    print(f"Saved features to {X_path}")

    # Summary JSON
    summary = {
        'num_rows': int(len(y)),
        'num_features': int(X.shape[1]),
        'positive_rate': float(y.mean()),
        'by_type': {},
    }
    if 'project_type' in meta.columns:
        # Use a temporary column to avoid column name mismatch
        tmp = meta.copy()
        tmp['__label__'] = y.values
        for t, grp in tmp.groupby('project_type'):
            summary['by_type'][t] = {
                'n': int(len(grp)),
                'pos_rate': float(grp['__label__'].mean()),
            }
    with open(RESULTS_DIR / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved dataset summary to {RESULTS_DIR / 'dataset_summary.json'}")


def main():
    print("=" * 60)
    print("STEP 9.1: PREPARE ML DATASET")
    print("=" * 60)
    df = load_step7_dataset()
    X, y, meta = select_features(df)
    save_outputs(X, y, meta)
    print("Done.")


if __name__ == "__main__":
    main()


