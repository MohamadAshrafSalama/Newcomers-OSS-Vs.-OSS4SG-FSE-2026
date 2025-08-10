#!/usr/bin/env python3
"""
Step 9 Tester: Validate ML dataset and outputs
- Verifies Step 7 inputs and Step 9 prepared data
- Ensures labels/features sane; no NaN/Inf; reasonable class rates
- Confirms instant-core/early-fast exclusions were applied upstream (Step 6 v2)
- Checks existence of results for combined/OSS/OSS4SG runs
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
STEP6_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step6_contributor_transitions"
STEP7_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step7_ml_features"
STEP9_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step9_ml_modeling"


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_step6_exclusions():
    path = STEP6_DIR / "results/contributor_transitions.csv"
    df = pd.read_csv(path)
    # Step 6 v2 rule checks
    assert_true((df['became_core'].isin([0, 1])).all(), "Invalid became_core values")
    # Instant cores should be excluded in v2 main output
    if 'weeks_to_core' in df.columns:
        assert_true((df['weeks_to_core'] != 0).all(), "Instant core found in v2 dataset")
    # Early joiners who became core fast should be excluded
    if {'first_commit_week', 'weeks_to_core'}.issubset(df.columns):
        bad = (df['first_commit_week'] <= 8) & (df['weeks_to_core'] <= 4) & (df['became_core'] == 1)
        assert_true((~bad).all(), "Early+fast cores present; expected excluded in v2")


def test_step7_features():
    path = STEP7_DIR / "results/ml_features_dataset.csv"
    df = pd.read_csv(path)
    # Label should exist and be binary
    label_col = 'label_became_core' if 'label_became_core' in df.columns else 'became_core'
    assert_true(label_col in df.columns, "Label column missing in Step 7 dataset")
    assert_true(set(df[label_col].unique()).issubset({0, 1}), "Non-binary labels")
    # Basic NaN/Inf checks on numeric columns
    num = df.select_dtypes(include=[np.number])
    assert_true(np.isfinite(num.to_numpy()).all(), "NaN/Inf in numeric features")
    # Expected columns
    expected = ['project_name', 'project_type', 'contributor_email']
    for c in expected:
        assert_true(c in df.columns, f"Missing expected column: {c}")


def test_step9_prepared():
    X = pd.read_parquet(STEP9_DIR / "data/X_features.parquet")
    y = pd.read_parquet(STEP9_DIR / "data/y_labels.parquet")['label']
    meta = pd.read_parquet(STEP9_DIR / "data/meta.parquet")
    # Dimensions
    assert_true(len(X) == len(y) == len(meta), "X/y/meta length mismatch")
    assert_true(X.shape[1] >= 10, "Too few features selected")
    # No NaN/Inf
    assert_true(np.isfinite(X.to_numpy()).all(), "NaN/Inf in X")
    # Reasonable positive rate
    pos = y.mean()
    assert_true(0.05 <= pos <= 0.5, f"Unexpected class imbalance (pos_rate={pos:.3f})")


def test_results_presence():
    for tag in ["combined", "oss", "oss4sg"]:
        path = STEP9_DIR / f"results/results_{tag}.json"
        assert_true(path.exists(), f"Missing results file: {path}")
        with open(path) as f:
            res = json.load(f)
        # Basic fields
        assert_true('summary' in res and res['summary'], f"No summary in {path}")
        for model in ['logreg', 'rf', 'hgb']:
            assert_true(model in res['summary'], f"Model {model} missing in summary for {tag}")


def main():
    print("Running Step 9 tester...")
    test_step6_exclusions()
    print("  Step 6 exclusion rules OK")
    test_step7_features()
    print("  Step 7 features integrity OK")
    test_step9_prepared()
    print("  Step 9 prepared dataset OK")
    test_results_presence()
    print("  Step 9 results presence OK")
    print("All Step 9 tests passed.")


if __name__ == "__main__":
    main()


