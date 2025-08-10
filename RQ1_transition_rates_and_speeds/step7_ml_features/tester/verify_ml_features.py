#!/usr/bin/env python3
"""
Verification Script for Dataset 4: ML Features
==============================================

Validates the ml_features_dataset.csv for correctness and ML readiness.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from scipy import stats


class MLFeaturesVerifier:
    def __init__(self):
        step_dir = Path(__file__).resolve().parents[1]
        self.ml_file = step_dir / 'results/ml_features_dataset.csv'
        self.ml_df = None
        self.errors = []
        self.warnings = []

    def load_data(self):
        print('=' * 70)
        print('ML FEATURES DATASET VERIFICATION')
        print('=' * 70)
        if not self.ml_file.exists():
            print(f'❌ ERROR: ML features file not found: {self.ml_file}')
            sys.exit(1)
        print(f'Loading: {self.ml_file}')
        self.ml_df = pd.read_csv(self.ml_file, low_memory=False)
        print(f'✅ Loaded {len(self.ml_df):,} samples\n')

    def test_schema(self):
        print('Test 1: Schema and Feature Validation')
        print('-' * 40)
        required_base = ['project_name', 'project_type', 'contributor_email', 'label_became_core', 'total_weeks_observed']
        missing_base = set(required_base) - set(self.ml_df.columns)
        if missing_base:
            self.errors.append(f'Missing base columns: {missing_base}')
            print(f'❌ Missing base columns: {missing_base}')
        else:
            print('✅ All base columns present')

        expected = ['w1_commits', 'w1_4_total_commits', 'w1_8_total_commits', 'w1_12_total_commits', 'activity_regularity']
        missing = [c for c in expected if c not in self.ml_df.columns]
        if missing:
            self.warnings.append(f'Missing features: {missing}')
            print(f'⚠️ Missing features: {missing}')
        else:
            print('✅ Expected key features present')
        print()

    def test_label_distribution(self):
        print('Test 2: Label Distribution and Class Balance')
        print('-' * 40)
        total = len(self.ml_df)
        pos = int(self.ml_df['label_became_core'].sum())
        pct = pos / total * 100 if total else 0
        print(f'Total samples: {total:,}')
        print(f'Positive class: {pos:,} ({pct:.1f}%)')
        print('By project type:')
        for p in self.ml_df['project_type'].unique():
            d = self.ml_df[self.ml_df['project_type'] == p]
            print(f'  {p}: {int(d["label_became_core"].sum())}/{len(d)} ({d["label_became_core"].mean()*100:.1f}%)')
        print()

    def test_feature_validity(self):
        print('Test 3: Feature Validity Checks')
        print('-' * 40)
        numeric_cols = self.ml_df.select_dtypes(include=[np.number]).columns
        if self.ml_df[numeric_cols].isnull().any().any():
            self.warnings.append('NaNs present in numeric columns')
            print('⚠️ NaNs present in numeric columns')
        else:
            print('✅ No NaNs in numeric columns')
        if np.isinf(self.ml_df[numeric_cols]).any().any():
            self.errors.append('Infinities present in numeric columns')
            print('❌ Infinities present in numeric columns')
        else:
            print('✅ No infinities')
        # Consistency ranges
        for c in [col for col in self.ml_df.columns if 'consistency' in col]:
            mn, mx = self.ml_df[c].min(), self.ml_df[c].max()
            if mn < 0 or mx > 1:
                self.warnings.append(f'{c} outside [0,1]: [{mn},{mx}]')
                print(f'⚠️ {c} outside [0,1]: [{mn},{mx}]')
        print()

    def test_feature_discrimination(self):
        print('Test 4: Feature Discrimination (quick)')
        print('-' * 40)
        y = self.ml_df['label_became_core'].astype(int)
        good = 0
        for feat in ['w1_4_total_commits', 'w1_8_total_commits', 'w1_12_total_commits', 'activity_regularity']:
            if feat in self.ml_df.columns:
                t, p = stats.ttest_ind(self.ml_df[y == 1][feat], self.ml_df[y == 0][feat], equal_var=False, nan_policy='omit')
                if p < 0.05:
                    good += 1
                    print(f'✓ {feat}: p={p:.3e}')
                else:
                    print(f'  {feat}: p={p:.3e}')
        if good >= 2:
            print(f'✅ {good} features show discrimination')
        else:
            print(f'⚠️ Only {good} discriminative features found')
        print()

    def generate_report(self):
        print('=' * 70)
        print('VERIFICATION SUMMARY')
        print('=' * 70)
        if self.errors:
            print('❌ Issues found:')
            for e in self.errors:
                print(' -', e)
        else:
            print('✅ No blocking issues')

    def run_all_tests(self):
        self.load_data()
        self.test_schema()
        self.test_label_distribution()
        self.test_feature_validity()
        self.test_feature_discrimination()
        self.generate_report()


def main():
    verifier = MLFeaturesVerifier()
    verifier.run_all_tests()


if __name__ == '__main__':
    main()


