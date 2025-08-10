#!/usr/bin/env python3
"""
Re-analysis (Step 4 variant): Exclude initial-core months
=========================================================
Reads the baseline monthly_transitions.csv, removes months with no existing
core (i.e., initial core formation), recomputes monthly-level stats, and
compares to baseline results.
"""

from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = REPO_ROOT / 'RQ1_transition_rates_and_speeds' / 'step4_newcomer_transition_rates'
BASELINE_RESULTS_DIR = BASE_DIR / 'corrected_transition_results'
VARIANT_RESULTS_DIR = BASE_DIR / 'corrected_transition_results_excluding_initial_core'


def load_monthly_transitions() -> pd.DataFrame:
    path = BASELINE_RESULTS_DIR / 'monthly_transitions.csv'
    if not path.exists():
        raise FileNotFoundError(f"Baseline monthly_transitions.csv not found at: {path}")
    df = pd.read_csv(path)
    return df


def compute_monthly_level_stats(monthly_df: pd.DataFrame) -> Dict[str, Any]:
    # Only consider months with an existing core, and then non-zero transitions
    valid_df = monthly_df[monthly_df['existing_core_count'] > 0]
    non_zero_df = valid_df[valid_df['truly_new_core_count'] > 0]

    oss_rates = non_zero_df[non_zero_df['project_type'] == 'OSS']['transition_rate']
    oss4sg_rates = non_zero_df[non_zero_df['project_type'] == 'OSS4SG']['transition_rate']

    # If either is empty, return NaNs to avoid crashes
    if len(oss_rates) == 0 or len(oss4sg_rates) == 0:
        return {
            'oss_median': np.nan,
            'oss4sg_median': np.nan,
            'oss_mean': np.nan,
            'oss4sg_mean': np.nan,
            'p_value': np.nan,
            'cliff_delta': np.nan,
            'effect_magnitude': 'n/a',
            'significant': False,
            'oss_count': int(len(oss_rates)),
            'oss4sg_count': int(len(oss4sg_rates)),
            'total_valid_months': int(len(valid_df)),
            'total_non_zero_months': int(len(non_zero_df)),
        }

    # Mann-Whitney U
    _, pvalue = stats.mannwhitneyu(oss_rates, oss4sg_rates, alternative='two-sided')

    # Cliff's Delta
    nx = len(oss_rates)
    ny = len(oss4sg_rates)
    greater = 0
    for xi in oss_rates:
        for yi in oss4sg_rates:
            if xi > yi:
                greater += 1
    cliff_delta = (2 * greater / (nx * ny)) - 1

    abs_delta = abs(cliff_delta)
    if abs_delta < 0.147:
        magnitude = 'negligible'
    elif abs_delta < 0.33:
        magnitude = 'small'
    elif abs_delta < 0.474:
        magnitude = 'medium'
    else:
        magnitude = 'large'

    return {
        'oss_median': float(oss_rates.median()),
        'oss4sg_median': float(oss4sg_rates.median()),
        'oss_mean': float(oss_rates.mean()),
        'oss4sg_mean': float(oss4sg_rates.mean()),
        'p_value': float(pvalue),
        'cliff_delta': float(cliff_delta),
        'effect_magnitude': magnitude,
        'significant': bool(pvalue < 0.05),
        'oss_count': int(len(oss_rates)),
        'oss4sg_count': int(len(oss4sg_rates)),
        'total_valid_months': int(len(valid_df)),
        'total_non_zero_months': int(len(non_zero_df)),
    }


def load_baseline_results() -> Dict[str, Any]:
    path = BASELINE_RESULTS_DIR / 'monthly_analysis_results.csv'
    if not path.exists():
        raise FileNotFoundError(f"Baseline monthly_analysis_results.csv not found at: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Baseline monthly_analysis_results.csv is empty")
    row = df.iloc[0].to_dict()
    return {
        'oss_median': float(row.get('oss_median', float('nan'))),
        'oss4sg_median': float(row.get('oss4sg_median', float('nan'))),
        'oss_mean': float(row.get('oss_mean', float('nan'))),
        'oss4sg_mean': float(row.get('oss4sg_mean', float('nan'))),
        'p_value': float(row.get('p_value', float('nan'))),
        'cliff_delta': float(row.get('cliff_delta', float('nan'))),
        'effect_magnitude': row.get('effect_magnitude', 'n/a'),
        'significant': bool(row.get('significant', False)),
        'oss_count': int(row.get('oss_count', 0)),
        'oss4sg_count': int(row.get('oss4sg_count', 0)),
    }


def save_variant_outputs(filtered_monthly: pd.DataFrame, results: Dict[str, Any]) -> None:
    VARIANT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Save filtered monthly data
    filtered_monthly.to_csv(VARIANT_RESULTS_DIR / 'monthly_transitions_filtered.csv', index=False)
    # Save results
    pd.DataFrame([results]).to_csv(VARIANT_RESULTS_DIR / 'monthly_analysis_results.csv', index=False)
    # Save summary table (aligned with baseline style)
    summary_table = pd.DataFrame({
        'Metric': ['Monthly Transition Rate (Non-Zero Months, Existing Core Only)'],
        'OSS_Median': [results['oss_median']],
        'OSS4SG_Median': [results['oss4sg_median']],
        'P_Value': [results['p_value']],
        'Effect_Size': [results['effect_magnitude']],
        'Significant': [results['significant']],
    })
    summary_table.to_csv(VARIANT_RESULTS_DIR / 'summary_table.csv', index=False)


def main():
    print("=" * 80)
    print("STEP 4 VARIANT: Re-analysis excluding initial-core months")
    print("=" * 80)

    print(f"Baseline dir: {BASELINE_RESULTS_DIR}")
    print(f"Variant dir:  {VARIANT_RESULTS_DIR}")

    monthly_df = load_monthly_transitions()

    # Filter out initial-core months (no existing core)
    valid_df = monthly_df[monthly_df['existing_core_count'] > 0].copy()
    results = compute_monthly_level_stats(monthly_df)

    # Save filtered outputs
    save_variant_outputs(valid_df, results)

    # Compare to baseline
    baseline = load_baseline_results()
    print("\nBASELINE vs VARIANT (excluding initial-core months)")
    print("-" * 80)
    def fmt(x):
        return 'nan' if pd.isna(x) else f"{x:.6f}" if isinstance(x, float) else str(x)

    print(f"OSS median:    baseline={fmt(baseline['oss_median'])} -> variant={fmt(results['oss_median'])}")
    print(f"OSS4SG median: baseline={fmt(baseline['oss4sg_median'])} -> variant={fmt(results['oss4sg_median'])}")
    print(f"p-value:       baseline={fmt(baseline['p_value'])} -> variant={fmt(results['p_value'])}")
    print(f"Cliff's delta: baseline={fmt(baseline['cliff_delta'])} -> variant={fmt(results['cliff_delta'])}")
    print(f"Effect size:   baseline={baseline['effect_magnitude']} -> variant={results['effect_magnitude']}")
    print(f"Significant:   baseline={baseline['significant']} -> variant={results['significant']}")

    print("\nCOUNTS")
    print(f"- Valid months (existing core > 0): {results['total_valid_months']:,}")
    print(f"- Non-zero transition months:       {results['total_non_zero_months']:,}")
    print(f"- OSS observations used:            {results['oss_count']:,}")
    print(f"- OSS4SG observations used:         {results['oss4sg_count']:,}")

    # Simple verdict
    changed = any([
        not np.isclose(baseline['oss_median'], results['oss_median'], equal_nan=True),
        not np.isclose(baseline['oss4sg_median'], results['oss4sg_median'], equal_nan=True),
        not np.isclose(baseline['p_value'], results['p_value'], equal_nan=True),
        not np.isclose(baseline['cliff_delta'], results['cliff_delta'], equal_nan=True),
        baseline['effect_magnitude'] != results['effect_magnitude'],
        baseline['significant'] != results['significant'],
    ])
    print("\nRESULT:")
    if changed:
        print("There IS a difference after excluding initial-core months.")
    else:
        print("No difference detected after excluding initial-core months.")


if __name__ == '__main__':
    main()


