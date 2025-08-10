#!/usr/bin/env python3
"""
Step 3: Cox Proportional Hazards Regression
This analyzes which factors influence the hazard (instantaneous risk) of becoming core.
Cox regression can include multiple covariates and tells us their relative importance.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
STEP8_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step8_survival_analysis"
DATA_DIR = STEP8_DIR / "data"
RESULTS_DIR = STEP8_DIR / "results"
PLOTS_DIR = STEP8_DIR / "visualizations"


def load_and_prepare_data() -> pd.DataFrame:
    """Load survival data and prepare for Cox regression."""
    file_path = DATA_DIR / "survival_data.csv"
    print(f"Loading survival data from: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} contributors")

    # Create binary indicator for OSS4SG
    df['is_oss4sg'] = (df['project_type'] == 'OSS4SG').astype(int)

    # Handle missing values in covariates
    numeric_cols = [
        'total_commits',
        'total_lines_changed',
        'total_active_weeks',
        'activity_rate',
        'avg_commits_per_active_week_before_core',
        'commit_consistency_before_core',
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Log-transform skewed variables
    df['log_total_commits'] = np.log1p(df['total_commits']) if 'total_commits' in df.columns else 0.0
    df['log_total_lines'] = (
        np.log1p(df['total_lines_changed']) if 'total_lines_changed' in df.columns else 0.0
    )

    # Standardize continuous variables
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    continuous_vars = ['log_total_commits', 'log_total_lines', 'activity_rate']
    for var in continuous_vars:
        if var in df.columns:
            df[f'{var}_scaled'] = scaler.fit_transform(df[[var]])

    # Interaction term used in full model and validation
    if all(col in df.columns for col in ['is_oss4sg', 'log_total_commits_scaled']):
        df['oss4sg_x_commits'] = df['is_oss4sg'] * df['log_total_commits_scaled']

    return df


def simple_cox_model(df: pd.DataFrame) -> CoxPHFitter:
    """Simple Cox model with just project type."""
    print("\n=== Simple Cox Model (Project Type Only) ===")
    cox_data = df[['duration', 'event', 'is_oss4sg']].copy()

    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='duration', event_col='event')

    print("\nModel Summary:")
    print(cph.summary)

    hr = float(np.exp(cph.params_['is_oss4sg']))
    ci_lower = float(np.exp(cph.confidence_intervals_.loc['is_oss4sg', '95% lower-bound']))
    ci_upper = float(np.exp(cph.confidence_intervals_.loc['is_oss4sg', '95% upper-bound']))

    print(f"\nHazard Ratio for OSS4SG: {hr:.3f}")
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(
        f"Interpretation: OSS4SG contributors are {hr:.2f}x more likely to become core at any given time"
    )

    print("\n=== Testing Proportional Hazards Assumption ===")
    results = proportional_hazard_test(cph, cox_data, time_transform='rank')
    print(results.summary)

    return cph


def full_cox_model(df: pd.DataFrame):
    """Full Cox model with multiple covariates."""
    print("\n=== Full Cox Model (Multiple Covariates) ===")

    covariates = [
        'is_oss4sg',
        'log_total_commits_scaled',
        'log_total_lines_scaled',
        'activity_rate_scaled',
    ]

    df = df.copy()
    if all(c in df.columns for c in ['is_oss4sg', 'log_total_commits_scaled']):
        df['oss4sg_x_commits'] = df['is_oss4sg'] * df['log_total_commits_scaled']
        covariates.append('oss4sg_x_commits')

    covariates = [c for c in covariates if c in df.columns]
    cox_data = df[['duration', 'event'] + covariates].copy().dropna()

    print(f"Using {len(cox_data):,} complete cases")
    print(f"Covariates: {covariates}")

    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='duration', event_col='event')

    print("\nModel Summary:")
    print(cph.summary)

    print("\n=== Hazard Ratios ===")
    for covar in covariates:
        hr = float(np.exp(cph.params_[covar]))
        ci_lower = float(np.exp(cph.confidence_intervals_.loc[covar, '95% lower-bound']))
        ci_upper = float(np.exp(cph.confidence_intervals_.loc[covar, '95% upper-bound']))
        p_value = float(cph.summary.loc[covar, 'p'])
        print(f"\n{covar}:")
        print(f"  HR: {hr:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"  p-value: {p_value:.4f}")

    print("\n=== Model Diagnostics ===")
    print(f"Concordance Index: {cph.concordance_index_:.3f}")
    print(f"Log-Likelihood: {cph.log_likelihood_:.1f}")
    # For Cox PH, use partial AIC
    try:
        print(f"AIC (partial): {cph.AIC_partial_:.1f}")
    except Exception:
        pass

    return cph, covariates


def plot_hazard_ratios(cph: CoxPHFitter, save_path: Optional[Path] = None):
    """Forest plot of hazard ratios."""
    fig, ax = plt.subplots(figsize=(10, 6))

    hrs = np.exp(cph.params_)
    ci_lower = np.exp(cph.confidence_intervals_.iloc[:, 0])
    ci_upper = np.exp(cph.confidence_intervals_.iloc[:, 1])

    labels = []
    for var in hrs.index:
        if var == 'is_oss4sg':
            labels.append('OSS4SG (vs OSS)')
        elif var == 'log_total_commits_scaled':
            labels.append('Total Commits (log, scaled)')
        elif var == 'log_total_lines_scaled':
            labels.append('Total Lines (log, scaled)')
        elif var == 'activity_rate_scaled':
            labels.append('Activity Rate (scaled)')
        elif var == 'oss4sg_x_commits':
            labels.append('OSS4SG × Commits')
        else:
            labels.append(var)

    y_pos = np.arange(len(hrs))
    for i, (hr, lower, upper) in enumerate(zip(hrs, ci_lower, ci_upper)):
        ax.plot([lower, upper], [i, i], 'b-', linewidth=2)
        ax.plot(hr, i, 'bo', markersize=8)

    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
    ax.set_title('Cox Regression: Factors Influencing Transition to Core', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    if (ci_upper.max() / max(1e-8, ci_lower.min())) > 10:
        ax.set_xscale('log')
        ax.set_xlabel('Hazard Ratio (95% CI) - Log Scale', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved forest plot to: {save_path}")
    plt.close(fig)
    return fig


def stratified_cox_models(df: pd.DataFrame) -> dict:
    """Separate Cox models for OSS and OSS4SG to understand different predictors."""
    print("\n=== Stratified Cox Models (Separate for OSS and OSS4SG) ===")
    results: dict[str, dict] = {}

    for ptype in ['OSS', 'OSS4SG']:
        print(f"\n--- {ptype} Model ---")
        type_df = df[df['project_type'] == ptype].copy()
        print(f"Sample size: {len(type_df):,}")

        covariates = [
            'log_total_commits_scaled',
            'log_total_lines_scaled',
            'activity_rate_scaled',
        ]
        covariates = [c for c in covariates if c in type_df.columns]
        cox_data = type_df[['duration', 'event'] + covariates].dropna()

        if len(cox_data) > 100 and covariates:
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col='duration', event_col='event')
            print(cph.summary)
            results[ptype] = {
                'model': cph,
                'concordance': float(cph.concordance_index_),
                'n': int(len(cox_data)),
                'events': int(cox_data['event'].sum()),
            }

    return results


def survival_predictions(cph: CoxPHFitter, df: pd.DataFrame, covariates: list[str]):
    """Make predictions for example contributors and plot survival functions."""
    print("\n=== Survival Predictions for Example Contributors ===")

    examples = pd.DataFrame({
        'is_oss4sg': [0, 1, 0, 1],
        'log_total_commits_scaled': [0, 0, 1, 1],
        'log_total_lines_scaled': [0, 0, 0, 0],
        'activity_rate_scaled': [0, 0, 1, 1],
        'oss4sg_x_commits': [0, 0, 0, 1] if 'oss4sg_x_commits' in covariates else [0, 0, 0, 0],
    })

    survival_funcs = cph.predict_survival_function(examples[covariates])

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [
        'OSS - Average Activity',
        'OSS4SG - Average Activity',
        'OSS - High Activity',
        'OSS4SG - High Activity',
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (col, label, color) in enumerate(zip(survival_funcs.columns, labels, colors)):
        ax.plot(survival_funcs.index, survival_funcs[col], label=label, color=color, linewidth=2)

    ax.set_xlabel('Time (weeks)', fontsize=12)
    ax.set_ylabel('Probability of Remaining Non-Core', fontsize=12)
    ax.set_title('Predicted Survival Curves for Different Contributor Profiles', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 150)

    plt.tight_layout()
    save_path = PLOTS_DIR / "predicted_survival_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved predictions plot to: {save_path}")
    plt.close(fig)

    print("\nPredicted Median Time to Core:")
    for i, label in enumerate(labels):
        series = survival_funcs.iloc[:, i]
        median_time = series[series <= 0.5].index.min() if (series <= 0.5).any() else float('inf')
        if median_time != float('inf'):
            print(f"  {label}: {median_time:.1f} weeks")
        else:
            print(f"  {label}: >150 weeks")


def save_cox_results(simple_model: CoxPHFitter, full_model_tuple, stratified_results: dict) -> None:
    """Save all Cox regression results."""
    full_model, covariates = full_model_tuple
    results = {
        'simple_model': {
            'hazard_ratio_oss4sg': float(np.exp(simple_model.params_['is_oss4sg'])),
            'ci_lower': float(np.exp(simple_model.confidence_intervals_.loc['is_oss4sg', '95% lower-bound'])),
            'ci_upper': float(np.exp(simple_model.confidence_intervals_.loc['is_oss4sg', '95% upper-bound'])),
            'p_value': float(simple_model.summary.loc['is_oss4sg', 'p']),
            'concordance': float(simple_model.concordance_index_),
        },
        'full_model': {
            'concordance': float(full_model.concordance_index_),
            'aic_partial': float(getattr(full_model, 'AIC_partial_', float('nan'))),
            'log_likelihood': float(full_model.log_likelihood_),
            'covariates': {},
        },
    }

    for covar in covariates:
        results['full_model']['covariates'][covar] = {
            'coefficient': float(full_model.params_[covar]),
            'hazard_ratio': float(np.exp(full_model.params_[covar])),
            'p_value': float(full_model.summary.loc[covar, 'p']),
        }

    results['stratified_models'] = {}
    for ptype, res in stratified_results.items():
        results['stratified_models'][ptype] = {
            'concordance': float(res['concordance']),
            'n': int(res['n']),
            'events': int(res['events']),
        }

    output_path = RESULTS_DIR / "cox_regression_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Cox results to: {output_path}")

    report_path = RESULTS_DIR / "cox_regression_report.txt"
    with open(report_path, 'w') as f:
        f.write("COX PROPORTIONAL HAZARDS REGRESSION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write("KEY FINDING:\n")
        f.write(
            f"OSS4SG contributors have {results['simple_model']['hazard_ratio_oss4sg']:.2f}x higher hazard of becoming core\n"
        )
        f.write(
            f"(95% CI: {results['simple_model']['ci_lower']:.2f}-{results['simple_model']['ci_upper']:.2f}, p={results['simple_model']['p_value']:.3g})\n\n"
        )
        f.write("FULL MODEL RESULTS:\n")
        f.write(f"Concordance Index: {results['full_model']['concordance']:.3f}\n\n")
        f.write("Significant Predictors:\n")
        for covar, stats in results['full_model']['covariates'].items():
            if stats['p_value'] < 0.05:
                f.write(
                    f"- {covar}: HR={stats['hazard_ratio']:.3f} (p={stats['p_value']:.4f})\n"
                )
    print(f"Saved report to: {report_path}")


def test_model_validation(cph: CoxPHFitter, df: pd.DataFrame, covariates: list[str]) -> None:
    """Validate the Cox model using 5-fold cross-validation."""
    print("\n=== Model Validation (5-Fold Cross-Validation) ===")
    from sklearn.model_selection import KFold

    cox_data = df[['duration', 'event'] + covariates].dropna()
    if len(cox_data) < 100:
        print("Insufficient data for cross-validation; skipping")
        return

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    concordances = []
    for train_idx, test_idx in kf.split(cox_data):
        train = cox_data.iloc[train_idx]
        test = cox_data.iloc[test_idx]
        cph_cv = CoxPHFitter()
        cph_cv.fit(train, duration_col='duration', event_col='event')
        c_index = cph_cv.score(test, scoring_method='concordance_index')
        concordances.append(c_index)

    mean_c = float(np.mean(concordances))
    std_c = float(np.std(concordances))
    print(f"Cross-validated Concordance: {mean_c:.3f} (±{std_c:.3f})")
    print(f"Original Concordance: {cph.concordance_index_:.3f}")


def main():
    print("=" * 60)
    print("STEP 3: COX PROPORTIONAL HAZARDS REGRESSION")
    print("=" * 60)

    try:
        df = load_and_prepare_data()
        simple_model = simple_cox_model(df)
        full_model, covariates = full_cox_model(df)
        plot_path = PLOTS_DIR / "hazard_ratios_forest.png"
        plot_hazard_ratios(full_model, save_path=plot_path)
        stratified = stratified_cox_models(df)
        survival_predictions(full_model, df, covariates)
        test_model_validation(full_model, df, covariates)
        save_cox_results(simple_model, (full_model, covariates), stratified)
        print("\n" + "=" * 60)
        print("COX REGRESSION ANALYSIS COMPLETE")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


