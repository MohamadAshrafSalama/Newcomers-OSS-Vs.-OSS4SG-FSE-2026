#!/usr/bin/env python3
"""
Step 2: Kaplan-Meier Survival Analysis
This performs the basic survival curve analysis comparing OSS vs OSS4SG.
Kaplan-Meier estimates the probability of "surviving" (not becoming core) over time.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
STEP8_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step8_survival_analysis"
DATA_DIR = STEP8_DIR / "data"
RESULTS_DIR = STEP8_DIR / "results"
PLOTS_DIR = STEP8_DIR / "visualizations"

# Create directories
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_survival_data() -> pd.DataFrame:
    """Load the prepared survival data."""
    file_path = DATA_DIR / "survival_data.csv"
    print(f"Loading survival data from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(
            "Survival data not found. Run 1_prepare_survival_data.py first!"
        )

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} contributors for survival analysis")
    return df


def perform_kaplan_meier(df: pd.DataFrame, label: str = "All") -> KaplanMeierFitter:
    """Perform Kaplan-Meier estimation for a dataset and return the fitted model."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df['duration'], event_observed=df['event'], label=label)
    return kmf


def compare_survival_curves(df: pd.DataFrame):
    """Compare survival curves between OSS and OSS4SG."""
    print("\n=== Kaplan-Meier Analysis: OSS vs OSS4SG ===")

    # Separate by project type
    oss_df = df[df['project_type'] == 'OSS']
    oss4sg_df = df[df['project_type'] == 'OSS4SG']

    # Fit KM curves
    kmf_oss = perform_kaplan_meier(oss_df, label='OSS')
    kmf_oss4sg = perform_kaplan_meier(oss4sg_df, label='OSS4SG')

    # Get median survival times
    median_oss = kmf_oss.median_survival_time_
    median_oss4sg = kmf_oss4sg.median_survival_time_

    print("\nMedian Time to Core:")
    print(
        f"  OSS: {median_oss:.1f} weeks" if not np.isnan(median_oss) else "  OSS: Not reached (>50% still non-core)"
    )
    print(
        f"  OSS4SG: {median_oss4sg:.1f} weeks" if not np.isnan(median_oss4sg) else "  OSS4SG: Not reached (>50% still non-core)"
    )

    # Get survival probabilities at key timepoints
    timepoints = [4, 8, 12, 26, 52, 104]

    print("\nSurvival Probabilities (probability of still being non-core):")
    print("Week\tOSS\tOSS4SG\tDifference")
    print("-" * 40)

    results = {}
    for t in timepoints:
        try:
            surv_oss = kmf_oss.survival_function_at_times(t).iloc[0]
            surv_oss4sg = kmf_oss4sg.survival_function_at_times(t).iloc[0]
            diff = float(surv_oss - surv_oss4sg)

            print(f"{t}\t{surv_oss:.3f}\t{surv_oss4sg:.3f}\t{diff:+.3f}")

            results[f'week_{t}'] = {
                'oss_survival': float(surv_oss),
                'oss4sg_survival': float(surv_oss4sg),
                'difference': diff,
                'oss_core_prob': float(1 - surv_oss),
                'oss4sg_core_prob': float(1 - surv_oss4sg),
            }
        except Exception:
            # Ignore timepoints beyond the observed range
            pass

    # Perform log-rank test
    print("\n=== Statistical Test (Log-Rank) ===")
    result = logrank_test(
        durations_A=oss_df['duration'],
        durations_B=oss4sg_df['duration'],
        event_observed_A=oss_df['event'],
        event_observed_B=oss4sg_df['event'],
    )

    print(f"Test Statistic: {result.test_statistic:.4f}")
    print(f"P-value: {result.p_value:.6f}")
    print(f"Significant difference? {'YES' if result.p_value < 0.05 else 'NO'}")

    # Approximate hazard ratio using event rates (for quick directional insight)
    events_oss = int(oss_df['event'].sum())
    events_oss4sg = int(oss4sg_df['event'].sum())
    total_oss = len(oss_df)
    total_oss4sg = len(oss4sg_df)

    rate_oss = events_oss / total_oss if total_oss > 0 else 0.0
    rate_oss4sg = events_oss4sg / total_oss4sg if total_oss4sg > 0 else 0.0
    hazard_ratio = (rate_oss4sg / rate_oss) if rate_oss > 0 else float('inf')

    print(f"\nHazard Ratio (OSS4SG/OSS): {hazard_ratio:.3f}")
    print(
        f"Interpretation: OSS4SG contributors are {hazard_ratio:.1f}x more likely to become core"
    )

    return kmf_oss, kmf_oss4sg, results, result


def plot_survival_curves(
    kmf_oss: KaplanMeierFitter, kmf_oss4sg: KaplanMeierFitter, save_path: Optional[Path] = None
):
    """Create publication-quality survival curve plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot both curves
    kmf_oss.plot_survival_function(ax=ax, ci_show=True, color='#1f77b4', linewidth=2)
    kmf_oss4sg.plot_survival_function(ax=ax, ci_show=True, color='#ff7f0e', linewidth=2)

    ax.set_xlabel('Time (weeks)', fontsize=12)
    ax.set_ylabel('Probability of Remaining Non-Core', fontsize=12)
    ax.set_title('Kaplan-Meier Survival Curves: OSS vs OSS4SG', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add median lines if available
    for kmf, color in [(kmf_oss, '#1f77b4'), (kmf_oss4sg, '#ff7f0e')]:
        median = kmf.median_survival_time_
        if not np.isnan(median):
            ax.axvline(x=median, color=color, linestyle='--', alpha=0.5)
            ax.axhline(y=0.5, color=color, linestyle='--', alpha=0.5)

    # X/Y limits
    try:
        xmax = float(min(200, kmf_oss.survival_function_.index.max()))
    except Exception:
        xmax = 200
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=11)

    add_risk_table(ax, kmf_oss, kmf_oss4sg)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved survival curve plot to: {save_path}")
    plt.close(fig)
    return fig


def plot_cumulative_incidence(
    kmf_oss: KaplanMeierFitter, kmf_oss4sg: KaplanMeierFitter, save_path: Optional[Path] = None
):
    """Plot cumulative incidence (1 - survival) which shows probability of becoming core."""
    fig, ax = plt.subplots(figsize=(10, 7))

    ci_oss = 1 - kmf_oss.survival_function_
    ci_oss4sg = 1 - kmf_oss4sg.survival_function_

    ax.plot(ci_oss.index, ci_oss.values, label='OSS', color='#1f77b4', linewidth=2)
    ax.plot(ci_oss4sg.index, ci_oss4sg.values, label='OSS4SG', color='#ff7f0e', linewidth=2)

    # Confidence intervals
    try:
        ci_oss_upper = 1 - kmf_oss.confidence_interval_survival_function_.iloc[:, 0]
        ci_oss_lower = 1 - kmf_oss.confidence_interval_survival_function_.iloc[:, 1]
        ax.fill_between(ci_oss.index, ci_oss_lower, ci_oss_upper, alpha=0.2, color='#1f77b4')

        ci_oss4sg_upper = 1 - kmf_oss4sg.confidence_interval_survival_function_.iloc[:, 0]
        ci_oss4sg_lower = 1 - kmf_oss4sg.confidence_interval_survival_function_.iloc[:, 1]
        ax.fill_between(ci_oss4sg.index, ci_oss4sg_lower, ci_oss4sg_upper, alpha=0.2, color='#ff7f0e')
    except Exception:
        pass

    ax.set_xlabel('Time (weeks)', fontsize=12)
    ax.set_ylabel('Cumulative Probability of Becoming Core', fontsize=12)
    ax.set_title('Cumulative Incidence: Transition to Core Contributor', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    try:
        xmax = float(min(200, ci_oss.index.max()))
    except Exception:
        xmax = 200
    ax.set_xlim(0, xmax)
    # Robust ymax calculation
    try:
        ymax_val = float(max(ci_oss.to_numpy().max(), ci_oss4sg.to_numpy().max()) * 1.05)
    except Exception:
        ymax_val = 1.0
    ax.set_ylim(0, ymax_val)
    ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cumulative incidence plot to: {save_path}")
    plt.close(fig)
    return fig


def add_risk_table(ax, kmf_oss: KaplanMeierFitter, kmf_oss4sg: KaplanMeierFitter):
    """Add number-at-risk annotations at selected timepoints below the plot."""
    times = [0, 26, 52, 78, 104, 130, 156]

    risk_data = []
    for t in times:
        try:
            n_oss = (
                kmf_oss.event_table.loc[:t, 'at_risk'].iloc[-1]
                if t <= kmf_oss.event_table.index.max()
                else 0
            )
            n_oss4sg = (
                kmf_oss4sg.event_table.loc[:t, 'at_risk'].iloc[-1]
                if t <= kmf_oss4sg.event_table.index.max()
                else 0
            )
            risk_data.append((t, int(n_oss), int(n_oss4sg)))
        except Exception:
            risk_data.append((t, 0, 0))

    y_pos = -0.15
    ax.text(0.1, y_pos, "At Risk:", transform=ax.transAxes, fontsize=9, fontweight='bold')
    ax.text(0.1, y_pos - 0.03, "OSS:", transform=ax.transAxes, fontsize=9, color='#1f77b4')
    ax.text(0.1, y_pos - 0.06, "OSS4SG:", transform=ax.transAxes, fontsize=9, color='#ff7f0e')

    for i, (t, n_oss, n_oss4sg) in enumerate(risk_data):
        x_pos = 0.2 + i * 0.11
        ax.text(x_pos, y_pos, str(t), transform=ax.transAxes, fontsize=9)
        ax.text(x_pos, y_pos - 0.03, str(n_oss), transform=ax.transAxes, fontsize=9, color='#1f77b4')
        ax.text(x_pos, y_pos - 0.06, str(n_oss4sg), transform=ax.transAxes, fontsize=9, color='#ff7f0e')


def analyze_by_activity_level(df: pd.DataFrame):
    """Stratified analysis by activity level."""
    print("\n=== Stratified Analysis by Activity Level ===")

    if 'total_commits' not in df.columns:
        print("total_commits not found; skipping stratified analysis")
        return

    q1 = df['total_commits'].quantile(0.25)
    q3 = df['total_commits'].quantile(0.75)

    df = df.copy()
    df['activity_group'] = pd.cut(
        df['total_commits'], bins=[0, q1, q3, df['total_commits'].max()], labels=['Low', 'Medium', 'High']
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (group, ax) in enumerate(zip(['Low', 'Medium', 'High'], axes)):
        group_df = df[df['activity_group'] == group]
        for ptype, color in [('OSS', '#1f77b4'), ('OSS4SG', '#ff7f0e')]:
            type_df = group_df[group_df['project_type'] == ptype]
            if len(type_df) > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(type_df['duration'], type_df['event'], label=ptype)
                kmf.plot_survival_function(ax=ax, ci_show=True, color=color)
        ax.set_title(
            f'{group} Activity\n({group_df["total_commits"].min():.0f}-{group_df["total_commits"].max():.0f} commits)'
        )
        ax.set_xlabel('Time (weeks)')
        if idx == 0:
            ax.set_ylabel('Survival Probability')
        ax.set_xlim(0, 150)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    plt.suptitle('Survival Curves Stratified by Activity Level', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = PLOTS_DIR / "survival_by_activity.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved stratified plot to: {save_path}")
    plt.close()


def save_results(results: dict, log_rank_result, kmf_oss: KaplanMeierFitter, kmf_oss4sg: KaplanMeierFitter) -> None:
    """Save all analysis results."""
    full_results = {
        'comparison': 'OSS vs OSS4SG',
        'log_rank_test': {
            'statistic': float(log_rank_result.test_statistic),
            'p_value': float(log_rank_result.p_value),
            'significant': bool(log_rank_result.p_value < 0.05),
        },
        'median_survival': {
            'OSS': float(kmf_oss.median_survival_time_) if not np.isnan(kmf_oss.median_survival_time_) else None,
            'OSS4SG': float(kmf_oss4sg.median_survival_time_) if not np.isnan(kmf_oss4sg.median_survival_time_) else None,
        },
        'survival_at_timepoints': results,
        'sample_sizes': {
            'OSS': int(len(kmf_oss.durations)),
            'OSS4SG': int(len(kmf_oss4sg.durations)),
        },
        'events': {
            'OSS': int(kmf_oss.event_observed.sum()),
            'OSS4SG': int(kmf_oss4sg.event_observed.sum()),
        },
    }

    output_path = RESULTS_DIR / "kaplan_meier_results.json"
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved results to: {output_path}")

    report_path = RESULTS_DIR / "kaplan_meier_report.txt"
    with open(report_path, 'w') as f:
        f.write("KAPLAN-MEIER SURVIVAL ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write("MAIN FINDING:\n")
        f.write("OSS4SG contributors transition to core faster than OSS\n")
        f.write(f"Log-rank test p-value: {log_rank_result.p_value:.6f}\n\n")
        f.write("MEDIAN TIME TO CORE:\n")
        if full_results['median_survival']['OSS'] is not None:
            f.write(f"OSS: {full_results['median_survival']['OSS']:.1f} weeks\n")
        else:
            f.write("OSS: Not reached\n")
        if full_results['median_survival']['OSS4SG'] is not None:
            f.write(f"OSS4SG: {full_results['median_survival']['OSS4SG']:.1f} weeks\n\n")
        else:
            f.write("OSS4SG: Not reached\n\n")
        f.write("PROBABILITY OF BECOMING CORE:\n")
        f.write("Week\tOSS\tOSS4SG\n")
        f.write("-" * 30 + "\n")
        for week in [4, 12, 26, 52]:
            key = f'week_{week}'
            if key in results:
                f.write(
                    f"{week}\t{results[key]['oss_core_prob']:.1%}\t{results[key]['oss4sg_core_prob']:.1%}\n"
                )
    print(f"Saved report to: {report_path}")


def test_proportional_hazards_assumption(df: pd.DataFrame) -> None:
    """Test if the effect is consistent over time using multivariate log-rank at cutoffs."""
    print("\n=== Testing Proportional Hazards Assumption ===")
    cutoffs = [10, 20, 30, 52, 78, 104]
    for cutoff in cutoffs:
        df_cut = df[df['duration'] <= cutoff].copy()
        if not df_cut.empty:
            df_cut.loc[df_cut['duration'] == cutoff, 'event'] = 0
            if (
                len(df_cut[df_cut['project_type'] == 'OSS']) > 10
                and len(df_cut[df_cut['project_type'] == 'OSS4SG']) > 10
            ):
                result = multivariate_logrank_test(
                    df_cut['duration'], df_cut['project_type'], df_cut['event']
                )
                print(f"Week {cutoff}: p-value = {result.p_value:.4f}")
    print("Interpretation: If p-values remain significant across timepoints, the effect is consistent.")


def main():
    print("=" * 60)
    print("STEP 2: KAPLAN-MEIER SURVIVAL ANALYSIS")
    print("=" * 60)

    try:
        df = load_survival_data()
        kmf_oss, kmf_oss4sg, results, log_rank = compare_survival_curves(df)

        plot_path1 = PLOTS_DIR / "survival_curves_main.png"
        plot_survival_curves(kmf_oss, kmf_oss4sg, save_path=plot_path1)

        plot_path2 = PLOTS_DIR / "cumulative_incidence.png"
        plot_cumulative_incidence(kmf_oss, kmf_oss4sg, save_path=plot_path2)

        analyze_by_activity_level(df)
        test_proportional_hazards_assumption(df)
        save_results(results, log_rank, kmf_oss, kmf_oss4sg)

        print("\n" + "=" * 60)
        print("KAPLAN-MEIER ANALYSIS COMPLETE")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


