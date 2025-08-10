#!/usr/bin/env python3
"""
Step 1: Prepare Survival Analysis Data
This script loads the contributor transitions data and prepares it for survival analysis.
It validates the data and creates the necessary format for survival analysis libraries.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Define paths
BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
STEP6_RESULTS = BASE_DIR / "RQ1_transition_rates_and_speeds/step6_contributor_transitions/results"
STEP8_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step8_survival_analysis"
RESULTS_DIR = STEP8_DIR / "results"
DATA_DIR = STEP8_DIR / "data"

# Create directories
for dir_path in [STEP8_DIR, RESULTS_DIR, DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def load_transitions_data() -> pd.DataFrame:
    """Load the main contributor transitions dataset (v2)."""
    file_path = STEP6_RESULTS / "contributor_transitions.csv"
    print(f"Loading transitions data from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Transitions file not found at {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} contributor transitions")
    return df


def prepare_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for survival analysis.

    Key columns needed:
    - duration: time to event (or censoring time)
    - event: 1 if became core, 0 if censored
    - project_type: OSS or OSS4SG (for comparison)
    """

    print("\n=== Preparing Survival Data ===")

    survival_df = df.copy()

    # Duration: time_to_event_weeks is either weeks_to_core (if became core) or total_weeks_observed (if censored)
    survival_df['duration'] = survival_df['time_to_event_weeks']

    # Event: 1 if became core, 0 if censored
    survival_df['event'] = survival_df['became_core'].astype(int)

    # Keep relevant columns for analysis
    columns_to_keep = [
        'project_name', 'project_type', 'contributor_email',
        'duration', 'event',
        'first_commit_week', 'total_commits', 'total_lines_changed',
        'total_active_weeks', 'activity_rate',
        'weeks_to_core', 'commits_to_core', 'lines_changed_to_core',
        'avg_commits_per_active_week_before_core',
        'commit_consistency_before_core',
        'is_early_joiner', 'is_fast_core'
    ]

    columns_to_keep = [col for col in columns_to_keep if col in survival_df.columns]
    survival_df = survival_df[columns_to_keep]

    # Remove any rows with invalid duration (should be positive)
    initial_count = len(survival_df)
    survival_df = survival_df[survival_df['duration'] > 0]
    removed = initial_count - len(survival_df)
    if removed > 0:
        print(f"Removed {removed} rows with invalid duration")

    return survival_df


def validate_survival_data(df: pd.DataFrame) -> bool:
    """Validate the prepared survival data."""

    print("\n=== Data Validation ===")

    # Check for missing values in critical columns
    critical_cols = ['duration', 'event', 'project_type']
    for col in critical_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"WARNING: {missing} missing values in {col}")
        else:
            print(f"OK: No missing values in {col}")

    # Check event coding
    unique_events = df['event'].unique()
    assert set(unique_events).issubset({0, 1}), "Event must be 0 or 1"
    print(f"OK: Event values: {sorted(unique_events)}")

    # Check duration is positive
    assert (df['duration'] > 0).all(), "All durations must be positive"
    print(
        f"OK: All durations are positive (min: {df['duration'].min()}, max: {df['duration'].max()})"
    )

    # Check project types
    types = df['project_type'].unique()
    print(f"OK: Project types: {types}")

    # Validate consistency for successful transitions
    core_df = df[df['event'] == 1]
    mismatch_found = False
    for _, row in core_df.iterrows():
        if pd.notna(row.get('weeks_to_core')):
            if abs(row['duration'] - row['weeks_to_core']) > 0.01:
                print("WARNING: Duration mismatch for a core contributor row")
                mismatch_found = True
                break
    if not mismatch_found:
        print("OK: Duration values consistent for core contributors")

    return True


def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the survival data."""

    print("\n=== Summary Statistics ===")

    stats = {
        'total_contributors': len(df),
        'became_core': int(df['event'].sum()),
        'censored': int(len(df) - df['event'].sum()),
        'core_rate': float(df['event'].mean()),
        'median_observation_time': float(df['duration'].median()),
        'max_observation_time': float(df['duration'].max()),
    }

    # By project type
    for ptype in df['project_type'].unique():
        type_df = df[df['project_type'] == ptype]
        stats[f'{ptype}_contributors'] = int(len(type_df))
        stats[f'{ptype}_became_core'] = int(type_df['event'].sum())
        stats[f'{ptype}_core_rate'] = float(type_df['event'].mean())
        stats[f'{ptype}_median_time'] = float(type_df['duration'].median())

        core_type_df = type_df[type_df['event'] == 1]
        if len(core_type_df) > 0:
            stats[f'{ptype}_median_time_to_core'] = float(core_type_df['duration'].median())
            stats[f'{ptype}_mean_time_to_core'] = float(core_type_df['duration'].mean())

    # Print statistics
    print(f"\nTotal Contributors: {stats['total_contributors']:,}")
    print(f"Became Core: {stats['became_core']:,} ({stats['core_rate']:.1%})")
    print(f"Censored: {stats['censored']:,} ({(1 - stats['core_rate']):.1%})")

    print(f"\nMedian Observation Time: {stats['median_observation_time']:.1f} weeks")
    print(f"Max Observation Time: {stats['max_observation_time']:.1f} weeks")

    print("\n--- By Project Type ---")
    for ptype in df['project_type'].unique():
        print(f"\n{ptype}:")
        print(f"  Contributors: {stats[f'{ptype}_contributors']:,}")
        print(
            f"  Became Core: {stats[f'{ptype}_became_core']:,} ({stats[f'{ptype}_core_rate']:.1%})"
        )
        print(f"  Median Observation: {stats[f'{ptype}_median_time']:.1f} weeks")
        if f'{ptype}_median_time_to_core' in stats:
            print(
                f"  Median Time to Core (for successes): {stats[f'{ptype}_median_time_to_core']:.1f} weeks"
            )
            print(
                f"  Mean Time to Core (for successes): {stats[f'{ptype}_mean_time_to_core']:.1f} weeks"
            )

    return stats


def create_analysis_subsets(df: pd.DataFrame) -> dict:
    """Create different subsets for various analyses."""

    print("\n=== Creating Analysis Subsets ===")

    subsets: dict[str, pd.DataFrame] = {}

    # Full dataset
    subsets['all'] = df.copy()
    print(f"Created 'all' subset: {len(subsets['all']):,} contributors")

    # By project type
    for ptype in df['project_type'].unique():
        subsets[ptype.lower()] = df[df['project_type'] == ptype].copy()
        print(
            f"Created '{ptype.lower()}' subset: {len(subsets[ptype.lower()]):,} contributors"
        )

    # High activity contributors (top 25% by total commits)
    if 'total_commits' in df.columns:
        threshold_hi = df['total_commits'].quantile(0.75)
        subsets['high_activity'] = df[df['total_commits'] >= threshold_hi].copy()
        print(
            f"Created 'high_activity' subset: {len(subsets['high_activity']):,} contributors (>= {threshold_hi:.0f} commits)"
        )

        # Low activity contributors (bottom 25%)
        threshold_lo = df['total_commits'].quantile(0.25)
        subsets['low_activity'] = df[df['total_commits'] <= threshold_lo].copy()
        print(
            f"Created 'low_activity' subset: {len(subsets['low_activity']):,} contributors (<= {threshold_lo:.0f} commits)"
        )

    # Consistent contributors (high consistency before core)
    if 'commit_consistency_before_core' in df.columns:
        threshold = df['commit_consistency_before_core'].median()
        subsets['consistent'] = df[
            df['commit_consistency_before_core'] >= threshold
        ].copy()
        print(f"Created 'consistent' subset: {len(subsets['consistent']):,} contributors")

    return subsets


def save_prepared_data(survival_df: pd.DataFrame, subsets: dict, stats: dict) -> None:
    """Save all prepared data and statistics."""

    print("\n=== Saving Data ===")

    # Save main survival dataset
    output_path = DATA_DIR / "survival_data.csv"
    survival_df.to_csv(output_path, index=False)
    print(f"Saved main survival data to: {output_path}")

    # Save subsets
    for name, subset_df in subsets.items():
        subset_path = DATA_DIR / f"survival_data_{name}.csv"
        subset_df.to_csv(subset_path, index=False)
        print(f"Saved {name} subset to: {subset_path}")

    # Save statistics
    stats_path = RESULTS_DIR / "data_preparation_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_path}")

    # Create a summary report
    report_path = RESULTS_DIR / "data_preparation_report.txt"
    with open(report_path, 'w') as f:
        f.write("SURVIVAL ANALYSIS DATA PREPARATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Contributors: {stats['total_contributors']:,}\n")
        f.write(f"Became Core: {stats['became_core']:,} ({stats['core_rate']:.1%})\n")
        f.write(f"Censored: {stats['censored']:,}\n\n")
        f.write("SUBSETS CREATED:\n")
        for name in subsets.keys():
            f.write(f"- {name}: {len(subsets[name]):,} contributors\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("Data validated and ready for survival analysis\n")

    print(f"Saved preparation report to: {report_path}")


def main():
    """Main execution."""
    print("=" * 60)
    print("STEP 1: SURVIVAL ANALYSIS DATA PREPARATION")
    print("=" * 60)

    try:
        # Load data
        transitions_df = load_transitions_data()

        # Prepare for survival analysis
        survival_df = prepare_survival_data(transitions_df)

        # Validate
        is_valid = validate_survival_data(survival_df)
        if not is_valid:
            raise ValueError("Data validation failed")

        # Generate statistics
        stats = generate_summary_statistics(survival_df)

        # Create subsets
        subsets = create_analysis_subsets(survival_df)

        # Save everything
        save_prepared_data(survival_df, subsets, stats)

        print("\n" + "=" * 60)
        print("DATA PREPARATION COMPLETE")
        print("=" * 60)

        return survival_df, stats

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main()


