#!/usr/bin/env python3
"""
Analyze Early Project Contributors to Understand the Problem
============================================================

Identifies contributors who joined early in a project's life and became core quickly.
These are likely founding members, not true newcomers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def analyze_early_contributors():
    """Analyze early project contributors who became core quickly."""
    print("=" * 70)
    print("ANALYZING EARLY PROJECT CONTRIBUTORS")
    print("=" * 70)

    # Use results path
    base = Path(".") / "results"
    complete_file = base / "contributor_transitions_including_all.csv"
    filtered_file = base / "contributor_transitions.csv"

    if not complete_file.exists():
        print(f"Error: {complete_file} not found")
        return

    print(f"\nLoading complete dataset...")
    df_all = pd.read_csv(complete_file, low_memory=False)
    df_filtered = pd.read_csv(filtered_file, low_memory=False) if filtered_file.exists() else pd.DataFrame()

    print(f"Total contributors (all): {len(df_all):,}")
    if not df_filtered.empty:
        print(f"Total contributors (filtered): {len(df_filtered):,}")
        print(f"Instant core excluded: {len(df_all) - len(df_filtered):,}")

    print("\n" + "-" * 40)
    print("WHEN DID CONTRIBUTORS JOIN THE PROJECT?")
    print("-" * 40)

    core_all = df_all[df_all['became_core'] == True]
    core_filtered = df_filtered[df_filtered['became_core'] == True] if not df_filtered.empty else pd.DataFrame()

    print(f"\nCore contributors (all): {len(core_all):,}")
    if not core_filtered.empty:
        print(f"Core contributors (filtered): {len(core_filtered):,}")

    thresholds = [1, 4, 8, 12, 26, 52]
    print("\n" + "-" * 40)
    print("EARLY PROJECT JOINERS WHO BECAME CORE")
    print("-" * 40)
    for threshold in thresholds:
        early_joiners = core_all[core_all['first_commit_week'] <= threshold]
        pct = len(early_joiners) / len(core_all) * 100 if len(core_all) > 0 else 0
        if len(early_joiners) > 0:
            median_time = early_joiners['weeks_to_core'].median()
            instant = (early_joiners['weeks_to_core'] == 0).sum()
            week1 = (early_joiners['weeks_to_core'] <= 1).sum()
            print(f"\nJoined in first {threshold} weeks of project:")
            print(f"  Count: {len(early_joiners):,} ({pct:.1f}% of all core)")
            print(f"  Instant core (week 0): {instant:,}")
            print(f"  Core by week 1: {week1:,}")
            print(f"  Median time to core: {median_time:.1f} weeks")

    print("\n" + "-" * 40)
    print("DETAILED ANALYSIS: FIRST 8 WEEKS JOINERS")
    print("-" * 40)
    early_threshold = 8
    early_joiners = df_all[df_all['first_commit_week'] <= early_threshold]
    print(f"\nTotal who joined in first {early_threshold} weeks: {len(early_joiners):,}")
    early_core = early_joiners[early_joiners['became_core'] == True]
    early_core_rate = len(early_core) / len(early_joiners) * 100 if len(early_joiners) > 0 else 0
    print(f"Became core: {len(early_core):,} ({early_core_rate:.1f}%)")

    late_joiners = df_all[df_all['first_commit_week'] > early_threshold]
    late_core = late_joiners[late_joiners['became_core'] == True]
    late_core_rate = len(late_core) / len(late_joiners) * 100 if len(late_joiners) > 0 else 0
    print(f"\nLater joiners (week >{early_threshold}):")
    print(f"  Total: {len(late_joiners):,}")
    print(f"  Became core: {len(late_core):,} ({late_core_rate:.1f}%)")
    print(f"\nCore achievement rate difference:")
    print(f"  Early joiners: {early_core_rate:.1f}%")
    print(f"  Late joiners: {late_core_rate:.1f}%")
    if late_core_rate > 0:
        print(f"  Ratio: {early_core_rate/late_core_rate:.2f}x higher for early joiners")

    if len(early_core) > 0 and len(late_core) > 0:
        print(f"\nTime to core comparison:")
        print(f"  Early joiners median: {early_core['weeks_to_core'].median():.1f} weeks")
        print(f"  Late joiners median: {late_core['weeks_to_core'].median():.1f} weeks")
        print(f"\nEffort to core comparison:")
        print(f"  Early joiners median: {early_core['commits_to_core'].median():.0f} commits")
        print(f"  Late joiners median: {late_core['commits_to_core'].median():.0f} commits")

    print("\n" + "-" * 40)
    print("BY PROJECT TYPE")
    print("-" * 40)
    for ptype in df_all['project_type'].unique():
        ptype_data = df_all[df_all['project_type'] == ptype]
        ptype_early = ptype_data[ptype_data['first_commit_week'] <= early_threshold]
        ptype_early_core = ptype_early[ptype_early['became_core'] == True]
        if len(ptype_early) > 0:
            print(f"\n{ptype}:")
            print(f"  Early joiners: {len(ptype_early):,}")
            print(f"  Became core: {len(ptype_early_core):,} ({len(ptype_early_core)/len(ptype_early)*100:.1f}%)")
            if len(ptype_early_core) > 0:
                print(f"  Median time to core: {ptype_early_core['weeks_to_core'].median():.1f} weeks")
                print(f"  Instant core: {(ptype_early_core['weeks_to_core'] == 0).sum():,}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    strategies = [
        ("Current (instant only)", df_all['weeks_to_core'] == 0),
        ("Instant + week 1 core", df_all['weeks_to_core'] <= 1),
        ("Early joiners (≤8 weeks) who became core", (df_all['first_commit_week'] <= 8) & (df_all['became_core'] == True)),
        ("Early joiners (≤8) + fast core (≤4)", (df_all['first_commit_week'] <= 8) & (df_all['weeks_to_core'] <= 4)),
    ]
    print("\nExclusion Strategy Comparison:")
    for name, mask in strategies:
        excluded = mask.sum() if isinstance(mask, pd.Series) else 0
        remaining_df = df_all[~mask] if isinstance(mask, pd.Series) else df_all
        remaining_core = remaining_df[remaining_df['became_core'] == True]
        print(f"\n'{name}':")
        print(f"  Excluded: {excluded:,}")
        print(f"  Remaining: {len(remaining_df):,}")
        if len(remaining_core) > 0:
            new_core_rate = len(remaining_core) / len(remaining_df) * 100
            new_median_time = remaining_core['weeks_to_core'].median()
            new_median_commits = remaining_core['commits_to_core'].median()
            print(f"  New core rate: {new_core_rate:.1f}%")
            print(f"  New median time: {new_median_time:.1f} weeks")
            print(f"  New median commits: {new_median_commits:.0f}")

    print("\n🎯 RECOMMENDED APPROACH:")
    print("Exclude contributors who:")
    print("1. Are instant core (week 0) - already doing this")
    print("2. Join in first 8 weeks of project AND become core within 4 weeks")
    print("   These are likely founding team members")

    mask_recommended = (
        (df_all['weeks_to_core'] == 0) |
        ((df_all['first_commit_week'] <= 8) & (df_all['became_core'] == True) & (df_all['weeks_to_core'] <= 4))
    )
    excluded_count = mask_recommended.sum()
    print(f"\nThis would exclude: {excluded_count:,} contributors")
    print(f"Leaving: {len(df_all) - excluded_count:,} for analysis")


if __name__ == "__main__":
    analyze_early_contributors()


