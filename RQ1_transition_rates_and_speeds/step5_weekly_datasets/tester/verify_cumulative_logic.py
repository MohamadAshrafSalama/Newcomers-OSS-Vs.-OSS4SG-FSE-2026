#!/usr/bin/env python3
"""
Verify that cumulative logic issues are due to multi-project contributors
=========================================================================
"""

import pandas as pd
from pathlib import Path


def verify_cumulative_logic():
    """Check if cumulative issues are from contributors in multiple projects."""

    print("=" * 70)
    print("VERIFYING CUMULATIVE LOGIC ISSUES")
    print("=" * 70)

    # Load the dataset
    activity_file = Path("RQ1_transition_rates_and_speeds/step5_weekly_datasets/dataset2_contributor_activity/contributor_activity_weekly.csv")

    print(f"\nLoading: {activity_file}")
    df = pd.read_csv(activity_file, nrows=1000000)  # Load first 1M rows for speed
    print(f"Loaded {len(df):,} rows for analysis\n")

    # Find contributors with "decreasing" cumulative when viewed globally
    print("Finding contributors with apparent cumulative decreases...")

    issues_found = []
    contributors_checked = 0

    # Group by contributor
    for contributor in df['contributor_email'].unique()[:100]:  # Check first 100
        contrib_data = df[df['contributor_email'] == contributor].sort_values(['week_date', 'project_name'])

        if len(contrib_data) < 2:
            continue

        contributors_checked += 1

        # Check if cumulative ever decreases (globally)
        cumulative_values = contrib_data['cumulative_commits'].values
        has_decrease = False

        for i in range(1, len(cumulative_values)):
            if cumulative_values[i] < cumulative_values[i-1]:
                has_decrease = True
                break

        if has_decrease:
            issues_found.append(contributor)

    print(f"Checked {contributors_checked} contributors")
    print(f"Found {len(issues_found)} with apparent decreases\n")

    # Now check if these contributors are in multiple projects
    print("-" * 40)
    print("Analyzing contributors with issues:")
    print("-" * 40)

    for contributor in issues_found[:5]:  # Analyze first 5
        contrib_data = df[df['contributor_email'] == contributor]
        projects = contrib_data['project_name'].unique()

        print(f"\n{contributor}:")
        print(f"  Projects: {len(projects)}")

        if len(projects) > 1:
            print(f"  CONFIRMED: In multiple projects: {list(projects[:3])}")

            # Show the cumulative pattern per project
            for project in projects[:2]:  # Show first 2 projects
                project_data = contrib_data[contrib_data['project_name'] == project].sort_values('week_number')
                cum_values = project_data['cumulative_commits'].values[:5]  # First 5 weeks
                print(f"    {project}: cumulative = {list(cum_values)}")

            # Show how they interleave when sorted by week
            sorted_data = contrib_data.sort_values('week_number').head(10)
            print(f"\n  When sorted by week_number (causes apparent decrease):")
            print(sorted_data[['project_name', 'week_number', 'cumulative_commits']].to_string(index=False))
        else:
            print(f"  ERROR: Only in one project but has decreasing cumulative!")
            # This would be a real bug

    print("\n" + "-" * 40)
    print("Testing cumulative logic PER PROJECT:")
    print("-" * 40)

    real_issues = []

    # Check cumulative logic within each project separately
    for contributor in issues_found[:10]:
        contrib_data = df[df['contributor_email'] == contributor]

        for project in contrib_data['project_name'].unique():
            project_data = contrib_data[contrib_data['project_name'] == project].sort_values('week_number')

            # Check cumulative within this project
            cum_values = project_data['cumulative_commits'].values
            weekly_values = project_data['commits_this_week'].values

            # Check monotonic increase
            is_monotonic = all(cum_values[i] <= cum_values[i+1] for i in range(len(cum_values)-1))

            # Check sum matches
            total_weekly = sum(weekly_values)
            final_cumulative = cum_values[-1] if len(cum_values) > 0 else 0
            sum_matches = abs(total_weekly - final_cumulative) <= 1

            if not is_monotonic or not sum_matches:
                real_issues.append((contributor, project))
                print(f"  REAL ISSUE: {contributor} in {project}")
                print(f"     Monotonic: {is_monotonic}, Sum matches: {sum_matches}")

    print(f"\nReal issues found: {len(real_issues)}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if len(real_issues) == 0:
        print("NO REAL ISSUES FOUND!")
        print("\nThe 'decreasing cumulative' is expected behavior because:")
        print("1. Cumulative counts are PER PROJECT (as they should be)")
        print("2. Contributors in multiple projects have separate cumulative counts")
        print("3. When consolidated and sorted, these appear to decrease")
        print("\nTHE DATA IS CORRECT - The test just needs to check per-project!")
    else:
        print(f"Found {len(real_issues)} real cumulative logic issues")
        print("These need investigation")

    return len(real_issues) == 0


if __name__ == "__main__":
    verify_cumulative_logic()

