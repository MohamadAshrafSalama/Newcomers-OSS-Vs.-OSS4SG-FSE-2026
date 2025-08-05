#!/usr/bin/env python3
"""
Test Script: Demonstrate Transition Rate Calculation on Sample Projects
======================================================================
Shows exactly how the newcomer-to-core transition analysis works
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def get_core_contributors_at_time(commits_up_to_t):
    """Apply 80% rule to commits up to time T"""
    if len(commits_up_to_t) < 3:
        return []
    
    contributor_commits = commits_up_to_t.groupby('author_name')['commit_hash'].count()
    contributor_commits = contributor_commits.sort_values(ascending=False)
    
    total_commits = contributor_commits.sum()
    cumulative = contributor_commits.cumsum()
    cumulative_pct = cumulative / total_commits
    
    # Find core contributors (80% rule)
    core_mask = cumulative_pct <= 0.80
    if core_mask.sum() == 0:
        return [contributor_commits.index[0]]
    
    core_contributors = contributor_commits[core_mask].index.tolist()
    
    # Add the contributor that pushes us over 80%
    if core_contributors and cumulative_pct[core_contributors[-1]] < 0.80:
        remaining = contributor_commits.index.difference(core_contributors)
        if len(remaining) > 0:
            next_contrib = contributor_commits.index[len(core_contributors)]
            core_contributors.append(next_contrib)
    
    return core_contributors

def demonstrate_transition_calculation(project_name, project_commits, max_months=12):
    """
    Demonstrate the transition calculation step by step for one project
    """
    print(f"\n{'='*60}")
    print(f"DEMONSTRATING: {project_name}")
    print(f"{'='*60}")
    
    # Prepare data
    project_commits = project_commits.copy()
    project_commits['commit_date_clean'] = pd.to_datetime(
        project_commits['commit_date'], errors='coerce', utc=True
    )
    project_commits = project_commits.dropna(subset=['commit_date_clean'])
    project_commits = project_commits.sort_values('commit_date_clean')
    
    print(f"Total commits: {len(project_commits)}")
    print(f"Total contributors: {project_commits['author_name'].nunique()}")
    print(f"Time range: {project_commits['commit_date_clean'].min()} to {project_commits['commit_date_clean'].max()}")
    
    # Create monthly periods
    project_commits['year_month'] = project_commits['commit_date_clean'].dt.to_period('M')
    project_months = sorted(project_commits['year_month'].unique())
    
    print(f"Active months: {len(project_months)}")
    
    # Analyze month by month (limited for demo) with FIXED algorithm
    ever_been_core = set()  # FIXED: Track who has EVER been core
    previous_core = set()
    demo_months = project_months[:min(max_months, len(project_months))]
    
    print(f"\nMONTH-BY-MONTH ANALYSIS (showing first {len(demo_months)} months):")
    print("FIXED ALGORITHM: 'Once core, always core' - no double counting!")
    print("-" * 80)
    
    for i, month in enumerate(demo_months):
        print(f"\nMonth {i+1}: {month}")
        
        # Get commits up to this month
        commits_up_to_month = project_commits[project_commits['year_month'] <= month]
        commits_this_month = project_commits[project_commits['year_month'] == month]
        
        print(f"  Commits to date: {len(commits_up_to_month)}")
        print(f"  Commits this month: {len(commits_this_month)}")
        
        # Find current core contributors
        current_core_list = get_core_contributors_at_time(commits_up_to_month)
        current_core = set(current_core_list)
        
        print(f"  Core contributors now: {len(current_core)} - {current_core_list[:3]}{'...' if len(current_core_list) > 3 else ''}")
        
        # FIXED: Find TRULY NEW core contributors (never been core before)
        truly_new_core = current_core - ever_been_core
        existing_core_count = len(ever_been_core)  # Use ever_been_core for denominator
        
        # Update ever_been_core set
        ever_been_core.update(current_core)
        
        if existing_core_count > 0:  # Skip first month
            transition_rate = len(truly_new_core) / existing_core_count
            print(f"  TRULY NEW CORE: {len(truly_new_core)} contributors: {list(truly_new_core)}")
            print(f"  Ever been core before: {existing_core_count}")
            print(f"  Total ever been core now: {len(ever_been_core)}")
            print(f"  TRANSITION RATE: {len(truly_new_core)}/{existing_core_count} = {transition_rate:.4f}")
        else:
            print(f"  (First month - establishing initial core: {list(current_core)})")
        
        previous_core = current_core
        
        if i >= 5:  # Limit output for readability
            remaining = len(demo_months) - i - 1
            if remaining > 0:
                print(f"\n  ... (showing only first 6 months, {remaining} more months available)")
                break

def main():
    print("STEP 4 TEST: Demonstrating Transition Rate Calculation")
    print("="*60)
    
    # Load a small sample of data
    dataset_path = "../data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return
    
    # Load only a small sample for demonstration
    print("Loading sample data...")
    df = pd.read_csv(dataset_path)
    
    # Get a few example projects (mix of OSS and OSS4SG)
    project_counts = df.groupby(['project_name', 'project_type']).size().reset_index(name='commit_count')
    project_counts = project_counts.sort_values('commit_count', ascending=False)
    
    # Select 2 projects with decent activity
    sample_projects = project_counts.head(4)['project_name'].tolist()
    
    print(f"Demonstrating with {len(sample_projects)} sample projects:")
    for project in sample_projects:
        project_type = df[df['project_name'] == project]['project_type'].iloc[0]
        commit_count = df[df['project_name'] == project].shape[0]
        print(f"  - {project} ({project_type}): {commit_count:,} commits")
    
    # Demonstrate on each sample project
    for project in sample_projects[:2]:  # Limit to 2 for readability
        project_commits = df[df['project_name'] == project]
        demonstrate_transition_calculation(project, project_commits)
    
    print(f"\n{'='*60}")
    print("WHAT THIS ANALYSIS WILL PRODUCE:")
    print("="*60)
    print("1. For each project: Monthly transition rates over its lifetime")
    print("2. Project summaries: Average rates, frequency, total growth")
    print("3. Statistical comparison: OSS vs OSS4SG transition patterns")
    print("4. Visualizations: Time series, distributions, comparisons")
    print("\nExpected insights:")
    print("- Do OSS4SG projects attract core contributors faster?")
    print("- Are there seasonal patterns in contributor onboarding?")
    print("- How do transition rates relate to project characteristics?")
    
    print(f"\n{'='*60}")
    print("READY TO RUN FULL ANALYSIS")
    print("="*60)
    print("To run the complete analysis:")
    print("1. cd RQ1_transition_rates_and_speeds/step4_newcomer_transition_rates/")
    print("2. python calculate_transition_rates.py")
    print("3. python statistical_analysis.py")

if __name__ == "__main__":
    main()