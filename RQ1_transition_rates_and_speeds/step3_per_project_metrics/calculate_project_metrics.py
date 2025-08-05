#!/usr/bin/env python3
"""
Step 3: Calculate Per-Project Metrics
====================================

This script processes the master commits dataset and creates a new dataset
where each row is one project with all calculated metrics.

Core Contributors Definition: Smallest group responsible for 80% of commits
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_gini(commits_array):
    """
    Calculate Gini coefficient for contribution inequality
    """
    # Remove zeros and sort
    commits = np.array([c for c in commits_array if c > 0])
    commits = np.sort(commits)
    n = len(commits)
    
    if n == 0 or np.sum(commits) == 0:
        return 0
    
    # Calculate Gini using the formula
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * commits)) / (n * np.sum(commits)) - (n + 1) / n

def calculate_bus_factor(commit_counts):
    """
    Minimum contributors needed to account for 50% of commits
    """
    sorted_counts = commit_counts.sort_values(ascending=False)
    cumsum = sorted_counts.cumsum()
    total = sorted_counts.sum()
    
    # Find where cumsum exceeds 50%
    bus_factor = (cumsum <= 0.5 * total).sum() + 1
    return min(bus_factor, len(sorted_counts))

def identify_core_contributors(project_commits):
    """
    Find contributors responsible for 80% of commits using the 80/20 rule
    """
    # Count commits per contributor
    contributor_commits = project_commits.groupby('author_name')['commit_hash'].count()
    contributor_commits = contributor_commits.sort_values(ascending=False)
    
    total_commits = contributor_commits.sum()
    cumulative_commits = contributor_commits.cumsum()
    cumulative_percentage = cumulative_commits / total_commits
    
    # Find contributors up to 80%
    core_mask = cumulative_percentage <= 0.80
    
    if core_mask.sum() == 0:
        # Top contributor has >80%
        core_contributors = [contributor_commits.index[0]]
    else:
        core_contributors = contributor_commits[core_mask].index.tolist()
        
        # If we haven't reached 80%, add one more
        last_percentage = cumulative_percentage[core_contributors[-1]]
        if last_percentage < 0.80:
            remaining_contributors = contributor_commits.index.difference(core_contributors)
            if len(remaining_contributors) > 0:
                # Get the next highest contributor
                next_contributor = None
                for contrib in contributor_commits.index:
                    if contrib not in core_contributors:
                        next_contributor = contrib
                        break
                if next_contributor:
                    core_contributors.append(next_contributor)
    
    # Calculate actual percentage covered by core
    core_commits = contributor_commits[core_contributors].sum()
    core_percentage = core_commits / total_commits
    
    # Validation warning
    if core_percentage < 0.75:  # Allow some tolerance
        print(f"Warning: {project_commits['project_name'].iloc[0]} core contributors only account for {core_percentage:.1%} of commits")
    
    return {
        'core_contributors': core_contributors,
        'core_count': len(core_contributors),
        'total_contributors': len(contributor_commits),
        'core_commits': core_commits,
        'total_commits': total_commits,
        'core_percentage_actual': core_percentage  # Add for validation
    }

def calculate_project_metrics(project_name, project_commits):
    """
    Calculate all metrics for a single project
    """
    # Edge case handling: Skip very small projects
    if len(project_commits) < 10:  # Less than 10 commits
        print(f"Skipping {project_name}: Too few commits ({len(project_commits)})")
        return None
    
    commit_counts = project_commits.groupby('author_name')['commit_hash'].count()
    if len(commit_counts) < 3:  # Less than 3 contributors
        print(f"Skipping {project_name}: Too few contributors ({len(commit_counts)})")
        return None
    
    # Basic info
    project_type = project_commits['project_type'].iloc[0]
    total_commits = len(project_commits)
    
    # 1. Core contributors analysis
    core_info = identify_core_contributors(project_commits)
    
    # 2. One-time contributors
    one_time_contributors = (commit_counts == 1).sum()
    
    # 3. Gini coefficient
    gini = calculate_gini(commit_counts.values)
    
    # 4. Bus factor
    bus_factor = calculate_bus_factor(commit_counts)
    
    # 5. Active contributors (last 90 days)
    # Convert commit_date to datetime for calculation
    project_commits_clean = project_commits.copy()
    project_commits_clean['commit_date_clean'] = pd.to_datetime(project_commits_clean['commit_date'], errors='coerce')
    project_commits_clean = project_commits_clean.dropna(subset=['commit_date_clean'])
    
    if len(project_commits_clean) > 0:
        latest_date = project_commits_clean['commit_date_clean'].max()
        active_date = latest_date - pd.Timedelta(days=90)
        active_commits = project_commits_clean[project_commits_clean['commit_date_clean'] > active_date]
        active_contributors = active_commits['author_name'].nunique()
    else:
        active_contributors = 0
    
    # 6. Contributor categories
    casual = ((commit_counts > 1) & (commit_counts <= 10)).sum()
    regular = ((commit_counts > 10) & (commit_counts <= 100)).sum()
    heavy = (commit_counts > 100).sum()
    
    # Calculate ratios with proper bounds checking
    core_ratio = core_info['core_count'] / core_info['total_contributors']
    one_time_ratio = one_time_contributors / core_info['total_contributors']
    active_ratio = min(active_contributors / core_info['total_contributors'], 1.0)  # Cap at 1.0
    commits_per_contributor = total_commits / core_info['total_contributors']
    
    return {
        'project_name': project_name,
        'project_type': project_type,
        'total_contributors': core_info['total_contributors'],
        'core_contributors': core_info['core_count'],
        'core_ratio': core_ratio,
        'core_percentage_actual': core_info['core_percentage_actual'],  # Add validation metric
        'one_time_contributors': one_time_contributors,
        'one_time_ratio': one_time_ratio,
        'gini_coefficient': gini,
        'bus_factor': bus_factor,
        'active_contributors': active_contributors,
        'active_ratio': active_ratio,
        'casual_contributors': casual,
        'regular_contributors': regular,
        'heavy_contributors': heavy,
        'total_commits': total_commits,
        'core_commits': core_info['core_commits'],
        'commits_per_contributor': commits_per_contributor
    }

def main():
    """
    Main function to process all projects and create per-project dataset
    """
    print("="*60)
    print("STEP 3: CALCULATING PER-PROJECT METRICS")
    print("="*60)
    
    # Load the master commits dataset
    data_path = "../data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv"
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    
    print(f"Loaded {len(df):,} commits from {df['project_name'].nunique()} projects")
    
    # Get list of all projects
    projects = df['project_name'].unique()
    print(f"Processing {len(projects)} projects...")
    
    # Calculate metrics for each project with progress indicator
    project_metrics = []
    skipped_projects = 0
    
    for i, project in enumerate(projects, 1):
        if i % 25 == 0 or i == len(projects):
            progress = (i / len(projects)) * 100
            print(f"  Progress: {i}/{len(projects)} projects ({progress:.1f}%) - Current: {project}")
        
        # Get all commits for this project
        project_commits = df[df['project_name'] == project]
        
        # Calculate metrics
        metrics = calculate_project_metrics(project, project_commits)
        if metrics is not None:  # Only add if not skipped
            project_metrics.append(metrics)
        else:
            skipped_projects += 1
    
    # Create DataFrame
    project_df = pd.DataFrame(project_metrics)
    
    print(f"\nProcessed {len(project_df)} projects successfully")
    if skipped_projects > 0:
        print(f"Skipped {skipped_projects} projects (too small)")
    
    # Save to CSV
    output_path = "project_metrics.csv"
    project_df.to_csv(output_path, index=False)
    
    print(f"\n✅ COMPLETED: Per-project dataset saved to {output_path}")
    
    # Show summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"Total projects: {len(project_df)}")
    print(f"OSS projects: {len(project_df[project_df['project_type'] == 'OSS'])}")
    print(f"OSS4SG projects: {len(project_df[project_df['project_type'] == 'OSS4SG'])}")
    
    print(f"\n📊 CORE CONTRIBUTOR STATISTICS:")
    print(f"  Core Ratio - Mean: {project_df['core_ratio'].mean():.3f}, Median: {project_df['core_ratio'].median():.3f}")
    print(f"  Core Ratio - Min: {project_df['core_ratio'].min():.3f}, Max: {project_df['core_ratio'].max():.3f}")
    print(f"  Core Coverage (actual %) - Mean: {project_df['core_percentage_actual'].mean():.3f}, Median: {project_df['core_percentage_actual'].median():.3f}")
    
    print(f"\n📊 ONE-TIME CONTRIBUTOR STATISTICS:")
    print(f"  One-Time Ratio - Mean: {project_df['one_time_ratio'].mean():.3f}, Median: {project_df['one_time_ratio'].median():.3f}")
    
    print(f"\n📊 INEQUALITY METRICS:")
    print(f"  Gini Coefficient - Mean: {project_df['gini_coefficient'].mean():.3f}, Median: {project_df['gini_coefficient'].median():.3f}")
    print(f"  Bus Factor - Mean: {project_df['bus_factor'].mean():.1f}, Median: {project_df['bus_factor'].median():.1f}")
    
    print(f"\n📊 ACTIVITY METRICS:")
    print(f"  Active Ratio (90 days) - Mean: {project_df['active_ratio'].mean():.3f}, Median: {project_df['active_ratio'].median():.3f}")
    print(f"  Active Ratio Max: {project_df['active_ratio'].max():.3f} (should be ≤ 1.0)")
    
    print(f"\n📊 CONTRIBUTOR CATEGORIES:")
    print(f"  Casual (2-10 commits) - Mean: {project_df['casual_contributors'].mean():.1f}")
    print(f"  Regular (11-100 commits) - Mean: {project_df['regular_contributors'].mean():.1f}")
    print(f"  Heavy (100+ commits) - Mean: {project_df['heavy_contributors'].mean():.1f}")
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"  Contributors per project - Mean: {project_df['total_contributors'].mean():.1f}, Median: {project_df['total_contributors'].median():.1f}")
    print(f"  Commits per project - Mean: {project_df['total_commits'].mean():.1f}, Median: {project_df['total_commits'].median():.1f}")
    
    # Validation check
    low_coverage = project_df[project_df['core_percentage_actual'] < 0.75]
    if len(low_coverage) > 0:
        print(f"\n⚠️  WARNING: {len(low_coverage)} projects have core coverage < 75%")
    
    # Show sample with key metrics including validation
    print(f"\n📋 SAMPLE OF RESULTS (with validation):")
    sample_cols = ['project_name', 'project_type', 'total_contributors', 'core_contributors', 'core_ratio', 'core_percentage_actual', 'one_time_ratio', 'gini_coefficient']
    print(project_df[sample_cols].head(8).to_string(index=False))
    
    return project_df

if __name__ == "__main__":
    project_df = main()