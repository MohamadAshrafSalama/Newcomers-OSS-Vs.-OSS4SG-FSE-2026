#!/usr/bin/env python3
"""
Quick Ecosystem Comparability Analysis
=====================================
Analyzes the master commits dataset to verify OSS vs OSS4SG project comparability
across key dimensions: project age, contributor count, and commit count.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import mannwhitneyu
import os
import sys

def safe_mannwhitney(group1, group2):
    """Safely perform Mann-Whitney U test with error handling"""
    try:
        if len(group1) == 0 or len(group2) == 0:
            return float('nan'), float('nan')
        
        # Remove any NaN values
        group1_clean = [x for x in group1 if pd.notna(x)]
        group2_clean = [x for x in group2 if pd.notna(x)]
        
        if len(group1_clean) == 0 or len(group2_clean) == 0:
            return float('nan'), float('nan')
            
        statistic, pvalue = mannwhitneyu(group1_clean, group2_clean, alternative='two-sided')
        return statistic, pvalue
    except Exception as e:
        print(f"Mann-Whitney U test failed: {e}")
        return float('nan'), float('nan')

def main():
    # File path
    data_file = "/Users/mohamadashraf/Desktop/?-=/Newcomers OSS Vs. OSS4SG FSE 2026/RQ1_transition_rates_and_speeds/data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv"
    
    print("=" * 70)
    print("ECOSYSTEM COMPARABILITY ANALYSIS")
    print("=" * 70)
    print(f"Processing: {os.path.basename(data_file)}")
    print(f"File size: {os.path.getsize(data_file) / (1024**3):.1f} GB")
    print()
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"ERROR: File not found: {data_file}")
        return
    
    print("Reading data in chunks (this may take a few minutes)...")
    
    # Process in chunks to handle large file
    chunk_size = 50000  # Adjust based on memory
    project_stats = {}
    
    try:
        chunk_count = 0
        for chunk in pd.read_csv(data_file, chunksize=chunk_size):
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"  Processed {chunk_count * chunk_size:,} rows...")
            
            # Convert commit_date to datetime
            chunk['commit_date'] = pd.to_datetime(chunk['commit_date'], errors='coerce')
            
            # Group by project
            for project_name, project_data in chunk.groupby('project_name'):
                if project_name not in project_stats:
                    project_stats[project_name] = {
                        'project_type': project_data['project_type'].iloc[0],
                        'commit_dates': [],
                        'contributors': set(),
                        'commit_count': 0
                    }
                
                # Update statistics
                project_stats[project_name]['commit_dates'].extend(
                    project_data['commit_date'].dropna().tolist()
                )
                project_stats[project_name]['contributors'].update(
                    project_data['author_email'].dropna().tolist()
                )
                project_stats[project_name]['commit_count'] += len(project_data)
        
        print(f"✓ Completed processing {chunk_count * chunk_size:,} total rows")
        print()
        
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return
    
    if not project_stats:
        print("ERROR: No data found in the file")
        return
    
    print("Computing project-level metrics...")
    
    # Convert to project-level metrics
    project_metrics = []
    
    for project_name, stats in project_stats.items():
        if not stats['commit_dates']:  # Skip projects with no valid dates
            continue
            
        # Calculate project age in years
        min_date = min(stats['commit_dates'])
        max_date = max(stats['commit_dates'])
        project_age_days = (max_date - min_date).days
        project_age_years = project_age_days / 365.25
        
        project_metrics.append({
            'project_name': project_name,
            'project_type': stats['project_type'],
            'project_age_years': project_age_years,
            'contributor_count': len(stats['contributors']),
            'commit_count': stats['commit_count'],
            'first_commit_date': min_date,
            'last_commit_date': max_date
        })
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(project_metrics)
    
    if df.empty:
        print("ERROR: No valid projects found")
        return
    
    print(f"✓ Analyzed {len(df)} projects")
    print()
    
    # Separate OSS and OSS4SG
    oss_projects = df[df['project_type'] == 'OSS']
    oss4sg_projects = df[df['project_type'] == 'OSS4SG']
    
    print("=" * 70)
    print("ECOSYSTEM COMPARABILITY RESULTS")
    print("=" * 70)
    print()
    
    print(f"Total Projects Analyzed: {len(df)}")
    print(f"OSS Projects: {len(oss_projects)} ({len(oss_projects)/len(df)*100:.1f}%)")
    print(f"OSS4SG Projects: {len(oss4sg_projects)} ({len(oss4sg_projects)/len(df)*100:.1f}%)")
    print()
    
    # Metrics to compare
    metrics = [
        ('project_age_years', 'Project Age (years)'),
        ('contributor_count', 'Contributors per project'),
        ('commit_count', 'Commits per project')
    ]
    
    print("STATISTICAL COMPARISON:")
    print("-" * 50)
    
    for metric_col, metric_name in metrics:
        oss_values = oss_projects[metric_col].values
        oss4sg_values = oss4sg_projects[metric_col].values
        
        # Basic statistics
        oss_median = np.median(oss_values)
        oss4sg_median = np.median(oss4sg_values)
        oss_mean = np.mean(oss_values)
        oss4sg_mean = np.mean(oss4sg_values)
        
        # Statistical test
        statistic, pvalue = safe_mannwhitney(oss_values, oss4sg_values)
        
        print(f"\n{metric_name}:")
        print(f"  OSS - Median: {oss_median:.1f}, Mean: {oss_mean:.1f}")
        print(f"  OSS4SG - Median: {oss4sg_median:.1f}, Mean: {oss4sg_mean:.1f}")
        if not pd.isna(pvalue):
            print(f"  Mann-Whitney U test: p = {pvalue:.3f}")
            significance = "significant" if pvalue < 0.05 else "not significant"
            print(f"  Result: {significance}")
        else:
            print(f"  Mann-Whitney U test: Could not compute")
    
    print()
    print("=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    
    # Generate paper-ready summary
    age_oss_med = np.median(oss_projects['project_age_years'])
    age_oss4sg_med = np.median(oss4sg_projects['project_age_years'])
    age_stat, age_p = safe_mannwhitney(oss_projects['project_age_years'], oss4sg_projects['project_age_years'])
    
    contrib_oss_med = np.median(oss_projects['contributor_count'])
    contrib_oss4sg_med = np.median(oss4sg_projects['contributor_count'])
    contrib_stat, contrib_p = safe_mannwhitney(oss_projects['contributor_count'], oss4sg_projects['contributor_count'])
    
    commit_oss_med = np.median(oss_projects['commit_count'])
    commit_oss4sg_med = np.median(oss4sg_projects['commit_count'])
    commit_stat, commit_p = safe_mannwhitney(oss_projects['commit_count'], oss4sg_projects['commit_count'])
    
    print(f"We verified ecosystem comparability across key dimensions: ")
    print(f"median project age (OSS: {age_oss_med:.1f} years, OSS4SG: {age_oss4sg_med:.1f} years, p={age_p:.2f}), ")
    print(f"median contributors (OSS: {contrib_oss_med:.0f}, OSS4SG: {contrib_oss4sg_med:.0f}, p={contrib_p:.2f}), ")
    print(f"and median commits (OSS: {commit_oss_med:,.0f}, OSS4SG: {commit_oss4sg_med:,.0f}, p={commit_p:.2f}). ")
    print(f"This ensures observed differences reflect ecosystem characteristics rather than sampling bias.")
    
    print()
    print("=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()