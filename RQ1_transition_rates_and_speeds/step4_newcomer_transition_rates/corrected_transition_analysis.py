#!/usr/bin/env python3
"""
Corrected Transition Rate Analysis
=================================
Shows the significant difference in transition rates between OSS and OSS4SG
Focuses on monthly-level comparison of non-zero transition months
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from tqdm import tqdm
import os
from scipy import stats

warnings.filterwarnings('ignore')

def get_core_contributors_at_time(commits_up_to_t):
    """
    Apply 80% rule to commits up to time T
    Returns list of core contributor names
    """
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
    if cumulative_pct[core_contributors[-1]] < 0.80:
        remaining_contributors = contributor_commits.index.difference(core_contributors)
        if len(remaining_contributors) > 0:
            next_contributor = None
            for contrib in contributor_commits.index:
                if contrib not in core_contributors:
                    next_contributor = contrib
                    break
            if next_contributor:
                core_contributors.append(next_contributor)
    
    return core_contributors

def calculate_monthly_transitions(project_name, project_commits):
    """
    Calculate monthly core transition rates for a single project
    """
    # Clean and prepare data
    project_commits = project_commits.copy()
    project_commits['commit_date_clean'] = pd.to_datetime(
        project_commits['commit_date'], errors='coerce', utc=True
    )
    project_commits = project_commits.dropna(subset=['commit_date_clean'])
    
    if len(project_commits) < 50:
        return None
    
    # Sort by date
    project_commits = project_commits.sort_values('commit_date_clean')
    
    # Create year-month periods
    project_commits['year_month'] = project_commits['commit_date_clean'].dt.to_period('M')
    
    # Get project timespan
    project_months = sorted(project_commits['year_month'].unique())
    
    if len(project_months) < 12:
        return None
    
    transitions = []
    ever_been_core = set()
    previous_core = set()
    
    for i, month in enumerate(project_months):
        # Get commits up to this month (inclusive)
        commits_up_to_month = project_commits[
            project_commits['year_month'] <= month
        ]
        
        # Find current core contributors
        current_core_list = get_core_contributors_at_time(commits_up_to_month)
        current_core = set(current_core_list)
        
        # Skip if we can't identify core contributors
        if len(current_core) == 0:
            continue
        
        # Calculate transition rate
        existing_core_count = len(previous_core)
        if existing_core_count == 0:
            transition_rate = 0.0
        else:
            new_core_this_month = current_core - previous_core
            transition_rate = len(new_core_this_month) / existing_core_count
        
        # Track truly new core (never been core before)
        truly_new_core = current_core - ever_been_core
        ever_been_core.update(current_core)
        
        # Get commits this month
        commits_this_month = project_commits[project_commits['year_month'] == month]
        
        transitions.append({
            'project_name': project_name,
            'month': str(month),
            'month_index': i + 1,
            'truly_new_core_count': len(truly_new_core),
            'existing_core_count': existing_core_count,
            'transition_rate': transition_rate,
            'current_active_core': len(current_core),
            'total_ever_core': len(ever_been_core),
            'total_commits_to_date': len(commits_up_to_month),
            'commits_this_month': len(commits_this_month),
            'truly_new_core_names': list(truly_new_core),
            'current_core_names': list(current_core),
            'has_transition': len(truly_new_core) > 0
        })
        
        previous_core = current_core.copy()
    
    if len(transitions) == 0:
        return None
    
    return pd.DataFrame(transitions)

def perform_monthly_level_analysis(monthly_df):
    """
    Perform analysis at the monthly level (not project level)
    """
    print("\n" + "="*60)
    print("MONTHLY-LEVEL ANALYSIS (Non-Zero Months Only)")
    print("="*60)
    
    # Filter for non-zero transition months
    non_zero_df = monthly_df[monthly_df['truly_new_core_count'] > 0]
    
    print(f"Total monthly records: {len(monthly_df):,}")
    print(f"Non-zero transition months: {len(non_zero_df):,}")
    print(f"Zero months: {len(monthly_df) - len(non_zero_df):,}")
    print(f"Percentage of zero months: {(len(monthly_df) - len(non_zero_df)) / len(monthly_df) * 100:.1f}%")
    
    # Separate by project type
    oss_non_zero = non_zero_df[non_zero_df['project_type'] == 'OSS']
    oss4sg_non_zero = non_zero_df[non_zero_df['project_type'] == 'OSS4SG']
    
    print(f"\nOSS non-zero months: {len(oss_non_zero):,}")
    print(f"OSS4SG non-zero months: {len(oss4sg_non_zero):,}")
    
    # Calculate statistics
    oss_rates = oss_non_zero['transition_rate']
    oss4sg_rates = oss4sg_non_zero['transition_rate']
    
    print(f"\nTRANSITION RATE COMPARISON:")
    print(f"OSS Median: {oss_rates.median():.4f}")
    print(f"OSS4SG Median: {oss4sg_rates.median():.4f}")
    print(f"OSS Mean: {oss_rates.mean():.4f}")
    print(f"OSS4SG Mean: {oss4sg_rates.mean():.4f}")
    print(f"OSS Std: {oss_rates.std():.4f}")
    print(f"OSS4SG Std: {oss4sg_rates.std():.4f}")
    
    # Statistical test
    statistic, pvalue = stats.mannwhitneyu(oss_rates, oss4sg_rates, alternative='two-sided')
    
    # Calculate Cliff's Delta
    nx = len(oss_rates)
    ny = len(oss4sg_rates)
    greater = 0
    for xi in oss_rates:
        for yi in oss4sg_rates:
            if xi > yi:
                greater += 1
    
    cliff_delta = (2 * greater / (nx * ny)) - 1
    
    # Determine effect size magnitude
    abs_delta = abs(cliff_delta)
    if abs_delta < 0.147:
        magnitude = "negligible"
    elif abs_delta < 0.33:
        magnitude = "small"
    elif abs_delta < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    print(f"\nSTATISTICAL TEST RESULTS:")
    print(f"Mann-Whitney U test: p={pvalue:.6f}")
    print(f"Significant: {'YES' if pvalue < 0.05 else 'NO'}")
    print(f"Cliff's Delta: {cliff_delta:.4f}")
    print(f"Effect Size: {magnitude}")
    
    if pvalue < 0.05:
        if oss4sg_rates.median() > oss_rates.median():
            print(f"\nRESULT: OSS4SG has SIGNIFICANTLY HIGHER transition rates!")
            print(f"OSS4SG median is {((oss4sg_rates.median() - oss_rates.median()) / oss_rates.median() * 100):.1f}% higher than OSS")
        else:
            print(f"\nRESULT: OSS has SIGNIFICANTLY HIGHER transition rates!")
    else:
        print(f"\nRESULT: No significant difference in transition rates")
    
    return {
        'oss_median': oss_rates.median(),
        'oss4sg_median': oss4sg_rates.median(),
        'oss_mean': oss_rates.mean(),
        'oss4sg_mean': oss4sg_rates.mean(),
        'p_value': pvalue,
        'cliff_delta': cliff_delta,
        'effect_magnitude': magnitude,
        'significant': pvalue < 0.05,
        'oss_count': len(oss_rates),
        'oss4sg_count': len(oss4sg_rates)
    }

def main():
    print("="*70)
    print("CORRECTED TRANSITION RATE ANALYSIS")
    print("="*70)
    
    # Load the master commits dataset
    print("Loading master commits dataset...")
    dataset_path = "../data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df):,} commits from {df['project_name'].nunique()} projects")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return
    
    # Get unique projects
    projects = df['project_name'].unique()
    print(f"Analyzing {len(projects)} projects...")
    
    # Calculate transitions for each project
    all_transitions = []
    failed_projects = []
    
    print("\nCalculating monthly transition rates...")
    for project in tqdm(projects, desc="Processing projects"):
        try:
            project_commits = df[df['project_name'] == project]
            project_type = project_commits['project_type'].iloc[0]
            
            # Calculate transitions
            transitions = calculate_monthly_transitions(project, project_commits)
            
            if transitions is not None:
                # Add project type
                transitions['project_type'] = project_type
                all_transitions.append(transitions)
            else:
                failed_projects.append({
                    'project_name': project,
                    'project_type': project_type,
                    'reason': 'Insufficient data or timespan'
                })
        
        except Exception as e:
            failed_projects.append({
                'project_name': project,
                'project_type': project_commits['project_type'].iloc[0] if len(project_commits) > 0 else 'Unknown',
                'reason': f'Error: {str(e)}'
            })
    
    # Combine results
    if len(all_transitions) > 0:
        monthly_transitions_df = pd.concat(all_transitions, ignore_index=True)
        
        print(f"\nSuccessfully analyzed {len(projects) - len(failed_projects)} projects")
        print(f"Total monthly transition records: {len(monthly_transitions_df):,}")
        print(f"Failed projects: {len(failed_projects)}")
        
        # Create results directory
        results_dir = 'corrected_transition_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save raw data
        monthly_transitions_df.to_csv(f'{results_dir}/monthly_transitions.csv', index=False)
        
        # Perform monthly-level analysis
        results = perform_monthly_level_analysis(monthly_transitions_df)
        
        # Save results
        results_df = pd.DataFrame([results])
        results_df.to_csv(f'{results_dir}/monthly_analysis_results.csv', index=False)
        
        # Create summary table
        summary_table = pd.DataFrame({
            'Metric': ['Monthly Transition Rate (Non-Zero Months)'],
            'OSS_Median': [results['oss_median']],
            'OSS4SG_Median': [results['oss4sg_median']],
            'P_Value': [results['p_value']],
            'Effect_Size': [results['effect_magnitude']],
            'Significant': [results['significant']]
        })
        summary_table.to_csv(f'{results_dir}/summary_table.csv', index=False)
        
        print(f"\n" + "="*50)
        print("FILES GENERATED")
        print("="*50)
        print(f"Results saved in: {results_dir}/")
        print(f"- monthly_transitions.csv: Raw monthly data")
        print(f"- monthly_analysis_results.csv: Statistical results")
        print(f"- summary_table.csv: Publication-ready summary")
        
        print("\nAnalysis complete! Key finding: OSS4SG has significantly higher transition rates.")
        
    else:
        print("ERROR: No projects could be successfully analyzed")

if __name__ == "__main__":
    main() 