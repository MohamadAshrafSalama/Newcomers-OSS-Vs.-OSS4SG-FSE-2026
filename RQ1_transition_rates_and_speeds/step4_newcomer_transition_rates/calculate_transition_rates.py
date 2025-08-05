#!/usr/bin/env python3
"""
Step 4: Newcomer-to-Core Transition Rate Analysis
================================================
Analyzes how frequently newcomers become core contributors over time,
comparing OSS vs OSS4SG projects.

Key Metric: Monthly Core Transition Rate = (New core this month) / (Existing core before month)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')

def get_core_contributors_at_time(commits_up_to_t):
    """
    Apply 80% rule to commits up to time T
    Returns list of core contributor names
    """
    if len(commits_up_to_t) < 3:  # Need minimum commits
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
    IMPORTANT: "Once core, always core" - contributors who were core before 
    are never counted as "new" again, even if they temporarily dropped out.
    """
    # Clean and prepare data
    project_commits = project_commits.copy()
    project_commits['commit_date_clean'] = pd.to_datetime(
        project_commits['commit_date'], errors='coerce', utc=True
    )
    project_commits = project_commits.dropna(subset=['commit_date_clean'])
    
    if len(project_commits) < 50:  # Minimum commits threshold
        return None
    
    # Sort by date
    project_commits = project_commits.sort_values('commit_date_clean')
    
    # Create year-month periods
    project_commits['year_month'] = project_commits['commit_date_clean'].dt.to_period('M')
    
    # Get project timespan
    project_months = sorted(project_commits['year_month'].unique())
    
    if len(project_months) < 12:  # Minimum 12 months
        return None
    
    transitions = []
    ever_been_core = set()  # FIXED: Track who has EVER been core
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
        
        # FIXED: Find TRULY NEW core contributors (never been core before)
        truly_new_core = current_core - ever_been_core
        existing_core_count = len(ever_been_core)  # Use ever_been_core for denominator
        
        # Update ever_been_core set
        ever_been_core.update(current_core)
        
        # Calculate transition rate (skip first month)
        if existing_core_count > 0:
            transition_rate = len(truly_new_core) / existing_core_count
            
            transitions.append({
                'project_name': project_name,
                'month': month,
                'month_index': i,
                'truly_new_core_count': len(truly_new_core),
                'existing_core_count': existing_core_count,
                'transition_rate': transition_rate,
                'current_active_core': len(current_core),  # Core contributors active this month
                'total_ever_core': len(ever_been_core),    # Total who have ever been core
                'total_commits_to_date': len(commits_up_to_month),
                'commits_this_month': len(project_commits[project_commits['year_month'] == month]),
                'truly_new_core_names': list(truly_new_core),  # Store names for detailed analysis
                'current_core_names': current_core_list         # Store current core names
            })
        
        previous_core = current_core
    
    if len(transitions) == 0:
        return None
        
    return pd.DataFrame(transitions)

def calculate_project_transition_summary(project_transitions):
    """
    Calculate summary statistics for a project's transition rates
    """
    if project_transitions is None or len(project_transitions) == 0:
        return None
    
    return {
        'project_name': project_transitions['project_name'].iloc[0],
        'total_months_analyzed': len(project_transitions),
        'avg_transition_rate': project_transitions['transition_rate'].mean(),
        'median_transition_rate': project_transitions['transition_rate'].median(),
        'max_transition_rate': project_transitions['transition_rate'].max(),
        'std_transition_rate': project_transitions['transition_rate'].std(),
        'total_truly_new_core_added': project_transitions['truly_new_core_count'].sum(),
        'avg_existing_core_count': project_transitions['existing_core_count'].mean(),
        'final_ever_core_count': project_transitions['total_ever_core'].iloc[-1],
        'final_active_core_count': project_transitions['current_active_core'].iloc[-1],
        'months_with_new_core': (project_transitions['truly_new_core_count'] > 0).sum(),
        'new_core_frequency': (project_transitions['truly_new_core_count'] > 0).mean(),
        'project_start_month': project_transitions['month'].iloc[0],
        'project_end_month': project_transitions['month'].iloc[-1],
        'total_commits_analyzed': project_transitions['total_commits_to_date'].iloc[-1]
    }

def main():
    print("="*70)
    print("STEP 4: NEWCOMER-TO-CORE TRANSITION RATE ANALYSIS")
    print("="*70)
    
    # Load the master commits dataset
    print("Loading master commits dataset...")
    dataset_path = "../data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please ensure the master commits dataset exists.")
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
    project_summaries = []
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
                
                # Calculate summary
                summary = calculate_project_transition_summary(transitions)
                if summary is not None:
                    summary['project_type'] = project_type
                    project_summaries.append(summary)
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
        project_summaries_df = pd.DataFrame(project_summaries)
        
        print(f"\nSuccessfully analyzed {len(project_summaries)} projects")
        print(f"Total monthly transition records: {len(monthly_transitions_df):,}")
        print(f"Failed projects: {len(failed_projects)}")
        
        # Create results directory
        results_dir = 'transition_analysis_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results with organized structure
        monthly_transitions_df.to_csv(f'{results_dir}/monthly_transition_rates.csv', index=False)
        project_summaries_df.to_csv(f'{results_dir}/project_transition_summaries.csv', index=False)
        
        if len(failed_projects) > 0:
            failed_df = pd.DataFrame(failed_projects)
            failed_df.to_csv(f'{results_dir}/failed_projects.csv', index=False)
        
        # Save detailed core contributor evolution for further analysis
        detailed_analysis = []
        for _, row in monthly_transitions_df.iterrows():
            if row['truly_new_core_names']:  # If there are new core contributors
                for name in row['truly_new_core_names']:
                    detailed_analysis.append({
                        'project_name': row['project_name'], 
                        'project_type': row['project_type'],
                        'month': row['month'],
                        'new_core_contributor': name,
                        'existing_core_count_when_joined': row['existing_core_count'],
                        'total_commits_when_joined': row['total_commits_to_date']
                    })
        
        if detailed_analysis:
            detailed_df = pd.DataFrame(detailed_analysis)
            detailed_df.to_csv(f'{results_dir}/individual_core_transitions.csv', index=False)
            print(f"Tracked {len(detailed_analysis)} individual core contributor transitions")
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        oss_projects = project_summaries_df[project_summaries_df['project_type'] == 'OSS']
        oss4sg_projects = project_summaries_df[project_summaries_df['project_type'] == 'OSS4SG']
        
        print(f"OSS Projects: {len(oss_projects)}")
        print(f"OSS4SG Projects: {len(oss4sg_projects)}")
        
        print(f"\nAverage Monthly Transition Rates:")
        print(f"OSS: {oss_projects['avg_transition_rate'].mean():.4f} ± {oss_projects['avg_transition_rate'].std():.4f}")
        print(f"OSS4SG: {oss4sg_projects['avg_transition_rate'].mean():.4f} ± {oss4sg_projects['avg_transition_rate'].std():.4f}")
        
        print(f"\nNew Core Frequency (months with truly new core):")
        print(f"OSS: {oss_projects['new_core_frequency'].mean():.3f}")
        print(f"OSS4SG: {oss4sg_projects['new_core_frequency'].mean():.3f}")
        
        print(f"\nMedian Final Ever-Core Count:")
        print(f"OSS: {oss_projects['final_ever_core_count'].median():.1f}")
        print(f"OSS4SG: {oss4sg_projects['final_ever_core_count'].median():.1f}")
        
        print(f"\nTotal Truly New Core Contributors Added:")
        print(f"OSS: {oss_projects['total_truly_new_core_added'].sum()} across {len(oss_projects)} projects")
        print(f"OSS4SG: {oss4sg_projects['total_truly_new_core_added'].sum()} across {len(oss4sg_projects)} projects")
        
        print("\n" + "="*50)
        print("FILES GENERATED:")
        print("="*50)
        print(f"Results saved in: {results_dir}/")
        print(f"- monthly_transition_rates.csv: Monthly data for all projects ({len(monthly_transitions_df):,} records)")
        print(f"- project_transition_summaries.csv: Project-level statistics ({len(project_summaries_df)} projects)")
        if len(failed_projects) > 0:
            print(f"- failed_projects.csv: Projects that couldn't be analyzed ({len(failed_projects)} projects)")
        if detailed_analysis:
            print(f"- individual_core_transitions.csv: Individual transitions ({len(detailed_analysis)} records)")
        
        print(f"\nData Organization:")
        print(f"- Monthly rates: Transition dynamics over time")
        print(f"- Project summaries: Aggregated metrics per project") 
        print(f"- Individual transitions: Each person who became core")
        print(f"- All data includes project type (OSS vs OSS4SG) for comparison")
        
        print("\nNext step: Run statistical comparison and visualization script")
        
    else:
        print("ERROR: No projects could be successfully analyzed")
        if len(failed_projects) > 0:
            failed_df = pd.DataFrame(failed_projects)
            failed_df.to_csv('failed_projects.csv', index=False)
            print("See failed_projects.csv for details")

if __name__ == "__main__":
    main()