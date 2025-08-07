#!/usr/bin/env python3
"""
Test Script: Handling Zero Transition Months
===========================================
Explore different approaches to handle months with zero new core contributors
"""

import pandas as pd
import numpy as np
from scipy import stats

def analyze_zero_handling():
    """
    Test different approaches for handling zero transition months
    """
    print("ZERO MONTH HANDLING ANALYSIS")
    print("="*50)
    
    # Load the data
    results_dir = '../clean_transition_results'
    monthly_file = f'{results_dir}/monthly_transitions.csv'
    project_file = f'{results_dir}/project_summaries.csv'
    
    monthly_df = pd.read_csv(monthly_file)
    project_df = pd.read_csv(project_file)
    
    print(f"Total monthly records: {len(monthly_df):,}")
    print(f"Records with zero transitions: {(monthly_df['truly_new_core_count'] == 0).sum():,}")
    print(f"Records with non-zero transitions: {(monthly_df['truly_new_core_count'] > 0).sum():,}")
    print(f"Percentage of zero months: {(monthly_df['truly_new_core_count'] == 0).mean()*100:.1f}%")
    
    # Approach 1: Only analyze months with transitions
    print("\nAPPROACH 1: Only Non-Zero Months")
    print("-" * 30)
    
    non_zero_df = monthly_df[monthly_df['truly_new_core_count'] > 0]
    
    oss_non_zero = non_zero_df[non_zero_df['project_type'] == 'OSS']['transition_rate']
    oss4sg_non_zero = non_zero_df[non_zero_df['project_type'] == 'OSS4SG']['transition_rate']
    
    print(f"OSS non-zero months: {len(oss_non_zero)}")
    print(f"OSS4SG non-zero months: {len(oss4sg_non_zero)}")
    print(f"OSS median (non-zero only): {oss_non_zero.median():.4f}")
    print(f"OSS4SG median (non-zero only): {oss4sg_non_zero.median():.4f}")
    
    # Statistical test
    statistic, pvalue = stats.mannwhitneyu(oss_non_zero, oss4sg_non_zero, alternative='two-sided')
    print(f"Mann-Whitney U test: p={pvalue:.4f}")
    
    # Approach 2: Use transition frequency instead of rates
    print("\nAPPROACH 2: Transition Frequency (Project Level)")
    print("-" * 40)
    
    oss_freq = project_df[project_df['project_type'] == 'OSS']['new_core_frequency']
    oss4sg_freq = project_df[project_df['project_type'] == 'OSS4SG']['new_core_frequency']
    
    print(f"OSS transition frequency: {oss_freq.mean():.3f} ± {oss_freq.std():.3f}")
    print(f"OSS4SG transition frequency: {oss4sg_freq.mean():.3f} ± {oss4sg_freq.std():.3f}")
    
    statistic, pvalue = stats.mannwhitneyu(oss_freq, oss4sg_freq, alternative='two-sided')
    print(f"Mann-Whitney U test: p={pvalue:.4f}")
    
    # Approach 3: Time between transitions
    print("\nAPPROACH 3: Time Between Transitions")
    print("-" * 35)
    
    def calculate_time_between_transitions(project_data):
        """Calculate average months between transitions for a project"""
        transitions = project_data[project_data['truly_new_core_count'] > 0]
        if len(transitions) < 2:
            return np.nan
        
        # Calculate months between transitions
        months_between = []
        for i in range(1, len(transitions)):
            months_diff = transitions.iloc[i]['month_index'] - transitions.iloc[i-1]['month_index']
            months_between.append(months_diff)
        
        return np.mean(months_between) if months_between else np.nan
    
    # Calculate for each project
    project_timing = []
    for project in project_df['project_name'].unique():
        project_data = monthly_df[monthly_df['project_name'] == project]
        avg_months_between = calculate_time_between_transitions(project_data)
        project_type = project_data['project_type'].iloc[0]
        
        if not np.isnan(avg_months_between):
            project_timing.append({
                'project_name': project,
                'project_type': project_type,
                'avg_months_between_transitions': avg_months_between
            })
    
    timing_df = pd.DataFrame(project_timing)
    
    oss_timing = timing_df[timing_df['project_type'] == 'OSS']['avg_months_between_transitions']
    oss4sg_timing = timing_df[timing_df['project_type'] == 'OSS4SG']['avg_months_between_transitions']
    
    print(f"OSS avg months between transitions: {oss_timing.mean():.1f} ± {oss_timing.std():.1f}")
    print(f"OSS4SG avg months between transitions: {oss4sg_timing.mean():.1f} ± {oss4sg_timing.std():.1f}")
    
    statistic, pvalue = stats.mannwhitneyu(oss_timing, oss4sg_timing, alternative='two-sided')
    print(f"Mann-Whitney U test: p={pvalue:.4f}")
    
    # Approach 4: Total transitions per project
    print("\nAPPROACH 4: Total Transitions per Project")
    print("-" * 40)
    
    oss_total = project_df[project_df['project_type'] == 'OSS']['total_truly_new_core_added']
    oss4sg_total = project_df[project_df['project_type'] == 'OSS4SG']['total_truly_new_core_added']
    
    print(f"OSS total new core per project: {oss_total.mean():.1f} ± {oss_total.std():.1f}")
    print(f"OSS4SG total new core per project: {oss4sg_total.mean():.1f} ± {oss4sg_total.std():.1f}")
    
    statistic, pvalue = stats.mannwhitneyu(oss_total, oss4sg_total, alternative='two-sided')
    print(f"Mann-Whitney U test: p={pvalue:.4f}")
    
    # Save results
    timing_df.to_csv(f'{results_dir}/transition_timing_analysis.csv', index=False)
    
    print(f"\nRECOMMENDATIONS:")
    print(f"1. Use transition frequency (Approach 2) - most meaningful")
    print(f"2. Use total transitions per project (Approach 4) - absolute growth")
    print(f"3. Use time between transitions (Approach 3) - dynamics")
    print(f"4. Avoid raw transition rates - too many zeros")

if __name__ == "__main__":
    analyze_zero_handling() 