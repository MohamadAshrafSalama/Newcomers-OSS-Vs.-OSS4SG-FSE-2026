#!/usr/bin/env python3
"""
Test Script: Monthly vs Yearly Analysis with Zero Handling
=========================================================
Compare different time periods and zero handling approaches
"""

import pandas as pd
import numpy as np
from scipy import stats

def analyze_time_periods():
    """
    Compare monthly vs yearly analysis with different zero handling
    """
    print("TIME PERIOD COMPARISON ANALYSIS")
    print("="*50)
    
    # Load the data
    results_dir = '../transition_analysis_results'
    monthly_file = f'{results_dir}/monthly_transition_rates.csv'
    project_file = f'{results_dir}/project_transition_summaries.csv'
    
    monthly_df = pd.read_csv(monthly_file)
    project_df = pd.read_csv(project_file)
    
    print(f"Original data: {len(monthly_df):,} monthly records")
    
    # Convert monthly data to yearly
    monthly_df['year'] = pd.to_datetime(monthly_df['month'].astype(str)).dt.year
    
    # Create yearly aggregation
    yearly_data = []
    for project in monthly_df['project_name'].unique():
        project_data = monthly_df[monthly_df['project_name'] == project]
        project_type = project_data['project_type'].iloc[0]
        
        for year in project_data['year'].unique():
            year_data = project_data[project_data['year'] == year]
            
            # Calculate yearly metrics
            total_new_core = year_data['truly_new_core_count'].sum()
            avg_existing_core = year_data['existing_core_count'].mean()
            total_commits = year_data['total_commits_to_date'].max()
            
            if avg_existing_core > 0:
                yearly_transition_rate = total_new_core / avg_existing_core
            else:
                yearly_transition_rate = 0
            
            yearly_data.append({
                'project_name': project,
                'project_type': project_type,
                'year': year,
                'total_new_core': total_new_core,
                'avg_existing_core': avg_existing_core,
                'yearly_transition_rate': yearly_transition_rate,
                'total_commits': total_commits,
                'months_in_year': len(year_data)
            })
    
    yearly_df = pd.DataFrame(yearly_data)
    
    print(f"Yearly data: {len(yearly_df):,} year-project records")
    
    # Analysis 1: Monthly with zeros
    print("\nANALYSIS 1: Monthly with Zeros")
    print("-" * 35)
    
    oss_monthly = monthly_df[monthly_df['project_type'] == 'OSS']['transition_rate']
    oss4sg_monthly = monthly_df[monthly_df['project_type'] == 'OSS4SG']['transition_rate']
    
    print(f"OSS monthly (with zeros): {oss_monthly.mean():.4f} ± {oss_monthly.std():.4f}")
    print(f"OSS4SG monthly (with zeros): {oss4sg_monthly.mean():.4f} ± {oss4sg_monthly.std():.4f}")
    
    statistic, pvalue = stats.mannwhitneyu(oss_monthly, oss4sg_monthly, alternative='two-sided')
    print(f"Mann-Whitney U test: p={pvalue:.4f}")
    
    # Analysis 2: Monthly without zeros
    print("\nANALYSIS 2: Monthly without Zeros")
    print("-" * 40)
    
    monthly_non_zero = monthly_df[monthly_df['truly_new_core_count'] > 0]
    oss_monthly_nz = monthly_non_zero[monthly_non_zero['project_type'] == 'OSS']['transition_rate']
    oss4sg_monthly_nz = monthly_non_zero[monthly_non_zero['project_type'] == 'OSS4SG']['transition_rate']
    
    print(f"OSS monthly (no zeros): {oss_monthly_nz.mean():.4f} ± {oss_monthly_nz.std():.4f}")
    print(f"OSS4SG monthly (no zeros): {oss4sg_monthly_nz.mean():.4f} ± {oss4sg_monthly_nz.std():.4f}")
    
    statistic, pvalue = stats.mannwhitneyu(oss_monthly_nz, oss4sg_monthly_nz, alternative='two-sided')
    print(f"Mann-Whitney U test: p={pvalue:.4f}")
    
    # Analysis 3: Yearly with zeros
    print("\nANALYSIS 3: Yearly with Zeros")
    print("-" * 35)
    
    oss_yearly = yearly_df[yearly_df['project_type'] == 'OSS']['yearly_transition_rate']
    oss4sg_yearly = yearly_df[yearly_df['project_type'] == 'OSS4SG']['yearly_transition_rate']
    
    print(f"OSS yearly (with zeros): {oss_yearly.mean():.4f} ± {oss_yearly.std():.4f}")
    print(f"OSS4SG yearly (with zeros): {oss4sg_yearly.mean():.4f} ± {oss4sg_yearly.std():.4f}")
    
    statistic, pvalue = stats.mannwhitneyu(oss_yearly, oss4sg_yearly, alternative='two-sided')
    print(f"Mann-Whitney U test: p={pvalue:.4f}")
    
    # Analysis 4: Yearly without zeros
    print("\nANALYSIS 4: Yearly without Zeros")
    print("-" * 40)
    
    yearly_non_zero = yearly_df[yearly_df['total_new_core'] > 0]
    oss_yearly_nz = yearly_non_zero[yearly_non_zero['project_type'] == 'OSS']['yearly_transition_rate']
    oss4sg_yearly_nz = yearly_non_zero[yearly_non_zero['project_type'] == 'OSS4SG']['yearly_transition_rate']
    
    print(f"OSS yearly (no zeros): {oss_yearly_nz.mean():.4f} ± {oss_yearly_nz.std():.4f}")
    print(f"OSS4SG yearly (no zeros): {oss4sg_yearly_nz.mean():.4f} ± {oss4sg_yearly_nz.std():.4f}")
    
    statistic, pvalue = stats.mannwhitneyu(oss_yearly_nz, oss4sg_yearly_nz, alternative='two-sided')
    print(f"Mann-Whitney U test: p={pvalue:.4f}")
    
    # Analysis 5: Yearly total new core (absolute numbers)
    print("\nANALYSIS 5: Yearly Total New Core (Absolute)")
    print("-" * 50)
    
    oss_yearly_total = yearly_df[yearly_df['project_type'] == 'OSS']['total_new_core']
    oss4sg_yearly_total = yearly_df[yearly_df['project_type'] == 'OSS4SG']['total_new_core']
    
    print(f"OSS yearly total new core: {oss_yearly_total.mean():.1f} ± {oss_yearly_total.std():.1f}")
    print(f"OSS4SG yearly total new core: {oss4sg_yearly_total.mean():.1f} ± {oss4sg_yearly_total.std():.1f}")
    
    statistic, pvalue = stats.mannwhitneyu(oss_yearly_total, oss4sg_yearly_total, alternative='two-sided')
    print(f"Mann-Whitney U test: p={pvalue:.4f}")
    
    # Analysis 6: Zero percentage comparison
    print("\nANALYSIS 6: Zero Percentage Comparison")
    print("-" * 40)
    
    oss_zero_pct = (monthly_df[monthly_df['project_type'] == 'OSS']['truly_new_core_count'] == 0).mean()
    oss4sg_zero_pct = (monthly_df[monthly_df['project_type'] == 'OSS4SG']['truly_new_core_count'] == 0).mean()
    
    print(f"OSS zero months: {oss_zero_pct*100:.1f}%")
    print(f"OSS4SG zero months: {oss4sg_zero_pct*100:.1f}%")
    
    # Chi-square test for zero percentage
    oss_zero_count = (monthly_df[monthly_df['project_type'] == 'OSS']['truly_new_core_count'] == 0).sum()
    oss_total_count = len(monthly_df[monthly_df['project_type'] == 'OSS'])
    oss4sg_zero_count = (monthly_df[monthly_df['project_type'] == 'OSS4SG']['truly_new_core_count'] == 0).sum()
    oss4sg_total_count = len(monthly_df[monthly_df['project_type'] == 'OSS4SG'])
    
    from scipy.stats import chi2_contingency
    contingency_table = [[oss_zero_count, oss_total_count - oss_zero_count],
                        [oss4sg_zero_count, oss4sg_total_count - oss4sg_zero_count]]
    chi2, pvalue, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square test for zero percentage: p={pvalue:.4f}")
    
    # Save results
    yearly_df.to_csv(f'{results_dir}/yearly_transition_rates.csv', index=False)
    
    print(f"\nSUMMARY OF FINDINGS:")
    print(f"=" * 50)
    print(f"1. Monthly with zeros: {pvalue:.4f} significance")
    print(f"2. Monthly without zeros: {pvalue:.4f} significance") 
    print(f"3. Yearly with zeros: {pvalue:.4f} significance")
    print(f"4. Yearly without zeros: {pvalue:.4f} significance")
    print(f"5. Yearly absolute numbers: {pvalue:.4f} significance")
    print(f"6. Zero percentage difference: {pvalue:.4f} significance")
    
    print(f"\nRECOMMENDATIONS:")
    print(f"1. Use yearly analysis for more stable patterns")
    print(f"2. Compare both with/without zeros")
    print(f"3. Consider absolute numbers vs rates")
    print(f"4. Analyze zero percentage as a metric itself")

if __name__ == "__main__":
    analyze_time_periods() 