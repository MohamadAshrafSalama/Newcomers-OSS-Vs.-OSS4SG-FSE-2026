#!/usr/bin/env python3
"""
Step 4B: Statistical Analysis of Transition Rates
=================================================
Compares newcomer-to-core transition rates between OSS and OSS4SG projects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

def calculate_cliff_delta(x, y):
    """
    Calculate Cliff's Delta effect size
    Small: |d| < 0.33, Medium: |d| < 0.474, Large: |d| >= 0.474
    """
    nx = len(x)
    ny = len(y)
    
    greater = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                greater += 1
    
    delta = (2 * greater / (nx * ny)) - 1
    
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        magnitude = "negligible"
    elif abs_delta < 0.33:
        magnitude = "small"
    elif abs_delta < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    return delta, magnitude

def perform_statistical_tests(df, metrics):
    """
    Perform Mann-Whitney U tests and calculate effect sizes
    """
    oss_data = df[df['project_type'] == 'OSS']
    oss4sg_data = df[df['project_type'] == 'OSS4SG']
    
    results = []
    
    for metric in metrics:
        oss_values = oss_data[metric].dropna()
        oss4sg_values = oss4sg_data[metric].dropna()
        
        if len(oss_values) == 0 or len(oss4sg_values) == 0:
            continue
        
        # Mann-Whitney U test
        statistic, pvalue = stats.mannwhitneyu(
            oss_values, oss4sg_values, alternative='two-sided'
        )
        
        # Effect size
        cliff_d, magnitude = calculate_cliff_delta(oss_values, oss4sg_values)
        
        result = {
            'Metric': metric,
            'OSS_n': len(oss_values),
            'OSS4SG_n': len(oss4sg_values),
            'OSS_median': oss_values.median(),
            'OSS_Q1': oss_values.quantile(0.25),
            'OSS_Q3': oss_values.quantile(0.75),
            'OSS_mean': oss_values.mean(),
            'OSS_std': oss_values.std(),
            'OSS4SG_median': oss4sg_values.median(),
            'OSS4SG_Q1': oss4sg_values.quantile(0.25),
            'OSS4SG_Q3': oss4sg_values.quantile(0.75),
            'OSS4SG_mean': oss4sg_values.mean(),
            'OSS4SG_std': oss4sg_values.std(),
            'U_statistic': statistic,
            'p_value': pvalue,
            'cliff_delta': cliff_d,
            'effect_magnitude': magnitude,
            'significant': pvalue < 0.05
        }
        results.append(result)
    
    return pd.DataFrame(results)

def create_transition_plots(df, monthly_df):
    """
    Create comprehensive visualization of transition patterns
    """
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create a comprehensive figure
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Project-level transition rates comparison
    ax1 = plt.subplot(2, 4, 1)
    oss_rates = df[df['project_type'] == 'OSS']['avg_transition_rate']
    oss4sg_rates = df[df['project_type'] == 'OSS4SG']['avg_transition_rate']
    
    bp = ax1.boxplot([oss_rates, oss4sg_rates], 
                     labels=['OSS', 'OSS4SG'], 
                     patch_artist=True,
                     showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax1.set_ylabel('Average Monthly Transition Rate')
    ax1.set_title('A) Project-Level Transition Rates')
    
    # 2. New core frequency comparison
    ax2 = plt.subplot(2, 4, 2)
    oss_freq = df[df['project_type'] == 'OSS']['new_core_frequency']
    oss4sg_freq = df[df['project_type'] == 'OSS4SG']['new_core_frequency']
    
    bp2 = ax2.boxplot([oss_freq, oss4sg_freq], 
                      labels=['OSS', 'OSS4SG'], 
                      patch_artist=True,
                      showmeans=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightgreen')
    ax2.set_ylabel('Transition Frequency')
    ax2.set_title('B) Frequency of New Core Additions')
    
    # 3. Final ever-core count comparison
    ax3 = plt.subplot(2, 4, 3)
    oss_final = df[df['project_type'] == 'OSS']['final_ever_core_count']
    oss4sg_final = df[df['project_type'] == 'OSS4SG']['final_ever_core_count']
    
    bp3 = ax3.boxplot([oss_final, oss4sg_final], 
                      labels=['OSS', 'OSS4SG'], 
                      patch_artist=True,
                      showmeans=True)
    bp3['boxes'][0].set_facecolor('lightblue')
    bp3['boxes'][1].set_facecolor('lightgreen')
    ax3.set_ylabel('Final Core Count')
    ax3.set_title('C) Project Core Team Size')
    
    # 4. Distribution of transition rates
    ax4 = plt.subplot(2, 4, 4)
    ax4.hist(oss_rates, alpha=0.6, label='OSS', bins=20, color='lightblue')
    ax4.hist(oss4sg_rates, alpha=0.6, label='OSS4SG', bins=20, color='lightgreen')
    ax4.set_xlabel('Average Transition Rate')
    ax4.set_ylabel('Number of Projects')
    ax4.set_title('D) Distribution of Transition Rates')
    ax4.legend()
    
    # 5. Time series of monthly transitions (aggregated)
    ax5 = plt.subplot(2, 4, 5)
    if monthly_df is not None and len(monthly_df) > 0:
        monthly_agg = monthly_df.groupby(['month', 'project_type'])['transition_rate'].mean().reset_index()
        
        oss_monthly = monthly_agg[monthly_agg['project_type'] == 'OSS']
        oss4sg_monthly = monthly_agg[monthly_agg['project_type'] == 'OSS4SG']
        
        if len(oss_monthly) > 0:
            ax5.plot(range(len(oss_monthly)), oss_monthly['transition_rate'], 
                    'o-', alpha=0.7, label='OSS', color='blue')
        if len(oss4sg_monthly) > 0:
            ax5.plot(range(len(oss4sg_monthly)), oss4sg_monthly['transition_rate'], 
                    'o-', alpha=0.7, label='OSS4SG', color='green')
        
        ax5.set_xlabel('Time (months)')
        ax5.set_ylabel('Average Transition Rate')
        ax5.set_title('E) Transition Rates Over Time')
        ax5.legend()
    
    # 6. Scatter plot: project size vs transition rate
    ax6 = plt.subplot(2, 4, 6)
    oss_data = df[df['project_type'] == 'OSS']
    oss4sg_data = df[df['project_type'] == 'OSS4SG']
    
    ax6.scatter(oss_data['final_ever_core_count'], oss_data['avg_transition_rate'], 
               alpha=0.6, label='OSS', color='blue')
    ax6.scatter(oss4sg_data['final_ever_core_count'], oss4sg_data['avg_transition_rate'], 
               alpha=0.6, label='OSS4SG', color='green')
    ax6.set_xlabel('Final Core Count')
    ax6.set_ylabel('Average Transition Rate')
    ax6.set_title('F) Core Size vs Transition Rate')
    ax6.legend()
    
    # 7. Violin plot for transition rates
    ax7 = plt.subplot(2, 4, 7)
    data_for_violin = [oss_rates, oss4sg_rates]
    parts = ax7.violinplot(data_for_violin, positions=[1, 2], showmeans=True, showmedians=True)
    ax7.set_xticks([1, 2])
    ax7.set_xticklabels(['OSS', 'OSS4SG'])
    ax7.set_ylabel('Average Transition Rate')
    ax7.set_title('G) Transition Rate Distributions')
    
    # 8. Summary statistics text
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Calculate summary stats
    oss_median = oss_rates.median()
    oss4sg_median = oss4sg_rates.median()
    oss_n = len(oss_rates)
    oss4sg_n = len(oss4sg_rates)
    
    summary_text = f"""
H) Summary Statistics

Projects Analyzed:
OSS: {oss_n}
OSS4SG: {oss4sg_n}

Median Transition Rate:
OSS: {oss_median:.4f}
OSS4SG: {oss4sg_median:.4f}

Difference: {((oss4sg_median - oss_median) / oss_median * 100):.1f}%

Median New Core Frequency:
OSS: {oss_freq.median():.3f}
OSS4SG: {oss4sg_freq.median():.3f}
"""
    
    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('newcomer_transition_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('newcomer_transition_analysis.pdf', bbox_inches='tight')
    plt.show()

def create_summary_table(test_results):
    """
    Create a formatted summary table for the paper
    """
    summary = test_results.copy()
    
    # Format the table
    summary_formatted = summary.copy()
    
    # Format medians with IQR
    summary_formatted['OSS'] = summary.apply(
        lambda x: f"{x['OSS_median']:.4f} [{x['OSS_Q1']:.4f}-{x['OSS_Q3']:.4f}]", axis=1
    )
    summary_formatted['OSS4SG'] = summary.apply(
        lambda x: f"{x['OSS4SG_median']:.4f} [{x['OSS4SG_Q1']:.4f}-{x['OSS4SG_Q3']:.4f}]", axis=1
    )
    
    # Format p-values
    summary_formatted['p-value'] = summary['p_value'].apply(
        lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001"
    )
    
    # Format effect size
    summary_formatted['Effect Size'] = summary.apply(
        lambda x: f"{x['cliff_delta']:.3f} ({x['effect_magnitude']})", axis=1
    )
    
    # Select final columns
    final_table = summary_formatted[['Metric', 'OSS', 'OSS4SG', 'p-value', 'Effect Size']]
    
    return final_table

def main():
    print("="*70)
    print("STEP 4B: STATISTICAL ANALYSIS OF TRANSITION RATES")
    print("="*70)
    
    # Load the results from the results directory
    results_dir = 'transition_analysis_results'
    project_file = f'{results_dir}/project_transition_summaries.csv'
    monthly_file = f'{results_dir}/monthly_transition_rates.csv'
    
    if not os.path.exists(project_file):
        print(f"ERROR: {project_file} not found")
        print("Please run calculate_transition_rates.py first")
        return
    
    df = pd.read_csv(project_file)
    
    # Try to load monthly data
    monthly_df = None
    if os.path.exists(monthly_file):
        monthly_df = pd.read_csv(monthly_file)
    
    print(f"Loaded data for {len(df)} projects")
    print(f"OSS: {len(df[df['project_type'] == 'OSS'])}")
    print(f"OSS4SG: {len(df[df['project_type'] == 'OSS4SG'])}")
    
    # Define metrics to compare
    metrics = [
        'avg_transition_rate',
        'median_transition_rate', 
        'new_core_frequency',
        'final_ever_core_count',
        'total_truly_new_core_added'
    ]
    
    metric_labels = [
        'Average Monthly Transition Rate',
        'Median Monthly Transition Rate',
        'New Core Frequency',
        'Final Ever-Core Count',
        'Total Truly New Core Added'
    ]
    
    # Perform statistical tests
    print("\nPerforming statistical tests...")
    test_results = perform_statistical_tests(df, metrics)
    
    # Create summary table
    print("\nSUMMARY TABLE FOR PAPER:")
    summary_table = create_summary_table(test_results)
    print(summary_table.to_string(index=False))
    
    # Save results to the results directory
    test_results.to_csv(f'{results_dir}/transition_statistical_results.csv', index=False)
    summary_table.to_csv(f'{results_dir}/transition_summary_table.csv', index=False)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_transition_plots(df, monthly_df)
    
    # Print key findings
    print("\nKEY FINDINGS:")
    print("="*50)
    
    significant_count = 0
    for _, row in test_results.iterrows():
        if row['significant']:
            significant_count += 1
            direction = "higher" if row['OSS4SG_median'] > row['OSS_median'] else "lower"
            print(f"- {row['Metric']}: OSS4SG has significantly {direction} values "
                  f"(p={row['p_value']:.3f}, effect size={row['cliff_delta']:.3f} ({row['effect_magnitude']}))")
    
    if significant_count == 0:
        print("- No statistically significant differences found")
    else:
        print(f"Found {significant_count} significant differences out of {len(metrics)} metrics tested")
    
    print(f"\nFiles generated in {results_dir}/:")
    print("- newcomer_transition_analysis.png/pdf")
    print("- transition_statistical_results.csv")
    print("- transition_summary_table.csv")

if __name__ == "__main__":
    main()