#!/usr/bin/env python3
"""
Create Visualizations for Transition Rate Analysis
================================================
Generate publication-ready plots for the corrected transition analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style
plt.style.use('default')
sns.set_palette("Set2")

def create_transition_rate_plots(results_dir):
    """
    Create comprehensive visualizations for transition rate analysis
    """
    print("Creating visualizations...")
    
    # Load the data
    monthly_file = f'{results_dir}/monthly_transitions.csv'
    results_file = f'{results_dir}/monthly_analysis_results.csv'
    
    monthly_df = pd.read_csv(monthly_file)
    results_df = pd.read_csv(results_file)
    
    # Filter for non-zero months
    non_zero_df = monthly_df[monthly_df['truly_new_core_count'] > 0]
    
    # Separate by project type
    oss_non_zero = non_zero_df[non_zero_df['project_type'] == 'OSS']
    oss4sg_non_zero = non_zero_df[non_zero_df['project_type'] == 'OSS4SG']
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Box plot of transition rates
    ax1 = plt.subplot(2, 3, 1)
    bp1 = ax1.boxplot([oss_non_zero['transition_rate'], oss4sg_non_zero['transition_rate']], 
                      labels=['OSS', 'OSS4SG'], 
                      patch_artist=True,
                      showmeans=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('lightgreen')
    ax1.set_ylabel('Transition Rate')
    ax1.set_title('A) Transition Rate Comparison\n(Non-Zero Months Only)')
    ax1.grid(True, alpha=0.3)
    
    # Add significance annotation
    oss_median = oss_non_zero['transition_rate'].median()
    oss4sg_median = oss4sg_non_zero['transition_rate'].median()
    ax1.text(0.5, 0.95, f'OSS Median: {oss_median:.4f}\nOSS4SG Median: {oss4sg_median:.4f}\nDifference: {((oss4sg_median - oss_median) / oss_median * 100):.1f}%', 
             transform=ax1.transAxes, ha='center', va='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. Violin plot
    ax2 = plt.subplot(2, 3, 2)
    data_for_violin = [oss_non_zero['transition_rate'], oss4sg_non_zero['transition_rate']]
    vp = ax2.violinplot(data_for_violin, positions=[1, 2], showmeans=True)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['OSS', 'OSS4SG'])
    ax2.set_ylabel('Transition Rate')
    ax2.set_title('B) Distribution of Transition Rates')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram comparison
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(oss_non_zero['transition_rate'], alpha=0.6, label='OSS', bins=30, color='lightblue', density=True)
    ax3.hist(oss4sg_non_zero['transition_rate'], alpha=0.6, label='OSS4SG', bins=30, color='lightgreen', density=True)
    ax3.set_xlabel('Transition Rate')
    ax3.set_ylabel('Density')
    ax3.set_title('C) Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Zero vs Non-Zero months comparison
    ax4 = plt.subplot(2, 3, 4)
    zero_counts = [len(monthly_df[monthly_df['project_type'] == 'OSS']) - len(oss_non_zero),
                   len(monthly_df[monthly_df['project_type'] == 'OSS4SG']) - len(oss4sg_non_zero)]
    non_zero_counts = [len(oss_non_zero), len(oss4sg_non_zero)]
    
    x = np.arange(2)
    width = 0.35
    
    ax4.bar(x - width/2, zero_counts, width, label='Zero Months', color='lightcoral', alpha=0.7)
    ax4.bar(x + width/2, non_zero_counts, width, label='Non-Zero Months', color='lightblue', alpha=0.7)
    
    ax4.set_xlabel('Project Type')
    ax4.set_ylabel('Number of Months')
    ax4.set_title('D) Zero vs Non-Zero Months')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['OSS', 'OSS4SG'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Transition rate over time (aggregated)
    ax5 = plt.subplot(2, 3, 5)
    monthly_agg = non_zero_df.groupby(['month', 'project_type'])['transition_rate'].mean().reset_index()
    
    oss_monthly = monthly_agg[monthly_agg['project_type'] == 'OSS']
    oss4sg_monthly = monthly_agg[monthly_agg['project_type'] == 'OSS4SG']
    
    if len(oss_monthly) > 0:
        ax5.plot(range(len(oss_monthly)), oss_monthly['transition_rate'], 
                'o-', alpha=0.7, label='OSS', color='blue', markersize=3)
    if len(oss4sg_monthly) > 0:
        ax5.plot(range(len(oss4sg_monthly)), oss4sg_monthly['transition_rate'], 
                'o-', alpha=0.7, label='OSS4SG', color='green', markersize=3)
    
    ax5.set_xlabel('Time (months)')
    ax5.set_ylabel('Average Transition Rate')
    ax5.set_title('E) Transition Rates Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Get results
    results = results_df.iloc[0]
    
    summary_text = f"""SUMMARY STATISTICS

OSS Projects: {len(oss_non_zero['project_name'].unique())}
OSS4SG Projects: {len(oss4sg_non_zero['project_name'].unique())}

Transition Rate Comparison:
OSS Median: {oss_median:.4f}
OSS4SG Median: {oss4sg_median:.4f}
Difference: {((oss4sg_median - oss_median) / oss_median * 100):.1f}%

Statistical Test:
P-value: {results['p_value']:.2e}
Effect Size: {results['effect_magnitude']}
Significant: {'YES' if results['significant'] else 'NO'}

Key Finding:
OSS4SG has significantly higher
transition rates when transitions
actually occur!
"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plots
    plots_dir = f'{results_dir}/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.savefig(f'{plots_dir}/transition_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{plots_dir}/transition_rate_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"Plots saved to: {plots_dir}/")
    
    # Create additional detailed plots
    create_detailed_plots(results_dir, non_zero_df)

def create_detailed_plots(results_dir, non_zero_df):
    """
    Create additional detailed plots
    """
    # Separate by project type
    oss_non_zero = non_zero_df[non_zero_df['project_type'] == 'OSS']
    oss4sg_non_zero = non_zero_df[non_zero_df['project_type'] == 'OSS4SG']
    
    # 1. Transition rate by project size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Transition rate vs existing core count
    ax1.scatter(oss_non_zero['existing_core_count'], oss_non_zero['transition_rate'], 
               alpha=0.6, label='OSS', color='blue', s=20)
    ax1.scatter(oss4sg_non_zero['existing_core_count'], oss4sg_non_zero['transition_rate'], 
               alpha=0.6, label='OSS4SG', color='green', s=20)
    ax1.set_xlabel('Existing Core Count')
    ax1.set_ylabel('Transition Rate')
    ax1.set_title('Transition Rate vs Existing Core Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Transition rate vs total commits
    ax2.scatter(oss_non_zero['total_commits_to_date'], oss_non_zero['transition_rate'], 
               alpha=0.6, label='OSS', color='blue', s=20)
    ax2.scatter(oss4sg_non_zero['total_commits_to_date'], oss4sg_non_zero['transition_rate'], 
               alpha=0.6, label='OSS4SG', color='green', s=20)
    ax2.set_xlabel('Total Commits to Date')
    ax2.set_ylabel('Transition Rate')
    ax2.set_title('Transition Rate vs Project Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plots_dir = f'{results_dir}/plots'
    plt.savefig(f'{plots_dir}/transition_rate_factors.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{plots_dir}/transition_rate_factors.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"Additional plots saved to: {plots_dir}/")

def main():
    print("="*70)
    print("CREATING VISUALIZATIONS FOR TRANSITION RATE ANALYSIS")
    print("="*70)
    
    results_dir = 'corrected_transition_results'
    
    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found at {results_dir}")
        print("Please run corrected_transition_analysis.py first")
        return
    
    create_transition_rate_plots(results_dir)
    
    print("\nVisualization complete!")
    print(f"All plots saved in: {results_dir}/plots/")

if __name__ == "__main__":
    main() 