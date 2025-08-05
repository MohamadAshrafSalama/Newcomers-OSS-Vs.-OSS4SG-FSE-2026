#!/usr/bin/env python3
"""
Step 3B: Statistical Comparison of OSS vs OSS4SG
================================================
Compares community structure metrics between project types
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_cliff_delta(x, y):
    """
    Calculate Cliff's Delta effect size
    Small: |d| < 0.33
    Medium: |d| < 0.474  
    Large: |d| >= 0.474
    """
    nx = len(x)
    ny = len(y)
    
    # Count how many times x > y
    greater = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                greater += 1
    
    # Cliff's delta
    delta = (2 * greater / (nx * ny)) - 1
    
    # Interpret magnitude
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
    # Separate by project type
    oss_data = df[df['project_type'] == 'OSS']
    oss4sg_data = df[df['project_type'] == 'OSS4SG']
    
    results = []
    
    for metric in metrics:
        # Get values
        oss_values = oss_data[metric].dropna()
        oss4sg_values = oss4sg_data[metric].dropna()
        
        # Mann-Whitney U test
        statistic, pvalue = stats.mannwhitneyu(
            oss_values, oss4sg_values, 
            alternative='two-sided'
        )
        
        # Effect size
        cliff_d, magnitude = calculate_cliff_delta(oss_values, oss4sg_values)
        
        # Summary statistics
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

def create_comparison_boxplots(df, metrics, metric_labels, test_results):
    """
    Create publication-ready box plots with significance indicators
    """
    # Set style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        # Prepare data for box plot
        oss_data = df[df['project_type'] == 'OSS'][metric].dropna()
        oss4sg_data = df[df['project_type'] == 'OSS4SG'][metric].dropna()
        
        # Create box plot
        bp = ax.boxplot(
            [oss_data, oss4sg_data],
            labels=['OSS', 'OSS4SG'],
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=6)
        )
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add title and labels
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        
        # Add sample sizes
        ax.text(1, ax.get_ylim()[1]*0.95, f'n={len(oss_data)}', 
                ha='center', va='top', fontsize=10)
        ax.text(2, ax.get_ylim()[1]*0.95, f'n={len(oss4sg_data)}', 
                ha='center', va='top', fontsize=10)
        
        # Add significance stars
        test_result = test_results[test_results['Metric'] == metric].iloc[0]
        if test_result['significant']:
            y_max = ax.get_ylim()[1]
            ax.plot([1, 2], [y_max*0.9, y_max*0.9], 'k-', linewidth=1)
            
            # Determine number of stars
            p_val = test_result['p_value']
            if p_val < 0.001:
                stars = '***'
            elif p_val < 0.01:
                stars = '**'
            else:
                stars = '*'
            
            ax.text(1.5, y_max*0.91, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Add effect size info
        effect_info = f"Effect: {test_result['cliff_delta']:.3f} ({test_result['effect_magnitude']})"
        ax.text(0.02, 0.98, effect_info, transform=ax.transAxes, 
                ha='left', va='top', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
    # Remove empty subplot
    if len(metrics) < 6:
        fig.delaxes(axes[-1])
    
    # Add overall title
    fig.suptitle('Community Structure Comparison: OSS vs OSS4SG Projects', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig

def create_violin_plots(df, metrics, metric_labels):
    """
    Create violin plots with individual data points
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        # Create violin plot
        sns.violinplot(data=df, x='project_type', y=metric, ax=ax, 
                      palette=['lightblue', 'lightgreen'], inner='box')
        
        # Add individual points with jitter
        sns.stripplot(data=df, x='project_type', y=metric, ax=ax, 
                     size=3, alpha=0.3, color='black', jitter=True)
        
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xlabel('Project Type', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        
    # Remove empty subplot
    if len(metrics) < 6:
        fig.delaxes(axes[-1])
    
    fig.suptitle('Distribution Shapes: OSS vs OSS4SG Projects', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig

def create_summary_table(test_results):
    """
    Create a formatted summary table for the paper
    """
    # Select key columns and format
    summary = test_results[['Metric', 'OSS_median', 'OSS_Q1', 'OSS_Q3', 
                           'OSS4SG_median', 'OSS4SG_Q1', 'OSS4SG_Q3', 
                           'p_value', 'cliff_delta', 'effect_magnitude']]
    
    # Format the table
    summary_formatted = summary.copy()
    
    # Format medians with IQR
    summary_formatted['OSS'] = summary.apply(
        lambda x: f"{x['OSS_median']:.3f} [{x['OSS_Q1']:.3f}-{x['OSS_Q3']:.3f}]", 
        axis=1
    )
    summary_formatted['OSS4SG'] = summary.apply(
        lambda x: f"{x['OSS4SG_median']:.3f} [{x['OSS4SG_Q1']:.3f}-{x['OSS4SG_Q3']:.3f}]", 
        axis=1
    )
    
    # Format p-values
    summary_formatted['p-value'] = summary['p_value'].apply(
        lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001"
    )
    
    # Format effect size
    summary_formatted['Effect Size'] = summary.apply(
        lambda x: f"{x['cliff_delta']:.3f} ({x['effect_magnitude']})", 
        axis=1
    )
    
    # Select final columns
    final_table = summary_formatted[['Metric', 'OSS', 'OSS4SG', 'p-value', 'Effect Size']]
    
    return final_table

def create_results_folder(test_results, summary_table, fig_box, fig_violin, df):
    """
    Create showing_plotting_results folder with all outputs
    """
    import os
    
    # Create folder
    results_folder = "showing_plotting_results"
    os.makedirs(results_folder, exist_ok=True)
    
    # Save statistical results
    test_results.to_csv(f'{results_folder}/statistical_test_results.csv', index=False)
    summary_table.to_csv(f'{results_folder}/summary_table_for_paper.csv', index=False)
    
    # Save figures
    fig_box.savefig(f'{results_folder}/community_structure_boxplots.png', dpi=300, bbox_inches='tight')
    fig_box.savefig(f'{results_folder}/community_structure_boxplots.pdf', bbox_inches='tight')
    
    fig_violin.savefig(f'{results_folder}/community_structure_violins.png', dpi=300, bbox_inches='tight')
    fig_violin.savefig(f'{results_folder}/community_structure_violins.pdf', bbox_inches='tight')
    
    # Create analysis summary
    with open(f'{results_folder}/analysis_summary.txt', 'w') as f:
        f.write("COMMUNITY STRUCTURE ANALYSIS SUMMARY\\n")
        f.write("="*50 + "\\n\\n")
        
        # Dataset info
        f.write(f"Dataset: {len(df)} projects\\n")
        f.write(f"OSS Projects: {len(df[df['project_type'] == 'OSS'])}\\n")
        f.write(f"OSS4SG Projects: {len(df[df['project_type'] == 'OSS4SG'])}\\n\\n")
        
        # Significant findings
        f.write("SIGNIFICANT DIFFERENCES:\\n")
        f.write("-" * 30 + "\\n")
        
        significant_results = test_results[test_results['significant']]
        for _, row in significant_results.iterrows():
            direction = "higher" if row['OSS4SG_median'] > row['OSS_median'] else "lower"
            f.write(f"- {row['Metric']}: OSS4SG has significantly {direction} values\\n")
            f.write(f"  p-value: {row['p_value']:.3f}, Effect size: {row['cliff_delta']:.3f} ({row['effect_magnitude']})\\n\\n")
        
        # Non-significant findings
        non_sig = test_results[~test_results['significant']]
        if len(non_sig) > 0:
            f.write("NON-SIGNIFICANT DIFFERENCES:\\n")
            f.write("-" * 30 + "\\n")
            for _, row in non_sig.iterrows():
                f.write(f"- {row['Metric']}: p-value: {row['p_value']:.3f}\\n")
    
    # Create README
    with open(f'{results_folder}/README.md', 'w') as f:
        f.write("# Community Structure Analysis Results\\n\\n")
        f.write("This folder contains all outputs from the statistical comparison of OSS vs OSS4SG community structures.\\n\\n")
        f.write("## Files:\\n\\n")
        f.write("### Data Files:\\n")
        f.write("- `statistical_test_results.csv`: Complete statistical test results\\n")
        f.write("- `summary_table_for_paper.csv`: Publication-ready summary table\\n\\n")
        f.write("### Visualizations:\\n")
        f.write("- `community_structure_boxplots.png/pdf`: Box plot comparison\\n")
        f.write("- `community_structure_violins.png/pdf`: Violin plot with data points\\n\\n")
        f.write("### Analysis Summary:\\n")
        f.write("- `analysis_summary.txt`: Key findings summary\\n")
        f.write("- `README.md`: This file\\n\\n")
        f.write("## Methodology:\\n")
        f.write("- Statistical Test: Mann-Whitney U (non-parametric)\\n")
        f.write("- Effect Size: Cliff's Delta\\n")
        f.write("- Primary Metric: Median with IQR\\n")
        f.write("- Significance Level: α = 0.05\\n")
    
    print(f"\\n📁 Results saved to: {results_folder}/")
    return results_folder

def main():
    """
    Main analysis function
    """
    print("="*60)
    print("STEP 3B: STATISTICAL COMPARISON OF OSS vs OSS4SG")
    print("="*60)
    
    # Load the project metrics
    df = pd.read_csv('project_metrics.csv')
    print(f"Loaded metrics for {len(df)} projects")
    print(f"OSS: {len(df[df['project_type'] == 'OSS'])}")
    print(f"OSS4SG: {len(df[df['project_type'] == 'OSS4SG'])}")
    
    # Define metrics to compare
    metrics = [
        'core_ratio',
        'one_time_ratio', 
        'gini_coefficient',
        'bus_factor',
        'active_ratio'
    ]
    
    metric_labels = [
        'Core Contributor Ratio',
        'One-Time Contributor Ratio',
        'Gini Coefficient',
        'Bus Factor',
        'Active Contributor Ratio'
    ]
    
    # Perform statistical tests
    print("\\n📊 PERFORMING STATISTICAL TESTS...")
    test_results = perform_statistical_tests(df, metrics)
    
    # Create summary table
    print("\\n📋 SUMMARY TABLE FOR PAPER:")
    summary_table = create_summary_table(test_results)
    print(summary_table.to_string(index=False))
    
    # Create visualizations
    print("\\n📊 CREATING VISUALIZATIONS...")
    
    # Convert ratios to percentages for plotting
    df_plot = df.copy()
    for col in ['core_ratio', 'one_time_ratio', 'active_ratio']:
        df_plot[col] = df_plot[col] * 100
    
    # Update labels for percentage metrics
    metric_labels_plot = [
        'Core Contributor Ratio (%)',
        'One-Time Contributor Ratio (%)',
        'Gini Coefficient',
        'Bus Factor',
        'Active Contributor Ratio (%)'
    ]
    
    fig_box = create_comparison_boxplots(df_plot, metrics, metric_labels_plot, test_results)
    fig_violin = create_violin_plots(df_plot, metrics, metric_labels_plot)
    
    # Create results folder with all outputs
    results_folder = create_results_folder(test_results, summary_table, fig_box, fig_violin, df)
    
    # Print key findings
    print("\\n🔍 KEY FINDINGS:")
    significant_count = test_results['significant'].sum()
    print(f"Found {significant_count} significant differences out of {len(metrics)} metrics tested")
    
    for _, row in test_results.iterrows():
        if row['significant']:
            direction = "higher" if row['OSS4SG_median'] > row['OSS_median'] else "lower"
            print(f"- {row['Metric']}: OSS4SG has significantly {direction} values "
                  f"(p={row['p_value']:.3f}, effect size={row['effect_magnitude']})")
    
    print("\\n✅ ANALYSIS COMPLETE!")
    print("Generated files in:", results_folder)
    
    # Close figures to prevent display issues
    plt.close('all')
    
    return test_results, summary_table

if __name__ == "__main__":
    test_results, summary_table = main()