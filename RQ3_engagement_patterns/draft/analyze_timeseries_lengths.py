import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_timeseries_lengths():
    """
    Analyze and visualize the distribution of time series lengths.
    """
    # Load the weekly timeseries data
    data_file = "../step1/results/rolling_4week/all_weekly_timeseries.csv"
    df = pd.read_csv(data_file)
    
    # Calculate the length of each contributor's time series
    contributor_weeks = df.groupby('contributor_id')['week_num'].max() + 1
    
    print("=== TIME SERIES LENGTH ANALYSIS ===")
    print(f"Total contributors: {len(contributor_weeks)}")
    print(f"Average length: {contributor_weeks.mean():.1f} weeks")
    print(f"Median length: {contributor_weeks.median():.1f} weeks")
    print(f"Min length: {contributor_weeks.min()} weeks")
    print(f"Max length: {contributor_weeks.max()} weeks")
    print(f"Standard deviation: {contributor_weeks.std():.1f} weeks")
    
    # Create detailed statistics
    print(f"\n=== PERCENTILES ===")
    print(f"25th percentile: {contributor_weeks.quantile(0.25):.1f} weeks")
    print(f"50th percentile: {contributor_weeks.quantile(0.50):.1f} weeks")
    print(f"75th percentile: {contributor_weeks.quantile(0.75):.1f} weeks")
    print(f"90th percentile: {contributor_weeks.quantile(0.90):.1f} weeks")
    print(f"95th percentile: {contributor_weeks.quantile(0.95):.1f} weeks")
    
    # Show distribution by project type
    project_info = df[['contributor_id', 'project_type']].drop_duplicates()
    contributor_data = pd.merge(contributor_weeks.reset_index(), project_info, on='contributor_id')
    contributor_data.columns = ['contributor_id', 'weeks', 'project_type']
    
    print(f"\n=== BY PROJECT TYPE ===")
    for ptype in contributor_data['project_type'].unique():
        subset = contributor_data[contributor_data['project_type'] == ptype]['weeks']
        print(f"{ptype}: {len(subset)} contributors, avg={subset.mean():.1f} weeks, median={subset.median():.1f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Time Series Length Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram of all lengths
    axes[0, 0].hist(contributor_weeks, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(contributor_weeks.mean(), color='red', linestyle='--', 
                       label=f'Mean: {contributor_weeks.mean():.1f}')
    axes[0, 0].axvline(contributor_weeks.median(), color='orange', linestyle='--', 
                       label=f'Median: {contributor_weeks.median():.1f}')
    axes[0, 0].set_xlabel('Time Series Length (weeks)')
    axes[0, 0].set_ylabel('Number of Contributors')
    axes[0, 0].set_title('Distribution of Time Series Lengths')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Log scale histogram for better view of distribution
    axes[0, 1].hist(contributor_weeks, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('Time Series Length (weeks)')
    axes[0, 1].set_ylabel('Number of Contributors (log scale)')
    axes[0, 1].set_title('Distribution (Log Scale)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot by project type
    project_types = contributor_data['project_type'].unique()
    data_by_type = [contributor_data[contributor_data['project_type'] == ptype]['weeks'] 
                    for ptype in project_types]
    
    axes[1, 0].boxplot(data_by_type, labels=project_types)
    axes[1, 0].set_ylabel('Time Series Length (weeks)')
    axes[1, 0].set_title('Length Distribution by Project Type')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_lengths = np.sort(contributor_weeks)
    cumulative_prob = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    
    axes[1, 1].plot(sorted_lengths, cumulative_prob, marker='o', markersize=3, linewidth=2)
    axes[1, 1].set_xlabel('Time Series Length (weeks)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution Function')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add percentile markers
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        val = contributor_weeks.quantile(p/100)
        axes[1, 1].axvline(val, color='red', linestyle=':', alpha=0.7)
        axes[1, 1].text(val, p/100, f'P{p}', rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "timeseries_length_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n=== PLOT SAVED ===")
    print(f"Saved visualization to: {output_file}")
    
    # Show detailed breakdown for short time series
    print(f"\n=== SHORT TIME SERIES BREAKDOWN ===")
    short_lengths = contributor_weeks[contributor_weeks <= 20].value_counts().sort_index()
    print("Weeks -> Count of Contributors")
    for weeks, count in short_lengths.items():
        print(f"{weeks:2d} weeks -> {count:2d} contributors")
    
    plt.show()
    
    return contributor_weeks, contributor_data

if __name__ == "__main__":
    lengths, data = analyze_timeseries_lengths()
