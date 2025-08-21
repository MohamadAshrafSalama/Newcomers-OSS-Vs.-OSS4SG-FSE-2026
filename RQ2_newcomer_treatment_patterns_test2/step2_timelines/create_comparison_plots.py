#!/usr/bin/env python3
"""
Create comparison plots: with and without outlier cleaning
to show the impact of outliers on the distributions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Load the timeline summary data
print("📊 Loading timeline summary data...")
df = pd.read_csv('results/timeline_summary_data_validated.csv')
print(f"✅ Loaded {len(df)} contributors")

# Function to remove outliers using IQR method
def remove_outliers_iqr(data, column, factor=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # Count outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    print(f"   Outliers removed: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
    
    # Return cleaned data
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Create comparison plots
print("🎨 Creating comparison plots...")

# Create figure with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Contributor Activity Distribution: With vs Without Outlier Cleaning', fontsize=16, fontweight='bold')

# Row 1: Original data (with outliers)
print("📈 Row 1: Original data (with outliers)")

# All contributors - original
sns.violinplot(data=df, y='total_events', ax=axes[0,0], color='lightblue')
axes[0,0].set_title(f'All Contributors (n={len(df)})', fontweight='bold')
axes[0,0].set_ylabel('Number of Events')
axes[0,0].set_xlabel('With Outliers')
axes[0,0].grid(True, alpha=0.3)

# OSS - original
oss_data = df[df['project_type'] == 'OSS']
sns.violinplot(data=oss_data, y='total_events', ax=axes[0,1], color='skyblue')
axes[0,1].set_title(f'OSS Contributors (n={len(oss_data)})', fontweight='bold')
axes[0,1].set_ylabel('Number of Events')
axes[0,1].set_xlabel('With Outliers')
axes[0,1].grid(True, alpha=0.3)

# OSS4SG - original
oss4sg_data = df[df['project_type'] == 'OSS4SG']
sns.violinplot(data=oss4sg_data, y='total_events', ax=axes[0,2], color='lightcoral')
axes[0,2].set_title(f'OSS4SG Contributors (n={len(oss4sg_data)})', fontweight='bold')
axes[0,2].set_ylabel('Number of Events')
axes[0,2].set_xlabel('With Outliers')
axes[0,2].grid(True, alpha=0.3)

# Row 2: Cleaned data (without outliers)
print("🧹 Row 2: Cleaned data (without outliers)")

# All contributors - cleaned
df_clean = remove_outliers_iqr(df, 'total_events')
sns.violinplot(data=df_clean, y='total_events', ax=axes[1,0], color='lightgreen')
axes[1,0].set_title(f'All Contributors - Cleaned (n={len(df_clean)})', fontweight='bold')
axes[1,0].set_ylabel('Number of Events')
axes[1,0].set_xlabel('Without Outliers')
axes[1,0].grid(True, alpha=0.3)

# OSS - cleaned
oss_clean = remove_outliers_iqr(oss_data, 'total_events')
sns.violinplot(data=oss_clean, y='total_events', ax=axes[1,1], color='lightgreen')
axes[1,1].set_title(f'OSS Contributors - Cleaned (n={len(oss_clean)})', fontweight='bold')
axes[1,1].set_ylabel('Number of Events')
axes[1,1].set_xlabel('Without Outliers')
axes[1,1].grid(True, alpha=0.3)

# OSS4SG - cleaned
oss4sg_clean = remove_outliers_iqr(oss4sg_data, 'total_events')
sns.violinplot(data=oss4sg_clean, y='total_events', ax=axes[1,2], color='lightgreen')
axes[1,2].set_title(f'OSS4SG Contributors - Cleaned (n={len(oss4sg_clean)})', fontweight='bold')
axes[1,2].set_ylabel('Number of Events')
axes[1,2].set_xlabel('Without Outliers')
axes[1,2].grid(True, alpha=0.3)

# Add statistics as text for each plot
datasets = [
    (df, 'All', axes[0,0], 'lightblue'),
    (oss_data, 'OSS', axes[0,1], 'skyblue'),
    (oss4sg_data, 'OSS4SG', axes[0,2], 'lightcoral'),
    (df_clean, 'All-Clean', axes[1,0], 'lightgreen'),
    (oss_clean, 'OSS-Clean', axes[1,1], 'lightgreen'),
    (oss4sg_clean, 'OSS4SG-Clean', axes[1,2], 'lightgreen')
]

for data, title, ax, color in datasets:
    mean_val = data['total_events'].mean()
    median_val = data['total_events'].median()
    std_val = data['total_events'].std()
    
    stats_text = f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}\nStd: {std_val:.1f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save the comparison plots
output_file = 'results/contributor_activity_comparison_plots.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Comparison plots saved to: {output_file}")

# Print comprehensive statistics
print(f"\n📊 COMPREHENSIVE ACTIVITY STATISTICS:")
print(f"{'='*60}")
print(f"ORIGINAL DATA (with outliers):")
print(f"   All contributors: Mean={df['total_events'].mean():.1f}, Median={df['total_events'].median():.1f}")
print(f"   OSS: Mean={oss_data['total_events'].mean():.1f}, Median={oss_data['total_events'].median():.1f}")
print(f"   OSS4SG: Mean={oss4sg_data['total_events'].mean():.1f}, Median={oss4sg_data['total_events'].median():.1f}")
print(f"\nCLEANED DATA (without outliers):")
print(f"   All contributors: Mean={df_clean['total_events'].mean():.1f}, Median={df_clean['total_events'].median():.1f}")
print(f"   OSS: Mean={oss_clean['total_events'].mean():.1f}, Median={oss_clean['total_events'].median():.1f}")
print(f"   OSS4SG: Mean={oss4sg_clean['total_events'].mean():.1f}, Median={oss4sg_clean['total_events'].median():.1f}")

# Create detailed comparison table
comparison_stats = pd.DataFrame({
    'Group': ['All-Original', 'OSS-Original', 'OSS4SG-Original', 'All-Cleaned', 'OSS-Cleaned', 'OSS4SG-Cleaned'],
    'Count': [len(df), len(oss_data), len(oss4sg_data), len(df_clean), len(oss_clean), len(oss4sg_clean)],
    'Mean_Events': [df['total_events'].mean(), oss_data['total_events'].mean(), oss4sg_data['total_events'].mean(),
                    df_clean['total_events'].mean(), oss_clean['total_events'].mean(), oss4sg_clean['total_events'].mean()],
    'Median_Events': [df['total_events'].median(), oss_data['total_events'].median(), oss4sg_data['total_events'].median(),
                      df_clean['total_events'].median(), oss_clean['total_events'].median(), oss4sg_clean['total_events'].median()],
    'Std_Events': [df['total_events'].std(), oss_data['total_events'].std(), oss4sg_data['total_events'].std(),
                   df_clean['total_events'].std(), oss_clean['total_events'].std(), oss4sg_clean['total_events'].std()],
    'Min_Events': [df['total_events'].min(), oss_data['total_events'].min(), oss4sg_data['total_events'].min(),
                   df_clean['total_events'].min(), oss_clean['total_events'].min(), oss4sg_clean['total_events'].min()],
    'Max_Events': [df['total_events'].max(), oss_data['total_events'].max(), oss4sg_data['total_events'].max(),
                   df_clean['total_events'].max(), oss_clean['total_events'].max(), oss4sg_clean['total_events'].max()]
})

comparison_file = 'results/contributor_activity_comparison_stats.csv'
comparison_stats.to_csv(comparison_file, index=False)
print(f"✅ Comparison statistics saved to: {comparison_file}")

print("\n🎯 Comparison plot creation completed successfully!")
