#!/usr/bin/env python3
"""
Create violin plots showing distribution of contributor activity
for all contributors, OSS only, and OSS4SG only.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Load the timeline summary data
print("📊 Loading timeline summary data...")
df = pd.read_csv('results/timeline_summary_data_validated.csv')
print(f"✅ Loaded {len(df)} contributors")

# Create violin plots
print("🎨 Creating violin plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Distribution of Contributor Activity (Number of Events)', fontsize=16, fontweight='bold')

# Plot 1: All contributors
sns.violinplot(data=df, y='total_events', ax=axes[0], color='lightblue')
axes[0].set_title('All Contributors (n=6,530)', fontweight='bold')
axes[0].set_ylabel('Number of Events')
axes[0].set_xlabel('')
axes[0].grid(True, alpha=0.3)

# Plot 2: OSS only
oss_data = df[df['project_type'] == 'OSS']
sns.violinplot(data=oss_data, y='total_events', ax=axes[1], color='skyblue')
axes[1].set_title(f'OSS Contributors (n={len(oss_data)})', fontweight='bold')
axes[1].set_ylabel('Number of Events')
axes[1].set_xlabel('')
axes[1].grid(True, alpha=0.3)

# Plot 3: OSS4SG only
oss4sg_data = df[df['project_type'] == 'OSS4SG']
sns.violinplot(data=oss4sg_data, y='total_events', ax=axes[2], color='lightcoral')
axes[2].set_title(f'OSS4SG Contributors (n={len(oss4sg_data)})', fontweight='bold')
axes[2].set_ylabel('Number of Events')
axes[2].set_xlabel('')
axes[2].grid(True, alpha=0.3)

# Add statistics as text
for i, (data, title) in enumerate([(df, 'All'), (oss_data, 'OSS'), (oss4sg_data, 'OSS4SG')]):
    mean_val = data['total_events'].mean()
    median_val = data['total_events'].median()
    std_val = data['total_events'].std()
    
    stats_text = f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}\nStd: {std_val:.1f}'
    axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save the plot
output_file = 'results/contributor_activity_violin_plots.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Violin plots saved to: {output_file}")

# Print statistics
print(f"\n📊 Activity Statistics:")
print(f"   All contributors: Mean={df['total_events'].mean():.1f}, Median={df['total_events'].median():.1f}")
print(f"   OSS: Mean={oss_data['total_events'].mean():.1f}, Median={oss_data['total_events'].median():.1f}")
print(f"   OSS4SG: Mean={oss4sg_data['total_events'].mean():.1f}, Median={oss4sg_data['total_events'].median():.1f}")

# Also create a summary table
summary_stats = pd.DataFrame({
    'Group': ['All Contributors', 'OSS', 'OSS4SG'],
    'Count': [len(df), len(oss_data), len(oss4sg_data)],
    'Mean_Events': [df['total_events'].mean(), oss_data['total_events'].mean(), oss4sg_data['total_events'].mean()],
    'Median_Events': [df['total_events'].median(), oss_data['total_events'].median(), oss4sg_data['total_events'].median()],
    'Std_Events': [df['total_events'].std(), oss_data['total_events'].std(), oss4sg_data['total_events'].std()],
    'Min_Events': [df['total_events'].min(), oss_data['total_events'].min(), oss4sg_data['total_events'].min()],
    'Max_Events': [df['total_events'].max(), oss_data['total_events'].max(), oss4sg_data['total_events'].max()]
})

summary_file = 'results/contributor_activity_summary_stats.csv'
summary_stats.to_csv(summary_file, index=False)
print(f"✅ Summary statistics saved to: {summary_file}")

print("\n🎯 Plot creation completed successfully!")
