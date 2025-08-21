#!/usr/bin/env python3
"""
Validation and Analysis Script for RQ2 Timeline Data
- Validates timeline creation process
- Generates comparison plots between OSS and OSS4SG
- Performs statistical analysis
- Creates paper-ready visualizations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
from scipy import stats
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Paths
BASE = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
TIMELINE_DIR = BASE / "RQ2_newcomer_treatment_patterns_test2" / "step2_timelines" / "from_cache_timelines"
OUTPUT_DIR = BASE / "paper_plots_and_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot styling
plt.style.use('default')
sns.set_palette("husl")

def validate_timeline_files() -> Dict:
    """Validate the timeline creation process and return summary stats."""
    print("🔍 Validating timeline files...")
    
    timeline_files = list(TIMELINE_DIR.glob("timeline_*.csv"))
    validation_results = {
        'total_files': len(timeline_files),
        'valid_files': 0,
        'empty_files': 0,
        'error_files': 0,
        'total_events': 0,
        'events_by_type': {'commit': 0, 'pull_request': 0, 'issue': 0},
        'projects_by_type': {'OSS': set(), 'OSS4SG': set()},
        'contributors_by_type': {'OSS': 0, 'OSS4SG': 0},
        'sample_issues': []
    }
    
    for file_path in tqdm(timeline_files[:100], desc="Validating files"):  # Sample first 100
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                validation_results['empty_files'] += 1
                continue
                
            validation_results['valid_files'] += 1
            validation_results['total_events'] += len(df)
            
            # Count events by type
            event_counts = df['event_type'].value_counts()
            for event_type, count in event_counts.items():
                if event_type in validation_results['events_by_type']:
                    validation_results['events_by_type'][event_type] += count
            
            # Track projects and contributors by type
            if not df.empty:
                project_type = df.iloc[0]['project_type']
                project_name = df.iloc[0]['project_name']
                
                if project_type in ['OSS', 'OSS4SG']:
                    validation_results['projects_by_type'][project_type].add(project_name)
                    validation_results['contributors_by_type'][project_type] += 1
                
                # Validate JSON data integrity (sample)
                if len(validation_results['sample_issues']) < 3:
                    pr_rows = df[df['event_type'] == 'pull_request']
                    if not pr_rows.empty:
                        try:
                            sample_json = json.loads(pr_rows.iloc[0]['event_data'])
                            validation_results['sample_issues'].append({
                                'type': 'PR_JSON_Valid',
                                'has_required_fields': all(k in sample_json for k in ['number', 'createdAt']),
                                'field_count': len(sample_json.keys())
                            })
                        except json.JSONDecodeError:
                            validation_results['sample_issues'].append({'type': 'PR_JSON_Invalid'})
                            
        except Exception as e:
            validation_results['error_files'] += 1
    
    # Convert sets to counts for JSON serialization
    for project_type in validation_results['projects_by_type']:
        validation_results['projects_by_type'][project_type] = len(validation_results['projects_by_type'][project_type])
    
    return validation_results

def load_all_timelines_summary() -> pd.DataFrame:
    """Load summary statistics from all timeline files."""
    print("📊 Loading timeline summaries...")
    
    summaries = []
    timeline_files = list(TIMELINE_DIR.glob("timeline_*.csv"))
    
    for file_path in tqdm(timeline_files, desc="Processing timelines"):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
                
            summary = {
                'contributor_email': df.iloc[0]['contributor_email'] if not df.empty else 'unknown',
                'project_name': df.iloc[0]['project_name'] if not df.empty else 'unknown',
                'project_type': df.iloc[0]['project_type'] if not df.empty else 'unknown',
                'username': df.iloc[0]['username'] if not df.empty else 'unknown',
                'total_events': len(df),
                'commits': len(df[df['event_type'] == 'commit']),
                'pull_requests': len(df[df['event_type'] == 'pull_request']),
                'issues': len(df[df['event_type'] == 'issue']),
                'pre_core_events': len(df[df['is_pre_core'] == True]) if 'is_pre_core' in df.columns else 0,
                'post_core_events': len(df[df['is_pre_core'] == False]) if 'is_pre_core' in df.columns else 0,
                'timeline_weeks': df['event_week'].max() - df['event_week'].min() + 1 if not df.empty else 0,
                'first_event_date': df.iloc[0]['event_timestamp'] if not df.empty else None,
                'last_event_date': df.iloc[-1]['event_timestamp'] if not df.empty else None,
            }
            summaries.append(summary)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return pd.DataFrame(summaries)

def create_interaction_comparison_plots(df: pd.DataFrame):
    """Create comprehensive interaction comparison plots."""
    print("📈 Creating interaction comparison plots...")
    
    # Filter valid data
    df_valid = df[(df['project_type'].isin(['OSS', 'OSS4SG'])) & (df['total_events'] > 0)].copy()
    
    # 1. Overall Interaction Distribution (Three Lines)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Distribution by project type
    all_interactions = df_valid['total_events']
    oss_interactions = df_valid[df_valid['project_type'] == 'OSS']['total_events']
    oss4sg_interactions = df_valid[df_valid['project_type'] == 'OSS4SG']['total_events']
    
    bins = np.logspace(0, np.log10(max(all_interactions)), 50)
    
    ax1.hist(all_interactions, bins=bins, alpha=0.7, label='All Contributors', color='gray', density=True)
    ax1.hist(oss_interactions, bins=bins, alpha=0.7, label='OSS', color='#2E86AB', density=True)
    ax1.hist(oss4sg_interactions, bins=bins, alpha=0.7, label='OSS4SG', color='#A23B72', density=True)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Total Interactions (log scale)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Total Interactions per Contributor')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Violin plots for event types
    event_types = ['commits', 'pull_requests', 'issues']
    violin_data = []
    
    for event_type in event_types:
        for project_type in ['OSS', 'OSS4SG']:
            data = df_valid[df_valid['project_type'] == project_type][event_type]
            violin_data.extend([(event_type, project_type, val) for val in data])
    
    violin_df = pd.DataFrame(violin_data, columns=['Event_Type', 'Project_Type', 'Count'])
    
    sns.violinplot(data=violin_df, x='Event_Type', y='Count', hue='Project_Type', 
                   palette=['#2E86AB', '#A23B72'], ax=ax2, inner='quartile')
    ax2.set_yscale('log')
    ax2.set_title('Distribution of Event Types by Project Type')
    ax2.set_ylabel('Count (log scale)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'interaction_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'interaction_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    metrics = ['total_events', 'commits', 'pull_requests', 'issues']
    titles = ['Total Interactions', 'Commits', 'Pull Requests', 'Issues']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sns.boxplot(data=df_valid, x='project_type', y=metric, palette=['#2E86AB', '#A23B72'], ax=axes[i])
        axes[i].set_yscale('log')
        axes[i].set_title(f'{title} by Project Type')
        axes[i].set_xlabel('Project Type')
        axes[i].set_ylabel(f'{title} (log scale)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'interaction_boxplots.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'interaction_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Interactive Plotly visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Events', 'Commits', 'Pull Requests', 'Issues'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = {'OSS': '#2E86AB', 'OSS4SG': '#A23B72'}
    
    for i, metric in enumerate(metrics):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        for project_type in ['OSS', 'OSS4SG']:
            data = df_valid[df_valid['project_type'] == project_type][metric]
            fig.add_trace(
                go.Violin(
                    y=data,
                    name=project_type,
                    legendgroup=project_type,
                    showlegend=(i == 0),
                    line_color=colors[project_type],
                    fillcolor=colors[project_type],
                    opacity=0.7
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title_text="Interactive Interaction Comparison",
        height=600,
        showlegend=True
    )
    
    fig.write_html(OUTPUT_DIR / 'interactive_comparison.html')

def perform_statistical_analysis(df: pd.DataFrame) -> Dict:
    """Perform comprehensive statistical analysis."""
    print("📊 Performing statistical analysis...")
    
    df_valid = df[(df['project_type'].isin(['OSS', 'OSS4SG'])) & (df['total_events'] > 0)].copy()
    
    results = {}
    metrics = ['total_events', 'commits', 'pull_requests', 'issues', 'timeline_weeks']
    
    for metric in metrics:
        oss_data = df_valid[df_valid['project_type'] == 'OSS'][metric].dropna()
        oss4sg_data = df_valid[df_valid['project_type'] == 'OSS4SG'][metric].dropna()
        
        # Descriptive statistics
        oss_stats = {
            'count': len(oss_data),
            'mean': oss_data.mean(),
            'median': oss_data.median(),
            'std': oss_data.std(),
            'min': oss_data.min(),
            'max': oss_data.max(),
            'q25': oss_data.quantile(0.25),
            'q75': oss_data.quantile(0.75)
        }
        
        oss4sg_stats = {
            'count': len(oss4sg_data),
            'mean': oss4sg_data.mean(),
            'median': oss4sg_data.median(),
            'std': oss4sg_data.std(),
            'min': oss4sg_data.min(),
            'max': oss4sg_data.max(),
            'q25': oss4sg_data.quantile(0.25),
            'q75': oss4sg_data.quantile(0.75)
        }
        
        # Statistical tests
        mannwhitney_stat, mannwhitney_p = stats.mannwhitneyu(oss_data, oss4sg_data, alternative='two-sided')
        
        # Effect size (Cliff's delta)
        def cliffs_delta(x, y):
            n1, n2 = len(x), len(y)
            delta = np.mean([np.sign(xi - yi) for xi in x for yi in y])
            return delta
        
        cliff_delta = cliffs_delta(oss_data.values, oss4sg_data.values)
        
        results[metric] = {
            'OSS': oss_stats,
            'OSS4SG': oss4sg_stats,
            'mannwhitney_statistic': float(mannwhitney_stat),
            'mannwhitney_p_value': float(mannwhitney_p),
            'cliffs_delta': float(cliff_delta),
            'significant': mannwhitney_p < 0.05,
            'effect_size_interpretation': (
                'negligible' if abs(cliff_delta) < 0.147 else
                'small' if abs(cliff_delta) < 0.33 else
                'medium' if abs(cliff_delta) < 0.474 else 'large'
            )
        }
    
    return results

def create_summary_tables(stats_results: Dict, df: pd.DataFrame):
    """Create publication-ready summary tables."""
    print("📋 Creating summary tables...")
    
    # Create descriptive statistics table
    desc_data = []
    for metric, results in stats_results.items():
        desc_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'OSS_Count': results['OSS']['count'],
            'OSS_Median': f"{results['OSS']['median']:.1f}",
            'OSS_IQR': f"{results['OSS']['q25']:.1f}-{results['OSS']['q75']:.1f}",
            'OSS4SG_Count': results['OSS4SG']['count'],
            'OSS4SG_Median': f"{results['OSS4SG']['median']:.1f}",
            'OSS4SG_IQR': f"{results['OSS4SG']['q25']:.1f}-{results['OSS4SG']['q75']:.1f}",
            'P_Value': f"{results['mannwhitney_p_value']:.2e}" if results['mannwhitney_p_value'] < 0.001 else f"{results['mannwhitney_p_value']:.3f}",
            'Cliffs_Delta': f"{results['cliffs_delta']:.3f}",
            'Effect_Size': results['effect_size_interpretation'],
            'Significant': '***' if results['mannwhitney_p_value'] < 0.001 else 
                          '**' if results['mannwhitney_p_value'] < 0.01 else
                          '*' if results['mannwhitney_p_value'] < 0.05 else 'ns'
        })
    
    desc_df = pd.DataFrame(desc_data)
    desc_df.to_csv(OUTPUT_DIR / 'descriptive_statistics_table.csv', index=False)
    
    # Create project-level summary
    project_summary = df.groupby(['project_type']).agg({
        'contributor_email': 'count',
        'project_name': 'nunique',
        'total_events': ['mean', 'median', 'std'],
        'commits': ['mean', 'median'],
        'pull_requests': ['mean', 'median'],
        'issues': ['mean', 'median']
    }).round(2)
    
    project_summary.to_csv(OUTPUT_DIR / 'project_level_summary.csv')

def main():
    """Main execution function."""
    print("🚀 Starting RQ2 Timeline Validation and Analysis")
    print("=" * 60)
    
    # Step 1: Validate timeline files
    validation_results = validate_timeline_files()
    
    # Save validation results
    with open(OUTPUT_DIR / 'validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"✅ Validation completed:")
    print(f"   Total files: {validation_results['total_files']}")
    print(f"   Valid files: {validation_results['valid_files']}")
    print(f"   Empty files: {validation_results['empty_files']}")
    print(f"   Error files: {validation_results['error_files']}")
    print(f"   Total events: {validation_results['total_events']:,}")
    print(f"   Contributors - OSS: {validation_results['contributors_by_type']['OSS']}, OSS4SG: {validation_results['contributors_by_type']['OSS4SG']}")
    print()
    
    # Step 2: Load timeline summaries
    df_summary = load_all_timelines_summary()
    
    if df_summary.empty:
        print("❌ No valid timeline data found!")
        return
    
    print(f"📊 Loaded {len(df_summary)} contributor timelines")
    print(f"   OSS contributors: {len(df_summary[df_summary['project_type'] == 'OSS'])}")
    print(f"   OSS4SG contributors: {len(df_summary[df_summary['project_type'] == 'OSS4SG'])}")
    print()
    
    # Step 3: Create plots
    create_interaction_comparison_plots(df_summary)
    
    # Step 4: Statistical analysis
    stats_results = perform_statistical_analysis(df_summary)
    
    # Save statistical results
    with open(OUTPUT_DIR / 'statistical_analysis_results.json', 'w') as f:
        json.dump(stats_results, f, indent=2, default=str)
    
    # Step 5: Create summary tables
    create_summary_tables(stats_results, df_summary)
    
    # Save full summary data
    df_summary.to_csv(OUTPUT_DIR / 'timeline_summary_data.csv', index=False)
    
    print("✅ Analysis completed! Results saved to:", OUTPUT_DIR)
    print("\nGenerated files:")
    for file_path in sorted(OUTPUT_DIR.glob('*')):
        print(f"   📄 {file_path.name}")


if __name__ == '__main__':
    main()
