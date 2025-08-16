#!/usr/bin/env python3
"""
COMPREHENSIVE RQ2 Timeline Validation with Outlier Analysis
- Validates timeline data integrity
- Performs statistical analysis with AND without outliers
- Generates box plots and violin plots
- Compares OSS vs OSS4SG with proper statistical tests

Author: Ultra-careful validation with outlier handling
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
from tqdm import tqdm
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("husl")

# CRITICAL: CORRECT PATHS - Output to LOCAL results folder
BASE = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
TIMELINE_DIR = BASE / "RQ2_newcomer_treatment_patterns_test2" / "step2_timelines" / "from_cache_timelines"
OUTPUT_DIR = BASE / "RQ2_newcomer_treatment_patterns_test2" / "step2_timelines" / "results"

def validate_environment() -> bool:
    """Validate that all required paths and files exist before proceeding."""
    print("🔍 Validating environment...")
    
    if not BASE.exists():
        print(f"❌ Base directory not found: {BASE}")
        return False
    
    if not TIMELINE_DIR.exists():
        print(f"❌ Timeline directory not found: {TIMELINE_DIR}")
        return False
    
    timeline_files = list(TIMELINE_DIR.glob("timeline_*.csv"))
    if len(timeline_files) == 0:
        print(f"❌ No timeline files found in: {TIMELINE_DIR}")
        return False
    
    print(f"✅ Found {len(timeline_files)} timeline files")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory ready: {OUTPUT_DIR}")
    
    return True

def safe_read_timeline(file_path: Path) -> Optional[pd.DataFrame]:
    """Safely read a timeline file with comprehensive error handling."""
    try:
        if not file_path.exists() or file_path.stat().st_size == 0:
            return None
            
        df = pd.read_csv(file_path)
        
        expected_columns = ['event_id', 'event_type', 'event_timestamp', 'event_week', 
                          'event_identifier', 'event_data', 'project_name', 'project_type', 
                          'contributor_email', 'username', 'is_pre_core']
        
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            return None
            
        return df
        
    except Exception:
        return None

def detect_outliers_iqr(data: pd.Series) -> Tuple[pd.Series, Dict]:
    """Detect outliers using IQR method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (data < lower_bound) | (data > upper_bound)
    
    outlier_info = {
        'method': 'IQR',
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'Q1': float(Q1),
        'Q3': float(Q3),
        'IQR': float(IQR),
        'n_outliers': int(outliers.sum()),
        'outlier_percentage': float(outliers.sum() / len(data) * 100),
        'outlier_indices': outliers.index[outliers].tolist()
    }
    
    return outliers, outlier_info

def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> Tuple[pd.Series, Dict]:
    """Detect outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(data))
    outliers = z_scores > threshold
    
    outlier_info = {
        'method': 'Z-score',
        'threshold': threshold,
        'n_outliers': int(outliers.sum()),
        'outlier_percentage': float(outliers.sum() / len(data) * 100),
        'outlier_indices': outliers.index[outliers].tolist(),
        'max_z_score': float(z_scores.max())
    }
    
    return outliers, outlier_info

def load_and_validate_timelines() -> Tuple[pd.DataFrame, Dict]:
    """Load and validate all timeline files."""
    print("📊 Loading and validating timeline files...")
    
    timeline_files = list(TIMELINE_DIR.glob("timeline_*.csv"))
    summaries = []
    validation_stats = {
        'total_files': len(timeline_files),
        'valid_files': 0,
        'empty_files': 0,
        'corrupted_files': 0,
        'project_type_counts': {'OSS': 0, 'OSS4SG': 0, 'UNKNOWN': 0, 'MISSING': 0},
        'total_events': 0,
        'event_type_counts': {'commit': 0, 'pull_request': 0, 'issue': 0},
        'temporal_issues': 0
    }
    
    for file_path in tqdm(timeline_files, desc="Processing timeline files"):
        df = safe_read_timeline(file_path)
        
        if df is None:
            validation_stats['corrupted_files'] += 1
            continue
            
        if len(df) == 0:
            validation_stats['empty_files'] += 1
            continue
            
        validation_stats['valid_files'] += 1
        validation_stats['total_events'] += len(df)
        
        try:
            # Extract summary information
            summary = {
                'file_name': file_path.name,
                'contributor_email': df.iloc[0]['contributor_email'] if len(df) > 0 else 'unknown',
                'project_name': df.iloc[0]['project_name'] if len(df) > 0 else 'unknown',
                'project_type': df.iloc[0]['project_type'] if len(df) > 0 else 'MISSING',
                'username': df.iloc[0]['username'] if len(df) > 0 else 'unknown',
                'total_events': len(df),
                'commits': len(df[df['event_type'] == 'commit']) if 'event_type' in df.columns else 0,
                'pull_requests': len(df[df['event_type'] == 'pull_request']) if 'event_type' in df.columns else 0,
                'issues': len(df[df['event_type'] == 'issue']) if 'event_type' in df.columns else 0,
                'pre_core_events': len(df[df['is_pre_core'] == True]) if 'is_pre_core' in df.columns else 0,
                'timeline_weeks': (df['event_week'].max() - df['event_week'].min() + 1) if 'event_week' in df.columns and len(df) > 0 else 0,
            }
            
            # Count project types
            ptype = summary['project_type']
            if ptype in validation_stats['project_type_counts']:
                validation_stats['project_type_counts'][ptype] += 1
            else:
                validation_stats['project_type_counts']['UNKNOWN'] += 1
            
            # Count event types
            if 'event_type' in df.columns:
                event_counts = df['event_type'].value_counts()
                for event_type, count in event_counts.items():
                    if event_type in validation_stats['event_type_counts']:
                        validation_stats['event_type_counts'][event_type] += count
            
            summaries.append(summary)
            
        except Exception as e:
            print(f"⚠️  Error processing {file_path.name}: {e}")
            validation_stats['corrupted_files'] += 1
            continue
    
    df_summary = pd.DataFrame(summaries)
    
    print(f"\n📋 VALIDATION SUMMARY:")
    print(f"   Total files: {validation_stats['total_files']}")
    print(f"   Valid files: {validation_stats['valid_files']}")
    print(f"   Empty files: {validation_stats['empty_files']}")
    print(f"   Corrupted files: {validation_stats['corrupted_files']}")
    print(f"   Project types: {validation_stats['project_type_counts']}")
    print(f"   Total events: {validation_stats['total_events']:,}")
    
    return df_summary, validation_stats

def perform_outlier_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Perform comprehensive outlier analysis."""
    print("🔍 Performing outlier analysis...")
    
    # Filter to valid data only
    valid_df = df[
        (df['project_type'].isin(['OSS', 'OSS4SG'])) & 
        (df['total_events'] > 0) & 
        (df['total_events'].notna())
    ].copy()
    
    if len(valid_df) == 0:
        print("❌ No valid data for outlier analysis!")
        return df, df, {}
    
    metrics = ['total_events', 'commits', 'pull_requests', 'issues', 'timeline_weeks']
    outlier_summary = {}
    
    # Create a copy for outlier removal
    df_no_outliers = valid_df.copy()
    all_outlier_indices = set()
    
    for metric in metrics:
        if metric not in valid_df.columns:
            continue
            
        # Get clean data for this metric
        metric_data = valid_df[valid_df[metric].notna() & (valid_df[metric] >= 0)]
        
        if len(metric_data) < 10:  # Need minimum data points
            continue
            
        print(f"   Analyzing outliers for {metric}...")
        
        # IQR method
        iqr_outliers, iqr_info = detect_outliers_iqr(metric_data[metric])
        
        # Z-score method
        zscore_outliers, zscore_info = detect_outliers_zscore(metric_data[metric])
        
        # Combine outlier detection methods (union)
        combined_outliers = iqr_outliers | zscore_outliers
        
        outlier_summary[metric] = {
            'original_count': len(metric_data),
            'iqr_outliers': iqr_info,
            'zscore_outliers': zscore_info,
            'combined_outliers': {
                'n_outliers': int(combined_outliers.sum()),
                'outlier_percentage': float(combined_outliers.sum() / len(metric_data) * 100)
            }
        }
        
        # Collect outlier indices for removal
        outlier_indices = metric_data.index[combined_outliers]
        all_outlier_indices.update(outlier_indices)
        
        print(f"     {metric}: {combined_outliers.sum()} outliers ({combined_outliers.sum()/len(metric_data)*100:.1f}%)")
    
    # Remove all outliers (any row that was an outlier in any metric)
    if all_outlier_indices:
        df_no_outliers = valid_df.drop(index=list(all_outlier_indices)).copy()
        print(f"   📊 Removed {len(all_outlier_indices)} rows with outliers in any metric")
        print(f"   📊 Clean dataset: {len(df_no_outliers)} rows ({len(df_no_outliers)/len(valid_df)*100:.1f}% retained)")
    
    return valid_df, df_no_outliers, outlier_summary

def create_comparison_plots(df_with_outliers: pd.DataFrame, df_no_outliers: pd.DataFrame):
    """Create comprehensive comparison plots."""
    print("📈 Creating comparison plots...")
    
    metrics = ['total_events', 'commits', 'pull_requests', 'issues', 'timeline_weeks']
    metric_labels = ['Total Events', 'Commits', 'Pull Requests', 'Issues', 'Timeline (Weeks)']
    
    # Create figure with subplots for box plots
    fig, axes = plt.subplots(2, 5, figsize=(20, 12))
    fig.suptitle('OSS vs OSS4SG Interaction Comparison: With and Without Outliers', fontsize=16, fontweight='bold')
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if metric not in df_with_outliers.columns:
            continue
            
        # Top row: With outliers
        ax_with = axes[0, i]
        data_with = []
        labels_with = []
        
        for ptype in ['OSS', 'OSS4SG']:
            data_subset = df_with_outliers[
                (df_with_outliers['project_type'] == ptype) & 
                (df_with_outliers[metric].notna()) & 
                (df_with_outliers[metric] >= 0)
            ][metric]
            
            if len(data_subset) > 0:
                data_with.append(data_subset.values)
                labels_with.append(f'{ptype}\n(n={len(data_subset)})')
        
        if len(data_with) == 2:
            bp_with = ax_with.boxplot(data_with, labels=labels_with, patch_artist=True)
            bp_with['boxes'][0].set_facecolor('#2E86AB')  # OSS
            bp_with['boxes'][1].set_facecolor('#A23B72')  # OSS4SG
            
        ax_with.set_title(f'{label}\n(With Outliers)', fontweight='bold')
        ax_with.set_yscale('log')
        ax_with.grid(True, alpha=0.3)
        
        # Bottom row: Without outliers
        ax_no = axes[1, i]
        data_no = []
        labels_no = []
        
        for ptype in ['OSS', 'OSS4SG']:
            data_subset = df_no_outliers[
                (df_no_outliers['project_type'] == ptype) & 
                (df_no_outliers[metric].notna()) & 
                (df_no_outliers[metric] >= 0)
            ][metric]
            
            if len(data_subset) > 0:
                data_no.append(data_subset.values)
                labels_no.append(f'{ptype}\n(n={len(data_subset)})')
        
        if len(data_no) == 2:
            bp_no = ax_no.boxplot(data_no, labels=labels_no, patch_artist=True)
            bp_no['boxes'][0].set_facecolor('#2E86AB')  # OSS
            bp_no['boxes'][1].set_facecolor('#A23B72')  # OSS4SG
            
        ax_no.set_title(f'{label}\n(Without Outliers)', fontweight='bold')
        ax_no.set_yscale('log')
        ax_no.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'interaction_comparison_boxplots.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'interaction_comparison_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create violin plots
    fig, axes = plt.subplots(2, 5, figsize=(20, 12))
    fig.suptitle('OSS vs OSS4SG Distribution Shapes: With and Without Outliers', fontsize=16, fontweight='bold')
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if metric not in df_with_outliers.columns:
            continue
            
        # Top row: With outliers
        ax_with = axes[0, i]
        plot_data_with = df_with_outliers[
            (df_with_outliers['project_type'].isin(['OSS', 'OSS4SG'])) & 
            (df_with_outliers[metric].notna()) & 
            (df_with_outliers[metric] > 0)  # Remove zeros for log scale
        ].copy()
        
        if len(plot_data_with) > 0:
            # Add small constant to avoid log(0)
            plot_data_with[metric] = plot_data_with[metric] + 1
            
            sns.violinplot(
                data=plot_data_with, 
                x='project_type', 
                y=metric, 
                palette=['#2E86AB', '#A23B72'], 
                ax=ax_with,
                inner='quartile'
            )
        
        ax_with.set_title(f'{label}\n(With Outliers)', fontweight='bold')
        ax_with.set_yscale('log')
        ax_with.set_xlabel('Project Type')
        ax_with.set_ylabel(f'{label} (log scale)')
        ax_with.grid(True, alpha=0.3)
        
        # Bottom row: Without outliers
        ax_no = axes[1, i]
        plot_data_no = df_no_outliers[
            (df_no_outliers['project_type'].isin(['OSS', 'OSS4SG'])) & 
            (df_no_outliers[metric].notna()) & 
            (df_no_outliers[metric] > 0)
        ].copy()
        
        if len(plot_data_no) > 0:
            plot_data_no[metric] = plot_data_no[metric] + 1
            
            sns.violinplot(
                data=plot_data_no, 
                x='project_type', 
                y=metric, 
                palette=['#2E86AB', '#A23B72'], 
                ax=ax_no,
                inner='quartile'
            )
        
        ax_no.set_title(f'{label}\n(Without Outliers)', fontweight='bold')
        ax_no.set_yscale('log')
        ax_no.set_xlabel('Project Type')
        ax_no.set_ylabel(f'{label} (log scale)')
        ax_no.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'interaction_comparison_violinplots.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'interaction_comparison_violinplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def perform_statistical_analysis(df_with_outliers: pd.DataFrame, df_no_outliers: pd.DataFrame) -> Dict:
    """Perform comprehensive statistical analysis with and without outliers."""
    print("📊 Performing statistical analysis...")
    
    metrics = ['total_events', 'commits', 'pull_requests', 'issues', 'timeline_weeks']
    results = {'with_outliers': {}, 'without_outliers': {}}
    
    for dataset_name, dataset in [('with_outliers', df_with_outliers), ('without_outliers', df_no_outliers)]:
        print(f"   Analyzing {dataset_name}...")
        
        for metric in metrics:
            if metric not in dataset.columns:
                continue
                
            # Get data for each project type
            oss_data = dataset[
                (dataset['project_type'] == 'OSS') & 
                (dataset[metric].notna()) & 
                (dataset[metric] >= 0)
            ][metric]
            
            oss4sg_data = dataset[
                (dataset['project_type'] == 'OSS4SG') & 
                (dataset[metric].notna()) & 
                (dataset[metric] >= 0)
            ][metric]
            
            if len(oss_data) < 3 or len(oss4sg_data) < 3:
                continue
                
            try:
                # Descriptive statistics
                oss_stats = {
                    'count': len(oss_data),
                    'mean': float(oss_data.mean()),
                    'median': float(oss_data.median()),
                    'std': float(oss_data.std()),
                    'min': float(oss_data.min()),
                    'max': float(oss_data.max()),
                    'q25': float(oss_data.quantile(0.25)),
                    'q75': float(oss_data.quantile(0.75))
                }
                
                oss4sg_stats = {
                    'count': len(oss4sg_data),
                    'mean': float(oss4sg_data.mean()),
                    'median': float(oss4sg_data.median()),
                    'std': float(oss4sg_data.std()),
                    'min': float(oss4sg_data.min()),
                    'max': float(oss_data.max()),
                    'q25': float(oss4sg_data.quantile(0.25)),
                    'q75': float(oss4sg_data.quantile(0.75))
                }
                
                # Statistical tests
                mannwhitney_stat, mannwhitney_p = stats.mannwhitneyu(
                    oss_data, oss4sg_data, alternative='two-sided'
                )
                
                # Effect size (Cliff's delta)
                def safe_cliffs_delta(x, y):
                    try:
                        if len(x) * len(y) > 50000:  # Sample for performance
                            x_sample = np.random.choice(x, size=min(1000, len(x)), replace=False)
                            y_sample = np.random.choice(y, size=min(1000, len(y)), replace=False)
                            x, y = x_sample, y_sample
                        
                        delta = np.mean([np.sign(xi - yi) for xi in x for yi in y])
                        return float(delta)
                    except Exception:
                        return 0.0
                
                cliff_delta = safe_cliffs_delta(oss_data.values, oss4sg_data.values)
                
                results[dataset_name][metric] = {
                    'OSS': oss_stats,
                    'OSS4SG': oss4sg_stats,
                    'mannwhitney_statistic': float(mannwhitney_stat),
                    'mannwhitney_p_value': float(mannwhitney_p),
                    'cliffs_delta': cliff_delta,
                    'significant': mannwhitney_p < 0.05,
                    'effect_size_interpretation': (
                        'negligible' if abs(cliff_delta) < 0.147 else
                        'small' if abs(cliff_delta) < 0.33 else
                        'medium' if abs(cliff_delta) < 0.474 else 'large'
                    ),
                    'median_ratio': float(oss_stats['median'] / oss4sg_stats['median']) if oss4sg_stats['median'] > 0 else float('inf')
                }
                
            except Exception as e:
                print(f"     Error analyzing {metric}: {e}")
                continue
    
    return results

def create_summary_tables(stats_results: Dict, outlier_summary: Dict):
    """Create comprehensive summary tables."""
    print("📋 Creating summary tables...")
    
    # Statistical comparison table
    comparison_data = []
    
    for dataset_name in ['with_outliers', 'without_outliers']:
        if dataset_name not in stats_results:
            continue
            
        for metric, results in stats_results[dataset_name].items():
            comparison_data.append({
                'Dataset': 'With Outliers' if dataset_name == 'with_outliers' else 'Without Outliers',
                'Metric': metric.replace('_', ' ').title(),
                'OSS_Count': results['OSS']['count'],
                'OSS_Median': f"{results['OSS']['median']:.1f}",
                'OSS4SG_Count': results['OSS4SG']['count'],
                'OSS4SG_Median': f"{results['OSS4SG']['median']:.1f}",
                'Median_Ratio_OSS_to_OSS4SG': f"{results['median_ratio']:.2f}",
                'P_Value': f"{results['mannwhitney_p_value']:.2e}" if results['mannwhitney_p_value'] < 0.001 else f"{results['mannwhitney_p_value']:.4f}",
                'Cliffs_Delta': f"{results['cliffs_delta']:.3f}",
                'Effect_Size': results['effect_size_interpretation'],
                'Significant': '***' if results['mannwhitney_p_value'] < 0.001 else 
                              '**' if results['mannwhitney_p_value'] < 0.01 else
                              '*' if results['mannwhitney_p_value'] < 0.05 else 'ns'
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(OUTPUT_DIR / 'statistical_comparison_with_without_outliers.csv', index=False)
    
    # Outlier summary table
    outlier_data = []
    for metric, info in outlier_summary.items():
        outlier_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Original_Count': info['original_count'],
            'IQR_Outliers': info['iqr_outliers']['n_outliers'],
            'IQR_Percentage': f"{info['iqr_outliers']['outlier_percentage']:.1f}%",
            'ZScore_Outliers': info['zscore_outliers']['n_outliers'],
            'ZScore_Percentage': f"{info['zscore_outliers']['outlier_percentage']:.1f}%",
            'Combined_Outliers': info['combined_outliers']['n_outliers'],
            'Combined_Percentage': f"{info['combined_outliers']['outlier_percentage']:.1f}%"
        })
    
    outlier_df = pd.DataFrame(outlier_data)
    outlier_df.to_csv(OUTPUT_DIR / 'outlier_analysis_summary.csv', index=False)

def main():
    """Main execution with comprehensive analysis."""
    print("🚀 Starting COMPREHENSIVE RQ2 Timeline Analysis")
    print("=" * 80)
    
    # Step 1: Environment validation
    if not validate_environment():
        print("❌ Environment validation failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Load and validate data
    try:
        df_summary, validation_stats = load_and_validate_timelines()
        
        if df_summary.empty:
            print("❌ No valid timeline data loaded. Exiting.")
            sys.exit(1)
        
        # Save validation results
        with open(OUTPUT_DIR / 'validation_comprehensive.json', 'w') as f:
            json.dump(validation_stats, f, indent=2, default=str)
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        sys.exit(1)
    
    # Step 3: Outlier analysis
    try:
        df_with_outliers, df_no_outliers, outlier_summary = perform_outlier_analysis(df_summary)
        
        print(f"\n📊 OUTLIER ANALYSIS SUMMARY:")
        print(f"   Original dataset: {len(df_with_outliers)} contributors")
        print(f"   Clean dataset: {len(df_no_outliers)} contributors")
        print(f"   Retention rate: {len(df_no_outliers)/len(df_with_outliers)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Outlier analysis failed: {e}")
        df_with_outliers = df_summary
        df_no_outliers = df_summary
        outlier_summary = {}
    
    # Step 4: Create plots
    try:
        create_comparison_plots(df_with_outliers, df_no_outliers)
        print("✅ Comparison plots created successfully")
        
    except Exception as e:
        print(f"⚠️  Plot creation failed: {e}")
    
    # Step 5: Statistical analysis
    try:
        stats_results = perform_statistical_analysis(df_with_outliers, df_no_outliers)
        
        # Save statistical results
        with open(OUTPUT_DIR / 'comprehensive_statistical_results.json', 'w') as f:
            json.dump(stats_results, f, indent=2, default=str)
        
        print("\n📈 STATISTICAL ANALYSIS SUMMARY:")
        for dataset_name in ['with_outliers', 'without_outliers']:
            if dataset_name in stats_results:
                print(f"\n   {dataset_name.replace('_', ' ').title()}:")
                for metric, result in stats_results[dataset_name].items():
                    p_val = result.get('mannwhitney_p_value', 1.0)
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    ratio = result.get('median_ratio', 0)
                    print(f"     {metric}: OSS/OSS4SG ratio={ratio:.2f}, p={p_val:.4f} {significance}, effect={result.get('effect_size_interpretation', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Statistical analysis failed: {e}")
        stats_results = {}
    
    # Step 6: Create summary tables
    try:
        create_summary_tables(stats_results, outlier_summary)
        print("✅ Summary tables created successfully")
        
    except Exception as e:
        print(f"⚠️  Table creation failed: {e}")
    
    # Step 7: Save all data
    try:
        df_with_outliers.to_csv(OUTPUT_DIR / 'timeline_data_with_outliers.csv', index=False)
        df_no_outliers.to_csv(OUTPUT_DIR / 'timeline_data_without_outliers.csv', index=False)
        
        with open(OUTPUT_DIR / 'outlier_analysis_detailed.json', 'w') as f:
            json.dump(outlier_summary, f, indent=2, default=str)
        
    except Exception as e:
        print(f"⚠️  Data saving failed: {e}")
    
    print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETED!")
    print(f"📁 All results saved to: {OUTPUT_DIR}")
    
    # List generated files
    print(f"\n📄 Generated files:")
    for file_path in sorted(OUTPUT_DIR.glob('*')):
        if file_path.is_file():
            size_kb = file_path.stat().st_size / 1024
            print(f"   {file_path.name} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
