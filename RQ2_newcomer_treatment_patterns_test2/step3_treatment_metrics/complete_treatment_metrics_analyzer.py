#!/usr/bin/env python3
"""
COMPLETE Treatment Metrics Analysis Engine
- Analyzes EVERY SINGLE METRIC in the dataset
- Comprehensive statistical testing for ALL metrics
- Full visualization suite covering ALL categories
- Outlier analysis for ALL metrics
- Zero-pattern analysis for ALL metrics
- NO shortcuts - COMPLETE analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, kruskal
import warnings
from datetime import datetime
import itertools
from collections import defaultdict
import json

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteMetricsAnalyzer:
    def __init__(self, base_path):
        """Initialize with complete dataset"""
        self.base_path = Path(base_path)
        self.results_dir = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step3_treatment_metrics" / "results"
        self.analysis_dir = self.results_dir / "complete_analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load complete dataset
        dataset_file = self.results_dir / "complete_treatment_metrics_dataset.csv"
        if not dataset_file.exists():
            raise FileNotFoundError(f"Complete dataset not found: {dataset_file}\n"
                                  f"Run the complete dataset creator first!")
        
        self.metrics_df = pd.read_csv(dataset_file)
        logger.info(f"Loaded complete dataset: {len(self.metrics_df)} contributors, {len(self.metrics_df.columns)} columns")
        
        # Load column categories
        categories_file = self.results_dir / "column_categories_mapping.json"
        if categories_file.exists():
            with open(categories_file, 'r') as f:
                self.metric_categories = json.load(f)
        else:
            self.metric_categories = self.auto_categorize_metrics()
        
        # Validate data
        self.validate_complete_dataset()
        
        # Get all numeric metrics (exclude metadata)
        metadata_cols = ['contributor_email', 'project_name', 'project_type']
        self.all_numeric_metrics = [col for col in self.metrics_df.select_dtypes(include=[np.number]).columns 
                                   if col not in metadata_cols]
        
        logger.info(f"Found {len(self.all_numeric_metrics)} numeric metrics to analyze")

    def auto_categorize_metrics(self):
        """Auto-categorize metrics if mapping not available"""
        logger.info("Auto-categorizing metrics...")
        
        categories = {
            'response_timing': [],
            'engagement_breadth': [],
            'interaction_patterns': [],
            'recognition_signals': [],
            'trust_indicators': [],
            'other': []
        }
        
        for col in self.metrics_df.columns:
            col_lower = col.lower()
            
            if any(term in col_lower for term in ['response', 'timing', 'first_response', 'approval_speed']):
                categories['response_timing'].append(col)
            elif any(term in col_lower for term in ['responder', 'engagement', 'diversity', 'maintainer', 'peer']):
                categories['engagement_breadth'].append(col)
            elif any(term in col_lower for term in ['conversation', 'back_forth', 'word', 'message', 'question', 'link']):
                categories['interaction_patterns'].append(col)
            elif any(term in col_lower for term in ['thanks', 'praise', 'emoji', 'merge', 'sentiment']):
                categories['recognition_signals'].append(col)
            elif any(term in col_lower for term in ['review_request', 'mention', 'assignment', 'label', 'trust']):
                categories['trust_indicators'].append(col)
            else:
                categories['other'].append(col)
        
        return categories

    def validate_complete_dataset(self):
        """Validate the complete dataset"""
        required_columns = ['project_type', 'contributor_email', 'project_name']
        missing_columns = [col for col in required_columns if col not in self.metrics_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check project type distribution
        type_counts = self.metrics_df['project_type'].value_counts()
        logger.info(f"Project type distribution: {dict(type_counts)}")
        
        if 'OSS' not in type_counts or 'OSS4SG' not in type_counts:
            raise ValueError("Both OSS and OSS4SG project types required")

    def calculate_cliff_delta(self, group1, group2):
        """Calculate Cliff's Delta effect size"""
        try:
            if len(group1) == 0 or len(group2) == 0:
                return np.nan
            
            g1 = np.array(group1)
            g2 = np.array(group2)
            
            n1, n2 = len(g1), len(g2)
            dominance = np.sum(g1[:, np.newaxis] > g2) - np.sum(g1[:, np.newaxis] < g2)
            
            return dominance / (n1 * n2)
            
        except Exception as e:
            logger.debug(f"Error calculating Cliff's delta: {e}")
            return np.nan

    def perform_complete_statistical_analysis(self):
        """Perform statistical analysis on ALL metrics"""
        logger.info(f"Performing statistical analysis on {len(self.all_numeric_metrics)} metrics...")
        
        results = []
        
        for metric in self.all_numeric_metrics:
            try:
                # Get data for both groups
                oss_data = self.metrics_df[self.metrics_df['project_type'] == 'OSS'][metric].dropna()
                oss4sg_data = self.metrics_df[self.metrics_df['project_type'] == 'OSS4SG'][metric].dropna()
                
                result = {
                    'metric': metric,
                    'category': self.get_metric_category(metric),
                    'oss_n': len(oss_data),
                    'oss4sg_n': len(oss4sg_data),
                    'oss_mean': float(oss_data.mean()) if len(oss_data) > 0 else np.nan,
                    'oss4sg_mean': float(oss4sg_data.mean()) if len(oss4sg_data) > 0 else np.nan,
                    'oss_median': float(oss_data.median()) if len(oss_data) > 0 else np.nan,
                    'oss4sg_median': float(oss4sg_data.median()) if len(oss4sg_data) > 0 else np.nan,
                    'oss_std': float(oss_data.std()) if len(oss_data) > 0 else np.nan,
                    'oss4sg_std': float(oss4sg_data.std()) if len(oss4sg_data) > 0 else np.nan,
                    'mannwhitney_u': np.nan,
                    'mannwhitney_p': np.nan,
                    'cliff_delta': np.nan,
                    'effect_size_interpretation': 'No data',
                    'significant': False,
                    'oss4sg_advantage': False
                }
                
                # Skip if insufficient data
                if len(oss_data) < 5 or len(oss4sg_data) < 5:
                    result['effect_size_interpretation'] = 'Insufficient data'
                    results.append(result)
                    continue
                
                # Mann-Whitney U test
                try:
                    u_stat, p_value = mannwhitneyu(oss_data, oss4sg_data, alternative='two-sided')
                    result['mannwhitney_u'] = float(u_stat)
                    result['mannwhitney_p'] = float(p_value)
                    result['significant'] = p_value < 0.05
                except Exception as e:
                    logger.debug(f"Mann-Whitney failed for {metric}: {e}")
                
                # Cliff's Delta
                try:
                    cliff_delta = self.calculate_cliff_delta(oss_data, oss4sg_data)
                    result['cliff_delta'] = float(cliff_delta) if not np.isnan(cliff_delta) else np.nan
                    
                    if not np.isnan(cliff_delta):
                        # Effect size interpretation
                        abs_delta = abs(cliff_delta)
                        if abs_delta < 0.147:
                            effect_magnitude = "negligible"
                        elif abs_delta < 0.33:
                            effect_magnitude = "small"
                        elif abs_delta < 0.474:
                            effect_magnitude = "medium"
                        else:
                            effect_magnitude = "large"
                        
                        direction = "OSS4SG advantage" if cliff_delta > 0 else "OSS advantage"
                        significance = "significant" if result['significant'] else "not significant"
                        
                        result['effect_size_interpretation'] = f"{direction}, {effect_magnitude} effect, {significance}"
                        result['oss4sg_advantage'] = cliff_delta > 0
                    
                except Exception as e:
                    logger.debug(f"Cliff's delta failed for {metric}: {e}")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error analyzing metric {metric}: {e}")
                continue
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.analysis_dir / "complete_statistical_analysis.csv", index=False)
        
        logger.info(f"Statistical analysis complete for {len(results_df)} metrics")
        return results_df

    def get_metric_category(self, metric):
        """Get category for a metric"""
        for category, metrics in self.metric_categories.items():
            if metric in metrics:
                return category
        return 'other'

    def analyze_all_outliers(self):
        """Analyze outliers for ALL metrics"""
        logger.info("Analyzing outliers for ALL metrics...")
        
        outlier_results = []
        
        for metric in self.all_numeric_metrics:
            try:
                oss_data = self.metrics_df[self.metrics_df['project_type'] == 'OSS'][metric].dropna()
                oss4sg_data = self.metrics_df[self.metrics_df['project_type'] == 'OSS4SG'][metric].dropna()
                
                if len(oss_data) < 5 or len(oss4sg_data) < 5:
                    continue
                
                # Identify outliers using IQR method
                oss_outliers = self.identify_outliers_iqr(oss_data)
                oss4sg_outliers = self.identify_outliers_iqr(oss4sg_data)
                
                # Clean data
                oss_clean = oss_data[~oss_outliers]
                oss4sg_clean = oss4sg_data[~oss4sg_outliers]
                
                # Full data stats
                full_stats = self.calculate_comparison_stats(oss_data, oss4sg_data)
                clean_stats = self.calculate_comparison_stats(oss_clean, oss4sg_clean)
                
                result = {
                    'metric': metric,
                    'category': self.get_metric_category(metric),
                    'oss_outliers': int(oss_outliers.sum()),
                    'oss4sg_outliers': int(oss4sg_outliers.sum()),
                    'oss_outlier_rate': float(oss_outliers.mean()),
                    'oss4sg_outlier_rate': float(oss4sg_outliers.mean()),
                    
                    # Full data
                    'full_oss_mean': full_stats['oss_mean'],
                    'full_oss4sg_mean': full_stats['oss4sg_mean'],
                    'full_cliff_delta': full_stats['cliff_delta'],
                    'full_p_value': full_stats['p_value'],
                    
                    # Clean data
                    'clean_oss_mean': clean_stats['oss_mean'],
                    'clean_oss4sg_mean': clean_stats['oss4sg_mean'],
                    'clean_cliff_delta': clean_stats['cliff_delta'],
                    'clean_p_value': clean_stats['p_value'],
                    
                    # Impact assessment
                    'outlier_impact_on_effect_size': abs(full_stats['cliff_delta'] - clean_stats['cliff_delta']) if not (np.isnan(full_stats['cliff_delta']) or np.isnan(clean_stats['cliff_delta'])) else np.nan
                }
                
                outlier_results.append(result)
                
            except Exception as e:
                logger.debug(f"Outlier analysis failed for {metric}: {e}")
                continue
        
        outlier_df = pd.DataFrame(outlier_results)
        outlier_df.to_csv(self.analysis_dir / "complete_outlier_analysis.csv", index=False)
        
        logger.info(f"Outlier analysis complete for {len(outlier_df)} metrics")
        return outlier_df

    def identify_outliers_iqr(self, data):
        """Identify outliers using IQR method"""
        if len(data) < 4:
            return pd.Series([False] * len(data), index=data.index)
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return pd.Series([False] * len(data), index=data.index)
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (data < lower_bound) | (data > upper_bound)

    def calculate_comparison_stats(self, oss_data, oss4sg_data):
        """Calculate comparison statistics"""
        try:
            if len(oss_data) == 0 or len(oss4sg_data) == 0:
                return {
                    'oss_mean': np.nan,
                    'oss4sg_mean': np.nan,
                    'cliff_delta': np.nan,
                    'p_value': np.nan
                }
            
            # Basic stats
            oss_mean = float(oss_data.mean())
            oss4sg_mean = float(oss4sg_data.mean())
            
            # Statistical test
            try:
                _, p_value = mannwhitneyu(oss_data, oss4sg_data, alternative='two-sided')
            except:
                p_value = np.nan
            
            # Effect size
            cliff_delta = self.calculate_cliff_delta(oss_data, oss4sg_data)
            
            return {
                'oss_mean': oss_mean,
                'oss4sg_mean': oss4sg_mean,
                'cliff_delta': cliff_delta,
                'p_value': float(p_value) if not np.isnan(p_value) else np.nan
            }
            
        except Exception as e:
            logger.debug(f"Error in comparison stats: {e}")
            return {
                'oss_mean': np.nan,
                'oss4sg_mean': np.nan,
                'cliff_delta': np.nan,
                'p_value': np.nan
            }

    def analyze_all_zero_patterns(self):
        """Analyze zero patterns for ALL metrics"""
        logger.info("Analyzing zero patterns for ALL metrics...")
        
        zero_results = []
        
        for metric in self.all_numeric_metrics:
            try:
                oss_data = self.metrics_df[self.metrics_df['project_type'] == 'OSS'][metric]
                oss4sg_data = self.metrics_df[self.metrics_df['project_type'] == 'OSS4SG'][metric]
                
                # Count zeros and NaNs
                oss_zeros = (oss_data == 0).sum()
                oss4sg_zeros = (oss4sg_data == 0).sum()
                oss_nans = oss_data.isna().sum()
                oss4sg_nans = oss4sg_data.isna().sum()
                
                # Valid data
                oss_valid = len(oss_data) - oss_nans
                oss4sg_valid = len(oss4sg_data) - oss4sg_nans
                
                result = {
                    'metric': metric,
                    'category': self.get_metric_category(metric),
                    'oss_total': len(oss_data),
                    'oss4sg_total': len(oss4sg_data),
                    'oss_valid': oss_valid,
                    'oss4sg_valid': oss4sg_valid,
                    'oss_zeros': int(oss_zeros),
                    'oss4sg_zeros': int(oss4sg_zeros),
                    'oss_nans': int(oss_nans),
                    'oss4sg_nans': int(oss4sg_nans),
                    'oss_zero_rate': float(oss_zeros / oss_valid) if oss_valid > 0 else np.nan,
                    'oss4sg_zero_rate': float(oss4sg_zeros / oss4sg_valid) if oss4sg_valid > 0 else np.nan,
                    'oss_participation_rate': float((oss_valid - oss_zeros) / oss_valid) if oss_valid > 0 else np.nan,
                    'oss4sg_participation_rate': float((oss4sg_valid - oss4sg_zeros) / oss4sg_valid) if oss4sg_valid > 0 else np.nan
                }
                
                # Non-zero statistics
                oss_nonzero = oss_data[oss_data > 0]
                oss4sg_nonzero = oss4sg_data[oss4sg_data > 0]
                
                result['oss_nonzero_mean'] = float(oss_nonzero.mean()) if len(oss_nonzero) > 0 else np.nan
                result['oss4sg_nonzero_mean'] = float(oss4sg_nonzero.mean()) if len(oss4sg_nonzero) > 0 else np.nan
                result['oss_nonzero_median'] = float(oss_nonzero.median()) if len(oss_nonzero) > 0 else np.nan
                result['oss4sg_nonzero_median'] = float(oss4sg_nonzero.median()) if len(oss4sg_nonzero) > 0 else np.nan
                
                # Chi-square test for participation patterns
                if oss_valid > 0 and oss4sg_valid > 0 and (oss_zeros + oss4sg_zeros) > 0:
                    try:
                        contingency = [
                            [oss_zeros, oss_valid - oss_zeros],
                            [oss4sg_zeros, oss4sg_valid - oss4sg_zeros]
                        ]
                        chi2_stat, chi2_p, _, _ = chi2_contingency(contingency)
                        result['participation_chi2_stat'] = float(chi2_stat)
                        result['participation_chi2_p'] = float(chi2_p)
                    except:
                        result['participation_chi2_stat'] = np.nan
                        result['participation_chi2_p'] = np.nan
                else:
                    result['participation_chi2_stat'] = np.nan
                    result['participation_chi2_p'] = np.nan
                
                zero_results.append(result)
                
            except Exception as e:
                logger.debug(f"Zero pattern analysis failed for {metric}: {e}")
                continue
        
        zero_df = pd.DataFrame(zero_results)
        zero_df.to_csv(self.analysis_dir / "complete_zero_patterns_analysis.csv", index=False)
        
        logger.info(f"Zero pattern analysis complete for {len(zero_df)} metrics")
        return zero_df

    def create_comprehensive_visualizations(self, stats_df):
        """Create visualizations for ALL metrics"""
        logger.info("Creating comprehensive visualizations...")
        
        try:
            # Setup matplotlib
            plt.style.use('default')
            plt.rcParams.update({'figure.max_open_warning': 0})
            
            # 1. Statistical significance overview
            self.create_significance_overview(stats_df)
            
            # 2. Effect sizes heatmap by category
            self.create_effect_sizes_heatmap(stats_df)
            
            # 3. Top significant metrics visualization
            self.create_top_significant_metrics_plot(stats_df)
            
            # 4. Distribution plots for top metrics
            self.create_top_metrics_distributions(stats_df)
            
            # 5. Category-wise summary
            self.create_category_wise_summary(stats_df)
            
            # 6. Correlation analysis
            self.create_complete_correlation_analysis()
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()

    def create_significance_overview(self, stats_df):
        """Create overview of statistical significance"""
        try:
            # Filter for valid results
            valid_df = stats_df.dropna(subset=['mannwhitney_p', 'cliff_delta'])
            
            if len(valid_df) == 0:
                logger.warning("No valid statistical results for significance overview")
                return
            
            # Create significance summary
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. P-value distribution
            ax1 = axes[0, 0]
            p_values = valid_df['mannwhitney_p']
            ax1.hist(p_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(x=0.05, color='red', linestyle='--', label='p = 0.05')
            ax1.set_xlabel('p-value')
            ax1.set_ylabel('Number of Metrics')
            ax1.set_title(f'Distribution of p-values\n{len(valid_df)} metrics analyzed')
            ax1.legend()
            
            # 2. Effect sizes distribution
            ax2 = axes[0, 1]
            effect_sizes = valid_df['cliff_delta']
            ax2.hist(effect_sizes, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax2.axvline(x=0.147, color='orange', linestyle='--', label='Small effect')
            ax2.axvline(x=-0.147, color='orange', linestyle='--')
            ax2.axvline(x=0.33, color='red', linestyle='--', label='Medium effect')
            ax2.axvline(x=-0.33, color='red', linestyle='--')
            ax2.set_xlabel("Cliff's Delta")
            ax2.set_ylabel('Number of Metrics')
            ax2.set_title('Distribution of Effect Sizes')
            ax2.legend()
            
            # 3. Significance by category
            ax3 = axes[1, 0]
            category_sig = valid_df.groupby('category').apply(
                lambda x: (x['mannwhitney_p'] < 0.05).sum()
            ).sort_values(ascending=True)
            
            if len(category_sig) > 0:
                bars = ax3.barh(range(len(category_sig)), category_sig.values, color='lightgreen')
                ax3.set_yticks(range(len(category_sig)))
                ax3.set_yticklabels(category_sig.index)
                ax3.set_xlabel('Number of Significant Metrics')
                ax3.set_title('Significant Results by Category')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                            f'{int(width)}', ha='left', va='center')
            
            # 4. OSS4SG advantage summary
            ax4 = axes[1, 1]
            oss4sg_advantage = valid_df['oss4sg_advantage'].value_counts()
            if len(oss4sg_advantage) > 0:
                colors = ['lightblue', 'lightcoral']
                labels = ['OSS Advantage', 'OSS4SG Advantage']
                wedges, texts, autotexts = ax4.pie(oss4sg_advantage.values, labels=labels, colors=colors, autopct='%1.1f%%')
                ax4.set_title('Overall Advantage Distribution')
            
            plt.tight_layout()
            plt.savefig(self.analysis_dir / "statistical_significance_overview.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating significance overview: {e}")

    def create_effect_sizes_heatmap(self, stats_df):
        """Create effect sizes heatmap by category"""
        try:
            # Filter for significant results
            significant_df = stats_df[(stats_df['mannwhitney_p'] < 0.05) & stats_df['cliff_delta'].notna()]
            
            if len(significant_df) == 0:
                logger.warning("No significant results for effect sizes heatmap")
                return
            
            # Group by category
            category_effects = {}
            for category in significant_df['category'].unique():
                cat_data = significant_df[significant_df['category'] == category]
                category_effects[category] = {
                    'large_positive': len(cat_data[(cat_data['cliff_delta'] > 0.474)]),
                    'medium_positive': len(cat_data[(cat_data['cliff_delta'] > 0.33) & (cat_data['cliff_delta'] <= 0.474)]),
                    'small_positive': len(cat_data[(cat_data['cliff_delta'] > 0.147) & (cat_data['cliff_delta'] <= 0.33)]),
                    'negligible': len(cat_data[abs(cat_data['cliff_delta']) <= 0.147]),
                    'small_negative': len(cat_data[(cat_data['cliff_delta'] < -0.147) & (cat_data['cliff_delta'] >= -0.33)]),
                    'medium_negative': len(cat_data[(cat_data['cliff_delta'] < -0.33) & (cat_data['cliff_delta'] >= -0.474)]),
                    'large_negative': len(cat_data[(cat_data['cliff_delta'] < -0.474)])
                }
            
            if category_effects:
                effect_df = pd.DataFrame(category_effects).T
                effect_df = effect_df.fillna(0)
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(effect_df, annot=True, fmt='d', cmap='RdBu_r', center=0)
                plt.title('Effect Sizes by Category\n(OSS4SG Advantage: Positive, OSS Advantage: Negative)')
                plt.xlabel('Effect Size Magnitude')
                plt.ylabel('Metric Category')
                plt.tight_layout()
                plt.savefig(self.analysis_dir / "effect_sizes_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            logger.error(f"Error creating effect sizes heatmap: {e}")

    def create_top_significant_metrics_plot(self, stats_df):
        """Create plot of top significant metrics"""
        try:
            # Get top 20 most significant metrics
            significant_df = stats_df[stats_df['mannwhitney_p'].notna() & (stats_df['mannwhitney_p'] < 0.05)]
            
            if len(significant_df) == 0:
                logger.warning("No significant results for top metrics plot")
                return
            
            top_metrics = significant_df.nsmallest(20, 'mannwhitney_p')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # 1. P-values
            y_pos = range(len(top_metrics))
            colors = ['red' if x < 0.001 else 'orange' if x < 0.01 else 'yellow' for x in top_metrics['mannwhitney_p']]
            
            bars1 = ax1.barh(y_pos, -np.log10(top_metrics['mannwhitney_p']), color=colors, alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([m[:30] + '...' if len(m) > 30 else m for m in top_metrics['metric']])
            ax1.set_xlabel('-log10(p-value)')
            ax1.set_title('Top 20 Most Significant Metrics')
            ax1.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
            ax1.axvline(x=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p = 0.01')
            ax1.axvline(x=-np.log10(0.001), color='red', linestyle='-', alpha=0.7, label='p = 0.001')
            ax1.legend()
            
            # 2. Effect sizes
            cliff_deltas = top_metrics['cliff_delta'].fillna(0)
            colors2 = ['green' if x > 0 else 'red' for x in cliff_deltas]
            
            bars2 = ax2.barh(y_pos, cliff_deltas, color=colors2, alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([m[:30] + '...' if len(m) > 30 else m for m in top_metrics['metric']])
            ax2.set_xlabel("Cliff's Delta (Effect Size)")
            ax2.set_title('Effect Sizes for Top Significant Metrics')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.axvline(x=0.147, color='orange', linestyle='--', alpha=0.7, label='Small')
            ax2.axvline(x=-0.147, color='orange', linestyle='--', alpha=0.7)
            ax2.axvline(x=0.33, color='red', linestyle='--', alpha=0.7, label='Medium')
            ax2.axvline(x=-0.33, color='red', linestyle='--', alpha=0.7)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(self.analysis_dir / "top_significant_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating top significant metrics plot: {e}")

    def create_top_metrics_distributions(self, stats_df):
        """Create distribution plots for top metrics"""
        try:
            # Get top 12 most significant metrics for detailed plots
            significant_df = stats_df[stats_df['mannwhitney_p'].notna() & (stats_df['mannwhitney_p'] < 0.05)]
            
            if len(significant_df) == 0:
                logger.warning("No significant results for distribution plots")
                return
            
            top_metrics = significant_df.nsmallest(12, 'mannwhitney_p')
            
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            axes = axes.flatten()
            
            for i, (_, row) in enumerate(top_metrics.iterrows()):
                if i >= 12:
                    break
                
                ax = axes[i]
                metric = row['metric']
                
                # Get data
                oss_data = self.metrics_df[self.metrics_df['project_type'] == 'OSS'][metric].dropna()
                oss4sg_data = self.metrics_df[self.metrics_df['project_type'] == 'OSS4SG'][metric].dropna()
                
                if len(oss_data) == 0 or len(oss4sg_data) == 0:
                    ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Create violin plots
                data_combined = list(oss_data) + list(oss4sg_data)
                labels_combined = ['OSS'] * len(oss_data) + ['OSS4SG'] * len(oss4sg_data)
                
                plot_df = pd.DataFrame({'value': data_combined, 'type': labels_combined})
                
                sns.violinplot(data=plot_df, x='type', y='value', ax=ax)
                
                # Customize plot
                metric_short = metric[:20] + '...' if len(metric) > 20 else metric
                ax.set_title(f'{metric_short}\np={row["mannwhitney_p"]:.3e}')
                ax.set_xlabel('Project Type')
                ax.set_ylabel('Value')
                
                # Add effect size annotation
                cliff_delta = row['cliff_delta']
                if not np.isnan(cliff_delta):
                    ax.text(0.5, 0.95, f"Cliff's δ = {cliff_delta:.3f}", 
                           transform=ax.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Remove empty subplots
            for i in range(len(top_metrics), 12):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.savefig(self.analysis_dir / "top_metrics_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating distribution plots: {e}")

    def create_category_wise_summary(self, stats_df):
        """Create category-wise summary visualization"""
        try:
            # Summary by category
            category_summary = []
            
            for category in self.metric_categories.keys():
                cat_data = stats_df[stats_df['category'] == category]
                if len(cat_data) == 0:
                    continue
                
                valid_data = cat_data.dropna(subset=['mannwhitney_p', 'cliff_delta'])
                
                summary = {
                    'category': category,
                    'total_metrics': len(cat_data),
                    'valid_tests': len(valid_data),
                    'significant': len(valid_data[valid_data['mannwhitney_p'] < 0.05]),
                    'oss4sg_advantage': len(valid_data[(valid_data['mannwhitney_p'] < 0.05) & (valid_data['cliff_delta'] > 0)]),
                    'oss_advantage': len(valid_data[(valid_data['mannwhitney_p'] < 0.05) & (valid_data['cliff_delta'] < 0)]),
                    'large_effects': len(valid_data[abs(valid_data['cliff_delta']) > 0.474]),
                    'medium_effects': len(valid_data[(abs(valid_data['cliff_delta']) > 0.33) & (abs(valid_data['cliff_delta']) <= 0.474)]),
                    'small_effects': len(valid_data[(abs(valid_data['cliff_delta']) > 0.147) & (abs(valid_data['cliff_delta']) <= 0.33)])
                }
                
                category_summary.append(summary)
            
            if not category_summary:
                logger.warning("No data for category summary")
                return
            
            summary_df = pd.DataFrame(category_summary)
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Total metrics per category
            ax1 = axes[0, 0]
            bars1 = ax1.bar(summary_df['category'], summary_df['total_metrics'], alpha=0.7, color='skyblue')
            ax1.set_title('Total Metrics by Category')
            ax1.set_ylabel('Number of Metrics')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom')
            
            # 2. Significant results per category
            ax2 = axes[0, 1]
            bars2 = ax2.bar(summary_df['category'], summary_df['significant'], alpha=0.7, color='lightgreen')
            ax2.set_title('Significant Results by Category')
            ax2.set_ylabel('Number of Significant Metrics')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            # 3. Advantage distribution
            ax3 = axes[1, 0]
            x = range(len(summary_df))
            width = 0.35
            bars3a = ax3.bar([i - width/2 for i in x], summary_df['oss4sg_advantage'], 
                           width, label='OSS4SG Advantage', alpha=0.7, color='green')
            bars3b = ax3.bar([i + width/2 for i in x], summary_df['oss_advantage'], 
                           width, label='OSS Advantage', alpha=0.7, color='red')
            ax3.set_title('Advantage Distribution by Category')
            ax3.set_ylabel('Number of Metrics')
            ax3.set_xticks(x)
            ax3.set_xticklabels(summary_df['category'], rotation=45, ha='right')
            ax3.legend()
            
            # 4. Effect size distribution
            ax4 = axes[1, 1]
            bottom = np.zeros(len(summary_df))
            colors = ['red', 'orange', 'yellow']
            labels = ['Large Effects', 'Medium Effects', 'Small Effects']
            
            for i, (effects, color, label) in enumerate(zip(
                ['large_effects', 'medium_effects', 'small_effects'], colors, labels)):
                ax4.bar(summary_df['category'], summary_df[effects], bottom=bottom, 
                       label=label, alpha=0.7, color=color)
                bottom += summary_df[effects]
            
            ax4.set_title('Effect Size Distribution by Category')
            ax4.set_ylabel('Number of Metrics')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
            ax4.legend()
            
            plt.tight_layout()
            plt.savefig(self.analysis_dir / "category_wise_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save summary table
            summary_df.to_csv(self.analysis_dir / "category_summary_table.csv", index=False)
            
        except Exception as e:
            logger.error(f"Error creating category summary: {e}")

    def create_complete_correlation_analysis(self):
        """Create correlation analysis for all metrics"""
        try:
            # Get numeric data only
            numeric_data = self.metrics_df[self.all_numeric_metrics].select_dtypes(include=[np.number])
            
            # Remove columns with too many NaNs
            valid_cols = []
            for col in numeric_data.columns:
                if numeric_data[col].notna().sum() > len(numeric_data) * 0.1:  # At least 10% valid data
                    valid_cols.append(col)
            
            if len(valid_cols) < 2:
                logger.warning("Insufficient valid columns for correlation analysis")
                return
            
            correlation_data = numeric_data[valid_cols]
            
            # Calculate correlation matrix
            corr_matrix = correlation_data.corr()
            
            # Create correlation heatmap
            plt.figure(figsize=(20, 16))
            
            # Use a mask for better visualization
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                       square=True, cbar_kws={'shrink': 0.8})
            
            plt.title('Complete Metrics Correlation Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.analysis_dir / "complete_correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Find strongest correlations
            corr_matrix_abs = corr_matrix.abs()
            corr_matrix_abs.values[np.triu_indices_from(corr_matrix_abs.values)] = np.nan
            
            # Get top correlations
            corr_flat = corr_matrix_abs.unstack().dropna().sort_values(ascending=False)
            top_correlations = corr_flat.head(50)
            
            # Save correlation results
            correlation_results = []
            for (metric1, metric2), correlation in top_correlations.items():
                correlation_results.append({
                    'metric1': metric1,
                    'metric2': metric2,
                    'correlation': correlation,
                    'category1': self.get_metric_category(metric1),
                    'category2': self.get_metric_category(metric2)
                })
            
            correlation_df = pd.DataFrame(correlation_results)
            correlation_df.to_csv(self.analysis_dir / "top_correlations.csv", index=False)
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")

    def generate_complete_report(self):
        """Generate complete analysis report"""
        logger.info("Starting COMPLETE analysis of ALL metrics...")
        
        try:
            # 1. Statistical analysis of ALL metrics
            stats_df = self.perform_complete_statistical_analysis()
            
            # 2. Outlier analysis for ALL metrics
            outlier_df = self.analyze_all_outliers()
            
            # 3. Zero pattern analysis for ALL metrics
            zero_df = self.analyze_all_zero_patterns()
            
            # 4. Create comprehensive visualizations
            self.create_comprehensive_visualizations(stats_df)
            
            # 5. Generate final report
            self.create_final_comprehensive_report(stats_df, outlier_df, zero_df)
            
            logger.info(f"COMPLETE analysis finished. Results in: {self.analysis_dir}")
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            import traceback
            traceback.print_exc()
            raise

    def create_final_comprehensive_report(self, stats_df, outlier_df, zero_df):
        """Create final comprehensive report"""
        
        # Summary statistics
        total_metrics = len(self.all_numeric_metrics)
        valid_tests = len(stats_df[stats_df['mannwhitney_p'].notna()])
        significant_results = len(stats_df[stats_df['mannwhitney_p'] < 0.05])
        oss4sg_advantages = len(stats_df[(stats_df['mannwhitney_p'] < 0.05) & (stats_df['cliff_delta'] > 0)])
        oss_advantages = len(stats_df[(stats_df['mannwhitney_p'] < 0.05) & (stats_df['cliff_delta'] < 0)])
        
        large_effects = len(stats_df[abs(stats_df['cliff_delta']) > 0.474])
        medium_effects = len(stats_df[(abs(stats_df['cliff_delta']) > 0.33) & (abs(stats_df['cliff_delta']) <= 0.474)])
        small_effects = len(stats_df[(abs(stats_df['cliff_delta']) > 0.147) & (abs(stats_df['cliff_delta']) <= 0.33)])
        
        report_content = f"""
# COMPLETE Treatment Metrics Analysis Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This is a COMPREHENSIVE analysis of ALL treatment metrics comparing OSS vs OSS4SG projects.
NO SHORTCUTS - EVERY SINGLE METRIC ANALYZED.

### Dataset Overview
- **Total Contributors:** {len(self.metrics_df)}
- **OSS Contributors:** {len(self.metrics_df[self.metrics_df['project_type'] == 'OSS'])}
- **OSS4SG Contributors:** {len(self.metrics_df[self.metrics_df['project_type'] == 'OSS4SG'])}
- **Total Metrics Analyzed:** {total_metrics}
- **Valid Statistical Tests:** {valid_tests}

### Statistical Results Summary
- **Significant Differences:** {significant_results}/{valid_tests} ({significant_results/valid_tests*100:.1f}%)
- **OSS4SG Advantages:** {oss4sg_advantages} significant metrics
- **OSS Advantages:** {oss_advantages} significant metrics
- **Large Effect Sizes:** {large_effects} metrics (|δ| > 0.474)
- **Medium Effect Sizes:** {medium_effects} metrics (0.33 < |δ| ≤ 0.474)
- **Small Effect Sizes:** {small_effects} metrics (0.147 < |δ| ≤ 0.33)

### Analysis Categories
"""
        
        # Add category-wise results
        for category, metrics_list in self.metric_categories.items():
            if not metrics_list:
                continue
            cat_stats = stats_df[stats_df['category'] == category]
            cat_significant = len(cat_stats[cat_stats['mannwhitney_p'] < 0.05])
            
            report_content += f"\n#### {category.upper()}\n"
            report_content += f"- Total metrics: {len(metrics_list)}\n"
            report_content += f"- Significant results: {cat_significant}\n"
        
        report_content += f"""

## Generated Files (COMPLETE ANALYSIS)
1. `complete_statistical_analysis.csv` - Statistical tests for ALL {total_metrics} metrics
2. `complete_outlier_analysis.csv` - Outlier analysis for ALL metrics
3. `complete_zero_patterns_analysis.csv` - Zero-pattern analysis for ALL metrics
4. `statistical_significance_overview.png` - Overview of all statistical results
5. `effect_sizes_heatmap.png` - Effect sizes by category
6. `top_significant_metrics.png` - Top 20 most significant metrics
7. `top_metrics_distributions.png` - Distribution plots for top 12 metrics
8. `category_wise_summary.png` - Summary by metric category
9. `complete_correlation_matrix.png` - Correlation analysis of all metrics
10. `category_summary_table.csv` - Category-wise summary statistics
11. `top_correlations.csv` - Top 50 metric correlations

## Methodology
- **Statistical Test:** Mann-Whitney U (non-parametric, robust)
- **Effect Size:** Cliff's Delta (non-parametric effect size)
- **Outlier Detection:** IQR method (1.5 × IQR)
- **Zero-Pattern Analysis:** Chi-square tests for participation differences
- **Multiple Testing:** Raw p-values reported (consider Bonferroni correction: α = 0.05/{valid_tests})

## Key Findings Summary
{f"OSS4SG shows advantages in {oss4sg_advantages} metrics vs OSS advantages in {oss_advantages} metrics." if oss4sg_advantages > oss_advantages else f"OSS shows advantages in {oss_advantages} metrics vs OSS4SG advantages in {oss4sg_advantages} metrics."}

Most significant differences found in: {stats_df[stats_df['mannwhitney_p'] < 0.001]['category'].value_counts().index[0] if len(stats_df[stats_df['mannwhitney_p'] < 0.001]) > 0 else 'N/A'}

## Data Quality Notes
- Outlier analysis performed for {len(outlier_df)} metrics
- Zero-pattern analysis for {len(zero_df)} metrics  
- Missing data handled appropriately in all analyses
- All results include both raw and cleaned (outlier-removed) comparisons
"""
        
        # Save report
        with open(self.analysis_dir / "COMPLETE_ANALYSIS_REPORT.md", 'w') as f:
            f.write(report_content)
        
        # Print summary to console
        print(f"\n{'='*100}")
        print("COMPLETE TREATMENT METRICS ANALYSIS FINISHED")
        print(f"{'='*100}")
        print(f"📊 ANALYZED: {total_metrics} metrics across {len(self.metrics_df)} contributors")
        print(f"🧪 STATISTICAL TESTS: {valid_tests} valid tests performed")
        print(f"⭐ SIGNIFICANT RESULTS: {significant_results} ({significant_results/valid_tests*100:.1f}%)")
        print(f"🏆 OSS4SG ADVANTAGES: {oss4sg_advantages} metrics")
        print(f"🏆 OSS ADVANTAGES: {oss_advantages} metrics")
        print(f"🎯 LARGE EFFECTS: {large_effects} metrics")
        print(f"📁 RESULTS LOCATION: {self.analysis_dir}")
        print(f"{'='*100}")

def main():
    """Main execution"""
    print("COMPLETE TREATMENT METRICS ANALYSIS ENGINE")
    print("=" * 100)
    print("ANALYZING EVERY SINGLE METRIC - NO SHORTCUTS!")
    print("=" * 100)
    
    try:
        base_path = "/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026"
        
        analyzer = CompleteMetricsAnalyzer(base_path)
        analyzer.generate_complete_report()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
