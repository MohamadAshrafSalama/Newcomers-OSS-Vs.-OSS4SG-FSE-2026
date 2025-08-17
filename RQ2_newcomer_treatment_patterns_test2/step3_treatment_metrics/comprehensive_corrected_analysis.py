#!/usr/bin/env python3
"""
COMPREHENSIVE CORRECTED Treatment Metrics Analysis for RQ2
- Analyzes ALL 91 metrics (not just 10)
- Fixes zero-activity bias by filtering for active contributors only
- Provides complete OSS vs OSS4SG comparisons
- Addresses methodological issues identified in original analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ComprehensiveCorrectedAnalyzer:
    def __init__(self, base_path):
        """Initialize the comprehensive corrected analyzer"""
        self.base_path = Path(base_path)
        self.input_file = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step3_treatment_metrics" / "results" / "complete_treatment_metrics_dataset.csv"
        self.output_dir = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step3_treatment_metrics" / "comprehensive_corrected_results"
        self.viz_dir = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step3_treatment_metrics" / "comprehensive_corrected_visualizations"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Define colors for consistency
        self.colors = {
            'OSS': '#2E86AB',
            'OSS4SG': '#A23B72'
        }
        
        # Metric categories for organization
        self.metric_categories = {
            'RESPONSE_TIMING': [
                'first_response_mean_hours', 'first_response_median_hours', 'first_response_std_hours',
                'first_response_min_hours', 'first_response_max_hours', 'first_response_25th_percentile',
                'first_response_75th_percentile', 'avg_response_mean_hours', 'avg_response_median_hours',
                'avg_response_std_hours', 'weekend_response_rate', 'off_hours_response_rate',
                'business_hours_response_rate', 'within_hour_response_rate', 'same_day_response_rate',
                'approval_speed_mean_hours', 'approval_speed_median_hours', 'approval_speed_std_hours',
                'approval_speed_min_hours', 'approval_speed_max_hours'
            ],
            'ENGAGEMENT_BREADTH': [
                'unique_responders', 'unique_pr_responders', 'repeat_engagers', 'avg_prs_per_responder',
                'max_prs_per_responder', 'response_diversity_index', 'response_concentration_ratio',
                'unique_responder_roles', 'role_diversity_index'
            ],
            'INTERACTION_PATTERNS': [
                'conversation_length_mean', 'conversation_length_median', 'conversation_length_std',
                'conversation_length_max', 'conversation_length_min', 'back_forth_turns_mean',
                'back_forth_turns_median', 'back_forth_turns_std', 'back_forth_turns_max',
                'word_count_mean', 'word_count_median', 'word_count_std', 'word_count_max',
                'sentence_count_mean', 'sentence_count_median', 'total_messages',
                'avg_words_per_message', 'question_rate', 'link_sharing_rate'
            ],
            'RECOGNITION_SIGNALS': [
                'thanks_rate', 'praise_rate', 'emoji_usage_rate', 'positive_sentiment_rate',
                'positive_emoji_count', 'negative_emoji_count', 'emoji_sentiment_ratio',
                'code_sharing_rate', 'mention_rate', 'code_snippets_total', 'external_links_total',
                'internal_links_total', 'at_mentions', 'review_requests_received'
            ],
            'TRUST_INDICATORS': [
                'issue_assignments', 'label_additions', 'milestone_assignments', 'project_assignments',
                'collaborative_edits', 'delegated_tasks', 'trusted_with_sensitive',
                'administrative_actions', 'trust_score', 'pr_with_labels_rate',
                'pr_with_milestones_rate', 'issue_assignment_rate'
            ],
            'PARTICIPATION_METRICS': [
                'total_items', 'items_with_responses', 'response_rate', 'total_responses',
                'first_response_count', 'maintainer_response_rate', 'peer_response_rate',
                'external_response_rate', 'total_prs', 'merge_rate', 'close_rate',
                'approval_rate', 'rejection_rate', 'author_attribution_rate', 'total_issues',
                're_engagement_rate', 're_engagement_cycles'
            ]
        }
        
    def load_and_filter_data(self):
        """Load data and apply proper filtering"""
        
        print("="*80)
        print("COMPREHENSIVE CORRECTED TREATMENT METRICS ANALYSIS")
        print("="*80)
        
        # Load original dataset
        df = pd.read_csv(self.input_file)
        print(f"Original dataset: {len(df)} contributors")
        print(f"OSS: {len(df[df['project_type'] == 'OSS'])}")
        print(f"OSS4SG: {len(df[df['project_type'] == 'OSS4SG'])}")
        
        # Get all metric columns
        non_metric_cols = ['contributor_email', 'project_name', 'project_type', 
                          'total_pr_events', 'total_issue_events', 'total_commit_events']
        self.all_metrics = [col for col in df.columns if col not in non_metric_cols]
        print(f"Total metrics to analyze: {len(self.all_metrics)}")
        
        # Identify zero-activity contributors
        zero_activity = df[(df['total_pr_events'] == 0) & (df['total_issue_events'] == 0)]
        print(f"\nZero-activity contributors: {len(zero_activity)} ({len(zero_activity)/len(df)*100:.1f}%)")
        print(f"OSS zero-activity: {len(zero_activity[zero_activity['project_type'] == 'OSS'])}")
        print(f"OSS4SG zero-activity: {len(zero_activity[zero_activity['project_type'] == 'OSS4SG'])}")
        
        # Filter for active contributors only
        self.active_df = df[(df['total_pr_events'] > 0) | (df['total_issue_events'] > 0)].copy()
        print(f"\nActive contributors: {len(self.active_df)} ({len(self.active_df)/len(df)*100:.1f}%)")
        print(f"OSS active: {len(self.active_df[self.active_df['project_type'] == 'OSS'])}")
        print(f"OSS4SG active: {len(self.active_df[self.active_df['project_type'] == 'OSS4SG'])}")
        
        # Store both datasets for comparison
        self.original_df = df
        
        return self.active_df
    
    def analyze_all_metrics(self):
        """Analyze ALL metrics with proper methodology"""
        
        results = []
        
        print(f"\n{'='*80}")
        print(f"ANALYZING ALL {len(self.all_metrics)} METRICS")
        print(f"{'='*80}")
        
        for i, metric in enumerate(self.all_metrics, 1):
            print(f"Processing {i:2d}/{len(self.all_metrics)}: {metric}")
            
            # Get data for both groups
            oss_data = self.active_df[self.active_df['project_type'] == 'OSS'][metric].dropna()
            oss4sg_data = self.active_df[self.active_df['project_type'] == 'OSS4SG'][metric].dropna()
            
            if len(oss_data) == 0 or len(oss4sg_data) == 0:
                print(f"  Skipping {metric} - insufficient data")
                continue
            
            # Calculate statistics
            oss_mean = oss_data.mean()
            oss4sg_mean = oss4sg_data.mean()
            oss_median = oss_data.median()
            oss4sg_median = oss4sg_data.median()
            oss_std = oss_data.std()
            oss4sg_std = oss4sg_data.std()
            
            # Statistical test
            try:
                statistic, p_value = stats.mannwhitneyu(oss_data, oss4sg_data, alternative='two-sided')
                
                # Calculate effect size (Cliff's Delta)
                def cliff_delta(x, y):
                    """Calculate Cliff's Delta effect size"""
                    n1, n2 = len(x), len(y)
                    greater = sum(xi > yi for xi in x for yi in y)
                    less = sum(xi < yi for xi in x for yi in y)
                    return (greater - less) / (n1 * n2)
                
                effect_size = cliff_delta(oss4sg_data, oss_data)  # Positive = OSS4SG advantage
                
                # Effect size interpretation
                if abs(effect_size) < 0.147:
                    effect_interpretation = "negligible"
                elif abs(effect_size) < 0.33:
                    effect_interpretation = "small"
                elif abs(effect_size) < 0.474:
                    effect_interpretation = "medium"
                else:
                    effect_interpretation = "large"
                
            except Exception as e:
                print(f"  Error in statistical test for {metric}: {e}")
                statistic, p_value, effect_size, effect_interpretation = np.nan, np.nan, np.nan, "error"
            
            # Determine advantage and category
            oss4sg_advantage = oss4sg_mean > oss_mean
            significant = p_value < 0.05 if not np.isnan(p_value) else False
            
            # Find metric category
            category = "OTHER"
            for cat_name, metrics_list in self.metric_categories.items():
                if metric in metrics_list:
                    category = cat_name
                    break
            
            result = {
                'metric': metric,
                'category': category,
                'oss_n': len(oss_data),
                'oss4sg_n': len(oss4sg_data),
                'oss_mean': oss_mean,
                'oss4sg_mean': oss4sg_mean,
                'oss_median': oss_median,
                'oss4sg_median': oss4sg_median,
                'oss_std': oss_std,
                'oss4sg_std': oss4sg_std,
                'mannwhitney_u': statistic,
                'mannwhitney_p': p_value,
                'cliff_delta': effect_size,
                'effect_size_interpretation': effect_interpretation,
                'significant': significant,
                'oss4sg_advantage': oss4sg_advantage
            }
            
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save comprehensive results
        results_file = self.output_dir / "comprehensive_corrected_metrics_analysis.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nComprehensive results saved to: {results_file}")
        
        return results_df
    
    def create_comprehensive_summary(self, results_df):
        """Create comprehensive summary of all results"""
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Overall summary
        total_metrics = len(results_df)
        significant_metrics = len(results_df[results_df['significant'] == True])
        oss4sg_advantages = len(results_df[results_df['oss4sg_advantage'] == True])
        oss_advantages = len(results_df[results_df['oss4sg_advantage'] == False])
        oss4sg_significant = len(results_df[(results_df['oss4sg_advantage'] == True) & (results_df['significant'] == True)])
        oss_significant = len(results_df[(results_df['oss4sg_advantage'] == False) & (results_df['significant'] == True)])
        
        print(f"Total metrics analyzed: {total_metrics}")
        print(f"Significant differences: {significant_metrics} ({significant_metrics/total_metrics*100:.1f}%)")
        print(f"OSS4SG advantages: {oss4sg_advantages} ({oss4sg_significant} significant)")
        print(f"OSS advantages: {oss_advantages} ({oss_significant} significant)")
        
        # Category breakdown
        print(f"\nResults by category:")
        category_summary = []
        
        for category in self.metric_categories.keys():
            cat_results = results_df[results_df['category'] == category]
            if len(cat_results) == 0:
                continue
                
            cat_total = len(cat_results)
            cat_significant = len(cat_results[cat_results['significant'] == True])
            cat_oss4sg_adv = len(cat_results[cat_results['oss4sg_advantage'] == True])
            cat_oss4sg_sig = len(cat_results[(cat_results['oss4sg_advantage'] == True) & (cat_results['significant'] == True)])
            
            print(f"  {category}: {cat_total} metrics, {cat_significant} significant, {cat_oss4sg_sig}/{cat_oss4sg_adv} OSS4SG advantages")
            
            category_summary.append({
                'category': category,
                'total_metrics': cat_total,
                'significant': cat_significant,
                'oss4sg_advantages': cat_oss4sg_adv,
                'oss4sg_significant': cat_oss4sg_sig,
                'oss_advantages': cat_total - cat_oss4sg_adv,
                'oss_significant': cat_significant - cat_oss4sg_sig
            })
        
        # Save category summary
        category_df = pd.DataFrame(category_summary)
        category_file = self.output_dir / "comprehensive_category_summary.csv"
        category_df.to_csv(category_file, index=False)
        
        # Top OSS4SG advantages
        print(f"\nTop OSS4SG significant advantages (by effect size):")
        oss4sg_sig = results_df[(results_df['oss4sg_advantage'] == True) & (results_df['significant'] == True)]
        oss4sg_sig = oss4sg_sig.sort_values('cliff_delta', ascending=False)
        
        for _, row in oss4sg_sig.head(10).iterrows():
            print(f"  {row['metric']}: δ={row['cliff_delta']:.3f}, p={row['mannwhitney_p']:.3f}")
        
        # Top OSS advantages
        print(f"\nTop OSS significant advantages (by effect size):")
        oss_sig = results_df[(results_df['oss4sg_advantage'] == False) & (results_df['significant'] == True)]
        oss_sig = oss_sig.sort_values('cliff_delta', ascending=True)
        
        for _, row in oss_sig.head(10).iterrows():
            print(f"  {row['metric']}: δ={row['cliff_delta']:.3f}, p={row['mannwhitney_p']:.3f}")
        
        return category_df
    
    def create_comprehensive_visualizations(self, results_df, category_df):
        """Create comprehensive visualizations"""
        
        print(f"\nCreating comprehensive visualizations...")
        
        # 1. Overview comparison - Original vs Corrected
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Load original results for comparison
        try:
            original_results = pd.read_csv(self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step3_treatment_metrics" / "results" / "complete_analysis" / "complete_statistical_analysis.csv")
            
            # Original results summary
            orig_oss4sg_adv = len(original_results[original_results['oss4sg_advantage'] == True])
            orig_oss_adv = len(original_results[original_results['oss4sg_advantage'] == False])
            
            # Corrected results summary
            corr_oss4sg_adv = len(results_df[results_df['oss4sg_advantage'] == True])
            corr_oss_adv = len(results_df[results_df['oss4sg_advantage'] == False])
            
            # Plot comparison
            categories = ['OSS4SG\nAdvantages', 'OSS\nAdvantages']
            original_counts = [orig_oss4sg_adv, orig_oss_adv]
            corrected_counts = [corr_oss4sg_adv, corr_oss_adv]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax1.bar(x - width/2, original_counts, width, label='Original (Biased)', alpha=0.7, color=['#FF6B6B', '#4ECDC4'])
            ax1.bar(x + width/2, corrected_counts, width, label='Corrected', alpha=1.0, color=['#A23B72', '#2E86AB'])
            
            ax1.set_ylabel('Number of Metrics')
            ax1.set_title('Treatment Advantages: Original vs Corrected Analysis')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (orig, corr) in enumerate(zip(original_counts, corrected_counts)):
                ax1.text(i - width/2, orig + 1, str(orig), ha='center', va='bottom')
                ax1.text(i + width/2, corr + 1, str(corr), ha='center', va='bottom')
                
        except Exception as e:
            print(f"Could not load original results for comparison: {e}")
        
        # 2. Category breakdown
        categories = category_df['category'].tolist()
        oss4sg_sig = category_df['oss4sg_significant'].tolist()
        oss_sig = category_df['oss_significant'].tolist()
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax2.bar(x - width/2, oss4sg_sig, width, label='OSS4SG Significant', color='#A23B72', alpha=0.8)
        ax2.bar(x + width/2, oss_sig, width, label='OSS Significant', color='#2E86AB', alpha=0.8)
        
        ax2.set_ylabel('Significant Metrics')
        ax2.set_title('Significant Advantages by Category\n(Corrected Analysis)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([cat.replace('_', '\n') for cat in categories], rotation=0, fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (oss4sg, oss) in enumerate(zip(oss4sg_sig, oss_sig)):
            if oss4sg > 0:
                ax2.text(i - width/2, oss4sg + 0.1, str(oss4sg), ha='center', va='bottom')
            if oss > 0:
                ax2.text(i + width/2, oss + 0.1, str(oss), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'comprehensive_corrected_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.viz_dir / 'comprehensive_corrected_overview.png'}")
        
        # 3. Effect sizes heatmap
        significant_results = results_df[results_df['significant'] == True].copy()
        if len(significant_results) > 0:
            
            # Create effect size matrix by category
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Sort by effect size within categories
            significant_results = significant_results.sort_values(['category', 'cliff_delta'], ascending=[True, False])
            
            # Create the heatmap data
            metrics = significant_results['metric'].tolist()
            effect_sizes = significant_results['cliff_delta'].tolist()
            
            # Limit to top 30 for readability
            if len(metrics) > 30:
                metrics = metrics[:30]
                effect_sizes = effect_sizes[:30]
            
            # Create color map
            colors = ['#2E86AB' if es < 0 else '#A23B72' for es in effect_sizes]
            
            y_pos = np.arange(len(metrics))
            bars = ax.barh(y_pos, effect_sizes, color=colors, alpha=0.8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([m.replace('_', ' ') for m in metrics], fontsize=8)
            ax.set_xlabel('Cliff\'s Delta (Effect Size)')
            ax.set_title('Effect Sizes of Significant Differences\n(Positive = OSS4SG Advantage, Negative = OSS Advantage)')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'comprehensive_effect_sizes.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {self.viz_dir / 'comprehensive_effect_sizes.png'}")

def main():
    """Main execution function"""
    
    base_path = "/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026"
    
    # Initialize comprehensive analyzer
    analyzer = ComprehensiveCorrectedAnalyzer(base_path)
    
    # Step 1: Load and filter data
    active_df = analyzer.load_and_filter_data()
    
    # Step 2: Analyze ALL metrics
    results_df = analyzer.analyze_all_metrics()
    
    # Step 3: Create comprehensive summary
    category_df = analyzer.create_comprehensive_summary(results_df)
    
    # Step 4: Create comprehensive visualizations
    analyzer.create_comprehensive_visualizations(results_df, category_df)
    
    print(f"\n{'='*80}")
    print("✅ COMPREHENSIVE CORRECTED ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"📁 Results: {analyzer.output_dir}")
    print(f"📊 Visualizations: {analyzer.viz_dir}")
    print(f"\nAnalyzed {len(results_df)} metrics total")
    print(f"Found {len(results_df[results_df['significant'] == True])} significant differences")

if __name__ == "__main__":
    main()
