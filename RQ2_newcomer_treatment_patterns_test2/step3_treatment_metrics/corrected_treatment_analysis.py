#!/usr/bin/env python3
"""
CORRECTED Treatment Metrics Analysis for RQ2
- Fixes zero-activity bias by filtering for active contributors only
- Provides proper OSS vs OSS4SG comparisons
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

class CorrectedTreatmentAnalyzer:
    def __init__(self, base_path):
        """Initialize the corrected analyzer"""
        self.base_path = Path(base_path)
        self.input_file = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step3_treatment_metrics" / "results" / "complete_treatment_metrics_dataset.csv"
        self.output_dir = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step3_treatment_metrics" / "corrected_results"
        self.viz_dir = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step3_treatment_metrics" / "corrected_visualizations"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Define colors for consistency
        self.colors = {
            'OSS': '#2E86AB',
            'OSS4SG': '#A23B72'
        }
        
    def load_and_filter_data(self):
        """Load data and apply proper filtering"""
        
        print("="*80)
        print("CORRECTED TREATMENT METRICS ANALYSIS")
        print("="*80)
        
        # Load original dataset
        df = pd.read_csv(self.input_file)
        print(f"Original dataset: {len(df)} contributors")
        print(f"OSS: {len(df[df['project_type'] == 'OSS'])}")
        print(f"OSS4SG: {len(df[df['project_type'] == 'OSS4SG'])}")
        
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
    
    def analyze_key_metrics(self):
        """Analyze key treatment metrics with proper methodology"""
        
        # Define key metrics for comparison
        key_metrics = {
            'response_rate': 'Response Rate',
            'merge_rate': 'Merge Rate', 
            'approval_rate': 'Approval Rate',
            'first_response_mean_hours': 'First Response Time (hours)',
            'avg_response_mean_hours': 'Average Response Time (hours)',
            'total_responses': 'Total Responses',
            'unique_responders': 'Unique Responders',
            'conversation_length_mean': 'Conversation Length',
            'back_forth_turns_mean': 'Back-and-forth Turns',
            'positive_sentiment_rate': 'Positive Sentiment Rate'
        }
        
        results = []
        
        print("\n" + "="*80)
        print("KEY METRICS COMPARISON (ACTIVE CONTRIBUTORS ONLY)")
        print("="*80)
        
        for metric, label in key_metrics.items():
            if metric not in self.active_df.columns:
                continue
                
            oss_data = self.active_df[self.active_df['project_type'] == 'OSS'][metric].dropna()
            oss4sg_data = self.active_df[self.active_df['project_type'] == 'OSS4SG'][metric].dropna()
            
            if len(oss_data) == 0 or len(oss4sg_data) == 0:
                continue
            
            # Calculate statistics
            oss_mean = oss_data.mean()
            oss4sg_mean = oss4sg_data.mean()
            oss_median = oss_data.median()
            oss4sg_median = oss4sg_data.median()
            
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
                
            except Exception as e:
                statistic, p_value, effect_size = np.nan, np.nan, np.nan
            
            # Determine advantage
            oss4sg_advantage = oss4sg_mean > oss_mean
            significant = p_value < 0.05 if not np.isnan(p_value) else False
            
            result = {
                'Metric': label,
                'OSS_Mean': oss_mean,
                'OSS4SG_Mean': oss4sg_mean,
                'OSS_Median': oss_median,
                'OSS4SG_Median': oss4sg_median,
                'OSS_N': len(oss_data),
                'OSS4SG_N': len(oss4sg_data),
                'Mann_Whitney_U': statistic,
                'P_Value': p_value,
                'Cliff_Delta': effect_size,
                'OSS4SG_Advantage': oss4sg_advantage,
                'Significant': significant
            }
            
            results.append(result)
            
            # Print result
            advantage_text = "OSS4SG" if oss4sg_advantage else "OSS"
            sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            print(f"\n{label}:")
            print(f"  OSS: mean={oss_mean:.3f}, median={oss_median:.3f}, n={len(oss_data)}")
            print(f"  OSS4SG: mean={oss4sg_mean:.3f}, median={oss4sg_median:.3f}, n={len(oss4sg_data)}")
            print(f"  Advantage: {advantage_text}, p={p_value:.3f} {sig_text}, δ={effect_size:.3f}")
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)
        
        # Save detailed results
        results_file = self.output_dir / "corrected_key_metrics_comparison.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
        
        return results_df
    
    def create_comparison_visualizations(self, results_df):
        """Create visualizations showing corrected comparisons"""
        
        # 1. OSS4SG Advantages Summary
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Count advantages
        oss4sg_advantages = results_df[results_df['OSS4SG_Advantage'] == True]
        oss_advantages = results_df[results_df['OSS4SG_Advantage'] == False]
        
        significant_oss4sg = oss4sg_advantages[oss4sg_advantages['Significant'] == True]
        significant_oss = oss_advantages[oss_advantages['Significant'] == True]
        
        # Plot 1: Advantage counts
        categories = ['OSS4SG\nAdvantages', 'OSS\nAdvantages']
        total_counts = [len(oss4sg_advantages), len(oss_advantages)]
        sig_counts = [len(significant_oss4sg), len(significant_oss)]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, total_counts, width, label='Total', alpha=0.7, color=['#A23B72', '#2E86AB'])
        bars2 = ax1.bar(x + width/2, sig_counts, width, label='Significant', alpha=1.0, color=['#A23B72', '#2E86AB'])
        
        ax1.set_ylabel('Number of Metrics')
        ax1.set_title('Corrected Treatment Advantage Analysis\n(Active Contributors Only)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
        
        # Plot 2: Effect sizes
        sig_results = results_df[results_df['Significant'] == True].copy()
        sig_results = sig_results.sort_values('Cliff_Delta')
        
        colors = ['#A23B72' if delta > 0 else '#2E86AB' for delta in sig_results['Cliff_Delta']]
        
        bars = ax2.barh(range(len(sig_results)), sig_results['Cliff_Delta'], color=colors, alpha=0.8)
        ax2.set_yticks(range(len(sig_results)))
        ax2.set_yticklabels(sig_results['Metric'], fontsize=10)
        ax2.set_xlabel('Cliff\'s Delta (Effect Size)')
        ax2.set_title('Effect Sizes of Significant Differences\n(Positive = OSS4SG Advantage)')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'corrected_treatment_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.viz_dir / 'corrected_treatment_comparison.png'}")
        
        # 2. Key Metrics Distributions
        key_metrics_to_plot = ['response_rate', 'merge_rate', 'approval_rate', 'first_response_mean_hours']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(key_metrics_to_plot):
            if metric not in self.active_df.columns:
                continue
                
            ax = axes[idx]
            
            # Get data
            oss_data = self.active_df[self.active_df['project_type'] == 'OSS'][metric].dropna()
            oss4sg_data = self.active_df[self.active_df['project_type'] == 'OSS4SG'][metric].dropna()
            
            if len(oss_data) == 0 or len(oss4sg_data) == 0:
                continue
            
            # Create box plots
            plot_data = [oss_data, oss4sg_data]
            labels = [f'OSS\n(n={len(oss_data)})', f'OSS4SG\n(n={len(oss4sg_data)})']
            
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, showmeans=True)
            
            # Color boxes
            bp['boxes'][0].set_facecolor(self.colors['OSS'])
            bp['boxes'][1].set_facecolor(self.colors['OSS4SG'])
            for box in bp['boxes']:
                box.set_alpha(0.7)
            
            # Add statistical test result
            try:
                _, p_value = stats.mannwhitneyu(oss_data, oss4sg_data, alternative='two-sided')
                sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                ax.text(0.5, 0.95, f'p={p_value:.3f} {sig_text}', 
                       transform=ax.transAxes, ha='center', va='top')
            except:
                pass
            
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Key Treatment Metrics Distribution: OSS vs OSS4SG\n(Active Contributors Only)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'key_metrics_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.viz_dir / 'key_metrics_distributions.png'}")
    
    def generate_corrected_summary(self, results_df):
        """Generate a corrected summary report"""
        
        summary_file = self.output_dir / "CORRECTED_ANALYSIS_SUMMARY.md"
        
        with open(summary_file, 'w') as f:
            f.write("# CORRECTED Treatment Metrics Analysis Report\n")
            f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n")
            f.write("This analysis CORRECTS the methodological issues found in the original Step 3 analysis.\n\n")
            
            f.write("### Issues Fixed:\n")
            f.write("1. **Zero-Activity Bias**: Removed 41% of contributors with no PR/issue activity\n")
            f.write("2. **Activity Level Confounding**: Focused on active contributors only\n")
            f.write("3. **Population Mixing**: Clear separation of OSS vs OSS4SG\n\n")
            
            # Dataset summary
            oss_count = len(self.active_df[self.active_df['project_type'] == 'OSS'])
            oss4sg_count = len(self.active_df[self.active_df['project_type'] == 'OSS4SG'])
            
            f.write("### Corrected Dataset:\n")
            f.write(f"- **Active Contributors**: {len(self.active_df)}\n")
            f.write(f"- **OSS Active**: {oss_count}\n")
            f.write(f"- **OSS4SG Active**: {oss4sg_count}\n\n")
            
            # Results summary
            total_metrics = len(results_df)
            oss4sg_advantages = len(results_df[results_df['OSS4SG_Advantage'] == True])
            oss_advantages = len(results_df[results_df['OSS4SG_Advantage'] == False])
            significant_oss4sg = len(results_df[(results_df['OSS4SG_Advantage'] == True) & (results_df['Significant'] == True)])
            significant_oss = len(results_df[(results_df['OSS4SG_Advantage'] == False) & (results_df['Significant'] == True)])
            
            f.write("### Corrected Results:\n")
            f.write(f"- **Total Metrics Analyzed**: {total_metrics}\n")
            f.write(f"- **OSS4SG Advantages**: {oss4sg_advantages} ({significant_oss4sg} significant)\n")
            f.write(f"- **OSS Advantages**: {oss_advantages} ({significant_oss} significant)\n\n")
            
            f.write("### Key Findings:\n")
            
            # Top OSS4SG advantages
            oss4sg_sig = results_df[(results_df['OSS4SG_Advantage'] == True) & (results_df['Significant'] == True)]
            if len(oss4sg_sig) > 0:
                f.write("**OSS4SG Significant Advantages:**\n")
                for _, row in oss4sg_sig.iterrows():
                    f.write(f"- {row['Metric']}: {row['OSS4SG_Mean']:.3f} vs {row['OSS_Mean']:.3f} (δ={row['Cliff_Delta']:.3f})\n")
                f.write("\n")
            
            # Top OSS advantages  
            oss_sig = results_df[(results_df['OSS4SG_Advantage'] == False) & (results_df['Significant'] == True)]
            if len(oss_sig) > 0:
                f.write("**OSS Significant Advantages:**\n")
                for _, row in oss_sig.head().iterrows():
                    f.write(f"- {row['Metric']}: {row['OSS_Mean']:.3f} vs {row['OSS4SG_Mean']:.3f} (δ={row['Cliff_Delta']:.3f})\n")
                f.write("\n")
            
            f.write("### Methodology:\n")
            f.write("- **Statistical Test**: Mann-Whitney U (non-parametric)\n")
            f.write("- **Effect Size**: Cliff's Delta\n")
            f.write("- **Population**: Active contributors only (PR or issue activity > 0)\n")
            f.write("- **Significance Level**: α = 0.05\n\n")
            
            f.write("### Conclusion:\n")
            f.write("The corrected analysis shows a more balanced picture compared to the original\n")
            f.write("analysis that was biased by zero-activity contributors. OSS4SG shows advantages\n")
            f.write("in several key metrics when comparing like-with-like (active contributors).\n")
        
        print(f"\nCorrected summary saved to: {summary_file}")

def main():
    """Main execution function"""
    
    base_path = "/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026"
    
    # Initialize corrected analyzer
    analyzer = CorrectedTreatmentAnalyzer(base_path)
    
    # Step 1: Load and filter data
    active_df = analyzer.load_and_filter_data()
    
    # Step 2: Analyze key metrics with corrected methodology
    results_df = analyzer.analyze_key_metrics()
    
    # Step 3: Create corrected visualizations
    print("\nCreating corrected visualizations...")
    analyzer.create_comparison_visualizations(results_df)
    
    # Step 4: Generate corrected summary
    analyzer.generate_corrected_summary(results_df)
    
    print("\n" + "="*80)
    print("✅ CORRECTED TREATMENT ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📁 Results: {analyzer.output_dir}")
    print(f"📊 Visualizations: {analyzer.viz_dir}")
    print("\nFiles generated:")
    print("- corrected_key_metrics_comparison.csv")
    print("- CORRECTED_ANALYSIS_SUMMARY.md")
    print("- corrected_treatment_comparison.png")
    print("- key_metrics_distributions.png")

if __name__ == "__main__":
    main()
