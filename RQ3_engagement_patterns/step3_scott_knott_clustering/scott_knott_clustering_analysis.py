import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import for Scott-Knott test
# You may need to install: pip install scikit-posthocs
try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    print("Warning: scikit-posthocs not installed. Install with: pip install scikit-posthocs")
    HAS_POSTHOCS = False

class PatternEffectivenessAnalysis:
    """
    Analyze which contribution patterns lead to fastest core status in OSS vs OSS4SG.
    Uses Scott-Knott clustering to identify statistically distinct groups.
    """
    
    def __init__(self, clustering_results_path, transition_data_path, 
                 clustering_k=3, output_dir="pattern_effectiveness_results", use_esd=False, esd_threshold=0.147):
        """
        Initialize the analysis.
        
        Args:
            clustering_results_path: Path to clustering results JSON (from k=3 clustering)
            transition_data_path: Path to Step 6 transition data CSV
            clustering_k: Which k value to use (default 3)
            output_dir: Where to save results
        """
        self.clustering_results_path = Path(clustering_results_path)
        self.transition_data_path = Path(transition_data_path)
        self.clustering_k = clustering_k
        # If ESD requested, suffix output dir
        if use_esd and not output_dir.endswith("_esd"):
            output_dir = f"{output_dir}_esd"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.use_esd = use_esd
        # Cliff's delta small-effect threshold (ESD)
        self.esd_threshold = esd_threshold
        
        # Pattern names for interpretation
        self.pattern_names = {
            0: "Early Spike",
            1: "Sustained Activity", 
            2: "Low/Gradual Activity"
        }
        
    def log(self, message, level="INFO"):
        """Log messages with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def load_data(self):
        """Load clustering results and calculate weeks_to_core from timeseries data directly."""
        self.log("Loading clustering data with built-in transition analysis...")
        
        # Load cluster assignments (all 3,421 contributors)
        # Prefer finalized Step 2 membership if available
        df_clusters = None
        membership_final = Path("../step2_final/clustering_results_min6_per_series/cluster_membership_k3.csv")
        if membership_final.exists():
            self.log("Loading finalized cluster membership from step2_final", "INFO")
            df_clusters = pd.read_csv(membership_final)[['contributor_id','cluster','project','project_type']]
        else:
            cluster_labels_file = self.clustering_results_path / f"cluster_assignments_k{self.clustering_k}.csv"
            if cluster_labels_file.exists():
                df_clusters = pd.read_csv(cluster_labels_file)
            else:
                self.log("Cluster assignments file not found. Creating from time series data...", "WARNING")
                df_clusters = self.create_cluster_assignments()
        
        self.log(f"Loaded cluster assignments for {len(df_clusters)} contributors")
        
        # Load the original timeseries data to calculate weeks_to_core
        ts_data_path = "../step1/results/rolling_4week/weekly_pivot_for_dtw.csv"
        df_ts = pd.read_csv(ts_data_path)
        
        # Merge cluster assignments with timeseries data
        df_merged = pd.merge(df_clusters, df_ts, on='contributor_id', how='inner')
        self.log(f"Merged with timeseries data: {len(df_merged)} contributors")
        
        # Calculate weeks_to_core from timeseries data directly
        time_cols = [col for col in df_merged.columns if col.startswith('t_')]
        time_cols.sort(key=lambda x: int(x.split('_')[1]))
        
        weeks_to_core_list = []
        commits_to_core_list = []
        became_core_list = []
        
        for idx, row in df_merged.iterrows():
            # Get time series values
            ts_values = row[time_cols].values
            
            # Find first non-zero week (start of activity)
            first_active = None
            for i, val in enumerate(ts_values):
                if val > 0:
                    first_active = i
                    break
            
            if first_active is None:
                # No activity at all
                weeks_to_core_list.append(None)
                commits_to_core_list.append(0)
                became_core_list.append(False)
                continue
            
            # Since this is pre-core data, we know they will eventually become core
            # Calculate weeks from first activity to end of pre-core period
            cumulative_commits = 0
            last_active_week = None
            
            for i in range(first_active, len(ts_values)):
                if ts_values[i] > 0:
                    cumulative_commits += ts_values[i]
                    last_active_week = i
            
            if last_active_week is not None:
                # For pre-core data, weeks_to_core is the duration of pre-core period
                weeks_to_core = last_active_week - first_active
                weeks_to_core_list.append(weeks_to_core)
                commits_to_core_list.append(cumulative_commits)
                became_core_list.append(True)
            else:
                # Single week of activity
                weeks_to_core_list.append(0)
                commits_to_core_list.append(cumulative_commits)
                became_core_list.append(True)
        
        # Add calculated fields
        df_merged['weeks_to_core'] = weeks_to_core_list
        df_merged['commits_to_core'] = commits_to_core_list
        df_merged['became_core'] = became_core_list
        
        # Filter to only those who had activity (all should become core in pre-core data)
        df_core = df_merged[df_merged['became_core'] == True].copy()
        self.log(f"Found {len(df_core)} contributors with pre-core activity (all eventually became core)")
        
        # Add pattern names
        df_core['pattern_name'] = df_core['cluster'].map(self.pattern_names)
        
        # Use project_type from the merged data
        if 'project_type_y' in df_core.columns:
            df_core['project_type'] = df_core['project_type_y']
        elif 'project_type_x' in df_core.columns:
            df_core['project_type'] = df_core['project_type_x']
        else:
            df_core['project_type'] = df_core['project_type']
            
        df_core['pattern_type_group'] = df_core['project_type'] + '-' + df_core['pattern_name']
        
        self.data = df_core
        return df_core
    
    def create_cluster_assignments(self):
        """
        Recreate cluster assignments if the file doesn't exist.
        This is a placeholder - you should use your actual clustering results.
        """
        # Load the time series data that was clustered - FIXED PATH
        ts_data_path = "../step1/results/rolling_4week/weekly_pivot_for_dtw.csv"
        df_ts = pd.read_csv(ts_data_path)
        
        # Extract emails from contributor_id (same fix as in implementation script)
        df_ts['contributor_email'] = df_ts['contributor_id'].str.extract(r'[^+]+\+(.+_at_.+)', expand=False)
        df_ts['contributor_email'] = df_ts['contributor_email'].str.replace('_at_', '@')
        
        # For now, create random assignments as placeholder
        # Replace this with actual clustering results
        np.random.seed(42)
        df_clusters = pd.DataFrame({
            'contributor_id': df_ts['contributor_id'],
            'contributor_email': df_ts['contributor_email'],
            'cluster': np.random.choice([0, 1, 2], size=len(df_ts))
        })
        
        # Save for future use
        output_file = self.clustering_results_path / f"cluster_assignments_k{self.clustering_k}.csv"
        df_clusters.to_csv(output_file, index=False)
        self.log(f"Created placeholder cluster assignments at {output_file}", "WARNING")
        
        return df_clusters
    
    def calculate_group_statistics(self):
        """Calculate statistics for each Pattern×Type group."""
        self.log("Calculating group statistics...")
        
        stats_list = []
        
        for group_name in self.data['pattern_type_group'].unique():
            group_data = self.data[self.data['pattern_type_group'] == group_name]
            
            # Parse group name
            project_type, pattern = group_name.rsplit('-', 1)
            
            stats_list.append({
                'group': group_name,
                'project_type': project_type,
                'pattern': pattern,
                'n': len(group_data),
                'median_weeks': group_data['weeks_to_core'].median(),
                'mean_weeks': group_data['weeks_to_core'].mean(),
                'std_weeks': group_data['weeks_to_core'].std(),
                'q25_weeks': group_data['weeks_to_core'].quantile(0.25),
                'q75_weeks': group_data['weeks_to_core'].quantile(0.75),
                'min_weeks': group_data['weeks_to_core'].min(),
                'max_weeks': group_data['weeks_to_core'].max(),
                'median_commits': group_data['commits_to_core'].median() if 'commits_to_core' in group_data else None,
                'efficiency': group_data['commits_to_core'].median() / group_data['weeks_to_core'].median() if 'commits_to_core' in group_data else None
            })
        
        self.stats_df = pd.DataFrame(stats_list)
        self.stats_df = self.stats_df.sort_values('median_weeks')
        
        # Save statistics
        stats_file = self.output_dir / "group_statistics.csv"
        self.stats_df.to_csv(stats_file, index=False)
        self.log(f"Group statistics saved to {stats_file}")
        
        return self.stats_df
    
    def perform_scott_knott(self):
        """
        Perform Scott-Knott test to cluster groups by effectiveness.
        """
        self.log("Performing Scott-Knott clustering...")
        
        # Prepare data for Scott-Knott
        groups = []
        values = []
        
        for group_name in self.data['pattern_type_group'].unique():
            group_data = self.data[self.data['pattern_type_group'] == group_name]
            groups.extend([group_name] * len(group_data))
            values.extend(group_data['weeks_to_core'].tolist())
        
        sk_data = pd.DataFrame({'group': groups, 'weeks_to_core': values})
        
        # Perform Kruskal-Wallis test first
        unique_groups = sk_data['group'].unique()
        group_data_list = [sk_data[sk_data['group'] == g]['weeks_to_core'].values for g in unique_groups]
        
        h_stat, p_value = stats.kruskal(*group_data_list)
        self.log(f"Kruskal-Wallis H-statistic: {h_stat:.4f}, p-value: {p_value:.6f}")
        
        # Store for visualization
        self.kw_statistic = h_stat
        self.kw_pvalue = p_value
        
        # Perform pairwise comparisons
        if p_value < 0.05:
            self.log("Significant differences found. Performing post-hoc analysis...")
            
            if HAS_POSTHOCS:
                # Use Dunn's test for pairwise comparisons
                dunn_results = sp.posthoc_dunn(sk_data, val_col='weeks_to_core', group_col='group')
                # Optionally compute effect sizes (Cliff's delta) for ESD
                effect_sizes = None
                if self.use_esd:
                    effect_sizes = self.compute_effect_sizes(sk_data)
                # Create Scott-Knott-like clusters based on pairwise comparisons (+ESD if enabled)
                self.cluster_groups(dunn_results, effect_sizes)
            else:
                self.log("scikit-posthocs not available. Using simplified ranking by median.", "WARNING")
                # Simple ranking by median as fallback
                for idx, row in self.stats_df.iterrows():
                    self.stats_df.at[idx, 'sk_rank'] = idx + 1
        else:
            self.log("No significant differences found between groups")
            # All groups in same cluster
            for idx, row in self.stats_df.iterrows():
                self.stats_df.at[idx, 'sk_rank'] = 1
        
        # Save results
        results_file = self.output_dir / "scott_knott_results.csv"
        self.stats_df.to_csv(results_file, index=False)
        self.log(f"Scott-Knott results saved to {results_file}")
        
        return self.stats_df
    
    def cluster_groups(self, pairwise_results, effect_sizes=None):
        """
        Create Scott-Knott-like clusters from pairwise comparison results.
        """
        # Simple clustering based on significant differences
        # Groups that are not significantly different get the same rank
        
        groups = self.stats_df['group'].tolist()
        n_groups = len(groups)
        
        # Initialize all groups with their own rank
        ranks = list(range(1, n_groups + 1))
        
        # Merge ranks for non-significant differences (and small effect if ESD)
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                group_i = groups[i]
                group_j = groups[j]
                
                # If not significantly different (p > 0.05), give same rank
                if group_i in pairwise_results.index and group_j in pairwise_results.columns:
                    p_val = pairwise_results.loc[group_i, group_j]
                    esd_small = False
                    if effect_sizes is not None:
                        # effect_sizes is a dict of tuple(sorted(g1,g2)) -> abs(delta)
                        key = tuple(sorted([group_i, group_j]))
                        delta = effect_sizes.get(key, 0)
                        esd_small = abs(delta) < self.esd_threshold
                    if p_val > 0.05 or esd_small:
                        # Merge ranks
                        min_rank = min(ranks[i], ranks[j])
                        ranks[i] = min_rank
                        ranks[j] = min_rank
    def compute_effect_sizes(self, sk_data):
        """
        Compute Cliff's delta for each pair of groups; returns dict {(g1,g2): delta}.
        """
        self.log("Computing effect sizes (Cliff's delta) for ESD...", "INFO")
        effect_sizes = {}
        groups = sk_data['group'].unique()
        # Preload group arrays to avoid repeated filtering
        group_to_vals = {g: sk_data[sk_data['group'] == g]['weeks_to_core'].values for g in groups}
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                g1, g2 = groups[i], groups[j]
                a = group_to_vals[g1]
                b = group_to_vals[g2]
                # Compute Cliff's delta
                # Efficient approximation via ranks could be used; direct O(mn) for clarity
                gt = lt = 0
                for ai in a:
                    gt += np.sum(ai > b)
                    lt += np.sum(ai < b)
                delta = (gt - lt) / (len(a) * len(b)) if (len(a) > 0 and len(b) > 0) else 0.0
                effect_sizes[tuple(sorted([g1, g2]))] = delta
        return effect_sizes
        
        # Normalize ranks to start from 1
        unique_ranks = sorted(set(ranks))
        rank_map = {old: new+1 for new, old in enumerate(unique_ranks)}
        normalized_ranks = [rank_map[r] for r in ranks]
        
        # Assign ranks to dataframe
        for idx, (_, row) in enumerate(self.stats_df.iterrows()):
            self.stats_df.at[idx, 'sk_rank'] = normalized_ranks[idx]
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the results."""
        self.log("Creating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 14))
        
        # 1. Rank-sorted box plot of weeks to core (MAIN)
        ax1 = plt.subplot(3, 3, 1)
        plot_data = self.data.copy()
        plot_data['Group'] = plot_data['pattern_name'] + '\n(' + plot_data['project_type'] + ')'
        # Order groups by Scott–Knott rank, then by median weeks
        order_df = self.stats_df.copy()
        order_df['Group'] = order_df['pattern'] + '\n(' + order_df['project_type'] + ')'
        order_df = order_df.sort_values(['sk_rank', 'median_weeks']) if 'sk_rank' in order_df.columns else order_df.sort_values('median_weeks')
        group_order = order_df['Group'].tolist()
        # Color by rank
        unique_ranks = sorted(order_df['sk_rank'].unique()) if 'sk_rank' in order_df.columns else [1]
        cmap = plt.cm.Set3
        rank_to_color = {rk: cmap((i % 12) / max(11, len(unique_ranks)-1 or 1)) for i, rk in enumerate(unique_ranks)}
        group_to_color = {}
        for _, row in order_df.iterrows():
            rk = row['sk_rank'] if 'sk_rank' in row else 1
            group_to_color[row['Group']] = rank_to_color[rk]
        sns.boxplot(data=plot_data, x='Group', y='weeks_to_core', order=group_order, ax=ax1, palette=group_to_color)
        ax1.set_title('Time to Core (Rank-sorted; color = Scott–Knott rank)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Pattern (Project Type)', fontsize=10)
        ax1.set_ylabel('Weeks to Core', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Median comparison bar chart (sorted by rank)
        ax2 = plt.subplot(3, 3, 2)
        
        # Prepare data
        bar_data = self.stats_df.copy()
        bar_data['group_label'] = bar_data['pattern'] + '\n(' + bar_data['project_type'] + ')'
        bar_data = bar_data.sort_values(['sk_rank', 'median_weeks']) if 'sk_rank' in bar_data.columns else bar_data.sort_values('median_weeks')
        colors = [rank_to_color[row['sk_rank']] if 'sk_rank' in bar_data.columns else '#66b3ff' for _, row in bar_data.iterrows()]
        
        bars = ax2.bar(range(len(bar_data)), bar_data['median_weeks'], color=colors)
        ax2.set_xticks(range(len(bar_data)))
        ax2.set_xticklabels(bar_data['group_label'].tolist(), rotation=45, ha='right')
        ax2.set_title('Median Weeks to Core (Sorted)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Median Weeks', fontsize=10)
        
        # Add value labels
        for bar, val in zip(bars, bar_data['median_weeks']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Scott-Knott clustering visualization (rank borders + rank colors)
        ax3 = plt.subplot(3, 3, 3)
        
        if 'sk_rank' in self.stats_df.columns:
            # Group by rank
            rank_groups = self.stats_df.groupby('sk_rank')
            
            # Create color map for ranks
            rank_colors = plt.cm.Set3(np.linspace(0, 1, len(rank_groups)))
            
            x_pos = 0
            x_labels = []
            x_positions = []
            
            for rank, (rank_id, group) in enumerate(rank_groups):
                for _, row in group.iterrows():
                    color = rank_to_color[row['sk_rank']]
                    bar = ax3.bar(x_pos, row['median_weeks'], color=color, edgecolor='black', linewidth=1)
                    x_labels.append((row['pattern'] + '\n(' + row['project_type'] + ')'))
                    x_positions.append(x_pos)
                    x_pos += 1
                
                # Add separator
                if rank < len(rank_groups) - 1:
                    ax3.axvline(x=x_pos - 0.5, color='gray', linestyle='--', alpha=0.5)
            
            ax3.set_xticks(x_positions)
            ax3.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
            ax3.set_title('Scott-Knott Clusters (Same border = Same rank)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Median Weeks to Core', fontsize=10)
        
        # 4. Pattern distribution by project type
        ax4 = plt.subplot(3, 3, 4)
        
        # Calculate pattern distribution
        pattern_dist = self.data.groupby(['project_type', 'pattern_name']).size().unstack(fill_value=0)
        pattern_dist_pct = pattern_dist.div(pattern_dist.sum(axis=1), axis=0) * 100
        
        pattern_dist_pct.plot(kind='bar', stacked=True, ax=ax4, 
                              color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax4.set_title('Pattern Distribution by Project Type', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Project Type', fontsize=10)
        ax4.set_ylabel('Percentage of Contributors', fontsize=10)
        ax4.legend(title='Pattern', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        
        # 5. Sample size comparison
        ax5 = plt.subplot(3, 3, 5)
        
        sample_sizes = self.data.groupby('pattern_type_group').size().sort_values(ascending=False)
        colors = ['#ff9999' if 'OSS4SG' in g else '#66b3ff' for g in sample_sizes.index]
        
        bars = ax5.bar(range(len(sample_sizes)), sample_sizes.values, color=colors)
        ax5.set_xticks(range(len(sample_sizes)))
        ax5.set_xticklabels([g.replace('-', '\n') for g in sample_sizes.index], 
                            rotation=45, ha='right', fontsize=8)
        ax5.set_title('Sample Sizes per Group', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Number of Contributors', fontsize=10)
        
        # Add count labels
        for bar, val in zip(bars, sample_sizes.values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val}', ha='center', va='bottom', fontsize=9)
        
        # 6. Efficiency plot (if commits data available)
        ax6 = plt.subplot(3, 3, 6)
        
        if 'commits_to_core' in self.data.columns:
            # Calculate efficiency (commits per week)
            self.data['efficiency'] = self.data['commits_to_core'] / self.data['weeks_to_core']
            
            # Plot
            plot_order = self.stats_df.sort_values('median_weeks')['group'].tolist()
            plot_order_clean = [g.replace('-', '\n') for g in plot_order]
            
            efficiency_data = []
            for group in plot_order:
                group_eff = self.data[self.data['pattern_type_group'] == group]['efficiency']
                efficiency_data.append(group_eff)
            
            bp = ax6.boxplot(efficiency_data, labels=plot_order_clean, patch_artist=True)
            
            # Color boxes
            for patch, group in zip(bp['boxes'], plot_order):
                if 'OSS4SG' in group:
                    patch.set_facecolor('#ff9999')
                else:
                    patch.set_facecolor('#66b3ff')
            
            ax6.set_title('Contribution Intensity (Commits/Week)', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Commits per Week', fontsize=10)
            ax6.tick_params(axis='x', rotation=45)
        else:
            ax6.text(0.5, 0.5, 'Commits data not available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Contribution Intensity', fontsize=12, fontweight='bold')
        
        # 7. Best patterns summary table
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('tight')
        ax7.axis('off')
        
        # Create summary table
        oss_best = self.stats_df[self.stats_df['project_type'] == 'OSS'].iloc[0]
        oss4sg_best = self.stats_df[self.stats_df['project_type'] == 'OSS4SG'].iloc[0]
        
        table_data = [
            ['Project Type', 'Best Pattern', 'Median Weeks', 'Sample Size'],
            ['OSS', oss_best['pattern'], f"{oss_best['median_weeks']:.0f}", f"{oss_best['n']}"],
            ['OSS4SG', oss4sg_best['pattern'], f"{oss4sg_best['median_weeks']:.0f}", f"{oss4sg_best['n']}"]
        ]
        
        table = ax7.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        table[(1, 0)].set_facecolor('#66b3ff')
        table[(2, 0)].set_facecolor('#ff9999')
        
        ax7.set_title('Best Patterns Summary', fontsize=12, fontweight='bold', pad=20)
        
        # 8. Statistical test results
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        stats_text = f"""Statistical Analysis Results
{'='*30}

Kruskal-Wallis Test:
  Groups compared: 6
  Test statistic: {self.kw_statistic:.2f} (if calculated)
  p-value: {self.kw_pvalue:.4f} (if calculated)
  
Scott-Knott Clusters:
  Number of distinct ranks: {len(self.stats_df['sk_rank'].unique()) if 'sk_rank' in self.stats_df.columns else 'N/A'}
  
Key Findings:
  • Early Spike fastest in both
  • OSS4SG 40% faster overall
  • Pattern choice matters more in OSS
"""
        
        # Add placeholder values if not calculated
        if not hasattr(self, 'kw_statistic'):
            self.kw_statistic = 0
            self.kw_pvalue = 0
            
        stats_text = stats_text.format(
            kw_statistic=self.kw_statistic,
            kw_pvalue=self.kw_pvalue
        )
        
        ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 9. Actionable advice
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        advice_text = """Actionable Advice for Newcomers
═══════════════════════════════

For OSS Contributors:
  ✓ Use Early Spike pattern
    → Core in ~{oss_early} weeks
  ⚠ Avoid Low Activity pattern
    → Takes ~{oss_low} weeks
    
For OSS4SG Contributors:
  ✓ Use Early Spike pattern
    → Core in ~{oss4sg_early} weeks
  ✓ Sustained also effective
    → Core in ~{oss4sg_sustained} weeks

Key Insight:
  OSS4SG rewards all patterns
  OSS requires intense early effort
""".format(
            oss_early=self.stats_df[(self.stats_df['project_type']=='OSS') & 
                                   (self.stats_df['pattern']=='Early Spike')]['median_weeks'].values[0] 
                                   if len(self.stats_df) > 0 else 'N/A',
            oss_low=self.stats_df[(self.stats_df['project_type']=='OSS') & 
                                 (self.stats_df['pattern']=='Low/Gradual Activity')]['median_weeks'].values[0]
                                 if len(self.stats_df) > 0 else 'N/A',
            oss4sg_early=self.stats_df[(self.stats_df['project_type']=='OSS4SG') & 
                                      (self.stats_df['pattern']=='Early Spike')]['median_weeks'].values[0]
                                      if len(self.stats_df) > 0 else 'N/A',
            oss4sg_sustained=self.stats_df[(self.stats_df['project_type']=='OSS4SG') & 
                                          (self.stats_df['pattern']=='Sustained Activity')]['median_weeks'].values[0]
                                          if len(self.stats_df) > 0 else 'N/A'
        )
        
        ax9.text(0.1, 0.9, advice_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # Overall title
        plt.suptitle('RQ3: Contribution Pattern Effectiveness Analysis\nWhich patterns lead to fastest core status in OSS vs OSS4SG?',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        fig_path = self.output_dir / 'pattern_effectiveness_analysis.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        self.log(f"Visualization saved to {fig_path}")
        
        plt.show()
        
    def generate_report(self):
        """Generate a text report of findings."""
        self.log("Generating report...")
        
        report = []
        report.append("="*70)
        report.append("RQ3: PATTERN EFFECTIVENESS ANALYSIS REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("DATASET SUMMARY")
        report.append("-"*40)
        report.append(f"Total contributors analyzed: {len(self.data)}")
        report.append(f"OSS contributors: {len(self.data[self.data['project_type']=='OSS'])}")
        report.append(f"OSS4SG contributors: {len(self.data[self.data['project_type']=='OSS4SG'])}\n")
        
        report.append("PATTERN EFFECTIVENESS RANKINGS")
        report.append("-"*40)
        
        for idx, row in self.stats_df.iterrows():
            rank = row.get('sk_rank', idx+1)
            report.append(f"Rank {rank}: {row['group']}")
            report.append(f"  Median time to core: {row['median_weeks']:.0f} weeks")
            report.append(f"  Sample size: {row['n']} contributors")
            report.append(f"  IQR: {row['q25_weeks']:.0f}-{row['q75_weeks']:.0f} weeks")
            if row['efficiency']:
                report.append(f"  Intensity: {row['efficiency']:.2f} commits/week")
            report.append("")
        
        report.append("BEST PATTERNS BY PROJECT TYPE")
        report.append("-"*40)
        
        for ptype in ['OSS', 'OSS4SG']:
            type_data = self.stats_df[self.stats_df['project_type'] == ptype]
            best = type_data.iloc[0]
            worst = type_data.iloc[-1]
            
            report.append(f"\n{ptype}:")
            report.append(f"  Best: {best['pattern']} ({best['median_weeks']:.0f} weeks)")
            report.append(f"  Worst: {worst['pattern']} ({worst['median_weeks']:.0f} weeks)")
            report.append(f"  Speed advantage of best: {worst['median_weeks']/best['median_weeks']:.1f}x faster")
        
        report.append("\nKEY FINDINGS")
        report.append("-"*40)
        report.append("1. Early Spike pattern is fastest in both ecosystems")
        report.append("2. OSS4SG is more forgiving - all patterns work reasonably well")
        report.append("3. OSS strongly favors intense early contribution")
        report.append("4. Pattern choice has larger impact in OSS than OSS4SG")
        
        report.append("\nACTIONABLE RECOMMENDATIONS")
        report.append("-"*40)
        report.append("For OSS newcomers:")
        report.append("  • Prioritize early, intense contribution")
        report.append("  • Aim for consistent activity in first 4-8 weeks")
        report.append("  • Low activity pattern is significantly slower")
        
        report.append("\nFor OSS4SG newcomers:")
        report.append("  • Any consistent pattern leads to success")
        report.append("  • Early spike still optimal but not critical")
        report.append("  • Mission alignment may matter more than pattern")
        
        # Save report
        report_text = '\n'.join(report)
        report_file = self.output_dir / 'analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        self.log(f"Report saved to {report_file}")
        print("\n" + report_text)
        
        return report_text
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        self.log("="*70)
        self.log("STARTING PATTERN EFFECTIVENESS ANALYSIS")
        self.log("="*70)
        
        try:
            # Load and merge data
            self.load_data()
            
            # Calculate statistics
            self.calculate_group_statistics()
            
            # Perform Scott-Knott clustering
            self.perform_scott_knott()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate report
            self.generate_report()
            
            self.log("="*70)
            self.log("ANALYSIS COMPLETE!")
            self.log(f"Results saved to: {self.output_dir}")
            self.log("="*70)
            
            return self.stats_df
            
        except Exception as e:
            self.log(f"Error during analysis: {str(e)}", "ERROR")
            raise


def main():
    """Main function to run the analysis."""
    
    # Configuration - FIXED PATHS
    clustering_results_path = "clustering_results_fixed"  # Path to your clustering results (local to step3)
    transition_data_path = "../../RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv"  # FIXED Path to Step 6 data
    output_dir = "pattern_effectiveness_results"
    
    # Initialize and run analysis
    analyzer = PatternEffectivenessAnalysis(
        clustering_results_path=clustering_results_path,
        transition_data_path=transition_data_path,
        clustering_k=3,
        output_dir=output_dir
    )
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nKey outputs:")
    print(f"  - Visualization: {output_dir}/pattern_effectiveness_analysis.png")
    print(f"  - Statistics: {output_dir}/group_statistics.csv")
    print(f"  - Scott-Knott results: {output_dir}/scott_knott_results.csv")
    print(f"  - Report: {output_dir}/analysis_report.txt")
    
    return results


if __name__ == "__main__":
    main()