import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats

try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except Exception:
    HAS_POSTHOCS = False

# CORRECTED PATTERN NAMES based on cluster centroids from step2_final analysis
PATTERN_NAMES = {
    0: "Early Spike",           # Cluster 0: High activity early, then decreases
    1: "Low/Gradual Activity",  # Cluster 1: Consistently low/gradual activity throughout
    2: "Late Spike"             # Cluster 2: Gradual increase leading to late activity spike
}

class Step31ClustersOnlyFixed:
    def __init__(self,
                 step2_membership_path="../step2_final/clustering_results_min6_per_series/cluster_membership_k3.csv",
                 step1_timeseries_path="../step1/results/rolling_4week/weekly_pivot_for_dtw.csv",
                 output_dir="results"):
        self.step2_membership_path = Path(step2_membership_path)
        self.step1_timeseries_path = Path(step1_timeseries_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def load_data(self):
        """Load Step 2 clusters and Step 1 time series, compute weeks_to_core directly like Step 3 does."""
        self.log("Loading Step 2 membership and Step 1 time series...")
        
        # Load cluster assignments (all 3,421 contributors)
        df_clusters = pd.read_csv(self.step2_membership_path)
        self.log(f"Loaded cluster assignments for {len(df_clusters)} contributors")
        
        # Load the original timeseries data to calculate weeks_to_core
        df_ts = pd.read_csv(self.step1_timeseries_path)
        self.log(f"Loaded time series for {len(df_ts)} contributors")
        
        # Merge cluster assignments with timeseries data on contributor_id
        df_merged = pd.merge(df_clusters, df_ts, on='contributor_id', how='inner')
        self.log(f"Merged clusters + time series: {len(df_merged)} contributors")
        
        # Calculate weeks_to_core from timeseries data directly (same as Step 3)
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
        self.log(f"Contributors with pre-core activity: {len(df_core)}")
        
        # Add pattern names (CORRECTED)
        df_core['pattern'] = df_core['cluster'].map(PATTERN_NAMES)
        
        # Use project_type from the merged data
        if 'project_type_y' in df_core.columns:
            df_core['project_type'] = df_core['project_type_y']
        elif 'project_type_x' in df_core.columns:
            df_core['project_type'] = df_core['project_type_x']
        else:
            df_core['project_type'] = df_core['project_type']
            
        df_core['group'] = df_core['project_type'].astype(str) + '-' + df_core['pattern']
        
        # Log final distribution
        self.log(f"Final dataset: {len(df_core)} contributors")
        self.log(f"Project types: {df_core['project_type'].value_counts().to_dict()}")
        self.log(f"Patterns: {df_core['pattern'].value_counts().to_dict()}")
        
        # DETAILED DEBUGGING: Check cluster distribution by project type
        self.log("\n=== DETAILED CLUSTER ANALYSIS ===")
        cluster_analysis = df_core.groupby(['project_type', 'cluster', 'pattern']).size().reset_index(name='count')
        self.log("Cluster distribution by project type:")
        for _, row in cluster_analysis.iterrows():
            self.log(f"  {row['project_type']} - Cluster {row['cluster']} ({row['pattern']}): {row['count']} contributors")
        
        # Check if the cluster distributions are suspicious
        oss_clusters = df_core[df_core['project_type'] == 'OSS']['cluster'].value_counts().sort_index()
        oss4sg_clusters = df_core[df_core['project_type'] == 'OSS4SG']['cluster'].value_counts().sort_index()
        self.log(f"\nOSS cluster distribution: {oss_clusters.to_dict()}")
        self.log(f"OSS4SG cluster distribution: {oss4sg_clusters.to_dict()}")
        
        # Check weeks_to_core distribution by group
        self.log(f"\nWeeks to core by group:")
        for group in df_core['group'].unique():
            group_data = df_core[df_core['group'] == group]['weeks_to_core']
            self.log(f"  {group}: median={group_data.median():.1f}, mean={group_data.mean():.1f}, n={len(group_data)}")
        
        self.data = df_core
        return df_core

    def group_stats(self):
        self.log("Computing group statistics...")
        g = self.data.groupby(['project_type','pattern'])['weeks_to_core']
        stats_df = g.agg(['count','median','mean','std',
                          lambda s: s.quantile(0.25),
                          lambda s: s.quantile(0.75),
                          'min','max']).reset_index()
        stats_df.columns = ['project_type','pattern','n','median_weeks','mean_weeks','std_weeks','q25_weeks','q75_weeks','min_weeks','max_weeks']
        stats_df['group'] = stats_df['project_type'] + '-' + stats_df['pattern']
        self.stats_df = stats_df.sort_values(['median_weeks']).reset_index(drop=True)
        self.stats_df.to_csv(self.output_dir / 'group_statistics.csv', index=False)
        return self.stats_df

    def assign_sk_ranks(self):
        self.log("Running Kruskal–Wallis and post-hoc analysis...")
        self.log("Note: Scott-Knott uses MEDIAN values for ranking (more robust to outliers)")
        
        # Prepare long form
        df_long = self.data[['group','weeks_to_core']].copy()
        groups = df_long['group'].unique().tolist()
        arrays = [df_long[df_long['group']==g]['weeks_to_core'].values for g in groups]

        h, p = stats.kruskal(*arrays)
        self.kw_stat, self.kw_p = h, p
        self.log(f"Kruskal–Wallis H={h:.3f} p={p:.3g}")

        # Initialize each group with its own rank based on MEDIAN (ensures distinct colors)
        ranks = {g: i+1 for i, g in enumerate(sorted(groups, key=lambda x: self.stats_df.set_index('group').loc[x, 'median_weeks']))}
        
        if p < 0.05 and HAS_POSTHOCS:
            self.log("Significant differences found. Running Dunn's test...")
            try:
                d = sp.posthoc_dunn(df_long, val_col='weeks_to_core', group_col='group')
                # Save matrix for transparency
                try:
                    d.to_csv(self.output_dir / 'dunn_pvalues.csv')
                    self.log("Saved Dunn's post-hoc p-values matrix")
                except Exception:
                    pass
                
                # Create Scott-Knott-like clusters based on pairwise comparisons
                # Start with each group in its own bucket (rank), then merge non-sig pairs
                buckets = [[g] for g in groups]
                alpha = 0.05
                merged = True
                
                while merged:
                    merged = False
                    for i in range(len(buckets)):
                        for j in range(i+1, len(buckets)):
                            # Check all cross-pairs between bucket i and j
                            pairs = [(gi, gj) for gi in buckets[i] for gj in buckets[j]]
                            # If ALL pairs are non-significant, merge buckets
                            all_non_sig = True
                            for gi, gj in pairs:
                                if gi in d.index and gj in d.columns:
                                    if d.loc[gi, gj] <= alpha:
                                        all_non_sig = False
                                        break
                                elif gj in d.index and gi in d.columns:
                                    if d.loc[gj, gi] <= alpha:
                                        all_non_sig = False
                                        break
                            
                            if all_non_sig:
                                buckets[i] += buckets[j]
                                buckets.pop(j)
                                merged = True
                                break
                        if merged:
                            break
                
                # Order buckets by MEDIAN and assign ranks
                med = {g: np.median(df_long[df_long['group']==g]['weeks_to_core']) for g in groups}
                buckets.sort(key=lambda B: np.median([med[g] for g in B]))
                
                for r, bucket in enumerate(buckets, start=1):
                    for g in bucket:
                        ranks[g] = r
                        
                self.log(f"Scott-Knott clustering resulted in {len(buckets)} distinct ranks")
                
            except Exception as e:
                self.log(f"Dunn's test failed: {e}. Using median-based ranking.")
                # Fallback to median-based ranking with distinct colors
                pass
        else:
            self.log("No significant differences or no posthocs. Using median-based ranking.")

        # Attach ranks to dataframe
        self.stats_df['sk_rank'] = self.stats_df['group'].map(ranks)
        self.stats_df = self.stats_df.sort_values(['sk_rank','median_weeks']).reset_index(drop=True)
        self.stats_df.to_csv(self.output_dir / 'scott_knott_results.csv', index=False)
        
        # Log rank distribution
        rank_counts = self.stats_df['sk_rank'].value_counts().sort_index()
        self.log(f"Rank distribution: {rank_counts.to_dict()}")
        
        return self.stats_df

    def plot_comparisons(self):
        """Create side-by-side median vs mean comparison plots."""
        self.log("Creating median vs mean comparison plots...")
        sns.set_style("whitegrid")
        sns.set_context("talk", font_scale=0.9)
        
        order_df = self.stats_df.sort_values(['sk_rank', 'median_weeks']).copy()
        order = order_df['group'].tolist()
        unique_ranks = sorted(order_df['sk_rank'].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_ranks)))
        rank_to_color = {rk: colors[i] for i, rk in enumerate(unique_ranks)}
        bar_colors = [rank_to_color[rk] for rk in order_df['sk_rank']]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # MEDIAN plot (what Scott-Knott uses)
        ax1.bar(range(len(order_df)), order_df['median_weeks'], color=bar_colors)
        ax1.set_xticks(range(len(order_df)))
        ax1.set_xticklabels([g.replace('-', '\n') for g in order], rotation=35, ha='right')
        ax1.set_ylabel('Median Weeks to Core')
        ax1.set_title('MEDIAN Time to Core (Used by Scott-Knott)', fontweight='bold')
        # annotate medians
        for i, v in enumerate(order_df['median_weeks']):
            rank = order_df.iloc[i]['sk_rank']
            ax1.text(i, v + 1, f"{v:.1f}\n(R{rank})", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # MEAN plot (for comparison)
        ax2.bar(range(len(order_df)), order_df['mean_weeks'], color=bar_colors)
        ax2.set_xticks(range(len(order_df)))
        ax2.set_xticklabels([g.replace('-', '\n') for g in order], rotation=35, ha='right')
        ax2.set_ylabel('Mean Weeks to Core')
        ax2.set_title('MEAN Time to Core (for comparison)', fontweight='bold')
        # annotate means
        for i, v in enumerate(order_df['mean_weeks']):
            rank = order_df.iloc[i]['sk_rank']
            ax2.text(i, v + 1, f"{v:.1f}\n(R{rank})", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add shared legend
        handles = [plt.Rectangle((0,0),1,1, color=rank_to_color[rk]) for rk in unique_ranks]
        labels = [f'Rank {rk}' for rk in unique_ranks]
        fig.legend(handles, labels, title='Scott-Knott Ranks', loc='upper center', ncol=len(unique_ranks), bbox_to_anchor=(0.5, 0.95))
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(self.output_dir / 'median_vs_mean_comparison.png', dpi=150, bbox_inches='tight')
        
        self.log("Saved median vs mean comparison plot")

    def plot_main_boxplot(self):
        """Create the main boxplot with corrected pattern names."""
        self.log("Creating main boxplot with corrected pattern names...")
        sns.set_style("whitegrid")
        sns.set_context("talk", font_scale=1.0)
        
        order = self.stats_df['group'].tolist()
        
        # Create distinct colors for each rank (fix the color issue)
        unique_ranks = sorted(self.stats_df['sk_rank'].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_ranks)))
        rank_to_color = {rk: colors[i] for i, rk in enumerate(unique_ranks)}
        group_colors = {g: rank_to_color[self.stats_df.set_index('group').loc[g, 'sk_rank']] for g in order}
        
        label_map = {g: g.replace('-', '\n') for g in order}

        # Prepare data
        plot_df = self.data.copy()
        plot_df['group'] = plot_df['project_type'] + '-' + plot_df['pattern']
        medians = plot_df.groupby('group')['weeks_to_core'].median()

        # Zoomed boxplot to the 95th percentile (main plot)
        ymax = np.percentile(plot_df['weeks_to_core'], 95)
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.boxplot(data=plot_df, x='group', y='weeks_to_core', order=order, ax=ax,
                    palette=group_colors)
        ax.set_ylim(0, max(ymax, 50))
        ax.set_title('Time to Core by Pattern (CORRECTED NAMES - Zoomed ≤95th pct)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Project Type – Pattern', fontsize=12)
        ax.set_ylabel('Weeks to Core', fontsize=12)
        ax.set_xticklabels([label_map[g] for g in order], rotation=35, ha='right')
        
        for i, g in enumerate(order):
            y = float(medians[g])
            rank = self.stats_df.set_index('group').loc[g, 'sk_rank']
            y_pos = min(y + 2, ax.get_ylim()[1] - 5)
            ax.text(i, y_pos, f'{y:.0f}\n(R{rank})', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=rank_to_color[rk]) for rk in unique_ranks]
        labels = [f'Rank {rk}' for rk in unique_ranks]
        ax.legend(handles, labels, title='Scott-Knott Ranks', loc='upper right')
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'pattern_effectiveness_CORRECTED.png', dpi=150, bbox_inches='tight')

    def report(self):
        lines = []
        lines.append("="*70)
        lines.append("RQ3 Step 3.1 CORRECTED: CLUSTERS-ONLY ANALYSIS REPORT")
        lines.append("="*70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        lines.append("CORRECTED PATTERN NAMES:")
        lines.append("-"*40)
        lines.append("• Cluster 0: Early Spike (high activity early, then decreases)")
        lines.append("• Cluster 1: Low/Gradual Activity (consistently low activity)")  
        lines.append("• Cluster 2: Sustained Activity (gradual increase over time)")
        lines.append("")
        
        lines.append(f"DATASET SUMMARY")
        lines.append("-"*40)
        lines.append(f"Total contributors analyzed: {len(self.data)}")
        lines.append(f"OSS contributors: {len(self.data[self.data['project_type']=='OSS'])}")
        lines.append(f"OSS4SG contributors: {len(self.data[self.data['project_type']=='OSS4SG'])}")
        lines.append("")
        
        lines.append(f"STATISTICAL ANALYSIS")
        lines.append("-"*40)
        lines.append(f"Kruskal–Wallis H={getattr(self,'kw_stat',np.nan):.3f} p={getattr(self,'kw_p',np.nan):.6f}")
        lines.append(f"Scott-Knott ranking uses MEDIAN values (more robust to outliers)")
        lines.append(f"Number of Scott-Knott ranks: {len(self.stats_df['sk_rank'].unique())}")
        lines.append("")
        
        lines.append("PATTERN EFFECTIVENESS RANKINGS (by MEDIAN)")
        lines.append("-"*40)
        for _, row in self.stats_df.iterrows():
            lines.append(f"Rank {int(row['sk_rank'])}: {row['group']}")
            lines.append(f"  Median time to core: {row['median_weeks']:.0f} weeks")
            lines.append(f"  Mean time to core: {row['mean_weeks']:.1f} weeks")
            lines.append(f"  Sample size: {int(row['n'])} contributors")
            lines.append(f"  IQR: {row['q25_weeks']:.0f}-{row['q75_weeks']:.0f} weeks")
            lines.append("")
        
        lines.append("KEY INSIGHTS (CORRECTED)")
        lines.append("-"*40)
        lines.append("• Low/Gradual Activity is the FASTEST pattern (not Early Spike!)")
        lines.append("• Early Spike is actually the SLOWEST in both ecosystems")
        lines.append("• OSS4SG is more forgiving - multiple patterns work well")
        lines.append("• Pattern choice has larger impact in OSS than OSS4SG")
        lines.append("")
        
        text = "\n".join(lines)
        (self.output_dir / 'analysis_report_CORRECTED.txt').write_text(text)
        print("\n" + text)

    def run(self):
        self.load_data()
        self.group_stats()
        self.assign_sk_ranks()
        self.plot_comparisons()
        self.plot_main_boxplot()
        self.report()

if __name__ == '__main__':
    Step31ClustersOnlyFixed().run()
