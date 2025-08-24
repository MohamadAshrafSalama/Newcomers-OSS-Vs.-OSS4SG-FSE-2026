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

PATTERN_NAMES = {0: "Early Spike", 1: "Sustained Activity", 2: "Low/Gradual Activity"}

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
        
        # Add pattern names
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
        # Prepare long form
        df_long = self.data[['group','weeks_to_core']].copy()
        groups = df_long['group'].unique().tolist()
        arrays = [df_long[df_long['group']==g]['weeks_to_core'].values for g in groups]

        h, p = stats.kruskal(*arrays)
        self.kw_stat, self.kw_p = h, p
        self.log(f"Kruskal–Wallis H={h:.3f} p={p:.3g}")

        # Initialize each group with its own rank (ensures distinct colors)
        ranks = {g: i+1 for i, g in enumerate(sorted(groups, key=lambda x: self.stats_df.set_index('group').loc[x, 'median_weeks']))}
        
        if p < 0.05 and HAS_POSTHOCS:
            self.log("Significant differences found. Running Dunn's test...")
            try:
                d = sp.posthoc_dunn(df_long, val_col='weeks_to_core', group_col='group')
                # Save matrix for transparency
                try:
                    d.to_csv(self.output_dir / 'dunn_pvalues.csv')
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
                
                # Order buckets by median and assign ranks
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

    def plot(self):
        self.log("Creating enhanced visualizations...")
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

        # Full-range boxplot
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.boxplot(data=plot_df, x='group', y='weeks_to_core', order=order, ax=ax,
                    palette=group_colors)
        ax.set_title('Time to Core by Pattern (Step 3.1 FIXED - Full range)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Project Type – Pattern', fontsize=12)
        ax.set_ylabel('Weeks to Core', fontsize=12)
        ax.set_xticklabels([label_map[g] for g in order], rotation=35, ha='right')
        
        # Add median annotations
        for i, g in enumerate(order):
            y = float(medians[g])
            rank = self.stats_df.set_index('group').loc[g, 'sk_rank']
            ax.text(i, y + 2, f'{y:.0f}\n(R{rank})', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add legend for ranks
        handles = [plt.Rectangle((0,0),1,1, color=rank_to_color[rk]) for rk in unique_ranks]
        labels = [f'Rank {rk}' for rk in unique_ranks]
        ax.legend(handles, labels, title='Scott-Knott Ranks', loc='upper right')
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'pattern_effectiveness.png', dpi=150, bbox_inches='tight')

        # Zoomed boxplot to the 95th percentile
        ymax = np.percentile(plot_df['weeks_to_core'], 95)
        fig2, ax2 = plt.subplots(figsize=(15, 7))
        sns.boxplot(data=plot_df, x='group', y='weeks_to_core', order=order, ax=ax2,
                    palette=group_colors)
        ax2.set_ylim(0, max(ymax, 50))
        ax2.set_title('Time to Core by Pattern (Step 3.1 FIXED - Zoomed ≤95th pct)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Project Type – Pattern', fontsize=12)
        ax2.set_ylabel('Weeks to Core', fontsize=12)
        ax2.set_xticklabels([label_map[g] for g in order], rotation=35, ha='right')
        
        for i, g in enumerate(order):
            y = float(medians[g])
            rank = self.stats_df.set_index('group').loc[g, 'sk_rank']
            y_pos = min(y + 2, ax2.get_ylim()[1] - 5)
            ax2.text(i, y_pos, f'{y:.0f}\n(R{rank})', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=rank_to_color[rk]) for rk in unique_ranks]
        labels = [f'Rank {rk}' for rk in unique_ranks]
        ax2.legend(handles, labels, title='Scott-Knott Ranks', loc='upper right')
        
        fig2.tight_layout()
        fig2.savefig(self.output_dir / 'pattern_effectiveness_zoom.png', dpi=150, bbox_inches='tight')

        # Sorted individual time-to-core plots per group (faceted)
        num_groups = len(order)
        cols = min(3, num_groups)
        rows = int(np.ceil(num_groups / cols))
        fig3, axes = plt.subplots(rows, cols, figsize=(5 * cols + 2, 3.8 * rows), sharey=False)
        if rows == 1 and cols == 1:
            axes = [axes]
        else:
            axes = np.atleast_1d(axes).reshape(rows, cols)
        
        for idx, g in enumerate(order):
            r = idx // cols
            c = idx % cols
            if rows == 1:
                axg = axes[c]
            else:
                axg = axes[r][c]
            
            vals = np.sort(plot_df[plot_df['group'] == g]['weeks_to_core'].astype(float).values)
            x = np.arange(1, len(vals) + 1)
            rank = self.stats_df.set_index('group').loc[g, 'sk_rank']
            
            axg.plot(x, vals, color=group_colors[g], marker='o', markersize=3, linewidth=1.5)
            axg.set_title(f"{label_map[g]}  (n={len(vals)}, med={int(medians[g])}, R{rank})", fontsize=11)
            axg.set_xlabel('Contributors (sorted by time)')
            axg.set_ylabel('Weeks to Core')
            axg.grid(True, alpha=0.3)
        
        # Hide any empty subplots
        for j in range(num_groups, rows * cols):
            r = j // cols
            c = j % cols
            if rows == 1:
                axes[c].axis('off') if c < len(axes) else None
            else:
                axes[r][c].axis('off')
        
        fig3.tight_layout()
        fig3.savefig(self.output_dir / 'time_to_core_sorted.png', dpi=150, bbox_inches='tight')

        self.log(f"Saved 3 visualizations with {len(unique_ranks)} distinct ranks")

    def plot_mean_time(self):
        """Bar chart of average (mean) time to core per group, colored by rank."""
        sns.set_style("whitegrid")
        sns.set_context("talk", font_scale=1.0)
        order_df = self.stats_df.sort_values(['sk_rank', 'mean_weeks']).copy()
        order = order_df['group'].tolist()
        unique_ranks = sorted(order_df['sk_rank'].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_ranks)))
        rank_to_color = {rk: colors[i] for i, rk in enumerate(unique_ranks)}
        bar_colors = [rank_to_color[rk] for rk in order_df['sk_rank']]

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.bar(range(len(order_df)), order_df['mean_weeks'], color=bar_colors)
        ax.set_xticks(range(len(order_df)))
        ax.set_xticklabels([g.replace('-', '\n') for g in order], rotation=35, ha='right')
        ax.set_ylabel('Mean Weeks to Core')
        ax.set_title('Average Time to Core by Pattern (colored by Scott–Knott rank)')
        # annotate means
        for i, v in enumerate(order_df['mean_weeks']):
            ax.text(i, v + 1, f"{v:.1f}", ha='center', va='bottom', fontsize=9)
        # legend
        handles = [plt.Rectangle((0,0),1,1, color=rank_to_color[rk]) for rk in unique_ranks]
        labels = [f'Rank {rk}' for rk in unique_ranks]
        ax.legend(handles, labels, title='Scott–Knott Ranks', loc='upper right')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'mean_time_to_core.png', dpi=150, bbox_inches='tight')

    def report(self):
        lines = []
        lines.append("="*70)
        lines.append("RQ3 Step 3.1 FIXED: CLUSTERS-ONLY ANALYSIS REPORT")
        lines.append("="*70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        lines.append(f"DATASET SUMMARY")
        lines.append("-"*40)
        lines.append(f"Total contributors analyzed: {len(self.data)}")
        lines.append(f"OSS contributors: {len(self.data[self.data['project_type']=='OSS'])}")
        lines.append(f"OSS4SG contributors: {len(self.data[self.data['project_type']=='OSS4SG'])}")
        lines.append("")
        
        lines.append(f"STATISTICAL ANALYSIS")
        lines.append("-"*40)
        lines.append(f"Kruskal–Wallis H={getattr(self,'kw_stat',np.nan):.3f} p={getattr(self,'kw_p',np.nan):.6f}")
        lines.append(f"Number of Scott-Knott ranks: {len(self.stats_df['sk_rank'].unique())}")
        lines.append("")
        
        lines.append("PATTERN EFFECTIVENESS RANKINGS")
        lines.append("-"*40)
        for _, row in self.stats_df.iterrows():
            lines.append(f"Rank {int(row['sk_rank'])}: {row['group']}")
            lines.append(f"  Median time to core: {row['median_weeks']:.0f} weeks")
            lines.append(f"  Sample size: {int(row['n'])} contributors")
            lines.append(f"  IQR: {row['q25_weeks']:.0f}-{row['q75_weeks']:.0f} weeks")
            lines.append("")
        
        text = "\n".join(lines)
        (self.output_dir / 'analysis_report.txt').write_text(text)
        print("\n" + text)

    def run(self):
        self.load_data()
        self.group_stats()
        self.assign_sk_ranks()
        self.plot()
        self.plot_mean_time()
        self.report()

if __name__ == '__main__':
    Step31ClustersOnlyFixed().run()