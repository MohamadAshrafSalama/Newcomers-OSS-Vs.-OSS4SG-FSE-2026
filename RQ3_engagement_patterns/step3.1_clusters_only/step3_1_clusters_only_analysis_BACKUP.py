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

class Step31ClustersOnly:
    def __init__(self,
                 step2_membership_path="../step2_final/clustering_results_min6_per_series/cluster_membership_k3.csv",
                 transitions_path="../../RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv",
                 output_dir="results"):
        self.step2_membership_path = Path(step2_membership_path)
        self.transitions_path = Path(transitions_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def load_data(self):
        self.log("Loading Step 2 membership and RQ1 transitions...")
        df_clusters = pd.read_csv(self.step2_membership_path)
        df_trans = pd.read_csv(self.transitions_path)

        # Normalize ids
        for df in (df_clusters, df_trans):
            if 'contributor_email' in df.columns:
                df['merge_id'] = df['contributor_email'].astype(str).str.strip().str.lower()
            else:
                df['merge_id'] = df['contributor_id'].astype(str).str.strip().str.lower()

        # Prepare cluster info
        col_map = {}
        if 'cluster' in df_clusters.columns:
            col_map['cluster'] = 'cluster'
        elif 'label' in df_clusters.columns:
            col_map['cluster'] = 'label'
        else:
            raise RuntimeError("No cluster column in Step 2 membership")

        # Keep only needed columns
        keep_cols = ['merge_id', col_map['cluster']]
        if 'project_type' in df_clusters.columns: keep_cols.append('project_type')
        if 'project' in df_clusters.columns: keep_cols.append('project')
        df_clusters = df_clusters[keep_cols].rename(columns={col_map['cluster']: 'cluster'})

        # Filter transitions to became_core True if column exists
        if 'became_core' in df_trans.columns:
            df_trans = df_trans[df_trans['became_core'] == True].copy()

        # Determine weeks_to_core
        if 'weeks_to_core' not in df_trans.columns:
            # Fallback using week columns if available
            if 'first_core_week' in df_trans.columns and 'first_commit_week' in df_trans.columns:
                df_trans['weeks_to_core'] = df_trans['first_core_week'] - df_trans['first_commit_week']
            elif 'time_to_event_weeks' in df_trans.columns:
                df_trans['weeks_to_core'] = df_trans['time_to_event_weeks']
            else:
                raise RuntimeError("Cannot determine weeks_to_core from transitions")

        # Join
        df = pd.merge(df_clusters, df_trans[['merge_id', 'weeks_to_core']], on='merge_id', how='inner')

        # Clean
        df = df.dropna(subset=['weeks_to_core'])
        df['weeks_to_core'] = df['weeks_to_core'].astype(float)
        df['pattern'] = df['cluster'].map(PATTERN_NAMES)
        if 'project_type' not in df.columns:
            df['project_type'] = np.where(df['project'].astype(str).str.contains('/'),
                                          df['project_type'], df['project_type']) if 'project' in df.columns else 'OSS'
        df['group'] = df['project_type'].astype(str) + '-' + df['pattern']
        self.data = df
        return df

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
        self.log("Running Kruskal–Wallis and post-hoc Dunn, assigning ranks...")
        # Prepare long form
        df_long = self.data[['group','weeks_to_core']].copy()
        groups = df_long['group'].unique().tolist()
        arrays = [df_long[df_long['group']==g]['weeks_to_core'].values for g in groups]

        h, p = stats.kruskal(*arrays)
        self.kw_stat, self.kw_p = h, p
        self.log(f"Kruskal–Wallis H={h:.3f} p={p:.3g}")

        ranks = {g: 1 for g in groups}
        if p < 0.05 and HAS_POSTHOCS:
            d = sp.posthoc_dunn(df_long, val_col='weeks_to_core', group_col='group')
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
                        # If ALL pairs are non-significant, merge
                        if all(((gi in d.index and gj in d.columns and d.loc[gi, gj] > alpha) or
                                (gj in d.index and gi in d.columns and d.loc[gj, gi] > alpha)) for gi, gj in pairs):
                            buckets[i] += buckets[j]
                            buckets.pop(j)
                            merged = True
                            break
                    if merged:
                        break
            # Order buckets by median
            med = {g: np.median(df_long[df_long['group']==g]['weeks_to_core']) for g in groups}
            buckets.sort(key=lambda B: np.median([med[g] for g in B]))
            for r, B in enumerate(buckets, start=1):
                for g in B:
                    ranks[g] = r
        else:
            # No significance or no posthocs: rank by median order (distinct colors)
            med = self.stats_df.set_index('group')['median_weeks'].to_dict()
            order = sorted(groups, key=lambda g: med[g])
            ranks = {g: i+1 for i, g in enumerate(order)}

        # Attach
        self.stats_df['sk_rank'] = self.stats_df['group'].map(ranks)
        self.stats_df = self.stats_df.sort_values(['sk_rank','median_weeks']).reset_index(drop=True)
        self.stats_df.to_csv(self.output_dir / 'scott_knott_results.csv', index=False)
        return self.stats_df

    def plot(self):
        self.log("Plotting...")
        sns.set_style("whitegrid")
        sns.set_context("talk", font_scale=1.0)
        order = self.stats_df['group'].tolist()
        rank_colors = {rk: plt.cm.Set3((i % 12)/11) for i, rk in enumerate(sorted(self.stats_df['sk_rank'].unique()))}
        group_colors = {g: rank_colors[self.stats_df.set_index('group').loc[g, 'sk_rank']] for g in order}
        label_map = {g: g.replace('-', '\n') for g in order}

        # Prepare data
        plot_df = self.data.copy()
        plot_df['group'] = plot_df['project_type'] + '-' + plot_df['pattern']
        medians = plot_df.groupby('group')['weeks_to_core'].median()

        # Full-range boxplot
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=plot_df, x='group', y='weeks_to_core', order=order, ax=ax,
                    palette=group_colors)
        ax.set_title('Time to Core by Pattern (Step 3.1 - Full range)')
        ax.set_xlabel('Project Type – Pattern')
        ax.set_ylabel('Weeks to Core')
        ax.set_xticklabels([label_map[g] for g in order], rotation=35, ha='right')
        for i, g in enumerate(order):
            y = float(medians[g])
            ax.text(i, y + 0.5, f'{y:.0f}', ha='center', va='bottom', fontsize=9)
        fig.tight_layout()
        fig.savefig(self.output_dir / 'pattern_effectiveness.png', dpi=150, bbox_inches='tight')

        # Zoomed boxplot to the 95th percentile
        ymax = np.percentile(plot_df['weeks_to_core'], 95)
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=plot_df, x='group', y='weeks_to_core', order=order, ax=ax2,
                    palette=group_colors)
        ax2.set_ylim(0, max(ymax, 30))
        ax2.set_title('Time to Core by Pattern (Step 3.1 - Zoomed ≤95th pct)')
        ax2.set_xlabel('Project Type – Pattern')
        ax2.set_ylabel('Weeks to Core')
        ax2.set_xticklabels([label_map[g] for g in order], rotation=35, ha='right')
        for i, g in enumerate(order):
            y = float(medians[g])
            ax2.text(i, min(y, ax2.get_ylim()[1]) + 0.5, f'{y:.0f}', ha='center', va='bottom', fontsize=9)
        fig2.tight_layout()
        fig2.savefig(self.output_dir / 'pattern_effectiveness_zoom.png', dpi=150, bbox_inches='tight')

        # Sorted individual time-to-core plots per group (faceted)
        num_groups = len(order)
        cols = min(3, num_groups)
        rows = int(np.ceil(num_groups / cols))
        fig3, axes = plt.subplots(rows, cols, figsize=(5 * cols + 2, 3.8 * rows), sharey=False)
        axes = np.atleast_1d(axes).reshape(rows, cols)
        for idx, g in enumerate(order):
            r = idx // cols
            c = idx % cols
            axg = axes[r][c]
            vals = np.sort(plot_df[plot_df['group'] == g]['weeks_to_core'].astype(float).values)
            x = np.arange(1, len(vals) + 1)
            axg.plot(x, vals, color=group_colors[g], marker='o', markersize=2, linewidth=1)
            axg.set_title(f"{label_map[g]}  (n={len(vals)}, med={int(medians[g])})", fontsize=11)
            axg.set_xlabel('Contributors (sorted by time)')
            axg.set_ylabel('Weeks to Core')
        # Hide any empty subplots
        for j in range(num_groups, rows * cols):
            r = j // cols
            c = j % cols
            axes[r][c].axis('off')
        fig3.tight_layout()
        fig3.savefig(self.output_dir / 'time_to_core_sorted.png', dpi=150, bbox_inches='tight')

        # Zoomed version of the sorted plots (≤95th percentile overall)
        ymax_z = np.percentile(plot_df['weeks_to_core'], 95)
        fig4, axes2 = plt.subplots(rows, cols, figsize=(5 * cols + 2, 3.8 * rows), sharey=False)
        axes2 = np.atleast_1d(axes2).reshape(rows, cols)
        for idx, g in enumerate(order):
            r = idx // cols
            c = idx % cols
            axg2 = axes2[r][c]
            vals = np.sort(plot_df[plot_df['group'] == g]['weeks_to_core'].astype(float).values)
            x = np.arange(1, len(vals) + 1)
            axg2.plot(x, vals, color=group_colors[g], marker='o', markersize=2, linewidth=1)
            axg2.set_ylim(0, max(30, ymax_z))
            axg2.set_title(f"{label_map[g]}  (n={len(vals)}, med={int(medians[g])})", fontsize=11)
            axg2.set_xlabel('Contributors (sorted by time)')
            axg2.set_ylabel('Weeks to Core')
        for j in range(num_groups, rows * cols):
            r = j // cols
            c = j % cols
            axes2[r][c].axis('off')
        fig4.tight_layout()
        fig4.savefig(self.output_dir / 'time_to_core_sorted_zoom.png', dpi=150, bbox_inches='tight')

    def report(self):
        lines = []
        lines.append("="*70)
        lines.append("RQ3 Step 3.1: CLUSTERS-ONLY ANALYSIS REPORT")
        lines.append("="*70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"Kruskal–Wallis H={getattr(self,'kw_stat',np.nan):.3f} p={getattr(self,'kw_p',np.nan):.3g}")
        lines.append("")
        for _, row in self.stats_df.iterrows():
            lines.append(f"Rank {int(row['sk_rank'])}: {row['group']}  (median={row['median_weeks']:.0f}, n={int(row['n'])})")
        text = "\n".join(lines)
        (self.output_dir / 'analysis_report.txt').write_text(text)
        print("\n" + text)

    def run(self):
        self.load_data()
        # Build n column for report
        self.data['group'] = self.data['project_type'] + '-' + self.data['pattern']
        self.stats_df = self.group_stats()
        self.assign_sk_ranks()
        self.plot()
        self.report()

if __name__ == '__main__':
    Step31ClustersOnly().run()
