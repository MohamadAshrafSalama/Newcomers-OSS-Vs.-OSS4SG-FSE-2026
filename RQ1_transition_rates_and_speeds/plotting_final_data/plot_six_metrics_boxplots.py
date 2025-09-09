#!/usr/bin/env python3
"""
Final 6-Metric Boxplots (OSS vs OSS4SG)
=======================================

Creates a single 2x3 boxplot figure containing:
1) Core Contributor Ratio (%)
2) One-Time Contributor Ratio (%)
3) Gini Coefficient
4) Bus Factor (robust y-axis cap)
5) Active Contributor Ratio (%)
6) Transition Rate (%) [non-zero months]

Inputs:
- step3_per_project_metrics/project_metrics.csv
- step4_newcomer_transition_rates/corrected_transition_results/monthly_transitions.csv

Outputs (saved here):
- plotting_final_data/six_metrics_boxplots.png
- plotting_final_data/six_metrics_boxplots.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_project_metrics() -> pd.DataFrame:
    root = _project_root()
    path = os.path.join(root, 'step3_per_project_metrics', 'project_metrics.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"project_metrics.csv not found at {path}")
    df = pd.read_csv(path)
    return df


def _load_transition_monthly() -> pd.DataFrame:
    root = _project_root()
    path = os.path.join(
        root,
        'step4_newcomer_transition_rates',
        'corrected_transition_results',
        'monthly_transitions.csv'
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            "monthly_transitions.csv not found. Run step4 corrected_transition_analysis.py first."
        )
    df = pd.read_csv(path)
    # non-zero transition months only
    return df[df['truly_new_core_count'] > 0].copy()


def _add_significance(ax, oss_values: pd.Series, oss4sg_values: pd.Series, alpha_adjusted: float = 0.01) -> None:
    oss_values = pd.Series(oss_values).dropna()
    oss4sg_values = pd.Series(oss4sg_values).dropna()
    if len(oss_values) == 0 or len(oss4sg_values) == 0:
        return
    try:
        _, p_val = stats.mannwhitneyu(oss_values, oss4sg_values, alternative='two-sided')
    except ValueError:
        return
    if p_val < alpha_adjusted:
        y_max = ax.get_ylim()[1]
        ax.plot([1, 2], [y_max * 0.9, y_max * 0.9], 'k-', linewidth=1)
        if p_val < 0.001:
            stars = '***'
        elif p_val < 0.01:
            stars = '**'
        else:
            stars = '*'
        ax.text(1.5, y_max * 0.91, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')


def _cap_bus_factor_axis(ax, oss_values: pd.Series, oss4sg_values: pd.Series) -> None:
    combined = pd.concat([pd.Series(oss_values).astype(float), pd.Series(oss4sg_values).astype(float)])
    if len(combined) == 0:
        return
    try:
        robust_max = np.nanpercentile(combined, 95)
        median_max = float(np.nanmax([np.nanmedian(oss_values), np.nanmedian(oss4sg_values)]))
        fallback = max(median_max * 2.0, 8.0)
        # Hard ceiling to keep the plot interpretable
        y_upper = min(max(robust_max, fallback), 15.0)
        ax.set_ylim(0, y_upper)
        # annotate clipped counts
        clipped_oss = int((pd.Series(oss_values) > y_upper).sum())
        clipped_oss4sg = int((pd.Series(oss4sg_values) > y_upper).sum())
        if (clipped_oss + clipped_oss4sg) > 0:
            ax.text(1, y_upper * 0.98, f'clipped={clipped_oss}', ha='center', va='top', fontsize=8)
            ax.text(2, y_upper * 0.98, f'clipped={clipped_oss4sg}', ha='center', va='top', fontsize=8)
    except Exception:
        pass


def create_six_boxplots() -> str:
    # Style (can be updated later as requested)
    plt.style.use('default')
    # Blue for OSS, Orange for OSS4SG
    custom_palette = ['#1f77b4', '#ff7f0e']
    sns.set_theme(style='whitegrid', palette=custom_palette, font_scale=1.0)

    project_df = _load_project_metrics()
    monthly_df = _load_transition_monthly()

    # Metrics configuration
    metrics_cfg = [
        {'key': 'core_ratio', 'title': 'Core Contributor Ratio (%)', 'src': 'project', 'as_percent': True},
        {'key': 'one_time_ratio', 'title': 'One-Time Contributor Ratio (%)', 'src': 'project', 'as_percent': True},
        {'key': 'gini_coefficient', 'title': 'Gini Coefficient', 'src': 'project', 'as_percent': False},
        {'key': 'bus_factor', 'title': 'Bus Factor', 'src': 'project', 'as_percent': False},
        {'key': 'active_ratio', 'title': 'Active Contributor Ratio (%)', 'src': 'project', 'as_percent': True},
        {'key': 'transition_rate', 'title': 'Transition Rate (%)', 'src': 'monthly', 'as_percent': True},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    # Panel statistic annotations removed per request

    for i, cfg in enumerate(metrics_cfg):
        ax = axes[i]
        if cfg['src'] == 'project':
            oss_data = project_df[project_df['project_type'] == 'OSS'][cfg['key']].dropna()
            oss4sg_data = project_df[project_df['project_type'] == 'OSS4SG'][cfg['key']].dropna()
            if cfg['as_percent']:
                oss_data = oss_data * 100.0
                oss4sg_data = oss4sg_data * 100.0
        else:  # monthly transition
            oss_data = monthly_df[monthly_df['project_type'] == 'OSS']['transition_rate'].dropna()
            oss4sg_data = monthly_df[monthly_df['project_type'] == 'OSS4SG']['transition_rate'].dropna()
            if cfg['as_percent']:
                oss_data = oss_data * 100.0
                oss4sg_data = oss4sg_data * 100.0

        bp = ax.boxplot(
            [oss_data, oss4sg_data],
            tick_labels=['OSS', 'OSS4SG'],
            patch_artist=True,
            showmeans=False,
            showfliers=False,
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='#666666'),
            capprops=dict(color='#666666')
        )
        colors = ['#1f77b4', '#ff7f0e']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(cfg['title'], fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)

        # no sample size labels per requirement

        # bus factor axis fix
        if cfg['key'] == 'bus_factor':
            _cap_bus_factor_axis(ax, oss_data, oss4sg_data)
        # transition rate axis: keep interpretable (percent scale)
        if cfg['key'] == 'transition_rate':
            try:
                combined = pd.concat([pd.Series(oss_data), pd.Series(oss4sg_data)])
                y_upper = min(max(np.nanpercentile(combined, 95), 25.0), 150.0)
                y_lower = 0.0
                ax.set_ylim(y_lower, y_upper)
            except Exception:
                pass

        # remove significance stars/lines per requirement

        # No per-panel numeric annotations

    # Shared legend (single, clean)
    # Create proxy artists for legend
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='#1f77b4', label='OSS'),
        mpatches.Patch(color='#ff7f0e', label='OSS4SG')
    ]
    # Single global legend with only colors
    fig.legend(handles=legend_handles, loc='upper center', ncol=2, frameon=False, borderaxespad=0.5)
    # No figure title per requirement
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.join(_project_root(), 'plotting_final_data')
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'six_metrics_boxplots.png')
    out_pdf = os.path.join(out_dir, 'six_metrics_boxplots.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    return out_png


def main():
    print("Generating final 6-metric boxplots...")
    try:
        output = create_six_boxplots()
        print(f"Saved: {output}")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == '__main__':
    main()


