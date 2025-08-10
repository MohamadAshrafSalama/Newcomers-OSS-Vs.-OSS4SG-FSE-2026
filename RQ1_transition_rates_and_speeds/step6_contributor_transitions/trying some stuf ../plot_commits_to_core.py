#!/usr/bin/env python3
"""
Exploratory Plots: Commits-To-Core Distribution for Core Contributors
=====================================================================

Creates frequency and cumulative plots of commits_to_core for all core
contributors, split by project_type (OSS vs OSS4SG). Also creates a
weeks_to_core vs commits_to_core density heatmap for each type.

Outputs are saved in this folder.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def resolve_input_file(script_dir: Path) -> Path:
    # Prefer step6 results (v2 main dataset)
    candidate = (script_dir.parent / 'results' / 'contributor_transitions.csv').resolve()
    if candidate.exists():
        return candidate
    # Fallback to legacy paths if needed
    alt = (script_dir.parent / 'contributor_transitions.csv').resolve()
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not locate contributor_transitions.csv near {script_dir}")


def load_core_df(input_file: Path) -> pd.DataFrame:
    df = pd.read_csv(input_file, low_memory=False)
    core_df = df[df['became_core'] == True].copy()
    # Ensure numeric
    core_df['commits_to_core'] = pd.to_numeric(core_df['commits_to_core'], errors='coerce')
    core_df['weeks_to_core'] = pd.to_numeric(core_df['weeks_to_core'], errors='coerce')
    core_df = core_df.dropna(subset=['commits_to_core', 'weeks_to_core'])
    core_df['commits_to_core'] = core_df['commits_to_core'].astype(int)
    core_df['weeks_to_core'] = core_df['weeks_to_core'].astype(int)
    return core_df


def save_distribution_csv(core_df: pd.DataFrame, out_path: Path) -> None:
    dist = (
        core_df.groupby(['project_type', 'commits_to_core'])
        .size()
        .reset_index(name='count')
        .sort_values(['project_type', 'commits_to_core'])
    )
    dist.to_csv(out_path, index=False)


def plot_frequency_and_cdf(core_df: pd.DataFrame, out_file: Path) -> None:
    # Truncate visible range to reduce extreme tail influence
    max_commits_for_plot = 200
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Histogram (counts)
    ax = axes[0]
    bins = np.arange(0, max_commits_for_plot + 1, 1)
    for ptype, label in [('OSS', 'OSS'), ('OSS4SG', 'OSS4SG')]:
        data = core_df[(core_df['project_type'] == ptype) & (core_df['commits_to_core'] <= max_commits_for_plot)]['commits_to_core']
        ax.hist(data, bins=bins, alpha=0.6, label=f"{label}")
    ax.set_xlabel('Commits to Core')
    ax.set_ylabel('Number of Core Contributors')
    ax.set_title('Frequency (≤200 commits)')
    ax.legend()

    # Vertical reference thresholds
    for t in [1, 2, 5, 10, 20, 50, 100, 150, 200]:
        ax.axvline(t, color='grey', linestyle=':', alpha=0.2)

    # Panel 2: Empirical CDF (up to 200)
    ax = axes[1]
    for ptype, label in [('OSS', 'OSS'), ('OSS4SG', 'OSS4SG')]:
        data = core_df[(core_df['project_type'] == ptype) & (core_df['commits_to_core'] <= max_commits_for_plot)]['commits_to_core']
        data = np.sort(data.values)
        if len(data) == 0:
            continue
        y = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, y, label=label)
    ax.set_xlabel('Commits to Core')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('CDF (≤200 commits)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 3: Log-scale box/violin of commits
    ax = axes[2]
    # Add 1 to avoid log(0)
    tmp = core_df.copy()
    tmp['log10_commits_plus1'] = np.log10(tmp['commits_to_core'] + 1)
    sns.violinplot(data=tmp, x='project_type', y='log10_commits_plus1', ax=ax, inner='box')
    ax.set_xlabel('Project Type')
    ax.set_ylabel('log10(Commits to Core + 1)')
    ax.set_title('Distribution (log scale)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    try:
        plt.show()
    except Exception:
        pass


def plot_weeks_vs_commits_heatmap(core_df: pd.DataFrame, out_file: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    max_weeks = 120
    max_commits = 200

    for ax, ptype in zip(axes, ['OSS', 'OSS4SG']):
        data = core_df[(core_df['project_type'] == ptype)]
        x = data['weeks_to_core'].clip(upper=max_weeks)
        y = data['commits_to_core'].clip(upper=max_commits)
        hb = ax.hexbin(x, y, gridsize=50, cmap='viridis', norm=LogNorm())
        ax.set_title(f'{ptype}: Weeks vs Commits (clipped)')
        ax.set_xlabel('Weeks to Core')
        ax.set_ylabel('Commits to Core')
        ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(hb, ax=axes.ravel().tolist())
    cbar.set_label('Count (log scale)')

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    try:
        plt.show()
    except Exception:
        pass


def main():
    script_dir = Path(__file__).resolve().parent
    input_file = resolve_input_file(script_dir)
    print(f"Loading transitions (core only) from: {input_file}")
    core_df = load_core_df(input_file)

    # Save distribution table
    dist_csv = script_dir / 'commits_to_core_distribution.csv'
    save_distribution_csv(core_df, dist_csv)
    print(f"Saved distribution table: {dist_csv}")

    # Plots
    freq_cdf_png = script_dir / 'commits_to_core_freq_and_cdf.png'
    plot_frequency_and_cdf(core_df, freq_cdf_png)
    print(f"Saved frequency/CDF plot: {freq_cdf_png}")

    heatmap_png = script_dir / 'weeks_vs_commits_heatmap.png'
    plot_weeks_vs_commits_heatmap(core_df, heatmap_png)
    print(f"Saved heatmap: {heatmap_png}")


if __name__ == '__main__':
    main()


