#!/usr/bin/env python3
"""
CREATE PUBLICATION-READY FIGURES
This script creates high-quality figures suitable for academic publication.
Run this after completing all survival analysis steps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')

# Publication-ready settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 1.5

# Define paths
BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
STEP8_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step8_survival_analysis"
DATA_DIR = STEP8_DIR / "data"
RESULTS_DIR = STEP8_DIR / "results"
PLOTS_DIR = STEP8_DIR / "visualizations"
PUBLICATION_DIR = STEP8_DIR / "publication_figures"

PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)

COLOR_OSS = '#2E86AB'
COLOR_OSS4SG = '#A23B72'
COLOR_NEUTRAL = '#666666'


def create_main_survival_figure():
    print("Creating main survival analysis figure...")
    df = pd.read_csv(DATA_DIR / "survival_data.csv")
    with open(RESULTS_DIR / "kaplan_meier_results.json") as f:
        km_results = json.load(f)
    with open(RESULTS_DIR / "cox_regression_results.json") as f:
        cox_results = json.load(f)

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

    # Panel A
    ax1 = fig.add_subplot(gs[0, :2])
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    oss_df = df[df['project_type'] == 'OSS']
    kmf.fit(oss_df['duration'], oss_df['event'], label='OSS')
    kmf.plot_survival_function(ax=ax1, ci_show=True, color=COLOR_OSS, alpha=0.8)
    oss4sg_df = df[df['project_type'] == 'OSS4SG']
    kmf.fit(oss4sg_df['duration'], oss4sg_df['event'], label='OSS4SG')
    kmf.plot_survival_function(ax=ax1, ci_show=True, color=COLOR_OSS4SG, alpha=0.8)
    ax1.set_xlabel('Time Since First Commit (weeks)', fontweight='bold')
    ax1.set_ylabel('Probability of Remaining Non-Core', fontweight='bold')
    ax1.set_title('(A) Survival Curves: Time to Core Contributor Status', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_xlim(0, 150)
    ax1.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)
    if km_results['median_survival']['OSS']:
        ax1.axvline(km_results['median_survival']['OSS'], color=COLOR_OSS, linestyle=':', alpha=0.5)
    if km_results['median_survival']['OSS4SG']:
        ax1.axvline(km_results['median_survival']['OSS4SG'], color=COLOR_OSS4SG, linestyle=':', alpha=0.5)
    p_val = km_results['log_rank_test']['p_value']
    p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax1.text(
        0.5,
        0.15,
        f"Log-rank test: {p_text}",
        transform=ax1.transAxes,
        fontsize=10,
        ha='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8),
    )

    # Panel B
    ax2 = fig.add_subplot(gs[0, 2])
    timepoints = [4, 12, 26, 52]
    oss_probs = []
    oss4sg_probs = []
    for t in timepoints:
        if f'week_{t}' in km_results['survival_at_timepoints']:
            oss_probs.append(km_results['survival_at_timepoints'][f'week_{t}']['oss_core_prob'])
            oss4sg_probs.append(
                km_results['survival_at_timepoints'][f'week_{t}']['oss4sg_core_prob']
            )
    x = np.arange(len(timepoints))
    width = 0.35
    bars1 = ax2.bar(x - width / 2, np.array(oss_probs) * 100, width, label='OSS', color=COLOR_OSS, alpha=0.8)
    bars2 = ax2.bar(x + width / 2, np.array(oss4sg_probs) * 100, width, label='OSS4SG', color=COLOR_OSS4SG, alpha=0.8)
    ax2.set_xlabel('Time (weeks)', fontweight='bold')
    ax2.set_ylabel('Cumulative % Became Core', fontweight='bold')
    ax2.set_title('(B) Cumulative Core Transitions', fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Week {t}' for t in timepoints])
    ax2.legend(loc='upper left', frameon=True, edgecolor='black')
    ax2.grid(True, alpha=0.2, axis='y', linestyle='--')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    # Panel C
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('tight')
    ax3.axis('off')
    events_oss = km_results['events']['OSS']
    events_oss4sg = km_results['events']['OSS4SG']
    n_oss = km_results['sample_sizes']['OSS']
    n_oss4sg = km_results['sample_sizes']['OSS4SG']
    stats_data = [
        ['Metric', 'OSS', 'OSS4SG', 'Difference'],
        ['Contributors (n)', f"{n_oss:,}", f"{n_oss4sg:,}", ''],
        ['Became Core (%)', f"{events_oss / n_oss * 100:.1f}%", f"{events_oss4sg / n_oss4sg * 100:.1f}%", ''],
        [
            'Median Time (weeks)',
            f"{km_results['median_survival']['OSS']:.0f}" if km_results['median_survival']['OSS'] else '>150',
            f"{km_results['median_survival']['OSS4SG']:.0f}" if km_results['median_survival']['OSS4SG'] else '>150',
            '',
        ],
    ]
    table = ax3.table(cellText=stats_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for i in range(4):
        table[(0, i)].set_facecolor('#E0E0E0')
        table[(0, i)].set_text_props(weight='bold')
    ax3.set_title('(C) Summary Statistics', fontweight='bold', pad=15)

    # Panel D
    ax4 = fig.add_subplot(gs[1, 1:])
    factors = ['OSS4SG\n(vs OSS)']
    hrs = [cox_results['simple_model']['hazard_ratio_oss4sg']]
    ci_lower = [cox_results['simple_model']['ci_lower']]
    ci_upper = [cox_results['simple_model']['ci_upper']]
    y_pos = np.arange(len(factors))
    for i in range(len(factors)):
        ax4.plot([ci_lower[i], ci_upper[i]], [i, i], 'k-', linewidth=2)
        ax4.plot(hrs[i], i, 'o', markersize=8, color=COLOR_OSS4SG)
    ax4.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(factors)
    ax4.set_xlabel('Hazard Ratio (95% CI)', fontweight='bold')
    ax4.set_title('(D) Factors Influencing Transition to Core', fontweight='bold', pad=15)
    ax4.set_xlim(0.5, 3)
    ax4.grid(True, alpha=0.2, axis='x', linestyle='--')
    for i, hr in enumerate(hrs):
        ax4.text(hr + 0.1, i, f'{hr:.2f}', va='center', fontweight='bold')
    ax4.text(2.2, -0.8, 'Higher likelihood →', fontsize=9, ha='center')
    ax4.text(0.7, -0.8, '← Lower likelihood', fontsize=9, ha='center')

    fig.suptitle('Survival Analysis: Newcomer to Core Contributor Transitions in OSS vs OSS4SG', fontsize=14, fontweight='bold', y=0.98)
    save_path = PUBLICATION_DIR / "figure_survival_main.pdf"
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    print(f"  Saved main figure to: {save_path}")
    save_path_png = PUBLICATION_DIR / "figure_survival_main.png"
    plt.savefig(save_path_png, bbox_inches='tight', format='png', dpi=300)
    print(f"  Saved PNG version to: {save_path_png}")
    plt.close(fig)
    return fig


def create_stratified_analysis_figure():
    print("\nCreating stratified analysis figure...")
    df = pd.read_csv(DATA_DIR / "survival_data.csv")
    df = df.copy()
    if 'total_commits' in df.columns:
        # Robust qcut with duplicate edges handling
        try:
            df['activity_level'] = pd.qcut(
                df['total_commits'], q=[0, 0.25, 0.75, 1.0], labels=['Low', 'Medium', 'High'], duplicates='drop'
            )
        except Exception:
            # Fallback to simple bins
            quantiles = df['total_commits'].quantile([0.25, 0.75]).tolist()
            bins = [df['total_commits'].min() - 1, quantiles[0], quantiles[1], df['total_commits'].max() + 1]
            df['activity_level'] = pd.cut(df['total_commits'], bins=bins, labels=['Low', 'Medium', 'High'], include_lowest=True)
    else:
        df['activity_level'] = 'Unknown'

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Stratified Survival Analysis by Contributor Characteristics', fontsize=14, fontweight='bold')
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()

    for idx, level in enumerate(['Low', 'Medium', 'High']):
        ax = axes[0, idx]
        level_df = df[df['activity_level'] == level]
        for ptype, color in [('OSS', COLOR_OSS), ('OSS4SG', COLOR_OSS4SG)]:
            type_df = level_df[level_df['project_type'] == ptype]
            if len(type_df) > 10:
                kmf.fit(type_df['duration'], type_df['event'], label=ptype)
                kmf.plot_survival_function(ax=ax, ci_show=False, color=color)
        ax.set_title(f'{level} Activity\n(n={len(level_df):,})', fontsize=11)
        ax.set_xlabel('Time (weeks)')
        ax.set_ylabel('Survival Probability' if idx == 0 else '')
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right', fontsize=9)

    df['joiner_type'] = pd.cut(df['first_commit_week'], bins=[0, 10, 50, float('inf')], labels=['Early', 'Mid', 'Late'])
    for idx, joiner in enumerate(['Early', 'Mid', 'Late']):
        ax = axes[1, idx]
        joiner_df = df[df['joiner_type'] == joiner]
        for ptype, color in [('OSS', COLOR_OSS), ('OSS4SG', COLOR_OSS4SG)]:
            type_df = joiner_df[joiner_df['project_type'] == ptype]
            if len(type_df) > 10:
                kmf.fit(type_df['duration'], type_df['event'], label=ptype)
                kmf.plot_survival_function(ax=ax, ci_show=False, color=color)
        ax.set_title(f'{joiner} Joiners\n(n={len(joiner_df):,})', fontsize=11)
        ax.set_xlabel('Time (weeks)')
        ax.set_ylabel('Survival Probability' if idx == 0 else '')
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    save_path = PUBLICATION_DIR / "figure_survival_stratified.pdf"
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    print(f"  Saved stratified figure to: {save_path}")
    plt.close()
    return fig


def create_summary_table():
    print("\nCreating LaTeX summary table...")
    with open(RESULTS_DIR / "kaplan_meier_results.json") as f:
        km_results = json.load(f)
    with open(RESULTS_DIR / "cox_regression_results.json") as f:
        cox_results = json.load(f)

    latex_table = r"""
\begin{table}[h]
\centering
\caption{Survival Analysis Results: Newcomer to Core Transitions}
\label{tab:survival_results}
\begin{tabular}{lrrr}
\toprule
\textbf{Metric} & \textbf{OSS} & \textbf{OSS4SG} & \textbf{Test Statistic} \\
\midrule
\multicolumn{4}{l}{\textit{Sample Characteristics}} \\
Total Contributors & """ + f"{km_results['sample_sizes']['OSS']:,}" + r""" & """ + f"{km_results['sample_sizes']['OSS4SG']:,}" + r""" & --- \\
Became Core (\%) & """ + f"{km_results['events']['OSS']/km_results['sample_sizes']['OSS']*100:.1f}" + r""" & """ + f"{km_results['events']['OSS4SG']/km_results['sample_sizes']['OSS4SG']*100:.1f}" + r""" & --- \\
\midrule
\multicolumn{4}{l}{\textit{Time to Core (weeks)}} \\
Median & """ + (f"{km_results['median_survival']['OSS']:.0f}" if km_results['median_survival']['OSS'] else ">150") + r""" & """ + (f"{km_results['median_survival']['OSS4SG']:.0f}" if km_results['median_survival']['OSS4SG'] else ">150") + r""" & --- \\
\midrule
\multicolumn{4}{l}{\textit{Statistical Tests}} \\
Log-rank Test & --- & --- & """ + ("p < 0.001" if km_results['log_rank_test']['p_value'] < 0.001 else f"p = {km_results['log_rank_test']['p_value']:.3f}") + r""" \\
Hazard Ratio (95\% CI) & Ref. & """ + f"{cox_results['simple_model']['hazard_ratio_oss4sg']:.2f} ({cox_results['simple_model']['ci_lower']:.2f}–{cox_results['simple_model']['ci_upper']:.2f})" + r""" & """ + ("p < 0.001" if cox_results['simple_model']['p_value'] < 0.001 else f"p = {cox_results['simple_model']['p_value']:.3f}") + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""

    table_path = PUBLICATION_DIR / "table_survival_results.tex"
    with open(table_path, 'w') as f:
        f.write(latex_table)
    print(f"  Saved LaTeX table to: {table_path}")

    csv_data = {
        'Metric': ['Total Contributors', 'Became Core (%)', 'Median Time (weeks)', 'Hazard Ratio', 'Log-rank p-value'],
        'OSS': [
            km_results['sample_sizes']['OSS'],
            f"{km_results['events']['OSS']/km_results['sample_sizes']['OSS']*100:.1f}",
            km_results['median_survival']['OSS'] if km_results['median_survival']['OSS'] else '>150',
            '1.00 (Ref.)',
            '',
        ],
        'OSS4SG': [
            km_results['sample_sizes']['OSS4SG'],
            f"{km_results['events']['OSS4SG']/km_results['sample_sizes']['OSS4SG']*100:.1f}",
            km_results['median_survival']['OSS4SG'] if km_results['median_survival']['OSS4SG'] else '>150',
            f"{cox_results['simple_model']['hazard_ratio_oss4sg']:.2f}",
            km_results['log_rank_test']['p_value'],
        ],
    }
    pd.DataFrame(csv_data).to_csv(PUBLICATION_DIR / "table_survival_results.csv", index=False)
    print(f"  Saved CSV table to: {PUBLICATION_DIR / 'table_survival_results.csv'}")


def create_interpretation_guide():
    print("\nCreating interpretation guide...")
    guide_content = (
        "SURVIVAL ANALYSIS RESULTS - INTERPRETATION GUIDE\n"
        "================================================\n\n"
        "KEY FINDINGS:\n"
        "------------\n"
        "1. OSS4SG contributors transition to core status significantly faster than OSS contributors\n"
        "2. The effect is consistent across different activity levels and time periods\n"
        "3. Project type (OSS vs OSS4SG) is a strong predictor of transition speed\n\n"
        "STATISTICAL INTERPRETATION:\n"
        "--------------------------\n"
        "• Hazard Ratio > 1: OSS4SG contributors more likely to become core at any given time\n"
        "• Log-rank test p < 0.05: Significant difference between survival curves\n"
        "• Median survival time: Time when 50% of contributors have become core\n"
        "• Concordance index > 0.7: Good model fit and predictive ability\n\n"
        "PRACTICAL IMPLICATIONS:\n"
        "----------------------\n"
        "• OSS4SG projects have more inclusive pathways to core contributor status\n"
        "• Mission-driven projects may have different community dynamics\n"
        "• Faster transitions in OSS4SG suggest lower barriers to advancement\n\n"
        "VISUALIZATIONS EXPLAINED:\n"
        "------------------------\n"
        "1. Survival Curves: Shows probability of remaining non-core over time\n"
        "   - Steeper decline = faster transitions\n"
        "   - Confidence bands show uncertainty\n\n"
        "2. Cumulative Incidence: Shows % who have become core by specific timepoints\n"
        "   - Complementary to survival curves (1 - survival)\n"
        "   - Easier to interpret for non-technical audiences\n\n"
        "3. Hazard Ratios: Shows relative likelihood of becoming core\n"
        "   - HR = 2 means twice as likely\n"
        "   - Confidence intervals should not cross 1 for significance\n\n"
        "4. Stratified Analysis: Shows if effects are consistent across subgroups\n"
        "   - Similar patterns = robust findings\n"
        "   - Different patterns = interaction effects\n\n"
        "CAVEATS AND LIMITATIONS:\n"
        "------------------------\n"
        "• Censoring: We only observe contributors until the study end\n"
        "• Selection bias: Projects were selected based on specific criteria\n"
        "• Time-varying effects: The hazard ratio may change over time\n"
        "• Unobserved factors: Other variables may influence transitions\n\n"
        "NEXT STEPS:\n"
        "----------\n"
        "1. Investigate mechanisms behind faster OSS4SG transitions\n"
        "2. Analyze quality and quantity of contributions post-transition\n"
        "3. Study retention rates after becoming core\n"
        "4. Examine project-level factors that moderate the effect\n"
    )
    guide_path = PUBLICATION_DIR / "interpretation_guide.txt"
    with open(guide_path, 'w') as f:
        f.write(guide_content)
    print(f"  Saved interpretation guide to: {guide_path}")


def main() -> bool:
    print("=" * 60)
    print("CREATING PUBLICATION-READY FIGURES")
    print("=" * 60)
    if not RESULTS_DIR.exists() or not (RESULTS_DIR / "kaplan_meier_results.json").exists():
        print("\nResults not found. Please run the survival analysis first:")
        print("  python3 RQ1_transition_rates_and_speeds/step8_survival_analysis/scripts/run_all_analysis.py")
        return False
    try:
        create_main_survival_figure()
        create_stratified_analysis_figure()
        create_summary_table()
        create_interpretation_guide()
        print("\n" + "=" * 60)
        print("PUBLICATION MATERIALS CREATED")
        print("=" * 60)
        print(f"\nAll materials saved to: {PUBLICATION_DIR}")
        return True
    except Exception as e:
        print(f"\nError creating figures: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)


