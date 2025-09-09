#!/usr/bin/env python3
"""
Export Figure A only: Kaplan-Meier survival curves (OSS vs OSS4SG)
Minimal, publication-ready styling consistent with project palette.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> str:
    # Resolve step8 directory relative to this script
    step8_dir = Path(__file__).resolve().parents[1]
    data_dir = step8_dir / 'data'
    pub_dir = step8_dir / 'publication_figures'
    pub_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(data_dir / 'survival_data.csv')

    # Professional styling
    COLOR_OSS = '#1f77b4'   # blue
    COLOR_OSS4SG = '#ff7f0e'  # orange
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'axes.linewidth': 1.2,
        'lines.linewidth': 3.0,
        'savefig.dpi': 450,
        'figure.dpi': 150,
        'lines.antialiased': True,
        'patch.antialiased': True,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'text.color': '#333333'
    })

    # Fit Kaplan-Meier curves
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(8, 5))

    # OSS
    oss_df = df[df['project_type'] == 'OSS']
    kmf.fit(oss_df['duration'], oss_df['event'], label='OSS')
    kmf.plot_survival_function(ax=ax, ci_show=True, color=COLOR_OSS, linewidth=3.0)

    # OSS4SG
    oss4sg_df = df[df['project_type'] == 'OSS4SG']
    kmf.fit(oss4sg_df['duration'], oss4sg_df['event'], label='OSS4SG')
    kmf.plot_survival_function(ax=ax, ci_show=True, color=COLOR_OSS4SG, linewidth=3.0)

    # Rounded line caps for smoother appearance
    for line in ax.lines:
        line.set_solid_capstyle('round')

    # Soften confidence interval shading
    for coll in ax.collections:
        try:
            coll.set_alpha(0.15)
        except Exception:
            pass

    # Axes formatting
    ax.set_xlabel('Time Since First Commit (weeks)', fontweight='bold')
    ax.set_ylabel('Probability of Remaining Non-Core', fontweight='bold')
    # Focus on first year for clarity
    ax.set_xlim(0, 52)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_ylim(0.90, 1.0)
    ax.set_yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    # Professional grid and styling
    ax.grid(True, alpha=0.35, linestyle='-', linewidth=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    
    # Clean, professional spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#666666')
        ax.spines[spine].set_linewidth(1.0)
    
    # Professional legend
    legend = ax.legend(loc='lower left', frameon=True, fancybox=False, 
                      edgecolor='#666666', facecolor='white', framealpha=0.95)
    legend.get_frame().set_linewidth(1.0)
    
    ax.margins(x=0.01, y=0.005)

    out_png = pub_dir / 'figure_survival_A_only.png'
    out_pdf = pub_dir / 'figure_survival_A_only.pdf'
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    return str(out_png)


if __name__ == '__main__':
    path = main()
    print(f'Saved Figure A to: {path}')


