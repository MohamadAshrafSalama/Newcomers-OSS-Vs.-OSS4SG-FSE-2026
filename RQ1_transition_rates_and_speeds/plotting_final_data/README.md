# Final Plotting: 6-Metric Boxplots

This folder generates a single figure with six OSS vs OSS4SG boxplots:
- Core Contributor Ratio (%)
- One-Time Contributor Ratio (%)
- Gini Coefficient
- Bus Factor (robust y-axis cap)
- Active Contributor Ratio (%)
- Transition Rate (%) [non-zero months]

Run:
```bash
python3 plot_six_metrics_boxplots.py | cat
```

Outputs:
- six_metrics_boxplots.png
- six_metrics_boxplots.pdf

Requires (already produced by earlier steps):
- step3_per_project_metrics/project_metrics.csv
- step4_newcomer_transition_rates/corrected_transition_results/monthly_transitions.csv
