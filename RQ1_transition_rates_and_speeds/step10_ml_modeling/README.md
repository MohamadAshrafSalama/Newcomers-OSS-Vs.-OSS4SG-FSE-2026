# Step 10 — ML Modeling (RQ1): Comprehensive ML Pipeline

Generated: 2025-08-19

Summary
-------
- Total contributors analyzed: 23,069
- Core contributors: 8,045 (34.9%)
- Projects covered: 358

Project type breakdown
----------------------
- **OSS**: 15,858 contributors, 4,914 cores (31.0%)
- **OSS4SG**: 7,211 contributors, 3,131 cores (43.4%)

Model performance (5-fold CV)
----------------------------
- LogisticRegression — ROC AUC = 0.689 ± 0.009, PR AUC = 0.622 ± 0.011
- RandomForest — ROC AUC = 0.746 ± 0.008, PR AUC = 0.657 ± 0.008
- GradientBoosting — ROC AUC = 0.746 ± 0.009, PR AUC = 0.656 ± 0.009

Top predictive features (RandomForest importance)
-----------------------------------------------
1. `lines_changed_90d` — 22.24%
2. `files_modified_90d` — 10.60%
3. `commits_90d` — 7.01%
4. `avg_commits_per_day` — 6.84%
5. `avg_gap_days` — 6.67%

Files produced
--------------
- `features_90day_comprehensive.csv` — per-contributor features (90d window)
- `filtered_contributors.csv` — contributor filtering summary
- `all_core_contributors.csv` — core contributor records (from monthly transitions)
- `model_results_comprehensive.csv` — model metrics
- `feature_importance_comprehensive.csv` and `.png` — feature rankings and plot
- `comprehensive_analysis_figure.{png,pdf}` — visual summary

Reproducibility
---------------
See `RQ1_transition_rates_and_speeds/step10_ml_modeling/` for the pipeline scripts and logs: `simple_comprehensive_pipeline.py`, `comprehensive_ml_pipeline.py`, and `comprehensive_pipeline.log`.


