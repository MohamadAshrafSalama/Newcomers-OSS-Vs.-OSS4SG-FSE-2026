### Step 7 — Dataset 4: ML Features for Core Prediction

This step creates a machine-learning–ready dataset of early-behavior features for predicting whether a newcomer will become core. Features are extracted from the first 4, 8, and 12 observed weeks for each contributor, using only pre-core data to avoid label leakage. We also enforce a minimum of 12 pre-core weeks for positives, so very fast core achievers are excluded from the ML dataset.

Key inputs
- `step5_weekly_datasets/dataset2_contributor_activity/contributor_activity_weekly.csv`
- `step6_contributor_transitions/results/contributor_transitions.csv` (labels and first-core timing from Step 6 v2)

Outputs (saved under `results/` in this folder)
- `ml_features_dataset.csv` — one row per (`project_name`, `contributor_email`)
- `ml_dataset_statistics.json` — basic stats and label distribution
- `processing.log` — logs

Important notes
- We derive `lines_this_week` from `cumulative_lines_changed` within each contributor (first week uses its cumulative value). This avoids assuming unavailable per-week line counts.
- For contributors who became core, only pre-core weeks are used to compute features (no leakage).
- Minimal inclusion thresholds (configurable): at least 3 commits and 12 observed weeks pre-core (or pre-censoring).
- Result size (current run): 28,430 samples; positives 4,302 (15.1%). By type: OSS 14.1%, OSS4SG 16.9%.

How to run (do not run without large-memory environment)
```bash
cd "/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026"
python3 RQ1_transition_rates_and_speeds/step7_ml_features/ml_feature_extraction.py

# Verify the dataset
python3 RQ1_transition_rates_and_speeds/step7_ml_features/tester/verify_ml_features.py
```

Feature families
- First week: commits, lines changed, rank, contribution percentage
- Windows 4/8/12 weeks: totals (commits, lines), active weeks, consistency, avg/max/std commits, burst ratio, trend slope/R², rank start/end/improvement, end-of-window contribution
- Temporal: gaps between active weeks (avg/max/std), activity regularity, early-vs-late acceleration

Schema expectations (subset)
- Base: `project_name`, `project_type`, `contributor_email`, `label_became_core`, `total_weeks_observed`
- Windows: `w1_{4,8,12}_total_commits`, `w1_{4,8,12}_active_weeks`, `w1_{4,8,12}_consistency`, `w1_{4,8,12}_avg_commits`, `w1_{4,8,12}_trend_slope`, `w1_{4,8,12}_rank_improvement`, etc.

Repro/size cautions
- The activity CSV is large; the extractor reads in chunks and groups per contributor-project pair. Results are ignored by git via repository `.gitignore`.

Verification summary
- `tester/verify_ml_features.py` checks schema, label balance, numeric validity, and quick discrimination.
- Current run: 4 key features show strong discrimination (p << 0.01): `w1_4_total_commits`, `w1_8_total_commits`, `w1_12_total_commits`, `activity_regularity`.


