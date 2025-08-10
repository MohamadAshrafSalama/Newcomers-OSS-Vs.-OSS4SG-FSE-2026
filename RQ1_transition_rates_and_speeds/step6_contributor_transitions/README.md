### Step 6 — Dataset 3: Contributor Transitions (v2)

This step builds a per-contributor transition dataset from first commit to first time becoming core, suitable for RQ1 analyses (rates and speeds). The v2 pipeline adds explicit exclusion logic to focus on true newcomers and documents all issues, trials, and fixes we applied.

### Goals
- Create `contributor_transitions.csv`: one row per (`project_name`, `contributor_email`), summarizing the journey to first core status (or censoring if never core).
- Provide survival-analysis–ready fields (e.g., `time_to_event_weeks`, `censored`).
- Produce realistic transition metrics for newcomers (time and effort), not skewed by founders/early maintainers.

### History: problems, trials, and fixes
- Step number confusion (7 → 6): Renamed directories and fixed all internal paths.
- Path/CWD issues: Scripts originally assumed project-root execution; all paths now resolve relative to the Step 6 folder and outputs land under `results/`.
- Instant core inflation: Many contributors were core at week 0 with 1 commit (especially OSS), yielding 0-week, 1-commit medians. We added explicit exclusion of week-0 cores.
- Early project + fast core: Founders/maintainers joining early and becoming core quickly still polluted results. v2 adds exclusion for join ≤8 weeks AND core ≤4 weeks.
- Tester path fixes: Validation scripts now read from the `results/` folder and verify both “including all” and “filtered” datasets.
- Library gaps: Installed seaborn/scipy for investigation/plots.
- Interpretation vs bug: Commit medians stayed very low for OSS even after exclusions; validation showed logic was correct. The gap reflects project workflows/styles rather than a computation error.

### Produced files (under `results/`)
- `contributor_transitions.csv` — Main v2 dataset (excludes instant core + early-project-fast-core)
- `contributor_transitions_no_instant.csv` — Only excludes instant core
- `contributor_transitions_including_all.csv` — No exclusions (reference)
- `transition_statistics_v2.json` — Summary statistics for the main dataset
- `validation_report_v2.json` — Validator summary
- `processing.log` — Execution log

### Scripts
- Main v2 analysis: `contributor_transitions_analysis_v2.py`
- Validator v2: `tester/test_transitions_v2.py`
- Diagnostic (early joiners): `analyze_early_contributors.py`
- Commit gap investigation: `tester/test_commit_requirement_investigation.py`
- Threshold sweep (exclude cores ≤X weeks): `threshold_analysis.py` (outputs under `results/threshold_analysis/`)
- Exploratory plots (commits-to-core distributions): `trying some stuf ../plot_commits_to_core.py`

### How to run

```bash
cd "/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026"

# (One-time) dependencies for plots/stats
./.venv/bin/python3 -m pip install seaborn matplotlib scipy

# Build v2 datasets
python3 RQ1_transition_rates_and_speeds/step6_contributor_transitions/contributor_transitions_analysis_v2.py

# Validate v2 datasets
python3 RQ1_transition_rates_and_speeds/step6_contributor_transitions/tester/test_transitions_v2.py

# Commit requirement investigation
python3 RQ1_transition_rates_and_speeds/step6_contributor_transitions/tester/test_commit_requirement_investigation.py

# Threshold analysis across early-core windows
python3 RQ1_transition_rates_and_speeds/step6_contributor_transitions/threshold_analysis.py

# Exploratory commits-to-core plots
python3 "RQ1_transition_rates_and_speeds/step6_contributor_transitions/trying some stuf ../plot_commits_to_core.py"
```

Outputs are written to `RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/` and `results/threshold_analysis/`.

### Key validated results (v2 main dataset)
- Size: 85,763 transitions across 356 projects; 8,812 became core (10.3%).
- Time to core (weeks): median 48 (OSS4SG 34, OSS 53).
- Effort to core (commits): median 12 overall; by type — OSS 2 vs OSS4SG 59.
- All validation checks passed: no instant cores, correct censoring, no negatives/duplicates, `commits_to_core ≤ total_commits`.

### Threshold analysis (time to core)
Excluding increasingly longer “early core” windows raises medians for both types. After removing instant cores, OSS consistently needs ~20–30 more weeks than OSS4SG to reach core. See: `results/threshold_analysis/threshold_analysis_results.json` and `threshold_analysis_comprehensive.png`.

### Commit-count gap: what we learned
- The difference is structural, not a bug. Many OSS projects allow core with few commits (e.g., curated lists/docs, squashed PRs, 80% rule dynamics). OSS4SG generally requires sustained code contributions.
- Investigation script shows a statistically significant gap, project-level concentration (some OSS projects have median ≤5 commits), and robustness under thresholds.
- Exploratory plots (histogram/CDF/log-violin) show OSS mass at 1–5 commits, while OSS4SG is shifted to tens/hundreds.

### Normalization guidance (for fairness, not to hide effects)
- Report both raw and log-scale summaries (use `log10(commits_to_core+1)`).
- Project-level medians: compare distributions of per-project medians to reduce dominance of large repos.
- Relative-to-project view: fold-change vs project median and within-project percentile rank.
- Rate metrics: `commits_to_core / active_weeks_to_core` and `lines_changed_to_core / active_weeks_to_core` to separate pace from duration.
- Cross-check with `lines_changed_to_core` to distinguish lightweight commits vs lightweight lines.

### Why this matters
- Focuses analyses on true newcomer journeys.
- Provides defensible, reproducible metrics for RQ1 (transition rates and speeds).
- Documents data pitfalls and our remediation so reviewers can audit decisions.

### Repro checklist
- Input: `step5_weekly_datasets/dataset2_contributor_activity/contributor_activity_weekly.csv`
- Run v2 analysis → validate → (optional) investigation, threshold, and exploratory scripts.
- Cite outputs from `results/` in the paper; keep “including all” datasets only for reference/sensitivity.

