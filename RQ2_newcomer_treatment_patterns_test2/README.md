RQ2 — Newcomer Treatment Patterns (Test2)
========================================

Overview
--------
This folder contains the current, documented pipeline for RQ2 (treatment patterns), replacing the archived prior attempts. It resolves GitHub usernames via commit hashes, then extracts PR/issue treatment signals up to each contributor's first core date.

Key Changes vs. Old RQ2
-----------------------
- Username resolution now uses commit SHA → REST commits endpoint → `author.login` (canonical), with a zero-API fallback for GitHub noreply emails.
- GraphQL extraction uses `user.contributionsCollection` filtered by time window and repository (no unsupported `author:` args on `repository.pullRequests`).
- Robust resumption via `extraction_state.pkl`; strict use of identity mapping provided via `--identity` arg.

Data Inputs
-----------
- Contributors/timing: `RQ2_newcomer_treatment_patterns_test2/daraset/core_contributors_filtered.csv`
- Identity (original): `RQ2_newcomer_treatment_patterns/results/core_commit_identity.csv` (archived source)
- Identity (enriched usernames): `RQ2_newcomer_treatment_patterns_test2/daraset/core_commit_identity_enriched.csv` (produced here)

Username Enrichment
-------------------
Script: `scripts/enrich_identity_with_usernames.py`

Strategy per row:
- If `original_author_email` is a GitHub noreply (…@users.noreply.github.com), parse the login locally.
- Else call `GET /repos/{owner}/{repo}/commits/{sha}` and prefer `author.login`. If null, fallback to noreply from commit header if present.

Run:
```
python3 RQ2_newcomer_treatment_patterns_test2/scripts/enrich_identity_with_usernames.py \
  --tokens RQ2_newcomer_treatment_patterns_test2/github_tokens.txt \
  --input RQ2_newcomer_treatment_patterns/results/core_commit_identity.csv \
  --output RQ2_newcomer_treatment_patterns_test2/daraset/core_commit_identity_enriched.csv
```

Enrichment Coverage (current run)
---------------------------------
- Total rows: 7,633
- Resolved from noreply: 1,486
- Resolved from API: 5,235
- Unresolved: 912

Extraction (Step 1)
-------------------
Script: `scripts/rq2_treatment_extractor.py`

Inputs:
- `--contributors`: `daraset/core_contributors_filtered.csv`
- `--identity`: `daraset/core_commit_identity_enriched.csv` (strictly provides `resolved_username`)
- `--tokens`: `github_tokens.txt`
- `--output-dir`: results folder (state/cache/summary written here)

Run (sample):
```
python3 RQ2_newcomer_treatment_patterns_test2/scripts/rq2_treatment_extractor.py \
  --tokens RQ2_newcomer_treatment_patterns_test2/github_tokens.txt \
  --contributors RQ2_newcomer_treatment_patterns_test2/daraset/core_contributors_filtered.csv \
  --identity RQ2_newcomer_treatment_patterns_test2/daraset/core_commit_identity_enriched.csv \
  --output-dir RQ2_newcomer_treatment_patterns_test2/results/enriched_test \
  --sample 50
```

Run (full):
```
python3 RQ2_newcomer_treatment_patterns_test2/scripts/rq2_treatment_extractor.py \
  --tokens RQ2_newcomer_treatment_patterns_test2/github_tokens.txt \
  --contributors RQ2_newcomer_treatment_patterns_test2/daraset/core_contributors_filtered.csv \
  --identity RQ2_newcomer_treatment_patterns_test2/daraset/core_commit_identity_enriched.csv \
  --output-dir RQ2_newcomer_treatment_patterns_test2/results/full_enriched
```

Outputs
-------
- `results/<run>/cache/*.json`: one JSON per contributor with raw PRs/issues and computed metrics
- `results/<run>/extraction_state.pkl`: resumable state
- `results/<run>/treatment_metrics_summary.csv`: flattened summary for analysis

Timeline Generation (Step 2)
---------------------------
Script: `step2_timelines/convert_caches_to_timelines.py`

Purpose: Convert the JSON cache files from Step 1 into individual contributor timeline CSVs, integrating commit data from RQ1 master dataset.

Inputs:
- Cache files from Step 1: `results/full_enriched/cache/*.json`
- RQ1 master commits: `RQ1_transition_rates_and_speeds/data_mining/consolidating_master_dataset/master_commits_dataset.csv`

Outputs:
- Individual CSV files: `step2_timelines/from_cache_timelines/contributor_[email]_[project].csv`
- Each row represents one event (commit, PR, or issue) with complete metadata

Run:
```
python3 RQ2_newcomer_treatment_patterns_test2/step2_timelines/convert_caches_to_timelines.py
```

Treatment Metrics Calculation (Step 3)
-------------------------------------
Script: `step3_treatment_metrics/calculate_treatment_metrics.py`

Purpose: Calculate comprehensive treatment metrics from the timeline data, grouped by project type (OSS vs OSS4SG).

Inputs:
- Timeline CSVs from Step 2: `step2_timelines/from_cache_timelines/*.csv`
- Contributor transitions from RQ1: `RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv`

Outputs:
- `step3_treatment_metrics/results/treatment_metrics_per_contributor.csv`: Individual contributor metrics
- `step3_treatment_metrics/results/treatment_metrics_summary_by_type.csv`: OSS vs OSS4SG comparison

Run:
```
python3 RQ2_newcomer_treatment_patterns_test2/step3_treatment_metrics/calculate_treatment_metrics.py
```

Current Status (as of latest commit)
-----------------------------------
✅ **Step 1 (Extraction)**: COMPLETED
- Full extraction run completed for all 7,633 contributors
- Processed: ~6,721 contributors with resolved usernames
- Skipped: ~912 contributors (unresolved usernames)
- Generated: Cache files with PR/issue data and initial metrics

✅ **Step 2 (Timeline Generation)**: COMPLETED
- Converted all cache files to individual contributor timelines
- Integrated commit data from RQ1 master dataset
- Generated: Individual CSV files for each contributor-project pair

✅ **Step 3 (Treatment Metrics)**: COMPLETED
- Calculated comprehensive treatment metrics across 5 categories:
  - Response Timing: First response times, average response times
  - Engagement Breadth: Number of unique responders, interaction diversity
  - Interaction Patterns: Review patterns, comment patterns
  - Recognition Signals: Assignment patterns, mention patterns
  - Trust Indicators: Merge rates, approval patterns
- Generated: Per-contributor metrics and OSS vs OSS4SG comparison

📊 **Final Results Summary**:
- Total contributors analyzed: 6,721
- OSS contributors: ~3,200
- OSS4SG contributors: ~3,500
- Metrics available: 25+ treatment indicators per contributor

Known Limitations
-----------------
- Unresolved usernames (~912) typically arise from commits not linked to GitHub accounts or history issues (404). They are skipped (no heuristics beyond commit lookup and noreply parsing).
- Default lookback is 365 days pre-core; can be switched to use `first_commit_date` → `first_core_date` if needed.

Experiment Log (summary)
------------------------
- Initial GraphQL approach using `repository.pullRequests(author: ...)` failed (schema disallows `author`); fixed by switching to `contributionsCollection`.
- Skips previously high due to missing usernames; solved by enrichment step using commit hashes.
- Sample run (50): Processed 48, Skipped 1, Failed 1 (transient API/read).
- Full run completed successfully under `results/full_enriched/`.
- Timeline generation completed with commit integration.
- Treatment metrics calculation completed with comprehensive analysis.

Next Steps
----------
- Analyze treatment metrics differences between OSS and OSS4SG projects
- Generate publication-ready visualizations
- Statistical significance testing of treatment patterns
- Integration with RQ1 findings for comprehensive newcomer analysis


