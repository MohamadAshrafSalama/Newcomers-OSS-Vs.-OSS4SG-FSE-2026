# RQ2 — Newcomer Treatment Patterns

This step collects comprehensive interaction data from GitHub (issues, pull requests, reviews, and comments) to study how newcomers are treated in OSS vs OSS4SG projects.

We use Google BigQuery's public GitHub Archive dataset to avoid GitHub API rate limits. No GitHub tokens are required.

## What we collect
- Pull requests: creation metadata, titles/bodies, merge status
- PR reviews: states (APPROVED/CHANGES_REQUESTED/COMMENTED), bodies
- PR review comments: code review discussion
- Issues: creation metadata, titles/bodies, state
- Issue comments: full conversation text

These are sufficient for response-time analyses, sentiment, support/recognition patterns, and reviewer/assignee behaviors.

## Prerequisites (one-time on your Mac)
```bash
brew install --cask google-cloud-sdk
python3 -m pip install --upgrade google-cloud-bigquery pandas pyarrow tqdm
gcloud auth application-default login
gcloud config set project oss4sg-research
```

## Run book
From the project root:
```bash
RQ2_DATE_START=2023-01-01 RQ2_DATE_END=2023-01-03 \
  "./.venv/bin/python3" RQ2_newcomer_treatment_patterns/scripts/get_all_github_data.py
```

Outputs are written to `RQ2_newcomer_treatment_patterns/extracted_data/`:
- `github_interactions_complete.csv`
- `github_interactions_complete.parquet`
- `extraction_summary.json`

To run a full window later (example):
```bash
RQ2_DATE_START=2022-01-01 RQ2_DATE_END=2024-12-31 \
  "./.venv/bin/python3" RQ2_newcomer_treatment_patterns/scripts/get_all_github_data.py
```

Cost/time: Typical end-to-end extraction takes minutes and costs a few dollars from free credits. Adjust the date range in the script to control scope.

## Notes
- Large artifacts are ignored by git; do not commit datasets.
- If memory is constrained, we can switch to a batched extractor that writes results incrementally.
- Data source: `githubarchive.day.*` (public dataset on BigQuery).


