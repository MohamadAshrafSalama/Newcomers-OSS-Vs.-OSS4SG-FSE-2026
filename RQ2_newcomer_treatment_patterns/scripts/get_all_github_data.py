#!/usr/bin/env python3
"""
Complete GitHub data extraction for RQ2 - One script does everything!
No API keys, no rate limits, gets ALL the data you need via BigQuery GitHub Archive.

Usage:
  python3 get_all_github_data.py

Prereqs (run once):
  brew install --cask google-cloud-sdk
  python3 -m pip install --upgrade google-cloud-bigquery pandas pyarrow tqdm
  gcloud auth application-default login
  gcloud config set project oss4sg-research
"""

from pathlib import Path
import os
from datetime import datetime
import json
import sys

import pandas as pd
from google.cloud import bigquery


def main() -> None:
    print("Starting GitHub data extraction for OSS vs OSS4SG research...")

    # Initialize BigQuery client (uses ADC from gcloud auth application-default login)
    try:
        # Prefer explicit project to avoid ADC project inference issues
        bq_project = (
            os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("CLOUDSDK_CORE_PROJECT")
            or "oss4sg-research"
        )
        client = bigquery.Client(project=bq_project)
        print("Connected to BigQuery successfully.")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to connect to BigQuery: {e}")
        print("Did you run: gcloud auth application-default login ?")
        sys.exit(1)

    # Load project list
    projects_file = Path("preparing_dataset/data/final_balanced_dataset.csv")
    if not projects_file.exists():
        print(f"Cannot find {projects_file}. Run from project root.")
        sys.exit(1)

    print(f"Loading projects from {projects_file}...")
    projects_df = pd.read_csv(projects_file)

    # Parse repository names (supports multiple schemas)
    repositories = []
    has_github_url = "github_url" in projects_df.columns
    has_owner_repo = {"owner", "repository_name"}.issubset(projects_df.columns)
    has_project_name = "project_name" in projects_df.columns
    type_col = "project_type" if "project_type" in projects_df.columns else ("type" if "type" in projects_df.columns else None)

    for _, project in projects_df.iterrows():
        owner = None
        repo = None
        if has_github_url and pd.notna(project.get("github_url")):
            url_parts = str(project["github_url"]).strip("/").split("/")
            if len(url_parts) >= 2:
                owner, repo = url_parts[-2], url_parts[-1]
        elif has_owner_repo:
            owner = project.get("owner")
            repo = project.get("repository_name")
        elif has_project_name and pd.notna(project.get("project_name")) and "/" in str(project.get("project_name")):
            # Expecting form owner/repo
            parts = str(project.get("project_name")).strip().split("/")
            if len(parts) == 2:
                owner, repo = parts[0], parts[1]

        if owner and repo:
            repositories.append({
                "owner": owner,
                "repo": repo,
                "full_name": f"{owner}/{repo}",
                "type": project.get(type_col, "unknown") if type_col else "unknown",
            })

    if not repositories:
        print("No repositories parsed from dataset.")
        sys.exit(1)

    print(f"Found {len(repositories)} repositories to process")

    # Build the query (adjust dates to control scope/cost)
    full_names = [r["full_name"] for r in repositories]
    repo_list = "', '".join(full_names)
    repo_list_lower = "', '".join([s.lower() for s in full_names])
    date_start = os.getenv("RQ2_DATE_START", "2022-01-01")
    date_end = os.getenv("RQ2_DATE_END", "2024-12-31")
    start_sfx = date_start.replace("-", "")
    end_sfx = date_end.replace("-", "")

    query = f"""

    WITH github_data AS (
      SELECT 
        CAST(repo.name AS STRING) as repo_name,
        CAST(actor.login AS STRING) as user_login,
        CAST(actor.id AS INT64) as user_id,
        CAST(created_at AS TIMESTAMP) as created_at,
        CAST(type AS STRING) as event_type,
        payload
      -- Enumerate concrete day partitions to avoid views (e.g., 'yesterday')
      FROM UNNEST(GENERATE_DATE_ARRAY(DATE('{date_start}'), DATE('{date_end}'))) AS d
      JOIN `githubarchive.day.*` t
      ON _TABLE_SUFFIX = REPLACE(FORMAT_DATE('%Y%m%d', d), '-', '')
      WHERE 
        _TABLE_SUFFIX BETWEEN '{start_sfx}' AND '{end_sfx}'
        AND LOWER(repo.name) IN ('{repo_list_lower}')
        AND type IN (
          'PullRequestEvent', 
          'PullRequestReviewEvent', 
          'PullRequestReviewCommentEvent',
          'IssuesEvent', 
          'IssueCommentEvent'
        )
    ),
    pull_requests AS (
      SELECT
        CAST(repo_name AS STRING) as repo_name,
        CAST('pull_request' AS STRING) as interaction_type,
        SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.number') AS INT64) as number,
        CAST(created_at AS TIMESTAMP) as created_at,
        CAST(user_login AS STRING) as user_login,
        CAST(user_id AS INT64) as user_id,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') AS STRING) as pr_author,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.action') AS STRING) as action,
        CAST(NULL AS STRING) as review_state,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.title') AS STRING) as title,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.body') AS STRING) as body,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.state') AS STRING) as state,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') = 'true' AS BOOL) as is_merged,
        SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.additions') AS INT64) as additions,
        SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.deletions') AS INT64) as deletions,
        SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.changed_files') AS INT64) as changed_files,
        SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ', JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')) as pr_created_at,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.assignee.login') AS STRING) as assignee,
        CAST(JSON_EXTRACT(payload, '$.pull_request.labels') AS STRING) as labels_json
      FROM github_data 
      WHERE event_type = 'PullRequestEvent' AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'opened'
    ),
    pr_reviews AS (
      SELECT
        CAST(repo_name AS STRING) as repo_name,
        CAST('pr_review' AS STRING) as interaction_type,
        SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') AS INT64) as number,
        CAST(created_at AS TIMESTAMP) as created_at,
        CAST(user_login AS STRING) as user_login,
        CAST(user_id AS INT64) as user_id, 
        CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') AS STRING) as pr_author,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.action') AS STRING) as action,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.review.state') AS STRING) as review_state,
        CAST(NULL AS STRING) as title,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.review.body') AS STRING) as body,
        CAST(NULL AS STRING) as state,
        CAST(NULL AS BOOL) as is_merged,
        CAST(NULL AS INT64) as additions, 
        CAST(NULL AS INT64) as deletions, 
        CAST(NULL AS INT64) as changed_files, 
        CAST(NULL AS TIMESTAMP) as pr_created_at, 
        CAST(NULL AS STRING) as assignee, 
        CAST(NULL AS STRING) as labels_json
      FROM github_data 
      WHERE event_type = 'PullRequestReviewEvent'
    ),
    pr_comments AS (
      SELECT
        CAST(repo_name AS STRING) as repo_name,
        CAST('pr_comment' AS STRING) as interaction_type,
        SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') AS INT64) as number,
        CAST(created_at AS TIMESTAMP) as created_at,
        CAST(user_login AS STRING) as user_login,
        CAST(user_id AS INT64) as user_id,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') AS STRING) as pr_author,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.action') AS STRING) as action,
        CAST(NULL AS STRING) as review_state,
        CAST(NULL AS STRING) as title,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.comment.body') AS STRING) as body,
        CAST(NULL AS STRING) as state,
        CAST(NULL AS BOOL) as is_merged,
        CAST(NULL AS INT64) as additions,
        CAST(NULL AS INT64) as deletions,
        CAST(NULL AS INT64) as changed_files,
        CAST(NULL AS TIMESTAMP) as pr_created_at,
        CAST(NULL AS STRING) as assignee,
        CAST(NULL AS STRING) as labels_json
      FROM github_data 
      WHERE event_type = 'PullRequestReviewCommentEvent'
    ),
    issues AS (
      SELECT
        CAST(repo_name AS STRING) as repo_name,
        CAST('issue' AS STRING) as interaction_type,
        SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.number') AS INT64) as number,
        CAST(created_at AS TIMESTAMP) as created_at,
        CAST(user_login AS STRING) as user_login,
        CAST(user_id AS INT64) as user_id,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.user.login') AS STRING) as pr_author,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.action') AS STRING) as action,
        CAST(NULL AS STRING) as review_state,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.title') AS STRING) as title,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.body') AS STRING) as body,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.state') AS STRING) as state,
        CAST(NULL AS BOOL) as is_merged,
        CAST(NULL AS INT64) as additions,
        CAST(NULL AS INT64) as deletions,
        CAST(NULL AS INT64) as changed_files,
        SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ', JSON_EXTRACT_SCALAR(payload, '$.issue.created_at')) as pr_created_at,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.assignee.login') AS STRING) as assignee,
        CAST(JSON_EXTRACT(payload, '$.issue.labels') AS STRING) as labels_json
      FROM github_data 
      WHERE event_type = 'IssuesEvent' AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'opened'
    ),
    issue_comments AS (
      SELECT
        CAST(repo_name AS STRING) as repo_name,
        CAST('issue_comment' AS STRING) as interaction_type,
        SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.number') AS INT64) as number,
        CAST(created_at AS TIMESTAMP) as created_at,
        CAST(user_login AS STRING) as user_login,
        CAST(user_id AS INT64) as user_id,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.user.login') AS STRING) as pr_author,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.action') AS STRING) as action,
        CAST(NULL AS STRING) as review_state,
        CAST(NULL AS STRING) as title,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.comment.body') AS STRING) as body,
        CAST(NULL AS STRING) as state,
        CAST(NULL AS BOOL) as is_merged,
        CAST(NULL AS INT64) as additions,
        CAST(NULL AS INT64) as deletions,
        CAST(NULL AS INT64) as changed_files,
        CAST(NULL AS TIMESTAMP) as pr_created_at,
        CAST(NULL AS STRING) as assignee,
        CAST(NULL AS STRING) as labels_json
      FROM github_data 
      WHERE event_type = 'IssueCommentEvent'
    )
    SELECT * FROM pull_requests
    UNION ALL SELECT * FROM pr_reviews  
    UNION ALL SELECT * FROM pr_comments
    UNION ALL SELECT * FROM issues
    UNION ALL SELECT * FROM issue_comments
    ORDER BY repo_name, created_at
    """

    # Chunked extraction per day to avoid dataset wildcard views
    repo_list_lower_vals = ", ".join([f"'{s.lower()}'" for s in full_names])
    date_range = pd.date_range(start=date_start, end=date_end, freq="D")
    day_suffixes = [d.strftime("%Y%m%d") for d in date_range]
    days_per_query = int(os.getenv("RQ2_DAYS_PER_QUERY", "7"))
    chunks = [day_suffixes[i:i+days_per_query] for i in range(0, len(day_suffixes), days_per_query)]

    out_dir = Path("RQ2_newcomer_treatment_patterns/extracted_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "github_interactions_complete.csv"
    parquet_file = out_dir / "github_interactions_complete.parquet"

    first_write = True
    collected_frames = []
    for chunk in chunks:
        selects = []
        for sfx in chunk:
            selects.append(f"""
            SELECT 
              CAST(repo.name AS STRING) as repo_name,
              CAST(actor.login AS STRING) as user_login,
              CAST(actor.id AS INT64) as user_id,
              CAST(created_at AS TIMESTAMP) as created_at,
              CAST(type AS STRING) as event_type,
              payload
            FROM `githubarchive.day.{sfx}`
            WHERE LOWER(repo.name) IN ({repo_list_lower_vals})
              AND type IN ('PullRequestEvent','PullRequestReviewEvent','PullRequestReviewCommentEvent','IssuesEvent','IssueCommentEvent')
            """)
        base_union = "\nUNION ALL\n".join(selects)

        query_chunk = f"""
        WITH github_data AS (
          {base_union}
        ),
        pull_requests AS (
          SELECT
            CAST(repo_name AS STRING) as repo_name,
            CAST('pull_request' AS STRING) as interaction_type,
            SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.number') AS INT64) as number,
            CAST(created_at AS TIMESTAMP) as created_at,
            CAST(user_login AS STRING) as user_login,
            CAST(user_id AS INT64) as user_id,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') AS STRING) as pr_author,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.action') AS STRING) as action,
            CAST(NULL AS STRING) as review_state,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.title') AS STRING) as title,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.body') AS STRING) as body,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.state') AS STRING) as state,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') = 'true' AS BOOL) as is_merged,
            SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.additions') AS INT64) as additions,
            SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.deletions') AS INT64) as deletions,
            SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.changed_files') AS INT64) as changed_files,
            SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ', JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')) as pr_created_at,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.assignee.login') AS STRING) as assignee,
            CAST(JSON_EXTRACT(payload, '$.pull_request.labels') AS STRING) as labels_json
          FROM github_data 
          WHERE event_type = 'PullRequestEvent' AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'opened'
        ),
        pr_reviews AS (
          SELECT
            CAST(repo_name AS STRING) as repo_name,
            CAST('pr_review' AS STRING) as interaction_type,
            SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') AS INT64) as number,
            CAST(created_at AS TIMESTAMP) as created_at,
            CAST(user_login AS STRING) as user_login,
            CAST(user_id AS INT64) as user_id, 
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') AS STRING) as pr_author,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.action') AS STRING) as action,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.review.state') AS STRING) as review_state,
            CAST(NULL AS STRING) as title,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.review.body') AS STRING) as body,
            CAST(NULL AS STRING) as state,
            CAST(NULL AS BOOL) as is_merged,
            CAST(NULL AS INT64) as additions, 
            CAST(NULL AS INT64) as deletions, 
            CAST(NULL AS INT64) as changed_files, 
            CAST(NULL AS TIMESTAMP) as pr_created_at, 
            CAST(NULL AS STRING) as assignee, 
            CAST(NULL AS STRING) as labels_json
          FROM github_data 
          WHERE event_type = 'PullRequestReviewEvent'
        ),
        pr_comments AS (
          SELECT
            CAST(repo_name AS STRING) as repo_name,
            CAST('pr_comment' AS STRING) as interaction_type,
            SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') AS INT64) as number,
            CAST(created_at AS TIMESTAMP) as created_at,
            CAST(user_login AS STRING) as user_login,
            CAST(user_id AS INT64) as user_id,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') AS STRING) as pr_author,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.action') AS STRING) as action,
            CAST(NULL AS STRING) as review_state,
            CAST(NULL AS STRING) as title,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.comment.body') AS STRING) as body,
            CAST(NULL AS STRING) as state,
            CAST(NULL AS BOOL) as is_merged,
            CAST(NULL AS INT64) as additions,
            CAST(NULL AS INT64) as deletions,
            CAST(NULL AS INT64) as changed_files,
            CAST(NULL AS TIMESTAMP) as pr_created_at,
            CAST(NULL AS STRING) as assignee,
            CAST(NULL AS STRING) as labels_json
          FROM github_data 
          WHERE event_type = 'PullRequestReviewCommentEvent'
        ),
        issues AS (
          SELECT
            CAST(repo_name AS STRING) as repo_name,
            CAST('issue' AS STRING) as interaction_type,
            SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.number') AS INT64) as number,
            CAST(created_at AS TIMESTAMP) as created_at,
            CAST(user_login AS STRING) as user_login,
            CAST(user_id AS INT64) as user_id,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.user.login') AS STRING) as pr_author,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.action') AS STRING) as action,
            CAST(NULL AS STRING) as review_state,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.title') AS STRING) as title,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.body') AS STRING) as body,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.state') AS STRING) as state,
            CAST(NULL AS BOOL) as is_merged,
            CAST(NULL AS INT64) as additions,
            CAST(NULL AS INT64) as deletions,
            CAST(NULL AS INT64) as changed_files,
            SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ', JSON_EXTRACT_SCALAR(payload, '$.issue.created_at')) as pr_created_at,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.assignee.login') AS STRING) as assignee,
            CAST(JSON_EXTRACT(payload, '$.issue.labels') AS STRING) as labels_json
          FROM github_data 
          WHERE event_type = 'IssuesEvent' AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'opened'
        ),
        issue_comments AS (
          SELECT
            CAST(repo_name AS STRING) as repo_name,
            CAST('issue_comment' AS STRING) as interaction_type,
            SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.number') AS INT64) as number,
            CAST(created_at AS TIMESTAMP) as created_at,
            CAST(user_login AS STRING) as user_login,
            CAST(user_id AS INT64) as user_id,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.user.login') AS STRING) as pr_author,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.action') AS STRING) as action,
            CAST(NULL AS STRING) as review_state,
            CAST(NULL AS STRING) as title,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.comment.body') AS STRING) as body,
            CAST(NULL AS STRING) as state,
            CAST(NULL AS BOOL) as is_merged,
            CAST(NULL AS INT64) as additions,
            CAST(NULL AS INT64) as deletions,
            CAST(NULL AS INT64) as changed_files,
            CAST(NULL AS TIMESTAMP) as pr_created_at,
            CAST(NULL AS STRING) as assignee,
            CAST(NULL AS STRING) as labels_json
          FROM github_data 
          WHERE event_type = 'IssueCommentEvent'
        )
        SELECT * FROM pull_requests
        UNION ALL SELECT * FROM pr_reviews  
        UNION ALL SELECT * FROM pr_comments
        UNION ALL SELECT * FROM issues
        UNION ALL SELECT * FROM issue_comments
        ORDER BY repo_name, created_at
        """
        try:
            job = client.query(query_chunk)
            df = job.result().to_dataframe()
        except Exception as e:  # noqa: BLE001
            print(f"Chunk query failed: {e}")
            continue
        if df.empty:
            continue
        repo_to_type = {r["full_name"]: r["type"] for r in repositories}
        df["project_type"] = df["repo_name"].map(repo_to_type)
        df["extraction_date"] = datetime.utcnow().isoformat()
        df.to_csv(csv_file, mode="a", header=first_write, index=False)
        first_write = False
        collected_frames.append(df)

    if not collected_frames:
        print("No data returned — verify repositories exist in GitHub Archive and date window.")
        sys.exit(0)

    final_df = pd.concat(collected_frames, ignore_index=True)
    final_df.to_parquet(parquet_file, index=False)

    summary = {
        "extraction_date": datetime.utcnow().isoformat(),
        "total_interactions": int(len(final_df)),
        "unique_repositories": int(final_df["repo_name"].nunique()),
        "unique_users": int(final_df["user_login"].nunique()) if "user_login" in final_df.columns else None,
        "interaction_types": final_df["interaction_type"].value_counts().to_dict(),
        "project_types": final_df["project_type"].value_counts().to_dict(),
        "date_range": {
            "start": pd.to_datetime(final_df["created_at"]).min().isoformat(),
            "end": pd.to_datetime(final_df["created_at"]).max().isoformat(),
        },
        "outputs": {
            "csv": str(csv_file),
            "parquet": str(parquet_file),
        },
    }
    with open(out_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


