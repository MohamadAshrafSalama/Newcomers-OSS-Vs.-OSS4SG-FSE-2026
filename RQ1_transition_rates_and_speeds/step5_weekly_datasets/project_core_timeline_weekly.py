#!/usr/bin/env python3
"""
RQ1 Step 5: Project Core Timeline Weekly Analysis
=================================================

Creates project_core_timeline_weekly.csv tracking who is core in each project at each week.
Implements 80% cumulative commits rule for core contributor definition.

Memory-efficient single-pass per project (loads only needed columns once).
"""

from __future__ import annotations

import pandas as pd
from datetime import timedelta
from pathlib import Path
import json
from tqdm import tqdm
import logging
import sys
from typing import List


INPUT_FILE = "RQ1_transition_rates_and_speeds/data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv"
OUTPUT_DIR = Path("RQ1_transition_rates_and_speeds/step5_weekly_datasets/datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "project_core_timeline_weekly.csv"


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(OUTPUT_DIR.parent / 'step5_processing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def get_all_projects(df: pd.DataFrame) -> List[str]:
    projects = df['project_name'].unique().tolist()
    projects.sort()
    return projects


def compute_weekly_core_timeline(project_df: pd.DataFrame, core_threshold_percentile: int = 80) -> pd.DataFrame:
    # Early return for empty input
    if project_df is None or len(project_df) == 0:
        return pd.DataFrame()

    # Ensure expected columns and normalize types
    df = project_df[['project_name', 'project_type', 'author_email', 'commit_hash', 'commit_date']].copy()
    df['author_email'] = df['author_email'].astype(str).str.lower().str.strip()
    # Normalize to UTC to avoid mixed timezone errors
    df['commit_date'] = pd.to_datetime(df['commit_date'], utc=True)

    # Create weekly boundaries (Monday as start of week)
    min_date = df['commit_date'].min()
    max_date = df['commit_date'].max()
    if pd.isna(min_date) or pd.isna(max_date):
        return pd.DataFrame()

    start_monday = min_date - timedelta(days=min_date.weekday())
    end_monday = max_date - timedelta(days=max_date.weekday()) + timedelta(days=7)
    weeks = pd.date_range(start=start_monday, end=end_monday, freq='W-MON')

    results = []

    for week_num, week_date in enumerate(weeks[:-1]):  # Exclude last boundary
        week_end = week_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
        commits_to_date = df[df['commit_date'] <= week_end]

        if commits_to_date.empty:
            continue

        # Drop missing/blank emails to avoid empty groupby results
        valid_commits = commits_to_date.dropna(subset=['author_email'])
        valid_commits = valid_commits[valid_commits['author_email'].str.len() > 0]
        if valid_commits.empty:
            continue

        # Contributor commit counts to date (sorted descending)
        contributor_counts = valid_commits.groupby('author_email')['commit_hash'].count().sort_values(ascending=False)

        total_commits = len(valid_commits)
        
        # Calculate 80% threshold
        threshold_commits = total_commits * (core_threshold_percentile / 100.0)
        
        # Find core contributors using corrected 80% rule
        cumulative_commits = 0
        core_contributors = []
        core_threshold_commits = 0
        
        for contributor, commits in contributor_counts.items():
            core_contributors.append(contributor)
            cumulative_commits += int(commits)
            core_threshold_commits = int(commits)  # Track the minimum commits of core contributors
            
            # Stop when we've captured at least the threshold of commits
            if cumulative_commits >= threshold_commits:
                break
        
        # Handle edge case: if no contributors found (safety check)
        if not core_contributors:
            continue

        weekly_record = {
            'project_name': df['project_name'].iloc[0],
            'project_type': df['project_type'].iloc[0],
            'week_date': week_date.strftime('%Y-%m-%d'),
            'week_number': int(week_num + 1),
            'total_commits_to_date': int(total_commits),
            'total_contributors_to_date': int(len(contributor_counts)),
            'core_threshold_commits': int(core_threshold_commits),
            'core_contributors_count': int(len(core_contributors)),
            'core_contributors_emails': json.dumps(core_contributors),
            'core_method': 'cumulative_commits_to_date_percentile',
            'core_threshold_percentile': int(core_threshold_percentile)
        }

        results.append(weekly_record)

    return pd.DataFrame(results)


def main():
    logger = setup_logging()
    logger.info("Loading master commits dataset (selected columns only)...")

    usecols = ['project_name', 'project_type', 'commit_hash', 'author_email', 'commit_date']
    df = pd.read_csv(INPUT_FILE, usecols=usecols, low_memory=False)

    logger.info(f"Loaded {len(df):,} commits across {df['project_name'].nunique()} projects")

    projects = get_all_projects(df)
    logger.info(f"Processing {len(projects)} projects...")

    all_results = []
    for project_name in tqdm(projects, desc="Step 5: Core timeline"):
        project_df = df[df['project_name'] == project_name]
        project_df = project_df.sort_values('commit_date')

        if project_df.empty:
            continue

        weekly_df = compute_weekly_core_timeline(project_df, core_threshold_percentile=80)
        if not weekly_df.empty:
            all_results.append(weekly_df)

    if not all_results:
        logger.error("No results were generated. Aborting.")
        sys.exit(1)

    final_dataset = pd.concat(all_results, ignore_index=True)
    final_dataset = final_dataset.sort_values(['project_name', 'week_number'])
    final_dataset.to_csv(OUTPUT_FILE, index=False)

    logger.info(f"Saved weekly core timeline: {OUTPUT_FILE}")
    logger.info(f"Total records: {len(final_dataset):,}")


if __name__ == "__main__":
    main()

