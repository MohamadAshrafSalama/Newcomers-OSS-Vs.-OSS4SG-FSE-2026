#!/usr/bin/env python3
"""
Convert ALL contributor cache JSONs (from the current RQ2 run)
into simple event timelines (PRs + Issues + Commits) with compact columns.

Fixes applied:
- Case-insensitive join on emails when fetching commits from the master dataset
- Normalize all timestamps to timezone-naive UTC

Usage:
  python3 convert_caches_to_timelines.py
  python3 convert_caches_to_timelines.py --only-cache /path/to/specific_cache.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


BASE = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
CACHE_DIR = BASE / "RQ2_newcomer_treatment_patterns_test2" / "results" / "full_enriched" / "cache"
OUT_DIR = BASE / "RQ2_newcomer_treatment_patterns_test2" / "step2_timelines" / "from_cache_timelines"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MASTER_COMMITS = BASE / "RQ1_transition_rates_and_speeds" / "data_mining" / "step2_commit_analysis" / "consolidating_master_dataset" / "master_commits_dataset.csv"


def parse_iso(dt: str | None) -> datetime | None:
    if not dt:
        return None
    try:
        return datetime.fromisoformat(dt.replace('Z', '+00:00'))
    except Exception:
        return None


def to_naive_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def build_events_from_cache(cache: Dict, commits_for_pair: List[Dict] | None) -> pd.DataFrame:
    project = cache.get('project_name', '')
    project_type = cache.get('project_type', '')
    contributor_email = cache.get('contributor_email', '')
    username = cache.get('username', '')

    first_commit_dt = parse_iso(cache.get('first_commit_date'))
    first_core_dt = parse_iso(cache.get('first_core_date'))

    events: List[Dict] = []

    # Pull Requests
    for pr in (cache.get('raw_data', {}).get('pull_requests') or []):
        ts = parse_iso(pr.get('createdAt')) or parse_iso(pr.get('created_at'))
        if not ts:
            continue
        events.append({
            'event_type': 'pull_request',
            'event_timestamp': ts,
            'event_identifier': f"PR#{pr.get('number')}",
            'event_data': json.dumps(pr, ensure_ascii=False),
            'project_name': project,
            'project_type': project_type,
            'contributor_email': contributor_email,
            'username': username,
            'first_commit_dt': first_commit_dt,
            'first_core_dt': first_core_dt,
        })

    # Issues
    for issue in (cache.get('raw_data', {}).get('issues') or []):
        ts = parse_iso(issue.get('createdAt')) or parse_iso(issue.get('created_at'))
        if not ts:
            continue
        events.append({
            'event_type': 'issue',
            'event_timestamp': ts,
            'event_identifier': f"Issue#{issue.get('number')}",
            'event_data': json.dumps(issue, ensure_ascii=False),
            'project_name': project,
            'project_type': project_type,
            'contributor_email': contributor_email,
            'username': username,
            'first_commit_dt': first_commit_dt,
            'first_core_dt': first_core_dt,
        })

    # Commits from master dataset
    if commits_for_pair:
        for c in commits_for_pair:
            try:
                ts = pd.to_datetime(c.get('commit_date'), utc=True)
                # pd.Timestamp has tz_localize; remove timezone to make naive UTC
                if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None:
                    ts = ts.tz_localize(None)
            except Exception:
                continue
            payload = {
                'hash': c.get('commit_hash'),
                'message': c.get('commit_message'),
                'author_email': c.get('author_email'),
                'author_name': c.get('author_name'),
                'files_modified': c.get('files_modified_count'),
                'additions': c.get('total_insertions'),
                'deletions': c.get('total_deletions'),
                'lines_changed': c.get('total_lines_changed'),
                'commit_hour': c.get('commit_hour'),
                'commit_day_of_week': c.get('commit_day_of_week'),
                'is_weekend': c.get('commit_is_weekend'),
            }
            events.append({
                'event_type': 'commit',
                'event_timestamp': ts,
                'event_identifier': c.get('commit_hash'),
                'event_data': json.dumps(payload, ensure_ascii=False),
                'project_name': project,
                'project_type': project_type,
                'contributor_email': contributor_email,
                'username': username,
                'first_commit_dt': first_commit_dt,
                'first_core_dt': first_core_dt,
            })

    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)
    # Normalize timestamps to tz-naive UTC
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], utc=True).dt.tz_localize(None)
    df = df.sort_values('event_timestamp').reset_index(drop=True)

    # Baseline for week index: prefer first_commit_dt if available, else earliest event
    baseline = df.loc[0, 'event_timestamp']
    fc = to_naive_utc(first_commit_dt)
    if fc is not None:
        baseline = fc
    df['event_week'] = ((df['event_timestamp'] - baseline).dt.days // 7).astype(int)

    # Pre-core flag
    fcore = to_naive_utc(first_core_dt)
    if fcore is not None:
        df['is_pre_core'] = df['event_timestamp'] < fcore
    else:
        df['is_pre_core'] = True

    # Sequential id
    df['event_id'] = range(1, len(df) + 1)
    return df[['event_id','event_type','event_timestamp','event_week','event_identifier','event_data','project_name','project_type','contributor_email','username','is_pre_core']]


def main():
    ap = argparse.ArgumentParser(description='Convert cache JSONs to timelines (PRs + Issues + Commits)')
    ap.add_argument('--only-cache', help='Process only this single cache JSON filepath')
    ap.add_argument('--limit', type=int, help='Process only the first N cache files (sorted)')
    args = ap.parse_args()

    files: List[Path]
    if args.only_cache:
        files = [Path(args.only_cache)]
    else:
        files = sorted(CACHE_DIR.glob('*.json'))
        if args.limit is not None and args.limit > 0:
            files = files[: args.limit]

    # Preload caches and collect pairs for commit join
    caches: List[Tuple[Path, Dict]] = []
    needed_pairs_lower: set[Tuple[str, str]] = set()
    for fp in files:
        try:
            cache = json.loads(fp.read_text())
        except Exception:
            continue
        pn = str(cache.get('project_name', ''))
        em = str(cache.get('contributor_email', ''))
        if pn and em:
            needed_pairs_lower.add((pn, em.lower()))
        caches.append((fp, cache))

    # Load master commits filtered to needed pairs (case-insensitive on email)
    commits_by_pair: Dict[Tuple[str, str], List[Dict]] = {}
    if MASTER_COMMITS.exists() and needed_pairs_lower:
        usecols = [
            'project_name','author_email','author_name','commit_hash','commit_date','commit_message',
            'files_modified_count','total_insertions','total_deletions','total_lines_changed',
            'commit_hour','commit_day_of_week','commit_is_weekend'
        ]
        for chunk in pd.read_csv(MASTER_COMMITS, usecols=usecols, chunksize=200_000):
            lower_emails = chunk['author_email'].astype(str).str.lower()
            project_names = chunk['project_name'].astype(str)
            mask = [(pn, em) in needed_pairs_lower for pn, em in zip(project_names, lower_emails)]
            sub = chunk[mask]
            if sub.empty:
                continue
            for _, r in sub.iterrows():
                key = (str(r['project_name']), str(r['author_email']).lower())
                commits_by_pair.setdefault(key, []).append(r.to_dict())

    written = 0
    empty = 0
    for fp, cache in tqdm(caches, desc='Converting caches'):
        key = (str(cache.get('project_name', '')), str(cache.get('contributor_email', '')).lower())
        commit_list = commits_by_pair.get(key, [])
        df = build_events_from_cache(cache, commit_list)
        cid = cache.get('contributor_id','unknown').replace('/', '_').replace('@','_at_')
        out_fp = OUT_DIR / f"timeline_{cid}.csv"
        if df.empty:
            empty += 1
            # Write an empty CSV with headers for consistency
            df = pd.DataFrame(columns=['event_id','event_type','event_timestamp','event_week','event_identifier','event_data','project_name','project_type','contributor_email','username','is_pre_core'])
            df.to_csv(out_fp, index=False)
            continue
        df.to_csv(out_fp, index=False)
        written += 1

    print(f"Timelines written: {written}; empty: {empty}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == '__main__':
    main()


