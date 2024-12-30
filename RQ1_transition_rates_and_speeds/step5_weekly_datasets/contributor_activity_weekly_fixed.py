#!/usr/bin/env python3
"""
RQ1 Step 5 - Dataset 2: Contributor Activity Weekly (FIXED, FULL RUN)
=====================================================================
Generates the contributor activity dataset with correct week alignment and core lookups.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import logging
import sys
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
import gc
warnings.filterwarnings('ignore')

CONFIG = {
    'input_commits': "RQ1_transition_rates_and_speeds/data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv",
    'input_core_timeline': "RQ1_transition_rates_and_speeds/step5_weekly_datasets/datasets/project_core_timeline_weekly.csv",
    'output_dir': Path("RQ1_transition_rates_and_speeds/step5_weekly_datasets/dataset2_contributor_activity"),
    'checkpoint_file': 'dataset2_checkpoint.json',
    'save_frequency': 5,
    'chunk_size': 200000,
}

class ContributorActivityProcessor:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'project_results').mkdir(exist_ok=True)
        
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
            logging.FileHandler(self.output_dir / 'processing.log'),
            logging.StreamHandler(sys.stdout)
        ])
        self.logger = logging.getLogger(__name__)
        
        self.checkpoint_path = self.output_dir / self.config['checkpoint_file']
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                self.checkpoint = json.load(f)
                self.logger.info(f"Loaded checkpoint: {len(self.checkpoint['completed_projects'])} projects completed")
        else:
            self.checkpoint = {
                'completed_projects': [],
                'failed_projects': [],
                'total_processed': 0,
                'total_records_created': 0,
                'start_time': None,
                'last_updated': None
            }

        self.core_lookup = None
        self.project_list = None

    def save_checkpoint(self):
        self.checkpoint['last_updated'] = datetime.now().isoformat()
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)

    def load_core_timeline(self):
        self.logger.info("Loading core timeline lookup...")
        timeline = pd.read_csv(self.config['input_core_timeline'])
        timeline['week_date'] = pd.to_datetime(timeline['week_date'], utc=True)
        self.core_lookup = {}
        for _, row in timeline.iterrows():
            key = (row['project_name'], row['week_date'].strftime('%Y-%m-%d'))
            try:
                self.core_lookup[key] = json.loads(row['core_contributors_emails'])
            except Exception:
                self.core_lookup[key] = []
        self.logger.info(f"Loaded {len(self.core_lookup):,} core timeline entries")

    def get_project_list(self) -> Tuple[List[str], List[str]]:
        if self.project_list is None:
            self.logger.info("Scanning projects from commits dataset...")
            projects_set = set()
            chunk_iter = pd.read_csv(self.config['input_commits'], usecols=['project_name'], chunksize=self.config['chunk_size'])
            for chunk in chunk_iter:
                projects_set.update(chunk['project_name'].unique())
            self.project_list = sorted(list(projects_set))
            self.logger.info(f"Found {len(self.project_list)} unique projects")
        remaining_projects = [p for p in self.project_list if p not in self.checkpoint['completed_projects']]
        return self.project_list, remaining_projects

    def load_project_commits(self, project_name: str) -> pd.DataFrame:
        usecols = ['project_name','project_type','commit_hash','author_email','commit_date','total_insertions','total_deletions','files_modified_count']
        chunks = []
        for ch in pd.read_csv(self.config['input_commits'], usecols=usecols, chunksize=self.config['chunk_size'], low_memory=False):
            sub = ch[ch['project_name'] == project_name]
            if not sub.empty:
                chunks.append(sub)
        if not chunks:
            return pd.DataFrame()
        df = pd.concat(chunks, ignore_index=True)
        df['author_email'] = df['author_email'].astype(str).str.lower().str.strip()
        df['commit_date'] = pd.to_datetime(df['commit_date'], utc=True)
        # Monday 00:00 UTC week start, tz-aware
        commit_dt_utc = df['commit_date'].dt.tz_convert('UTC')
        df['week_start'] = (commit_dt_utc - pd.to_timedelta(commit_dt_utc.dt.weekday, unit='D')).dt.normalize()
        df['total_insertions'] = pd.to_numeric(df['total_insertions'], errors='coerce').fillna(0).astype('int32')
        df['total_deletions'] = pd.to_numeric(df['total_deletions'], errors='coerce').fillna(0).astype('int32')
        df['files_modified_count'] = pd.to_numeric(df['files_modified_count'], errors='coerce').fillna(0).astype('int16')
        df = df[(df['author_email'].str.len() > 0) & (df['author_email'] != 'nan')]
        return df.sort_values('commit_date')

    def compute_project_activity(self, project_name: str) -> Optional[pd.DataFrame]:
        try:
            self.logger.info(f"Processing project: {project_name}")
            df = self.load_project_commits(project_name)
            if df.empty:
                self.logger.warning(f"No commits found for project: {project_name}")
                return None
            project_type = df['project_type'].iloc[0]
            self.logger.info(f"  Loaded {len(df):,} commits from {df['author_email'].nunique()} contributors")

            min_week = df['week_start'].min(); max_week = df['week_start'].max()
            weeks = pd.date_range(start=min_week, end=max_week, freq='W-MON', tz='UTC')

            grouped = df.groupby(['author_email','week_start'])
            weekly = grouped.agg({
                'commit_hash': ['count', lambda x: list(x)],
                'total_insertions': 'sum',
                'total_deletions': 'sum',
                'files_modified_count': 'sum'
            }).reset_index()
            weekly.columns = ['author_email','week_start','commits_count','commit_hashes','lines_added','lines_deleted','files_modified']

            metrics_dict = {}
            for _, row in weekly.iterrows():
                ts = pd.Timestamp(row['week_start'])
                if ts.tzinfo is None:
                    ts = ts.tz_localize('UTC')
                else:
                    ts = ts.tz_convert('UTC')
                metrics_dict[(row['author_email'], ts)] = {
                    'commits': int(row['commits_count']),
                    'hashes': row['commit_hashes'],
                    'added': int(row['lines_added']),
                    'deleted': int(row['lines_deleted']),
                    'files': int(row['files_modified'])
                }

            project_weekly = df.groupby('week_start')['commit_hash'].count()
            project_cumulative = {}
            cum_sum = 0
            for w in weeks:
                cum_sum += project_weekly.get(w, 0)
                project_cumulative[w] = cum_sum

            records = []
            for contributor in tqdm(df['author_email'].unique(), desc=f"  {project_name}", leave=False):
                first_week = df.loc[df['author_email'] == contributor, 'week_start'].min()
                cumulative_commits = 0
                cumulative_lines = 0
                for idx, week in enumerate(weeks, start=1):
                    if week < first_week:
                        continue
                    wk = metrics_dict.get((contributor, week), {})
                    commits = wk.get('commits', 0)
                    hashes = wk.get('hashes', [])
                    added = wk.get('added', 0)
                    deleted = wk.get('deleted', 0)
                    files = wk.get('files', 0)
                    cumulative_commits += commits
                    cumulative_lines += (added + deleted)
                    proj_cum = project_cumulative.get(week, 0)
                    contrib_pct = (cumulative_commits / proj_cum * 100.0) if proj_cum > 0 else 0.0
                    week_str = week.strftime('%Y-%m-%d')
                    core_list = self.core_lookup.get((project_name, week_str), [])
                    is_core = contributor in core_list
                    records.append({
                        'project_name': project_name,
                        'project_type': project_type,
                        'contributor_email': contributor,
                        'week_date': week_str,
                        'week_number': idx,
                        'weeks_since_first_commit': int((week - first_week).days // 7),
                        'commits_this_week': commits,
                        'commit_hashes': json.dumps(hashes),
                        'lines_added_this_week': added,
                        'lines_deleted_this_week': deleted,
                        'files_modified_this_week': files,
                        'cumulative_commits': cumulative_commits,
                        'cumulative_lines_changed': cumulative_lines,
                        'project_commits_to_date': proj_cum,
                        'contribution_percentage': round(contrib_pct, 4),
                        'is_core_this_week': is_core,
                        'rank_this_week': 1
                    })
            if not records:
                return None
            result_df = pd.DataFrame(records)
            # Compute rank per (project, week)
            for week in result_df['week_date'].unique():
                mask = result_df['week_date'] == week
                ranks = result_df.loc[mask, 'cumulative_commits'].rank(method='min', ascending=False).astype('int32')
                result_df.loc[mask, 'rank_this_week'] = ranks
            # Types
            int_cols = ['week_number','weeks_since_first_commit','commits_this_week','lines_added_this_week','lines_deleted_this_week','files_modified_this_week','cumulative_commits','cumulative_lines_changed','project_commits_to_date','rank_this_week']
            for c in int_cols:
                result_df[c] = result_df[c].astype('int32')
            result_df['is_core_this_week'] = result_df['is_core_this_week'].astype('bool')
            result_df['contribution_percentage'] = result_df['contribution_percentage'].astype('float32')
            self.logger.info(f"  Generated {len(result_df):,} records")
            return result_df
        except Exception as e:
            self.logger.error(f"Error processing project {project_name}: {e}", exc_info=True)
            return None
        finally:
            gc.collect()

    def process_project(self, project_name: str) -> Tuple[bool, int]:
        df = self.compute_project_activity(project_name)
        if df is None:
            return False, 0
        safe = project_name.replace('/', '_').replace('\\', '_')
        out = self.output_dir / 'project_results' / f"{safe}_activity.csv"
        df.to_csv(out, index=False)
        self.logger.info(f"  Saved: {out}")
        return True, len(df)

    def consolidate_results(self) -> bool:
        files = list((self.output_dir / 'project_results').glob('*_activity.csv'))
        if not files:
            self.logger.error("No project result files found!")
            return False
        self.logger.info(f"Consolidating {len(files)} files...")
        chunks = []
        for fp in tqdm(files, desc="Loading files"):
            try:
                chunks.append(pd.read_csv(fp))
            except Exception as e:
                self.logger.error(f"Error reading {fp}: {e}")
        final = pd.concat(chunks, ignore_index=True).sort_values(['project_name','contributor_email','week_number'])
        out = self.output_dir / 'contributor_activity_weekly.csv'
        final.to_csv(out, index=False)
        stats = {
            'dataset_info': {
                'total_records': len(final),
                'unique_projects': final['project_name'].nunique(),
                'unique_contributors': final['contributor_email'].nunique(),
                'date_range': {'start': final['week_date'].min(), 'end': final['week_date'].max()}
            },
            'activity_metrics': {
                'active_weeks': int((final['commits_this_week'] > 0).sum()),
                'core_weeks': int(final['is_core_this_week'].sum()),
                'avg_commits_per_week': float(final['commits_this_week'].mean())
            },
            'completed_at': datetime.now().isoformat()
        }
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"Final dataset saved: {out}")
        self.logger.info(f"Total records: {len(final):,}")
        return True

    def run(self):
        if not self.checkpoint['start_time']:
            self.checkpoint['start_time'] = datetime.now().isoformat()
        self.load_core_timeline()
        all_projects, remaining = self.get_project_list()
        self.logger.info(f"Total projects: {len(all_projects)} | Remaining: {len(remaining)}")
        for i, project in enumerate(tqdm(remaining, desc="Processing projects")):
            ok, n = self.process_project(project)
            if ok:
                self.checkpoint['completed_projects'].append(project)
                self.checkpoint['total_records_created'] += n
            else:
                self.checkpoint['failed_projects'].append(project)
            self.checkpoint['total_processed'] += 1
            if (i + 1) % self.config['save_frequency'] == 0:
                self.save_checkpoint()
        self.save_checkpoint()
        self.logger.info("Starting consolidation...")
        self.consolidate_results()
        self.logger.info("Done.")

def main():
    if not Path(CONFIG['input_commits']).exists():
        print(f"Commits file not found: {CONFIG['input_commits']}")
        sys.exit(1)
    if not Path(CONFIG['input_core_timeline']).exists():
        print(f"Core timeline not found: {CONFIG['input_core_timeline']}")
        sys.exit(1)
    processor = ContributorActivityProcessor(CONFIG)
    processor.run()

if __name__ == '__main__':
    main()

