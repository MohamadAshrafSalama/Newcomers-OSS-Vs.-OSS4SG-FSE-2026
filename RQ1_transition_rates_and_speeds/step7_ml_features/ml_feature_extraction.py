#!/usr/bin/env python3
"""
RQ1 Step 7 — Dataset 4: ML Features for Core Prediction (v1)
============================================================

Builds ml_features_dataset.csv with early-behavior features from the first
4, 8, and 12 observed weeks, using only pre-core weeks for those who later
become core. Uses Step 6 v2 transitions for labels and first core timing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import sys
from tqdm import tqdm
from scipy import stats
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')

CONFIG = {
    'input_activity': '../step5_weekly_datasets/dataset2_contributor_activity/contributor_activity_weekly.csv',
    'input_transitions': '../step6_contributor_transitions/results/contributor_transitions.csv',
    'output_dir': Path('./results'),
    'output_file': 'ml_features_dataset.csv',
    'min_commits_for_ml': 3,
    'min_weeks_observed': 12,
    'feature_windows': [4, 8, 12],
    'chunk_size': 500000
}


class MLFeatureExtractor:
    def __init__(self, config: Dict):
        self.config = config
        self.base_dir = Path(__file__).resolve().parent
        # Resolve inputs relative to this script
        self.input_activity = (self.base_dir / config['input_activity']).resolve()
        self.input_transitions = (self.base_dir / config['input_transitions']).resolve()
        self.output_dir = (self.base_dir / config['output_dir']).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()

    def setup_logging(self) -> None:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.output_dir / 'processing.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _compute_lines_this_week(self, df: pd.DataFrame) -> pd.Series:
        """Derive lines changed this week from cumulative, per contributor.

        lines_this_week = cumulative.diff(); first row uses cumulative value.
        """
        # Assumes df is sorted by week_number
        cumulative = df['cumulative_lines_changed'].astype(float)
        delta = cumulative.diff()
        if len(delta) > 0:
            # Replace NaN for first row with its cumulative value
            delta.iloc[0] = cumulative.iloc[0]
        return delta.fillna(0)

    def extract_window_features(self, data: pd.DataFrame, window_weeks: int) -> Dict:
        window_data = data[data['week_number'] <= data['week_number'].min() + window_weeks - 1].copy()
        if len(window_data) == 0:
            return {}
        features: Dict = {}
        prefix = f'w1_{window_weeks}_'

        # Basic counts
        features[prefix + 'total_commits'] = int(window_data['commits_this_week'].sum())
        features[prefix + 'total_lines'] = int(window_data['lines_this_week'].sum())
        features[prefix + 'total_files'] = int(window_data.get('files_modified_this_week', pd.Series([0]*len(window_data))).sum())

        # Active weeks
        active_weeks = int((window_data['commits_this_week'] > 0).sum())
        features[prefix + 'active_weeks'] = active_weeks
        features[prefix + 'consistency'] = float(active_weeks / window_weeks)

        # Statistical measures
        active_data = window_data[window_data['commits_this_week'] > 0]
        if len(active_data) > 0:
            features[prefix + 'avg_commits'] = float(active_data['commits_this_week'].mean())
            features[prefix + 'max_commits'] = int(active_data['commits_this_week'].max())
            features[prefix + 'std_commits'] = float(active_data['commits_this_week'].std()) if len(active_data) > 1 else 0.0
            features[prefix + 'burst_ratio'] = float(features[prefix + 'max_commits'] / features[prefix + 'avg_commits']) if features[prefix + 'avg_commits'] > 0 else 0.0
        else:
            features[prefix + 'avg_commits'] = 0.0
            features[prefix + 'max_commits'] = 0
            features[prefix + 'std_commits'] = 0.0
            features[prefix + 'burst_ratio'] = 0.0

        # Growth trend (linear regression slope)
        if len(window_data) > 1 and features[prefix + 'total_commits'] > 0:
            x = np.arange(len(window_data))
            y = window_data['commits_this_week'].values
            if np.std(y) > 0:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                features[prefix + 'trend_slope'] = float(slope)
                features[prefix + 'trend_r2'] = float(r_value ** 2)
            else:
                features[prefix + 'trend_slope'] = 0.0
                features[prefix + 'trend_r2'] = 0.0
        else:
            features[prefix + 'trend_slope'] = 0.0
            features[prefix + 'trend_r2'] = 0.0

        # Rank progression & contribution at end
        first_rank = int(window_data.iloc[0]['rank_this_week'])
        last_rank = int(window_data.iloc[-1]['rank_this_week'])
        features[prefix + 'rank_start'] = first_rank
        features[prefix + 'rank_end'] = last_rank
        features[prefix + 'rank_improvement'] = int(first_rank - last_rank)
        features[prefix + 'contribution_pct_end'] = float(window_data.iloc[-1]['contribution_percentage'])

        # Inactive streaks
        commits_series = window_data['commits_this_week'].values
        inactive_streaks = []
        current = 0
        for c in commits_series:
            if c == 0:
                current += 1
            else:
                if current > 0:
                    inactive_streaks.append(current)
                current = 0
        if current > 0:
            inactive_streaks.append(current)
        features[prefix + 'longest_inactive_streak'] = int(max(inactive_streaks)) if inactive_streaks else 0
        features[prefix + 'num_inactive_streaks'] = int(len(inactive_streaks))

        return features

    def extract_first_week_features(self, data: pd.DataFrame) -> Dict:
        if len(data) == 0:
            return {}
        first = data.iloc[0]
        return {
            'w1_commits': int(first['commits_this_week']),
            'w1_lines_changed': int(first['lines_this_week']),
            'w1_rank': int(first['rank_this_week']),
            'w1_contribution_pct': float(first['contribution_percentage'])
        }

    def extract_temporal_patterns(self, data: pd.DataFrame) -> Dict:
        features: Dict = {}
        active_weeks = data[data['commits_this_week'] > 0]['week_number'].values
        if len(active_weeks) > 1:
            gaps = np.diff(active_weeks)
            features['avg_gap_between_active_weeks'] = float(gaps.mean())
            features['max_gap_between_active_weeks'] = int(gaps.max())
            features['std_gap_between_active_weeks'] = float(gaps.std()) if len(gaps) > 1 else 0.0
            features['activity_regularity'] = float(1.0 / (1.0 + features['std_gap_between_active_weeks']))
        else:
            features['avg_gap_between_active_weeks'] = 0.0
            features['max_gap_between_active_weeks'] = 0
            features['std_gap_between_active_weeks'] = 0.0
            features['activity_regularity'] = 0.0

        if len(data) >= 12:
            early_commits = int(data.iloc[:4]['commits_this_week'].sum())
            late_commits = int(data.iloc[8:12]['commits_this_week'].sum())
            features['activity_acceleration'] = float(late_commits / (early_commits + 1))
            features['activity_front_loaded'] = bool(early_commits > late_commits)
        else:
            features['activity_acceleration'] = 1.0
            features['activity_front_loaded'] = False
        return features

    def process_contributor(self, contrib_data: pd.DataFrame, transition_info: Dict) -> Optional[Dict]:
        contrib_data = contrib_data.sort_values('week_number').copy()
        # Derive lines_this_week from cumulative per contributor
        contrib_data['lines_this_week'] = self._compute_lines_this_week(contrib_data)

        total_commits = int(contrib_data['cumulative_commits'].max())
        if total_commits < self.config['min_commits_for_ml']:
            return None

        # Use only pre-core weeks if became_core
        if transition_info.get('became_core', False) and pd.notna(transition_info.get('first_core_week')):
            contrib_data = contrib_data[contrib_data['week_number'] < int(transition_info['first_core_week'])]

        if len(contrib_data) < self.config['min_weeks_observed']:
            return None

        features: Dict = {
            'project_name': contrib_data['project_name'].iloc[0],
            'project_type': contrib_data['project_type'].iloc[0],
            'contributor_email': contrib_data['contributor_email'].iloc[0],
            'label_became_core': bool(transition_info.get('became_core', False)),
            'weeks_to_core': int(transition_info['weeks_to_core']) if pd.notna(transition_info.get('weeks_to_core')) else np.nan,
            'total_weeks_observed': int(len(contrib_data))
        }

        # First-week features
        features.update(self.extract_first_week_features(contrib_data))

        # Window features
        for w in self.config['feature_windows']:
            features.update(self.extract_window_features(contrib_data, w))

        # Temporal patterns
        features.update(self.extract_temporal_patterns(contrib_data))

        return features

    def process_all_contributors(self) -> pd.DataFrame:
        self.logger.info("=" * 70)
        self.logger.info("STARTING ML FEATURE EXTRACTION (Step 7)")
        self.logger.info("=" * 70)

        # Load transitions for labels
        if not self.input_transitions.exists():
            self.logger.error(f"Transitions file not found: {self.input_transitions}")
            sys.exit(1)
        transitions_df = pd.read_csv(self.input_transitions, low_memory=False)
        transitions_lookup: Dict = {}
        for _, row in transitions_df.iterrows():
            transitions_lookup[(row['project_name'], row['contributor_email'])] = row.to_dict()
        self.logger.info(f"Loaded {len(transitions_lookup):,} transition records")

        # Stream activity data in chunks and group per contributor-project
        if not self.input_activity.exists():
            self.logger.error(f"Activity file not found: {self.input_activity}")
            sys.exit(1)
        self.logger.info(f"Loading activity data from: {self.input_activity}")

        all_features = []
        contributor_project_data: Dict = {}
        chunks = pd.read_csv(
            self.input_activity,
            chunksize=self.config['chunk_size'],
            usecols=[
                'project_name', 'project_type', 'contributor_email', 'week_number', 'cumulative_commits',
                'cumulative_lines_changed', 'commits_this_week', 'rank_this_week', 'contribution_percentage'
            ],
            low_memory=False
        )
        self.logger.info("Grouping activity data...")
        for chunk in tqdm(chunks, desc="Loading chunks"):
            for (project, contributor), group in chunk.groupby(['project_name', 'contributor_email']):
                key = (project, contributor)
                if key not in contributor_project_data:
                    contributor_project_data[key] = []
                contributor_project_data[key].append(group)

        self.logger.info(f"Processing {len(contributor_project_data):,} contributor-project pairs...")
        skipped_insufficient = 0
        skipped_no_transition = 0
        processed = 0

        for (project, contributor), data_chunks in tqdm(contributor_project_data.items(), desc="Extracting features"):
            transition_info = transitions_lookup.get((project, contributor))
            if transition_info is None:
                skipped_no_transition += 1
                continue
            contrib_data = pd.concat(data_chunks, ignore_index=True)
            try:
                features = self.process_contributor(contrib_data, transition_info)
                if features is None:
                    skipped_insufficient += 1
                else:
                    all_features.append(features)
                    processed += 1
            except Exception as e:
                self.logger.error(f"Error processing {contributor} in {project}: {str(e)}")
                continue

        self.logger.info(f"Creating ML dataset with {len(all_features):,} samples...")
        ml_df = pd.DataFrame(all_features)
        # Handle missing numerics
        if not ml_df.empty:
            numeric_cols = ml_df.select_dtypes(include=[np.number]).columns
            ml_df[numeric_cols] = ml_df[numeric_cols].fillna(0)

        # Sort and write
        ml_df = ml_df.sort_values(['project_name', 'label_became_core', 'contributor_email'])
        out_path = self.output_dir / self.config['output_file']
        ml_df.to_csv(out_path, index=False)
        self.logger.info(f"Saved ML features dataset: {out_path}")
        self.logger.info(f"Processed: {processed:,} | Skipped (insufficient): {skipped_insufficient:,} | Skipped (no transition): {skipped_no_transition:,}")

        self.generate_ml_stats(ml_df)
        return ml_df

    def generate_ml_stats(self, df: pd.DataFrame) -> None:
        stats_dict = {
            'dataset_info': {
                'total_samples': int(len(df)),
                'unique_projects': int(df['project_name'].nunique()) if not df.empty else 0,
                'unique_contributors': int(df['contributor_email'].nunique()) if not df.empty else 0
            },
            'label_distribution': {}
        }
        if not df.empty:
            stats_dict['label_distribution'] = {
                'became_core': int(df['label_became_core'].sum()),
                'never_core': int((~df['label_became_core']).sum()),
                'core_percentage': float(df['label_became_core'].mean() * 100)
            }
        stats_path = self.output_dir / 'ml_dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        self.logger.info(f"ML dataset statistics saved: {stats_path}")


def main() -> None:
    extractor = MLFeatureExtractor(CONFIG)
    extractor.process_all_contributors()


if __name__ == '__main__':
    main()


