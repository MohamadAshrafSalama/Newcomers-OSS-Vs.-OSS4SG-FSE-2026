#!/usr/bin/env python3
"""
Create Multiple Contributor Transition Datasets with Different Early-Core Thresholds
- Enriches each record with canonical identity from RQ2 (original_author_email/name, resolved_username)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import sys
from tqdm import tqdm
from typing import Dict, Optional
import matplotlib

# Use non-interactive backend for headless runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

CONFIG = {
    'input_activity': "RQ1_transition_rates_and_speeds/step5_weekly_datasets/dataset2_contributor_activity/contributor_activity_weekly.csv",
    'project_results_dir': "RQ1_transition_rates_and_speeds/step5_weekly_datasets/dataset2_contributor_activity/project_results",
    'identity_file': "RQ2_newcomer_treatment_patterns/data/core_contributors_filtered.csv",
    'output_dir': Path("RQ2_newcomer_treatment_patterns/threshold_analysis/results"),
    'early_core_thresholds': [0, 1, 2, 4, 8, 12, 26, 52],
    'min_commits_threshold': 1,
    'max_weeks_to_track': 156,
    'chunk_size': 500000
}

REQUIRED_COLS = [
    'project_name', 'project_type', 'contributor_email', 'week_date',
    'week_number', 'commits_this_week', 'cumulative_commits',
    'cumulative_lines_changed', 'is_core_this_week', 'rank_this_week',
    'contribution_percentage'
]


class ThresholdAnalysisProcessor:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_parent_dirs()
        self._setup_logging()
        self.threshold_results = {}
        self.identity_map: Dict = {}
        self._load_identity_map()

    def _ensure_parent_dirs(self):
        # Ensure threshold_analysis directory exists
        Path("RQ2_newcomer_treatment_patterns/threshold_analysis").mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.output_dir / 'threshold_analysis.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_identity_map(self):
        """Load core identity info keyed by (project_name, contributor_email)."""
        id_path = Path(self.config['identity_file'])
        if not id_path.exists():
            self.logger.warning(f"Identity file not found: {id_path} (proceeding without enrichment)")
            self.identity_map = {}
            return

        usecols = [
            'project_name', 'contributor_email',
            'original_author_name', 'original_author_email',
            'resolved_username', 'sample_commit_hash'
        ]
        df = pd.read_csv(id_path, usecols=usecols)

        # Prefer rows that have original_author_email
        df['has_canonical'] = df['original_author_email'].notna()
        df = df.sort_values(['project_name', 'contributor_email', 'has_canonical'], ascending=[True, True, False])
        df = df.drop_duplicates(['project_name', 'contributor_email'], keep='first')
        self.identity_map = df.set_index(['project_name', 'contributor_email']).to_dict('index')
        self.logger.info(f"Loaded identity map for {len(self.identity_map):,} project-contributor pairs")

    def _attach_identity(self, result: Dict):
        key = (result['project_name'], result['contributor_email'])
        info = self.identity_map.get(key)
        if info:
            canonical_email = info.get('original_author_email') or result['contributor_email']
            canonical_name = info.get('original_author_name')
            resolved_username = info.get('resolved_username')
            sample_commit_hash = info.get('sample_commit_hash')
        else:
            canonical_email = result['contributor_email']
            canonical_name = None
            resolved_username = None
            sample_commit_hash = None

        result.update({
            'canonical_email': canonical_email,
            'canonical_name': canonical_name,
            'resolved_username': resolved_username,
            'sample_commit_hash': sample_commit_hash
        })

    def _columns_ok(self, df: pd.DataFrame) -> bool:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return False
        return True

    def _process_contributor_project(self, contrib_data: pd.DataFrame) -> Optional[Dict]:
        contrib_data = contrib_data.sort_values('week_number')
        project_name = contrib_data['project_name'].iloc[0]
        project_type = contrib_data['project_type'].iloc[0]
        contributor_email = contrib_data['contributor_email'].iloc[0]

        first_week = contrib_data['week_number'].min()
        last_week = contrib_data['week_number'].max()
        first_date = contrib_data['week_date'].min()
        last_date = contrib_data['week_date'].max()

        total_commits = contrib_data['cumulative_commits'].max()
        if total_commits < self.config['min_commits_threshold']:
            return None

        total_lines = contrib_data['cumulative_lines_changed'].max()
        total_weeks_observed = last_week - first_week + 1

        active_weeks_mask = contrib_data['commits_this_week'] > 0
        total_active_weeks = active_weeks_mask.sum()
        became_core = contrib_data['is_core_this_week'].any()

        result = {
            'project_name': project_name,
            'project_type': project_type,
            'contributor_email': contributor_email,
            'first_commit_date': first_date,
            'first_commit_week': int(first_week),
            'last_observed_date': last_date,
            'last_observed_week': int(last_week),
            'total_weeks_observed': int(total_weeks_observed),
            'total_commits': int(total_commits),
            'total_lines_changed': int(total_lines),
            'total_active_weeks': int(total_active_weeks),
            'activity_rate': float(total_active_weeks / total_weeks_observed) if total_weeks_observed > 0 else 0,
            'became_core': became_core
        }

        if became_core:
            core_weeks = contrib_data[contrib_data['is_core_this_week']]
            first_core_row = core_weeks.iloc[0]
            first_core_week = first_core_row['week_number']
            first_core_date = first_core_row['week_date']
            weeks_to_core = first_core_week - first_week
            pre_core_data = contrib_data[contrib_data['week_number'] <= first_core_week]
            commits_to_core = first_core_row['cumulative_commits']
            lines_to_core = first_core_row['cumulative_lines_changed']
            pre_core_active = pre_core_data[pre_core_data['commits_this_week'] > 0]
            active_weeks_to_core = len(pre_core_active)

            if len(pre_core_active) > 0:
                avg_commits_per_active_week = pre_core_active['commits_this_week'].mean()
                max_commits_week = pre_core_active['commits_this_week'].max()
                std_commits = pre_core_active['commits_this_week'].std()
            else:
                avg_commits_per_active_week = 0
                max_commits_week = 0
                std_commits = 0

            weeks_to_core_inclusive = weeks_to_core + 1
            commit_consistency = active_weeks_to_core / weeks_to_core_inclusive if weeks_to_core_inclusive > 0 else 0
            rank_at_core = first_core_row['rank_this_week']
            contribution_at_core = first_core_row['contribution_percentage']

            result.update({
                'first_core_date': first_core_date,
                'first_core_week': int(first_core_week),
                'weeks_to_core': int(weeks_to_core),
                'commits_to_core': int(commits_to_core),
                'lines_changed_to_core': int(lines_to_core),
                'active_weeks_to_core': int(active_weeks_to_core),
                'avg_commits_per_active_week_before_core': float(avg_commits_per_active_week),
                'max_commits_week_before_core': int(max_commits_week),
                'std_commits_before_core': float(std_commits),
                'commit_consistency_before_core': float(commit_consistency),
                'rank_at_first_core': int(rank_at_core),
                'contribution_percentage_at_first_core': float(contribution_at_core),
                'censored': False,
                'time_to_event_weeks': int(weeks_to_core)
            })
        else:
            avg_commits = contrib_data[contrib_data['commits_this_week'] > 0]['commits_this_week'].mean() if total_active_weeks > 0 else 0
            max_commits = contrib_data['commits_this_week'].max()
            std_commits = contrib_data[contrib_data['commits_this_week'] > 0]['commits_this_week'].std() if total_active_weeks > 1 else 0
            result.update({
                'first_core_date': None,
                'first_core_week': None,
                'weeks_to_core': None,
                'commits_to_core': None,
                'lines_changed_to_core': None,
                'active_weeks_to_core': None,
                'avg_commits_per_active_week_before_core': float(avg_commits),
                'max_commits_week_before_core': int(max_commits),
                'std_commits_before_core': float(std_commits),
                'commit_consistency_before_core': float(total_active_weeks / total_weeks_observed) if total_weeks_observed > 0 else 0,
                'rank_at_first_core': None,
                'contribution_percentage_at_first_core': None,
                'censored': True,
                'time_to_event_weeks': int(total_weeks_observed),
                'weeks_observed_after_core': 0,
                'still_core_at_end': False,
                'total_weeks_as_core': 0,
                'core_retention_rate': 0.0
            })

        self._attach_identity(result)
        return result

    def load_and_process_data(self):
        self.logger.info("=" * 80)
        self.logger.info("LOADING AND PROCESSING ALL CONTRIBUTOR DATA")
        self.logger.info("=" * 80)

        in_path = Path(self.config['input_activity'])
        all_transitions = []

        if in_path.exists():
            self.logger.info(f"Loading consolidated activity data: {in_path}")
            chunks = pd.read_csv(in_path, chunksize=self.config['chunk_size'], usecols=REQUIRED_COLS)
            contributor_project_data = {}
            for chunk in tqdm(chunks, desc="Loading chunks"):
                if not self._columns_ok(chunk):
                    raise RuntimeError("Input file missing required columns")
                for (project, contributor), group in chunk.groupby(['project_name', 'contributor_email']):
                    contributor_project_data.setdefault((project, contributor), []).append(group)
            self.logger.info(f"Found {len(contributor_project_data):,} contributor-project pairs")
            for (_, _), data_chunks in tqdm(contributor_project_data.items(), desc="Processing contributors"):
                contrib_data = pd.concat(data_chunks, ignore_index=True)
                rec = self._process_contributor_project(contrib_data)
                if rec is not None:
                    all_transitions.append(rec)
        else:
            proj_dir = Path(self.config['project_results_dir'])
            if not proj_dir.exists():
                raise FileNotFoundError(f"Neither {in_path} nor {proj_dir} exists")
            self.logger.info(f"Consolidated file not found. Falling back to per-project results: {proj_dir}")
            files = list(proj_dir.glob("*_activity.csv"))
            self.logger.info(f"Found {len(files)} project files")
            for fp in tqdm(files, desc="Processing project files"):
                try:
                    df = pd.read_csv(fp, usecols=REQUIRED_COLS)
                    if not self._columns_ok(df):
                        continue
                    for _, group in df.groupby('contributor_email'):
                        rec = self._process_contributor_project(group)
                        if rec is not None:
                            all_transitions.append(rec)
                except Exception as e:
                    self.logger.error(f"Error reading {fp}: {e}")

        self.full_df = pd.DataFrame(all_transitions)
        complete_path = self.output_dir / 'contributor_transitions_COMPLETE.csv'
        self.full_df.to_csv(complete_path, index=False)
        self.logger.info(f"Saved complete dataset: {complete_path}")

    def apply_threshold_and_analyze(self, threshold_weeks: int):
        self.logger.info(f"\nProcessing threshold: ≤{threshold_weeks} weeks")
        if threshold_weeks == -1:
            filtered_df = self.full_df.copy()
            excluded_count = 0
        else:
            mask = (self.full_df['became_core'] == False) | \
                   ((self.full_df['became_core'] == True) &
                    (self.full_df['weeks_to_core'] > threshold_weeks))
            filtered_df = self.full_df[mask].copy()
            excluded_count = len(self.full_df) - len(filtered_df)

        stats = self.calculate_statistics(filtered_df, threshold_weeks, excluded_count)

        filename = 'contributor_transitions_NO_FILTER.csv' if threshold_weeks == -1 else f'contributor_transitions_exclude_{threshold_weeks}w.csv'
        output_path = self.output_dir / filename
        filtered_df.to_csv(output_path, index=False)
        self.logger.info(f"  Saved: {filename}")
        self.logger.info(f"  Total: {len(filtered_df):,} | Excluded: {excluded_count:,}")

        self.threshold_results[threshold_weeks] = {'df': filtered_df, 'stats': stats, 'filename': filename}
        return stats

    def calculate_statistics(self, df: pd.DataFrame, threshold: int, excluded: int) -> Dict:
        total = len(df)
        became_core = df['became_core'].sum()
        never_core = (~df['became_core']).sum()
        stats = {
            'threshold_weeks': threshold,
            'total_transitions': total,
            'excluded_count': excluded,
            'exclusion_percentage': (excluded / len(self.full_df) * 100) if len(self.full_df) > 0 else 0,
            'became_core_count': int(became_core),
            'became_core_percentage': float(became_core / total * 100) if total > 0 else 0,
            'never_core_count': int(never_core)
        }
        core_df = df[df['became_core'] == True]
        if len(core_df) > 0:
            stats['time_to_core'] = {
                'mean': float(core_df['weeks_to_core'].mean()),
                'median': float(core_df['weeks_to_core'].median()),
                'std': float(core_df['weeks_to_core'].std()),
                'min': int(core_df['weeks_to_core'].min()),
                'max': int(core_df['weeks_to_core'].max()),
                'q25': float(core_df['weeks_to_core'].quantile(0.25)),
                'q75': float(core_df['weeks_to_core'].quantile(0.75))
            }
            stats['commits_to_core'] = {
                'mean': float(core_df['commits_to_core'].mean()),
                'median': float(core_df['commits_to_core'].median()),
                'std': float(core_df['commits_to_core'].std()),
                'min': int(core_df['commits_to_core'].min()),
                'max': int(core_df['commits_to_core'].max()),
                'q25': float(core_df['commits_to_core'].quantile(0.25)),
                'q75': float(core_df['commits_to_core'].quantile(0.75))
            }
        stats['by_project_type'] = {}
        for ptype in df['project_type'].unique():
            ptype_data = df[df['project_type'] == ptype]
            ptype_core = ptype_data[ptype_data['became_core'] == True]
            ptype_stats = {
                'total': len(ptype_data),
                'became_core': len(ptype_core),
                'core_percentage': float(len(ptype_core) / len(ptype_data) * 100) if len(ptype_data) > 0 else 0
            }
            if len(ptype_core) > 0:
                ptype_stats['median_weeks'] = float(ptype_core['weeks_to_core'].median())
                ptype_stats['mean_weeks'] = float(ptype_core['weeks_to_core'].mean())
                ptype_stats['median_commits'] = float(ptype_core['commits_to_core'].median())
                ptype_stats['mean_commits'] = float(ptype_core['commits_to_core'].mean())
            stats['by_project_type'][ptype] = ptype_stats
        return stats

    def create_comprehensive_plots(self):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CREATING COMPREHENSIVE VISUALIZATIONS")
        self.logger.info("=" * 80)

        thresholds = []
        oss_medians_weeks, oss4sg_medians_weeks = [], []
        oss_medians_commits, oss4sg_medians_commits = [], []
        total_excluded, core_rates = [], []

        for threshold in sorted(self.threshold_results.keys()):
            stats = self.threshold_results[threshold]['stats']
            thresholds.append(threshold if threshold != -1 else 0)
            total_excluded.append(stats['excluded_count'])
            core_rates.append(stats['became_core_percentage'])

            if 'OSS' in stats['by_project_type'] and 'median_weeks' in stats['by_project_type']['OSS']:
                oss_medians_weeks.append(stats['by_project_type']['OSS']['median_weeks'])
                oss_medians_commits.append(stats['by_project_type']['OSS']['median_commits'])
            else:
                oss_medians_weeks.append(np.nan)
                oss_medians_commits.append(np.nan)

            if 'OSS4SG' in stats['by_project_type'] and 'median_weeks' in stats['by_project_type']['OSS4SG']:
                oss4sg_medians_weeks.append(stats['by_project_type']['OSS4SG']['median_weeks'])
                oss4sg_medians_commits.append(stats['by_project_type']['OSS4SG']['median_commits'])
            else:
                oss4sg_medians_weeks.append(np.nan)
                oss4sg_medians_commits.append(np.nan)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        ax = axes[0, 0]
        ax.bar(thresholds, total_excluded, color='coral', alpha=0.7)
        ax.set_xlabel('Threshold (weeks)')
        ax.set_ylabel('Contributors Excluded')
        ax.set_title('Number of Contributors Excluded by Threshold')
        ax.grid(True, alpha=0.3)
        for t, e in zip(thresholds, total_excluded):
            if len(self.full_df) > 0:
                pct = e / len(self.full_df) * 100
                ax.text(t, e, f'{pct:.1f}%', ha='center', va='bottom')

        ax = axes[0, 1]
        ax.plot(thresholds, core_rates, 'o-', color='darkblue', linewidth=2, markersize=8)
        ax.set_xlabel('Threshold (weeks)')
        ax.set_ylabel('Core Achievement Rate (%)')
        ax.set_title('Core Achievement Rate vs Exclusion Threshold')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        ax.plot(thresholds, oss_medians_weeks, 'o-', label='OSS', linewidth=2, markersize=8)
        ax.plot(thresholds, oss4sg_medians_weeks, 's-', label='OSS4SG', linewidth=2, markersize=8)
        ax.set_xlabel('Threshold (weeks)')
        ax.set_ylabel('Median Weeks to Core')
        ax.set_title('Median Time to Core vs Exclusion Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(thresholds, oss_medians_commits, 'o-', label='OSS', linewidth=2, markersize=8)
        ax.plot(thresholds, oss4sg_medians_commits, 's-', label='OSS4SG', linewidth=2, markersize=8)
        ax.set_xlabel('Threshold (weeks)')
        ax.set_ylabel('Median Commits to Core')
        ax.set_title('Median Effort to Core vs Exclusion Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ratios = []
        for oss, oss4sg in zip(oss_medians_commits, oss4sg_medians_commits):
            if pd.notna(oss) and pd.notna(oss4sg) and oss > 0:
                ratios.append(oss4sg / oss)
            else:
                ratios.append(np.nan)
        ax.plot(thresholds, ratios, 'o-', color='purple', linewidth=2, markersize=8)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Threshold (weeks)')
        ax.set_ylabel('Ratio (OSS4SG / OSS)')
        ax.set_title('Commit Requirement Ratio vs Exclusion Threshold')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        core_df = self.full_df[self.full_df['became_core'] == True]
        if len(core_df) > 0:
            bins = np.arange(0, min(52, core_df['weeks_to_core'].max()) + 1, 1)
            oss_data = core_df[core_df['project_type'] == 'OSS']['weeks_to_core']
            oss4sg_data = core_df[core_df['project_type'] == 'OSS4SG']['weeks_to_core']
            ax.hist([oss_data[oss_data <= 52], oss4sg_data[oss4sg_data <= 52]],
                    bins=bins, label=['OSS', 'OSS4SG'], alpha=0.6, density=False, stacked=False)
            for t in [4, 8, 12, 26, 52]:
                if t <= 52:
                    ax.axvline(x=t, color='red', linestyle='--', alpha=0.3)
                    ax.text(t, ax.get_ylim()[1]*0.95, f'{t}w', ha='center')
        ax.set_xlabel('Weeks to Core')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Time to Core (First Year)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Threshold Analysis: Impact of Excluding Early Core Contributors', fontsize=16, y=1.02)
        plt.tight_layout()
        output_file = self.output_dir / 'threshold_analysis_comprehensive.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved visualization: {output_file}")

    def generate_summary_report(self):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("THRESHOLD ANALYSIS SUMMARY")
        self.logger.info("=" * 80)

        summary_data = []
        for threshold in sorted(self.threshold_results.keys()):
            stats = self.threshold_results[threshold]['stats']
            row = {
                'Threshold': f'≤{threshold}w' if threshold != -1 else 'None',
                'Excluded': stats['excluded_count'],
                'Excluded%': f"{stats['exclusion_percentage']:.1f}%",
                'Core%': f"{stats['became_core_percentage']:.1f}%"
            }
            if 'OSS' in stats['by_project_type'] and 'median_weeks' in stats['by_project_type']['OSS']:
                row['OSS_Med_Weeks'] = f"{stats['by_project_type']['OSS']['median_weeks']:.0f}"
                row['OSS_Med_Commits'] = f"{stats['by_project_type']['OSS']['median_commits']:.0f}"
            else:
                row['OSS_Med_Weeks'] = 'N/A'
                row['OSS_Med_Commits'] = 'N/A'
            if 'OSS4SG' in stats['by_project_type'] and 'median_weeks' in stats['by_project_type']['OSS4SG']:
                row['OSS4SG_Med_Weeks'] = f"{stats['by_project_type']['OSS4SG']['median_weeks']:.0f}"
                row['OSS4SG_Med_Commits'] = f"{stats['by_project_type']['OSS4SG']['median_commits']:.0f}"
            else:
                row['OSS4SG_Med_Weeks'] = 'N/A'
                row['OSS4SG_Med_Commits'] = 'N/A'
            try:
                oss_commits = float(stats['by_project_type']['OSS']['median_commits'])
                oss4sg_commits = float(stats['by_project_type']['OSS4SG']['median_commits'])
                row['Ratio'] = f"{(oss4sg_commits/oss_commits):.1f}x" if oss_commits > 0 else 'N/A'
            except Exception:
                row['Ratio'] = 'N/A'
            summary_data.append(row)

        print("\n" + "=" * 100)
        print("SUMMARY TABLE: Impact of Different Thresholds")
        print("=" * 100)
        print(f"{'Threshold':>10} | {'Excluded':>8} | {'Excluded%':>10} | {'Core%':>7} | "
              f"{'OSS Weeks':>10} | {'OSS Commits':>12} | {'OSS4SG Weeks':>13} | "
              f"{'OSS4SG Commits':>15} | {'Ratio':>7}")
        print("-" * 100)
        for row in summary_data:
            print(f"{row['Threshold']:>10} | {row['Excluded']:>8} | {row['Excluded%']:>10} | "
                  f"{row['Core%']:>7} | {row['OSS_Med_Weeks']:>10} | {row['OSS_Med_Commits']:>12} | "
                  f"{row['OSS4SG_Med_Weeks']:>13} | {row['OSS4SG_Med_Commits']:>15} | {row['Ratio']:>7}")

        results = {
            'analysis_date': datetime.now().isoformat(),
            'thresholds_tested': list(self.threshold_results.keys()),
            'summary_table': summary_data,
            'detailed_stats': {k: v['stats'] for k, v in self.threshold_results.items()},
            'recommendations': self.generate_recommendations()
        }
        output_file = self.output_dir / 'threshold_analysis_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"\n📊 Complete results saved: {output_file}")

    def generate_recommendations(self):
        recommendations = []
        for threshold in [4, 8, 12, 26]:
            stats = self.threshold_results[threshold]['stats']
            if 'OSS' in stats['by_project_type'] and 'OSS4SG' in stats['by_project_type']:
                oss_commits = stats['by_project_type']['OSS'].get('median_commits', 0)
                oss4sg_commits = stats['by_project_type']['OSS4SG'].get('median_commits', 0)
                if oss_commits > 0:
                    ratio = oss4sg_commits / oss_commits
                    if ratio < 3:
                        recommendations.append(
                            f"Threshold ≤{threshold} weeks: Ratio = {ratio:.1f}x, "
                            f"Excludes {stats['exclusion_percentage']:.1f}% of data"
                        )
        return recommendations

    def run_analysis(self):
        self.load_and_process_data()
        self.logger.info("\n" + "=" * 80)
        self.logger.info("APPLYING DIFFERENT THRESHOLDS")
        self.logger.info("=" * 80)
        self.apply_threshold_and_analyze(-1)
        for threshold in self.config['early_core_thresholds']:
            self.apply_threshold_and_analyze(threshold)
        self.create_comprehensive_plots()
        self.generate_summary_report()
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ANALYSIS COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Created {len(self.threshold_results)} different datasets")
        self.logger.info(f"Results saved in: {self.output_dir}")


def main():
    processor = ThresholdAnalysisProcessor(CONFIG)
    processor.run_analysis()


if __name__ == "__main__":
    main()


