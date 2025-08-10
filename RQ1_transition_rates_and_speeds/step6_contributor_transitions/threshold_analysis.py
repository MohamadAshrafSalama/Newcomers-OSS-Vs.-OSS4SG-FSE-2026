#!/usr/bin/env python3
"""
Create Multiple Contributor Transition Datasets with Different Early-Core Thresholds
====================================================================================

Tests removing contributors who became core within different time periods:
- 1 week, 2 weeks, 4 weeks (1 month), 8 weeks (2 months),
- 12 weeks (3 months), 26 weeks (6 months), 52 weeks (1 year)

Creates separate datasets for each threshold and comprehensive statistics.
Outputs are saved under Step 6 results: results/threshold_analysis/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import sys
from tqdm import tqdm
from typing import Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Configuration (paths will be resolved relative to this script)
CONFIG: Dict = {
    # Thresholds to test (in weeks); 0 = exclude instant-only, up to 1 year
    'early_core_thresholds': [0, 1, 2, 4, 8, 12, 26, 52],
    'min_commits_threshold': 1,  # Minimum commits to include
    'max_weeks_to_track': 156,   # 3 years (unused but kept for parity)
    'chunk_size': 500000
}


class ThresholdAnalysisProcessor:
    def __init__(self, config: Dict):
        self.config = config
        # Resolve paths relative to this script directory
        self.base_dir: Path = Path(__file__).resolve().parent
        self.input_activity: Path = (self.base_dir / "../step5_weekly_datasets/dataset2_contributor_activity/contributor_activity_weekly.csv").resolve()
        self.output_dir: Path = (self.base_dir / "results/threshold_analysis").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()

        # Store results for all thresholds
        self.threshold_results: Dict[int, Dict] = {}

        # Placeholder for the full, once-processed transitions
        self.full_df: Optional[pd.DataFrame] = None

    def setup_logging(self) -> None:
        """Configure logging."""
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

    def process_contributor_project(self, contrib_data: pd.DataFrame) -> Optional[Dict]:
        """
        Process a single contributor-project pair.
        Returns full record with all metrics.
        """
        # Sort by week
        contrib_data = contrib_data.sort_values('week_number')

        # Basic info
        project_name = contrib_data['project_name'].iloc[0]
        project_type = contrib_data['project_type'].iloc[0]
        contributor_email = contrib_data['contributor_email'].iloc[0]

        # Get first and last weeks
        first_week = contrib_data['week_number'].min()
        last_week = contrib_data['week_number'].max()
        first_date = contrib_data['week_date'].min()
        last_date = contrib_data['week_date'].max()

        # Total metrics
        total_commits = contrib_data['cumulative_commits'].max()

        # Check minimum commits
        if total_commits < self.config['min_commits_threshold']:
            return None

        total_lines = contrib_data['cumulative_lines_changed'].max()
        total_weeks_observed = int(last_week - first_week + 1)

        # Active weeks
        active_weeks_mask = contrib_data['commits_this_week'] > 0
        total_active_weeks = int(active_weeks_mask.sum())

        # Check if ever became core
        became_core = bool(contrib_data['is_core_this_week'].any())

        # Initialize result
        result: Dict = {
            'project_name': project_name,
            'project_type': project_type,
            'contributor_email': contributor_email,
            'first_commit_date': first_date,
            'first_commit_week': int(first_week),
            'last_observed_date': last_date,
            'last_observed_week': int(last_week),
            'total_weeks_observed': total_weeks_observed,
            'total_commits': int(total_commits),
            'total_lines_changed': int(total_lines),
            'total_active_weeks': total_active_weeks,
            'activity_rate': float(total_active_weeks / total_weeks_observed) if total_weeks_observed > 0 else 0.0,
            'became_core': became_core
        }

        if became_core:
            # Find FIRST time became core
            core_weeks = contrib_data[contrib_data['is_core_this_week']]
            first_core_row = core_weeks.iloc[0]

            first_core_week = int(first_core_row['week_number'])
            first_core_date = first_core_row['week_date']

            # Calculate weeks to core
            weeks_to_core = int(first_core_week - first_week)

            # Get pre-core data
            pre_core_data = contrib_data[contrib_data['week_number'] <= first_core_week]

            # Metrics at core achievement
            commits_to_core = int(first_core_row['cumulative_commits'])
            lines_to_core = int(first_core_row['cumulative_lines_changed'])

            # Activity patterns
            pre_core_active = pre_core_data[pre_core_data['commits_this_week'] > 0]
            active_weeks_to_core = int(len(pre_core_active))

            if len(pre_core_active) > 0:
                avg_commits_per_active_week = float(pre_core_active['commits_this_week'].mean())
                max_commits_week = int(pre_core_active['commits_this_week'].max())
                std_commits = float(pre_core_active['commits_this_week'].std())
            else:
                avg_commits_per_active_week = 0.0
                max_commits_week = 0
                std_commits = 0.0

            # Consistency
            weeks_to_core_inclusive = weeks_to_core + 1
            commit_consistency = float(active_weeks_to_core / weeks_to_core_inclusive) if weeks_to_core_inclusive > 0 else 0.0

            # Rank and contribution
            rank_at_core = int(first_core_row['rank_this_week'])
            contribution_at_core = float(first_core_row['contribution_percentage'])

            result.update({
                'first_core_date': first_core_date,
                'first_core_week': first_core_week,
                'weeks_to_core': weeks_to_core,
                'commits_to_core': commits_to_core,
                'lines_changed_to_core': lines_to_core,
                'active_weeks_to_core': active_weeks_to_core,
                'avg_commits_per_active_week_before_core': avg_commits_per_active_week,
                'max_commits_week_before_core': max_commits_week,
                'std_commits_before_core': std_commits,
                'commit_consistency_before_core': commit_consistency,
                'rank_at_first_core': rank_at_core,
                'contribution_percentage_at_first_core': contribution_at_core,
                'censored': False,
                'time_to_event_weeks': weeks_to_core
            })

            # Post-core retention
            post_core_data = contrib_data[contrib_data['week_number'] > first_core_week]
            if len(post_core_data) > 0:
                weeks_after_core = int(len(post_core_data))
                still_core_at_end = bool(contrib_data.iloc[-1]['is_core_this_week'])
                core_weeks_total = int(contrib_data['is_core_this_week'].sum())
                core_retention_rate = float(core_weeks_total / len(contrib_data))

                result.update({
                    'weeks_observed_after_core': weeks_after_core,
                    'still_core_at_end': still_core_at_end,
                    'total_weeks_as_core': core_weeks_total,
                    'core_retention_rate': core_retention_rate
                })
            else:
                result.update({
                    'weeks_observed_after_core': 0,
                    'still_core_at_end': False,
                    'total_weeks_as_core': 1,
                    'core_retention_rate': float(1 / len(contrib_data))
                })
        else:
            # Never became core
            avg_commits = float(contrib_data[contrib_data['commits_this_week'] > 0]['commits_this_week'].mean()) if total_active_weeks > 0 else 0.0
            max_commits = int(contrib_data['commits_this_week'].max())
            std_commits = float(contrib_data[contrib_data['commits_this_week'] > 0]['commits_this_week'].std()) if total_active_weeks > 1 else 0.0

            result.update({
                'first_core_date': None,
                'first_core_week': None,
                'weeks_to_core': None,
                'commits_to_core': None,
                'lines_changed_to_core': None,
                'active_weeks_to_core': None,
                'avg_commits_per_active_week_before_core': avg_commits,
                'max_commits_week_before_core': max_commits,
                'std_commits_before_core': std_commits,
                'commit_consistency_before_core': float(total_active_weeks / total_weeks_observed) if total_weeks_observed > 0 else 0.0,
                'rank_at_first_core': None,
                'contribution_percentage_at_first_core': None,
                'censored': True,
                'time_to_event_weeks': total_weeks_observed,
                'weeks_observed_after_core': 0,
                'still_core_at_end': False,
                'total_weeks_as_core': 0,
                'core_retention_rate': 0.0
            })

        return result

    def load_and_process_data(self) -> None:
        """Load activity data and process all contributors once."""
        self.logger.info("=" * 80)
        self.logger.info("LOADING AND PROCESSING ALL CONTRIBUTOR DATA")
        self.logger.info("=" * 80)

        # Validate input file
        if not self.input_activity.exists():
            self.logger.error(f"Activity file not found: {self.input_activity}")
            sys.exit(1)

        # Load activity data
        self.logger.info(f"Loading activity data from: {self.input_activity}")

        # Process in chunks
        chunks = pd.read_csv(
            self.input_activity,
            chunksize=self.config['chunk_size'],
            usecols=[
                'project_name', 'project_type', 'contributor_email', 'week_date',
                'week_number', 'commits_this_week', 'cumulative_commits',
                'cumulative_lines_changed', 'is_core_this_week', 'rank_this_week',
                'contribution_percentage'
            ],
            low_memory=False
        )

        # Group by contributor-project
        contributor_project_data: Dict = {}

        self.logger.info("Loading and grouping data...")
        for chunk in tqdm(chunks, desc="Loading chunks"):
            for (project, contributor), group in chunk.groupby(['project_name', 'contributor_email']):
                key = (project, contributor)
                if key not in contributor_project_data:
                    contributor_project_data[key] = []
                contributor_project_data[key].append(group)

        self.logger.info(f"Found {len(contributor_project_data):,} contributor-project pairs")

        # Process each contributor once
        all_transitions = []

        for (project, contributor), data_chunks in tqdm(contributor_project_data.items(), desc="Processing contributors"):
            # Combine chunks
            contrib_data = pd.concat(data_chunks, ignore_index=True)

            # Process
            try:
                transition_record = self.process_contributor_project(contrib_data)
                if transition_record is not None:
                    all_transitions.append(transition_record)
            except Exception as e:
                self.logger.error(f"Error processing {contributor} in {project}: {str(e)}")
                continue

        # Convert to DataFrame
        self.full_df = pd.DataFrame(all_transitions)
        self.logger.info(f"Processed {len(self.full_df):,} total transitions")

        # Save complete dataset
        complete_path = self.output_dir / 'contributor_transitions_COMPLETE.csv'
        self.full_df.to_csv(complete_path, index=False)
        self.logger.info(f"Saved complete dataset: {complete_path}")

    def apply_threshold_and_analyze(self, threshold_weeks: int) -> Dict:
        """Apply a specific threshold and generate statistics."""
        self.logger.info(f"\nProcessing threshold: ≤{threshold_weeks} weeks")

        # Filter based on threshold
        if threshold_weeks == -1:
            # No filtering - use all data
            filtered_df = self.full_df.copy()
            excluded_count = 0
        else:
            # Exclude those who became core within threshold
            mask = (
                (self.full_df['became_core'] == False) |
                (
                    (self.full_df['became_core'] == True) &
                    (self.full_df['weeks_to_core'] > threshold_weeks)
                )
            )
            filtered_df = self.full_df[mask].copy()
            excluded_count = int(len(self.full_df) - len(filtered_df))

        # Calculate statistics
        stats = self.calculate_statistics(filtered_df, threshold_weeks, excluded_count)

        # Save filtered dataset
        filename = 'contributor_transitions_NO_FILTER.csv' if threshold_weeks == -1 else f'contributor_transitions_exclude_{threshold_weeks}w.csv'
        output_path = self.output_dir / filename
        filtered_df.to_csv(output_path, index=False)

        self.logger.info(f"  Saved: {filename}")
        self.logger.info(f"  Total: {len(filtered_df):,} | Excluded: {excluded_count:,}")

        # Store results
        self.threshold_results[threshold_weeks] = {
            'stats': stats,
            'filename': filename
        }

        return stats

    def calculate_statistics(self, df: pd.DataFrame, threshold: int, excluded: int) -> Dict:
        """Calculate comprehensive statistics for a dataset."""
        total = int(len(df))
        became_core = int(df['became_core'].sum())
        never_core = int((~df['became_core']).sum())

        stats: Dict = {
            'threshold_weeks': threshold,
            'total_transitions': total,
            'excluded_count': excluded,
            'exclusion_percentage': float(excluded / len(self.full_df) * 100) if self.full_df is not None and len(self.full_df) > 0 else 0.0,
            'became_core_count': became_core,
            'became_core_percentage': float(became_core / total * 100) if total > 0 else 0.0,
            'never_core_count': never_core
        }

        # Core achievement statistics
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

        # By project type
        stats['by_project_type'] = {}

        for ptype in df['project_type'].unique():
            ptype_data = df[df['project_type'] == ptype]
            ptype_core = ptype_data[ptype_data['became_core'] == True]

            ptype_stats: Dict = {
                'total': int(len(ptype_data)),
                'became_core': int(len(ptype_core)),
                'core_percentage': float(len(ptype_core) / len(ptype_data) * 100) if len(ptype_data) > 0 else 0.0
            }

            if len(ptype_core) > 0:
                ptype_stats['median_weeks'] = float(ptype_core['weeks_to_core'].median())
                ptype_stats['mean_weeks'] = float(ptype_core['weeks_to_core'].mean())
                ptype_stats['median_commits'] = float(ptype_core['commits_to_core'].median())
                ptype_stats['mean_commits'] = float(ptype_core['commits_to_core'].mean())

            stats['by_project_type'][ptype] = ptype_stats

        return stats

    def create_comprehensive_plots(self) -> None:
        """Create comprehensive visualization of all thresholds."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CREATING COMPREHENSIVE VISUALIZATIONS")
        self.logger.info("=" * 80)

        # Prepare data for plotting
        thresholds = []
        oss_medians_weeks = []
        oss4sg_medians_weeks = []
        oss_medians_commits = []
        oss4sg_medians_commits = []
        total_excluded = []
        core_rates = []

        for threshold in sorted(self.threshold_results.keys()):
            stats = self.threshold_results[threshold]['stats']
            thresholds.append(threshold if threshold != -1 else 0)
            total_excluded.append(stats['excluded_count'])
            core_rates.append(stats['became_core_percentage'])

            # Get medians by project type
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

        # Create figure with 6 subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Excluded count by threshold
        ax = axes[0, 0]
        ax.bar(thresholds, total_excluded, color='coral', alpha=0.7)
        ax.set_xlabel('Threshold (weeks)')
        ax.set_ylabel('Contributors Excluded')
        ax.set_title('Number of Contributors Excluded by Threshold')
        ax.grid(True, alpha=0.3)
        for t, e in zip(thresholds, total_excluded):
            if self.full_df is not None and len(self.full_df) > 0:
                pct = e / len(self.full_df) * 100
                ax.text(t, e, f'{pct:.1f}%', ha='center', va='bottom')

        # 2. Core achievement rate by threshold
        ax = axes[0, 1]
        ax.plot(thresholds, core_rates, 'o-', color='darkblue', linewidth=2, markersize=8)
        ax.set_xlabel('Threshold (weeks)')
        ax.set_ylabel('Core Achievement Rate (%)')
        ax.set_title('Core Achievement Rate vs Exclusion Threshold')
        ax.grid(True, alpha=0.3)

        # 3. Median weeks to core by threshold
        ax = axes[0, 2]
        ax.plot(thresholds, oss_medians_weeks, 'o-', label='OSS', linewidth=2, markersize=8)
        ax.plot(thresholds, oss4sg_medians_weeks, 's-', label='OSS4SG', linewidth=2, markersize=8)
        ax.set_xlabel('Threshold (weeks)')
        ax.set_ylabel('Median Weeks to Core')
        ax.set_title('Median Time to Core vs Exclusion Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Median commits to core by threshold
        ax = axes[1, 0]
        ax.plot(thresholds, oss_medians_commits, 'o-', label='OSS', linewidth=2, markersize=8)
        ax.plot(thresholds, oss4sg_medians_commits, 's-', label='OSS4SG', linewidth=2, markersize=8)
        ax.set_xlabel('Threshold (weeks)')
        ax.set_ylabel('Median Commits to Core')
        ax.set_title('Median Effort to Core vs Exclusion Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Ratio of medians (commits)
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

        # 6. Distribution of weeks_to_core (original data, first year)
        ax = axes[1, 2]
        core_df = self.full_df[self.full_df['became_core'] == True]
        if len(core_df) > 0:
            bins = np.arange(0, min(52, int(core_df['weeks_to_core'].max())) + 1, 1)
            oss_data = core_df[core_df['project_type'] == 'OSS']['weeks_to_core']
            oss4sg_data = core_df[core_df['project_type'] == 'OSS4SG']['weeks_to_core']
            ax.hist([oss_data[oss_data <= 52], oss4sg_data[oss4sg_data <= 52]],
                    bins=bins, label=['OSS', 'OSS4SG'], alpha=0.6, density=False, stacked=False)
            for t in [4, 8, 12, 26, 52]:
                if t <= 52:
                    ax.axvline(x=t, color='red', linestyle='--', alpha=0.3)
                    ax.text(t, ax.get_ylim()[1] * 0.95, f'{t}w', ha='center')
            ax.set_xlabel('Weeks to Core')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Time to Core (First Year)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Threshold Analysis: Impact of Excluding Early Core Contributors', fontsize=16, y=1.02)
        plt.tight_layout()

        # Save figure
        output_file = self.output_dir / 'threshold_analysis_comprehensive.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved visualization: {output_file}")
        # Do not block in headless runs
        try:
            plt.show()
        except Exception:
            pass

    def generate_recommendations(self) -> list:
        """Generate recommendations based on analysis."""
        recommendations = []

        for threshold in [4, 8, 12, 26]:
            if threshold not in self.threshold_results:
                continue
            stats = self.threshold_results[threshold]['stats']
            if 'OSS' in stats['by_project_type'] and 'OSS4SG' in stats['by_project_type']:
                oss_commits = stats['by_project_type']['OSS'].get('median_commits', 0)
                oss4sg_commits = stats['by_project_type']['OSS4SG'].get('median_commits', 0)
                if oss_commits and oss_commits > 0:
                    ratio = oss4sg_commits / oss_commits
                    if ratio < 3:
                        recommendations.append(
                            f"Threshold ≤{threshold} weeks: Ratio = {ratio:.1f}x, Excludes {stats['exclusion_percentage']:.1f}% of data"
                        )
        return recommendations

    def generate_summary_report(self) -> None:
        """Generate comprehensive summary report."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("THRESHOLD ANALYSIS SUMMARY")
        self.logger.info("=" * 80)

        # Create summary table
        summary_data = []

        for threshold in sorted(self.threshold_results.keys()):
            stats = self.threshold_results[threshold]['stats']
            row = {
                'Threshold': f'≤{threshold}w' if threshold != -1 else 'None',
                'Excluded': stats['excluded_count'],
                'Excluded%': f"{stats['exclusion_percentage']:.1f}%",
                'Core%': f"{stats['became_core_percentage']:.1f}%"
            }

            # Add OSS stats
            if 'OSS' in stats['by_project_type'] and 'median_weeks' in stats['by_project_type']['OSS']:
                row['OSS_Med_Weeks'] = f"{stats['by_project_type']['OSS']['median_weeks']:.0f}"
                row['OSS_Med_Commits'] = f"{stats['by_project_type']['OSS']['median_commits']:.0f}"
            else:
                row['OSS_Med_Weeks'] = 'N/A'
                row['OSS_Med_Commits'] = 'N/A'

            # Add OSS4SG stats
            if 'OSS4SG' in stats['by_project_type'] and 'median_weeks' in stats['by_project_type']['OSS4SG']:
                row['OSS4SG_Med_Weeks'] = f"{stats['by_project_type']['OSS4SG']['median_weeks']:.0f}"
                row['OSS4SG_Med_Commits'] = f"{stats['by_project_type']['OSS4SG']['median_commits']:.0f}"
            else:
                row['OSS4SG_Med_Weeks'] = 'N/A'
                row['OSS4SG_Med_Commits'] = 'N/A'

            # Calculate ratio
            try:
                oss_commits = float(stats['by_project_type']['OSS']['median_commits'])
                oss4sg_commits = float(stats['by_project_type']['OSS4SG']['median_commits'])
                ratio = oss4sg_commits / oss_commits if oss_commits > 0 else np.nan
                row['Ratio'] = f"{ratio:.1f}x" if pd.notna(ratio) else 'N/A'
            except Exception:
                row['Ratio'] = 'N/A'

            summary_data.append(row)

        # Print table
        print("\n" + "=" * 100)
        print("SUMMARY TABLE: Impact of Different Thresholds")
        print("=" * 100)
        print(f"{'Threshold':>10} | {'Excluded':>8} | {'Excluded%':>10} | {'Core%':>7} | "
              f"{'OSS Weeks':>10} | {'OSS Commits':>12} | {'OSS4SG Weeks':>13} | "
              f"{'OSS4SG Commits':>15} | {'Ratio':>7}")
        print("-" * 100)
        for row in summary_data:
            print(
                f"{row['Threshold']:>10} | {row['Excluded']:>8} | {row['Excluded%']:>10} | {row['Core%']:>7} | "
                f"{row['OSS_Med_Weeks']:>10} | {row['OSS_Med_Commits']:>12} | {row['OSS4SG_Med_Weeks']:>13} | {row['OSS4SG_Med_Commits']:>15} | {row['Ratio']:>7}"
            )

        # Save complete results
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

    def run_analysis(self) -> None:
        """Run the complete threshold analysis."""
        # Load and process all data once
        self.load_and_process_data()

        # Apply each threshold
        self.logger.info("\n" + "=" * 80)
        self.logger.info("APPLYING DIFFERENT THRESHOLDS")
        self.logger.info("=" * 80)

        # Test no filter first
        self.apply_threshold_and_analyze(-1)  # No filtering

        # Then test each threshold
        for threshold in self.config['early_core_thresholds']:
            self.apply_threshold_and_analyze(threshold)

        # Create visualizations
        self.create_comprehensive_plots()

        # Generate summary report
        self.generate_summary_report()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("ANALYSIS COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Created {len(self.threshold_results)} different datasets")
        self.logger.info(f"Results saved in: {self.output_dir}")


def main() -> None:
    processor = ThresholdAnalysisProcessor(CONFIG)
    processor.run_analysis()


if __name__ == "__main__":
    main()


