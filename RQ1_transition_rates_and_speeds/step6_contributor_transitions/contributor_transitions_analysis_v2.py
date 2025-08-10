#!/usr/bin/env python3
"""
RQ1 Step 6 - Dataset 3: Contributor Transitions Analysis v2
===========================================================

Creates contributor_transitions.csv tracking true newcomer journeys.

IMPROVEMENTS:
1. Excludes instant core (week 0)
2. Excludes early project members (join ≤8 weeks AND core ≤4 weeks)
3. Tracks exclusion reasons for transparency
4. Creates multiple datasets for comparison

One row per contributor-project pair.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import sys
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'input_activity': "../step5_weekly_datasets/dataset2_contributor_activity/contributor_activity_weekly.csv",
    'output_dir': Path("results"),
    'output_file': 'contributor_transitions.csv',
    'output_file_all': 'contributor_transitions_including_all.csv',
    'output_file_instant_only': 'contributor_transitions_no_instant.csv',
    
    # Exclusion thresholds
    'min_commits_threshold': 1,  # Minimum commits to include
    'exclude_instant_core': True,  # Exclude week 0 core
    'exclude_early_project_core': True,  # Exclude early project members
    'early_project_weeks': 8,  # Consider first 8 weeks as "early project"
    'early_core_weeks': 4,  # If core within 4 weeks of joining early
    
    'max_weeks_to_track': 156,  # 3 years
    'chunk_size': 500000  # Process in chunks for memory efficiency
}

class TransitionAnalyzer:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # Detailed statistics tracking
        self.stats = {
            'total_processed': 0,
            'instant_core_excluded': 0,
            'early_project_core_excluded': 0,
            'low_activity_excluded': 0,
            'included_in_analysis': 0,
            'exclusion_reasons': {}  # Track why each was excluded
        }
        
    def setup_logging(self):
        """Configure logging."""
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
    
    def classify_contributor(self, contrib_data: pd.DataFrame, became_core: bool, 
                            first_week: int, weeks_to_core: Optional[int]) -> str:
        """
        Classify a contributor to determine if they should be excluded.
        
        Returns:
            Classification string: 'include', 'instant_core', 'early_project_core', 'low_activity'
        """
        # Check minimum commits
        total_commits = contrib_data['cumulative_commits'].max()
        if total_commits < self.config['min_commits_threshold']:
            return 'low_activity'
        
        if not became_core:
            return 'include'  # Never became core - always include
        
        # Check instant core (week 0)
        if weeks_to_core == 0:
            return 'instant_core'
        
        # Check early project member who became core quickly
        if (self.config['exclude_early_project_core'] and
            first_week <= self.config['early_project_weeks'] and
            weeks_to_core <= self.config['early_core_weeks']):
            return 'early_project_core'
        
        return 'include'
    
    def process_contributor_project(self, contrib_data: pd.DataFrame, 
                                   classification_mode: str = 'full') -> Optional[Dict]:
        """
        Process a single contributor-project pair.
        
        Args:
            contrib_data: DataFrame with all weeks for one contributor in one project
            classification_mode: 'full', 'no_instant', or 'all'
        
        Returns:
            Dictionary with transition metrics or None if excluded
        """
        # Sort by week to ensure chronological order
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
        
        # Check if ever became core
        became_core = contrib_data['is_core_this_week'].any()
        
        # Calculate weeks to core if applicable
        weeks_to_core = None
        if became_core:
            first_core_week_num = contrib_data[contrib_data['is_core_this_week']]['week_number'].min()
            weeks_to_core = first_core_week_num - first_week
        
        # Classify the contributor
        classification = self.classify_contributor(contrib_data, became_core, first_week, weeks_to_core)
        
        # Apply exclusion based on mode
        if classification_mode == 'full':
            # Full exclusion (instant + early project)
            if classification in ['instant_core', 'early_project_core', 'low_activity']:
                self.stats[f'{classification}_excluded'] += 1
                self.stats['exclusion_reasons'][contributor_email] = classification
                return None
                
        elif classification_mode == 'no_instant':
            # Only exclude instant core
            if classification in ['instant_core', 'low_activity']:
                self.stats[f'{classification}_excluded'] += 1
                return None
                
        # If we get here, include the contributor
        
        # Total metrics (entire observation period)
        total_commits = contrib_data['cumulative_commits'].max()
        total_lines = contrib_data['cumulative_lines_changed'].max()
        total_weeks_observed = last_week - first_week + 1
        
        # Active weeks (weeks with commits > 0)
        active_weeks_mask = contrib_data['commits_this_week'] > 0
        total_active_weeks = active_weeks_mask.sum()
        
        # Initialize result
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
            'became_core': became_core,
            'contributor_classification': classification  # Add classification for transparency
        }
        
        if became_core:
            # Find FIRST time became core
            core_weeks = contrib_data[contrib_data['is_core_this_week']]
            first_core_row = core_weeks.iloc[0]
            
            first_core_week = first_core_row['week_number']
            first_core_date = first_core_row['week_date']
            
            # Get data up to (and including) first core week
            pre_core_data = contrib_data[contrib_data['week_number'] <= first_core_week]
            
            # Commits at point of becoming core
            commits_to_core = first_core_row['cumulative_commits']
            lines_to_core = first_core_row['cumulative_lines_changed']
            
            # Activity patterns before core
            pre_core_active = pre_core_data[pre_core_data['commits_this_week'] > 0]
            active_weeks_to_core = len(pre_core_active)
            
            # Calculate metrics for pre-core period
            if len(pre_core_active) > 0:
                avg_commits_per_active_week = pre_core_active['commits_this_week'].mean()
                max_commits_week = pre_core_active['commits_this_week'].max()
                std_commits = pre_core_active['commits_this_week'].std()
                
                # Calculate burst pattern
                burst_ratio = max_commits_week / avg_commits_per_active_week if avg_commits_per_active_week > 0 else 0
            else:
                avg_commits_per_active_week = 0
                max_commits_week = 0
                std_commits = 0
                burst_ratio = 0
            
            # Consistency metrics
            weeks_to_core_inclusive = weeks_to_core + 1
            commit_consistency = active_weeks_to_core / weeks_to_core_inclusive if weeks_to_core_inclusive > 0 else 0
            
            # Rank and contribution at core achievement
            rank_at_core = first_core_row['rank_this_week']
            contribution_at_core = first_core_row['contribution_percentage']
            
            # Calculate growth rate
            if weeks_to_core >= 8:
                quarter_size = max(1, weeks_to_core_inclusive // 4)
                first_quarter = pre_core_data.iloc[:quarter_size]
                last_quarter = pre_core_data.iloc[-quarter_size:]
                
                first_quarter_commits = first_quarter['commits_this_week'].sum()
                last_quarter_commits = last_quarter['commits_this_week'].sum()
                
                growth_rate = (last_quarter_commits / (first_quarter_commits + 1)) - 1
            else:
                growth_rate = 0
            
            # Update result with core metrics
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
                'burst_ratio_before_core': float(burst_ratio),
                'commit_consistency_before_core': float(commit_consistency),
                'growth_rate_before_core': float(growth_rate),
                'rank_at_first_core': int(rank_at_core),
                'contribution_percentage_at_first_core': float(contribution_at_core),
                'censored': False,
                'time_to_event_weeks': int(weeks_to_core),
                
                # Additional classification info
                'is_early_joiner': first_week <= self.config['early_project_weeks'],
                'is_fast_core': weeks_to_core <= self.config['early_core_weeks']
            })
            
            # Post-core retention metrics
            post_core_data = contrib_data[contrib_data['week_number'] > first_core_week]
            if len(post_core_data) > 0:
                weeks_after_core = len(post_core_data)
                still_core_at_end = contrib_data.iloc[-1]['is_core_this_week']
                core_weeks_total = contrib_data['is_core_this_week'].sum()
                core_retention_rate = core_weeks_total / len(contrib_data)
                
                result.update({
                    'weeks_observed_after_core': int(weeks_after_core),
                    'still_core_at_end': bool(still_core_at_end),
                    'total_weeks_as_core': int(core_weeks_total),
                    'core_retention_rate': float(core_retention_rate)
                })
            else:
                result.update({
                    'weeks_observed_after_core': 0,
                    'still_core_at_end': False,
                    'total_weeks_as_core': 1,
                    'core_retention_rate': float(1 / len(contrib_data))
                })
            
        else:
            # Never became core - censored observation
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
                'burst_ratio_before_core': 0.0,
                'commit_consistency_before_core': float(total_active_weeks / total_weeks_observed) if total_weeks_observed > 0 else 0,
                'growth_rate_before_core': 0.0,
                'rank_at_first_core': None,
                'contribution_percentage_at_first_core': None,
                'censored': True,
                'time_to_event_weeks': int(total_weeks_observed),
                'is_early_joiner': first_week <= self.config['early_project_weeks'],
                'is_fast_core': False,
                'weeks_observed_after_core': 0,
                'still_core_at_end': False,
                'total_weeks_as_core': 0,
                'core_retention_rate': 0.0
            })
        
        self.stats['included_in_analysis'] += 1
        return result
    
    def process_all_contributors(self):
        """Process all contributor-project pairs with multiple exclusion strategies."""
        self.logger.info("=" * 70)
        self.logger.info("CONTRIBUTOR TRANSITIONS ANALYSIS v2")
        self.logger.info("=" * 70)
        self.logger.info(f"Exclusion settings:")
        self.logger.info(f"  - Instant core (week 0): EXCLUDE")
        self.logger.info(f"  - Early project members (≤{self.config['early_project_weeks']}w) who become core fast (≤{self.config['early_core_weeks']}w): EXCLUDE")
        self.logger.info("=" * 70)
        
        # Load activity data
        self.logger.info(f"Loading activity data from: {self.config['input_activity']}")
        
        # Storage for different datasets
        all_transitions_full = []  # With full exclusions
        all_transitions_no_instant = []  # Only instant excluded
        all_transitions_complete = []  # No exclusions
        
        # Get unique contributor-project pairs
        chunks = pd.read_csv(
            self.config['input_activity'],
            chunksize=self.config['chunk_size'],
            usecols=['project_name', 'project_type', 'contributor_email', 'week_date', 
                    'week_number', 'commits_this_week', 'cumulative_commits', 
                    'cumulative_lines_changed', 'is_core_this_week', 'rank_this_week',
                    'contribution_percentage']
        )
        
        # Process by grouping
        contributor_project_data = {}
        
        self.logger.info("Loading and grouping data...")
        for chunk in tqdm(chunks, desc="Loading chunks"):
            for (project, contributor), group in chunk.groupby(['project_name', 'contributor_email']):
                key = (project, contributor)
                if key not in contributor_project_data:
                    contributor_project_data[key] = []
                contributor_project_data[key].append(group)
        
        self.logger.info(f"Found {len(contributor_project_data):,} contributor-project pairs")
        
        # Process each contributor-project pair
        self.logger.info("Processing transitions...")
        
        for (project, contributor), data_chunks in tqdm(contributor_project_data.items(), 
                                                        desc="Processing contributors"):
            self.stats['total_processed'] += 1
            
            # Combine all chunks for this contributor-project
            contrib_data = pd.concat(data_chunks, ignore_index=True)
            
            try:
                # Process with full exclusions (main dataset)
                result_full = self.process_contributor_project(contrib_data, 'full')
                if result_full is not None:
                    all_transitions_full.append(result_full)
                
                # Process with only instant core excluded
                self.stats['instant_core_excluded'] = 0  # Reset counter
                result_no_instant = self.process_contributor_project(contrib_data, 'no_instant')
                if result_no_instant is not None:
                    all_transitions_no_instant.append(result_no_instant)
                
                # Process with no exclusions (complete dataset)
                result_all = self.process_contributor_project(contrib_data, 'all')
                if result_all is not None:
                    all_transitions_complete.append(result_all)
                
            except Exception as e:
                self.logger.error(f"Error processing {contributor} in {project}: {str(e)}")
                continue
        
        # Create and save datasets
        datasets = [
            (all_transitions_full, self.config['output_file'], "Main (Full Exclusions)"),
            (all_transitions_no_instant, self.config['output_file_instant_only'], "No Instant Core Only"),
            (all_transitions_complete, self.config['output_file_all'], "Complete (No Exclusions)")
        ]
        
        for transitions, filename, name in datasets:
            if transitions:
                self.logger.info(f"\nCreating {name} dataset with {len(transitions):,} transitions...")
                df = pd.DataFrame(transitions)
                df = df.sort_values(['project_name', 'became_core', 'contributor_email'])
                
                output_path = self.output_dir / filename
                df.to_csv(output_path, index=False)
                self.logger.info(f"Saved: {output_path}")
                
                # Generate statistics for main dataset
                if filename == self.config['output_file']:
                    self.generate_summary_stats(df, name)
        
        # Print exclusion statistics
        self.logger.info("\n" + "=" * 70)
        self.logger.info("EXCLUSION STATISTICS")
        self.logger.info("=" * 70)
        self.logger.info(f"Total contributor-project pairs processed: {self.stats['total_processed']:,}")
        self.logger.info(f"Instant core excluded: {self.stats.get('instant_core_excluded', 0):,}")
        self.logger.info(f"Early project core excluded: {self.stats.get('early_project_core_excluded', 0):,}")
        self.logger.info(f"Low activity excluded: {self.stats.get('low_activity_excluded', 0):,}")
        self.logger.info(f"Included in main analysis: {len(all_transitions_full):,}")
        
        # Show breakdown of exclusion reasons
        if self.stats['exclusion_reasons']:
            reasons_count = {}
            for reason in self.stats['exclusion_reasons'].values():
                reasons_count[reason] = reasons_count.get(reason, 0) + 1
            
            self.logger.info("\nExclusion reason breakdown:")
            for reason, count in sorted(reasons_count.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {reason}: {count:,}")
    
    def generate_summary_stats(self, df: pd.DataFrame, dataset_name: str = "Dataset"):
        """Generate and save comprehensive summary statistics."""
        if len(df) == 0:
            self.logger.warning("Empty dataframe - skipping statistics")
            return
        
        stats = {
            'dataset_name': dataset_name,
            'dataset_info': {
                'total_transitions': len(df),
                'unique_projects': df['project_name'].nunique(),
                'unique_contributors': df['contributor_email'].nunique(),
                'date_range': {
                    'earliest_first_commit': str(df['first_commit_date'].min()),
                    'latest_first_commit': str(df['first_commit_date'].max())
                }
            },
            'exclusion_settings': {
                'instant_core_excluded': self.config['exclude_instant_core'],
                'early_project_excluded': self.config['exclude_early_project_core'],
                'early_project_weeks_threshold': self.config['early_project_weeks'],
                'early_core_weeks_threshold': self.config['early_core_weeks']
            },
            'exclusion_stats': {
                'total_processed': self.stats['total_processed'],
                'instant_core_excluded': self.stats.get('instant_core_excluded', 0),
                'early_project_core_excluded': self.stats.get('early_project_core_excluded', 0),
                'included_in_analysis': len(df)
            },
            'core_achievement': {
                'became_core_count': int(df['became_core'].sum()),
                'became_core_percentage': float(df['became_core'].mean() * 100),
                'never_core_count': int((~df['became_core']).sum()),
                'never_core_percentage': float((~df['became_core']).mean() * 100)
            }
        }
        
        # Time to core statistics (only for those who became core)
        core_df = df[df['became_core']]
        if len(core_df) > 0:
            stats['time_to_core_stats'] = {
                'median_weeks': float(core_df['weeks_to_core'].median()),
                'mean_weeks': float(core_df['weeks_to_core'].mean()),
                'std_weeks': float(core_df['weeks_to_core'].std()),
                'min_weeks': int(core_df['weeks_to_core'].min()),
                'max_weeks': int(core_df['weeks_to_core'].max()),
                'q25_weeks': float(core_df['weeks_to_core'].quantile(0.25)),
                'q75_weeks': float(core_df['weeks_to_core'].quantile(0.75))
            }
            
            stats['effort_to_core_stats'] = {
                'median_commits': float(core_df['commits_to_core'].median()),
                'mean_commits': float(core_df['commits_to_core'].mean()),
                'std_commits': float(core_df['commits_to_core'].std()),
                'min_commits': int(core_df['commits_to_core'].min()),
                'max_commits': int(core_df['commits_to_core'].max())
            }
        
        # Project type comparison
        stats['project_type_comparison'] = {}
        for ptype in df['project_type'].unique():
            ptype_data = df[df['project_type'] == ptype]
            ptype_core = ptype_data[ptype_data['became_core']]
            
            stats['project_type_comparison'][ptype] = {
                'total_contributors': len(ptype_data),
                'became_core_count': len(ptype_core),
                'became_core_percentage': float(len(ptype_core) / len(ptype_data) * 100) if len(ptype_data) > 0 else 0
            }
            
            if len(ptype_core) > 0:
                stats['project_type_comparison'][ptype].update({
                    'median_weeks_to_core': float(ptype_core['weeks_to_core'].median()),
                    'mean_weeks_to_core': float(ptype_core['weeks_to_core'].mean()),
                    'median_commits_to_core': float(ptype_core['commits_to_core'].median()),
                    'mean_commits_to_core': float(ptype_core['commits_to_core'].mean()),
                    'min_weeks_to_core': int(ptype_core['weeks_to_core'].min())
                })
        
        # Save statistics
        stats_path = self.output_dir / 'transition_statistics_v2.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Summary statistics saved: {stats_path}")
        
        # Print summary
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"SUMMARY STATISTICS - {dataset_name}")
        self.logger.info("=" * 70)
        self.logger.info(f"Total contributor-project pairs: {len(df):,}")
        self.logger.info(f"Became core: {stats['core_achievement']['became_core_count']:,} ({stats['core_achievement']['became_core_percentage']:.1f}%)")
        self.logger.info(f"Never became core: {stats['core_achievement']['never_core_count']:,} ({stats['core_achievement']['never_core_percentage']:.1f}%)")
        
        if 'time_to_core_stats' in stats:
            self.logger.info(f"\nTime to core (weeks) - True Newcomers:")
            self.logger.info(f"  Median: {stats['time_to_core_stats']['median_weeks']:.1f}")
            self.logger.info(f"  Mean: {stats['time_to_core_stats']['mean_weeks']:.1f}")
            self.logger.info(f"  Range: {stats['time_to_core_stats']['min_weeks']}-{stats['time_to_core_stats']['max_weeks']}")
            
            self.logger.info(f"\nEffort to core (commits):")
            self.logger.info(f"  Median: {stats['effort_to_core_stats']['median_commits']:.0f}")
            self.logger.info(f"  Mean: {stats['effort_to_core_stats']['mean_commits']:.1f}")
        
        self.logger.info("\nProject type comparison:")
        for ptype, pstats in stats['project_type_comparison'].items():
            self.logger.info(f"\n{ptype}:")
            self.logger.info(f"  Contributors: {pstats['total_contributors']:,}")
            self.logger.info(f"  Became core: {pstats['became_core_count']:,} ({pstats['became_core_percentage']:.1f}%)")
            if 'median_weeks_to_core' in pstats:
                self.logger.info(f"  Median time: {pstats['median_weeks_to_core']:.1f} weeks")
                self.logger.info(f"  Median effort: {pstats['median_commits_to_core']:.0f} commits")
                self.logger.info(f"  Min weeks to core: {pstats['min_weeks_to_core']}")

def main():
    """Main entry point."""
    # Validate input file
    if not Path(CONFIG['input_activity']).exists():
        print(f"Error: Activity file not found: {CONFIG['input_activity']}")
        sys.exit(1)
    
    analyzer = TransitionAnalyzer(CONFIG)
    analyzer.process_all_contributors()

if __name__ == "__main__":
    main()


