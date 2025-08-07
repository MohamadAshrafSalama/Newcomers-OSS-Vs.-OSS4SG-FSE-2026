"""
Comprehensive Dataset Creator for RQ1: Newcomer to Core Analysis
Creates 4 interconnected datasets tracking contributor journeys with weekly granularity
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from tqdm import tqdm
import warnings
import hashlib
from collections import defaultdict
import gc
import sys

warnings.filterwarnings('ignore')

class WeeklyDatasetCreator:
    """
    Creates comprehensive weekly datasets for tracking contributor journeys to core status.
    """
    
    def __init__(self, commits_csv_path, output_dir):
        """
        Initialize the dataset creator.
        
        Parameters:
        -----------
        commits_csv_path : str
            Path to master_commits_dataset.csv
        output_dir : str
            Directory for output datasets
        """
        self.commits_csv_path = Path(commits_csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.datasets_dir = self.output_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        self.validation_dir = self.output_dir / "validation"
        self.validation_dir.mkdir(exist_ok=True)
        
        print("="*80)
        print("COMPREHENSIVE DATASET CREATOR FOR RQ1 ANALYSIS")
        print("="*80)
        print(f"Input: {self.commits_csv_path}")
        print(f"Output: {self.output_dir}")
        print("-"*80)
        
    def load_commits_data(self):
        """
        Load and preprocess the commits dataset with progress bar.
        """
        print("\n📂 LOADING COMMITS DATA...")
        
        # Get file size for progress bar
        file_size = self.commits_csv_path.stat().st_size / (1024**2)  # MB
        print(f"   File size: {file_size:.1f} MB")
        
        # Load with progress bar
        print("   Loading CSV...")
        self.commits_df = pd.read_csv(
            self.commits_csv_path, 
            low_memory=False,
            dtype={
                'project_name': str,
                'project_type': str,
                'commit_hash': str,
                'author_email': str,
                'author_name': str
            }
        )
        
        print(f"   Loaded {len(self.commits_df):,} commits")
        
        # Convert dates
        print("   Converting dates...")
        self.commits_df['commit_date'] = pd.to_datetime(self.commits_df['commit_date'], utc=True)
        
        # Add week information
        print("   Adding week information...")
        self.commits_df['week_date'] = self.commits_df['commit_date'].dt.to_period('W').dt.start_time
        self.commits_df['week_str'] = self.commits_df['week_date'].dt.strftime('%Y-%m-%d')
        
        # Clean email addresses (lowercase, strip)
        print("   Cleaning contributor emails...")
        self.commits_df['author_email'] = self.commits_df['author_email'].str.lower().str.strip()
        
        # Get unique projects
        self.projects = self.commits_df.groupby(['project_name', 'project_type']).size().reset_index()[['project_name', 'project_type']]
        print(f"   Found {len(self.projects)} unique projects")
        
        # Basic stats
        print("\n   📊 Basic Statistics:")
        print(f"      Total commits: {len(self.commits_df):,}")
        print(f"      Unique contributors: {self.commits_df['author_email'].nunique():,}")
        print(f"      Date range: {self.commits_df['commit_date'].min()} to {self.commits_df['commit_date'].max()}")
        
        return self.commits_df
    
    def create_project_core_timeline(self):
        """
        DATASET 1: Create weekly core contributor timeline for each project.
        Tracks who is core at each point in time using 80% rule.
        """
        print("\n" + "="*80)
        print("📊 CREATING DATASET 1: PROJECT CORE TIMELINE (WEEKLY)")
        print("="*80)
        
        all_timeline_records = []
        
        # Process each project
        for idx, project_row in tqdm(self.projects.iterrows(), 
                                     total=len(self.projects), 
                                     desc="Processing projects"):
            
            project_name = project_row['project_name']
            project_type = project_row['project_type']
            
            # Get project commits
            project_commits = self.commits_df[
                self.commits_df['project_name'] == project_name
            ].sort_values('commit_date')
            
            if len(project_commits) == 0:
                continue
            
            # Get all weeks for this project
            min_week = project_commits['week_date'].min()
            max_week = project_commits['week_date'].max()
            
            # Create week range
            week_range = pd.date_range(start=min_week, end=max_week, freq='W-MON')
            
            # Track cumulative commits per contributor
            cumulative_commits = defaultdict(int)
            
            for week_idx, current_week in enumerate(week_range):
                # Get all commits up to and including this week
                commits_to_date = project_commits[
                    project_commits['week_date'] <= current_week
                ]
                
                # Count commits per contributor
                contributor_counts = commits_to_date.groupby('author_email').size().sort_values(ascending=False)
                
                if len(contributor_counts) == 0:
                    continue
                
                # Calculate 80% threshold
                total_commits = contributor_counts.sum()
                cumsum = contributor_counts.cumsum()
                threshold_80 = total_commits * 0.8
                
                # Find core contributors (those needed for 80% of commits)
                core_mask = cumsum <= threshold_80
                
                # Include the contributor that pushes us over 80%
                if core_mask.any():
                    last_core_idx = core_mask.sum()
                    if last_core_idx < len(contributor_counts):
                        core_contributors = list(contributor_counts.iloc[:last_core_idx + 1].index)
                    else:
                        core_contributors = list(contributor_counts.index)
                else:
                    # If no one reaches 80%, take the top contributor
                    core_contributors = [contributor_counts.index[0]]
                
                # Create record for this week
                record = {
                    'project_name': project_name,
                    'project_type': project_type,
                    'week_date': current_week,
                    'week_number': week_idx + 1,
                    'total_commits_to_date': total_commits,
                    'total_contributors_to_date': len(contributor_counts),
                    'core_threshold_commits': int(threshold_80),
                    'core_contributors_count': len(core_contributors),
                    'core_contributors_emails': json.dumps(core_contributors)  # Store as JSON string
                }
                
                all_timeline_records.append(record)
        
        # Create DataFrame
        timeline_df = pd.DataFrame(all_timeline_records)
        
        # Save to CSV
        output_path = self.datasets_dir / "project_core_timeline_weekly.csv"
        timeline_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Created project_core_timeline_weekly.csv")
        print(f"   Rows: {len(timeline_df):,}")
        print(f"   Projects: {timeline_df['project_name'].nunique()}")
        print(f"   Saved to: {output_path}")
        
        # Validation
        self._validate_dataset_1(timeline_df)
        
        return timeline_df
    
    def create_contributor_activity_weekly(self, timeline_df):
        """
        DATASET 2: Create weekly activity records for each contributor.
        Tracks detailed activity metrics week by week.
        """
        print("\n" + "="*80)
        print("📊 CREATING DATASET 2: CONTRIBUTOR ACTIVITY (WEEKLY)")
        print("="*80)
        
        all_activity_records = []
        
        # Load core timeline for reference
        print("   Preparing core status lookup...")
        core_lookup = {}
        for _, row in timeline_df.iterrows():
            key = (row['project_name'], row['week_date'])
            core_lookup[key] = json.loads(row['core_contributors_emails'])
        
        # Process each project
        for idx, project_row in tqdm(self.projects.iterrows(), 
                                     total=len(self.projects), 
                                     desc="Processing projects"):
            
            project_name = project_row['project_name']
            project_type = project_row['project_type']
            
            # Get project commits
            project_commits = self.commits_df[
                self.commits_df['project_name'] == project_name
            ].sort_values('commit_date')
            
            if len(project_commits) == 0:
                continue
            
            # Get project week range
            min_week = project_commits['week_date'].min()
            max_week = project_commits['week_date'].max()
            week_range = pd.date_range(start=min_week, end=max_week, freq='W-MON')
            
            # Get unique contributors in this project
            contributors = project_commits['author_email'].unique()
            
            # Process each contributor
            for contributor in contributors:
                contributor_commits = project_commits[
                    project_commits['author_email'] == contributor
                ]
                
                first_commit_week = contributor_commits['week_date'].min()
                cumulative_commits = 0
                cumulative_lines = 0
                
                # Process each week
                for week_idx, current_week in enumerate(week_range):
                    # Skip weeks before contributor's first commit
                    if current_week < first_commit_week:
                        continue
                    
                    # Get commits for this week
                    week_commits = contributor_commits[
                        contributor_commits['week_date'] == current_week
                    ]
                    
                    # Calculate metrics for this week
                    commits_this_week = len(week_commits)
                    
                    if commits_this_week > 0:
                        commit_hashes = week_commits['commit_hash'].tolist()
                        lines_added = week_commits['total_insertions'].sum()
                        lines_deleted = week_commits['total_deletions'].sum()
                        files_modified = week_commits['files_modified_count'].sum()
                    else:
                        commit_hashes = []
                        lines_added = 0
                        lines_deleted = 0
                        files_modified = 0
                    
                    # Update cumulative metrics
                    cumulative_commits += commits_this_week
                    cumulative_lines += lines_added + lines_deleted
                    
                    # Get total project commits to date
                    project_commits_to_date = len(project_commits[
                        project_commits['week_date'] <= current_week
                    ])
                    
                    # Calculate contribution percentage
                    contribution_percentage = (cumulative_commits / project_commits_to_date * 100) if project_commits_to_date > 0 else 0
                    
                    # Check if core this week
                    core_list = core_lookup.get((project_name, current_week), [])
                    is_core_this_week = contributor in core_list
                    
                    # Calculate rank this week (by cumulative commits)
                    week_contributors_cumulative = project_commits[
                        project_commits['week_date'] <= current_week
                    ].groupby('author_email').size().sort_values(ascending=False)
                    
                    if contributor in week_contributors_cumulative.index:
                        rank = week_contributors_cumulative.index.get_loc(contributor) + 1
                    else:
                        rank = len(week_contributors_cumulative) + 1
                    
                    # Calculate weeks since first commit
                    weeks_since_first = (current_week - first_commit_week).days // 7
                    
                    # Create activity record
                    record = {
                        'project_name': project_name,
                        'project_type': project_type,
                        'contributor_email': contributor,
                        'week_date': current_week,
                        'week_number': week_idx + 1,
                        'weeks_since_first_commit': weeks_since_first,
                        'commits_this_week': commits_this_week,
                        'commit_hashes': json.dumps(commit_hashes),
                        'lines_added_this_week': int(lines_added),
                        'lines_deleted_this_week': int(lines_deleted),
                        'files_modified_this_week': int(files_modified),
                        'cumulative_commits': cumulative_commits,
                        'cumulative_lines_changed': cumulative_lines,
                        'project_commits_to_date': project_commits_to_date,
                        'contribution_percentage': round(contribution_percentage, 2),
                        'is_core_this_week': is_core_this_week,
                        'rank_this_week': rank
                    }
                    
                    all_activity_records.append(record)
        
        # Create DataFrame
        activity_df = pd.DataFrame(all_activity_records)
        
        # Save to CSV
        output_path = self.datasets_dir / "contributor_activity_weekly.csv"
        activity_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Created contributor_activity_weekly.csv")
        print(f"   Rows: {len(activity_df):,}")
        print(f"   Contributors: {activity_df['contributor_email'].nunique():,}")
        print(f"   Saved to: {output_path}")
        
        # Validation
        self._validate_dataset_2(activity_df)
        
        return activity_df
    
    def create_contributor_transitions(self, activity_df):
        """
        DATASET 3: Create summary of each contributor's journey.
        One row per contributor with their complete journey summary.
        """
        print("\n" + "="*80)
        print("📊 CREATING DATASET 3: CONTRIBUTOR TRANSITIONS")
        print("="*80)
        
        all_transition_records = []
        
        # Group by project and contributor
        grouped = activity_df.groupby(['project_name', 'project_type', 'contributor_email'])
        
        for (project_name, project_type, contributor), group in tqdm(grouped, 
                                                                     desc="Processing contributors"):
            
            # Sort by week
            group = group.sort_values('week_date')
            
            # Basic information
            first_week = group['week_date'].min()
            last_week = group['week_date'].max()
            
            # Activity metrics
            total_commits = group['cumulative_commits'].max()
            total_active_weeks = (group['commits_this_week'] > 0).sum()
            
            # Core status
            ever_core = group['is_core_this_week'].any()
            
            if ever_core:
                # Find first week became core
                first_core_idx = group['is_core_this_week'].idxmax()
                first_core_row = group.loc[first_core_idx]
                first_core_week = first_core_row['week_date']
                weeks_to_core = first_core_row['weeks_since_first_commit']
                commits_at_core = first_core_row['cumulative_commits']
                
                # Check if maintained core status
                last_status = group.iloc[-1]['is_core_this_week']
                maintained_core = last_status
                
                # Count total weeks as core
                total_weeks_as_core = group['is_core_this_week'].sum()
            else:
                first_core_week = None
                weeks_to_core = None
                commits_at_core = None
                maintained_core = False
                total_weeks_as_core = 0
            
            # Maximum contribution percentage
            max_contribution_pct = group['contribution_percentage'].max()
            
            # Determine final status
            last_activity = group.iloc[-1]
            if last_activity['is_core_this_week']:
                final_status = 'core'
            elif (last_week - group[group['commits_this_week'] > 0]['week_date'].max()).days < 90:
                final_status = 'active'
            else:
                final_status = 'inactive'
            
            # Create transition record
            record = {
                'project_name': project_name,
                'project_type': project_type,
                'contributor_email': contributor,
                'first_commit_date': first_week,
                'last_commit_date': last_week,
                'total_commits': int(total_commits),
                'total_active_weeks': int(total_active_weeks),
                'became_core': ever_core,
                'first_core_week_date': first_core_week,
                'weeks_to_core': int(weeks_to_core) if weeks_to_core is not None else None,
                'commits_at_core': int(commits_at_core) if commits_at_core is not None else None,
                'maintained_core_status': maintained_core,
                'total_weeks_as_core': int(total_weeks_as_core),
                'max_contribution_percentage': round(max_contribution_pct, 2),
                'final_status': final_status
            }
            
            all_transition_records.append(record)
        
        # Create DataFrame
        transitions_df = pd.DataFrame(all_transition_records)
        
        # Save to CSV
        output_path = self.datasets_dir / "contributor_transitions.csv"
        transitions_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Created contributor_transitions.csv")
        print(f"   Rows: {len(transitions_df):,}")
        print(f"   Core contributors: {transitions_df['became_core'].sum():,}")
        print(f"   Core ratio: {transitions_df['became_core'].mean():.1%}")
        print(f"   Saved to: {output_path}")
        
        # Validation
        self._validate_dataset_3(transitions_df)
        
        return transitions_df
    
    def create_weekly_commit_aggregates(self, activity_df):
        """
        DATASET 4: Create weekly commit statistics for ML features.
        Detailed commit patterns for each contributor each week.
        """
        print("\n" + "="*80)
        print("📊 CREATING DATASET 4: WEEKLY COMMIT AGGREGATES")
        print("="*80)
        
        all_aggregate_records = []
        
        # Process each project
        for idx, project_row in tqdm(self.projects.iterrows(), 
                                     total=len(self.projects), 
                                     desc="Processing projects"):
            
            project_name = project_row['project_name']
            
            # Get project commits
            project_commits = self.commits_df[
                self.commits_df['project_name'] == project_name
            ]
            
            if len(project_commits) == 0:
                continue
            
            # Get activity for this project
            project_activity = activity_df[
                activity_df['project_name'] == project_name
            ]
            
            # Process each contributor-week combination
            for _, activity_row in project_activity.iterrows():
                
                # Skip weeks with no commits
                if activity_row['commits_this_week'] == 0:
                    continue
                
                # Get the actual commits for this week
                commit_hashes = json.loads(activity_row['commit_hashes'])
                
                if len(commit_hashes) == 0:
                    continue
                
                week_commits = project_commits[
                    project_commits['commit_hash'].isin(commit_hashes)
                ]
                
                if len(week_commits) == 0:
                    continue
                
                # Calculate aggregate metrics
                avg_commit_size = week_commits['total_lines_changed'].mean()
                max_commit_size = week_commits['total_lines_changed'].max()
                
                # Commit frequency (commits per day in the week)
                commit_frequency = len(week_commits) / 7.0
                
                # Files per commit
                avg_files_per_commit = week_commits['files_modified_count'].mean()
                
                # Weekend vs weekday
                weekend_commits = week_commits['commit_is_weekend'].sum()
                weekday_commits = len(week_commits) - weekend_commits
                
                # Time of day analysis
                morning_commits = week_commits[week_commits['commit_hour'].between(6, 11)].shape[0]
                afternoon_commits = week_commits[week_commits['commit_hour'].between(12, 17)].shape[0]
                evening_commits = week_commits[week_commits['commit_hour'].between(18, 23)].shape[0]
                night_commits = week_commits[week_commits['commit_hour'].isin([0,1,2,3,4,5])].shape[0]
                
                # Message length
                avg_message_length = week_commits['message_length_chars'].mean()
                
                # Code churn
                code_churn = (week_commits['total_insertions'].sum() + 
                             week_commits['total_deletions'].sum()) / 2
                
                # Create aggregate record
                record = {
                    'project_name': project_name,
                    'contributor_email': activity_row['contributor_email'],
                    'week_date': activity_row['week_date'],
                    'avg_commit_size': round(avg_commit_size, 1),
                    'max_commit_size': int(max_commit_size),
                    'commit_frequency': round(commit_frequency, 2),
                    'avg_files_per_commit': round(avg_files_per_commit, 1),
                    'weekend_commits': int(weekend_commits),
                    'weekday_commits': int(weekday_commits),
                    'morning_commits': int(morning_commits),
                    'afternoon_commits': int(afternoon_commits),
                    'evening_commits': int(evening_commits),
                    'night_commits': int(night_commits),
                    'avg_message_length': round(avg_message_length, 1),
                    'code_churn': round(code_churn, 1)
                }
                
                all_aggregate_records.append(record)
        
        # Create DataFrame
        aggregates_df = pd.DataFrame(all_aggregate_records)
        
        # Save to CSV
        output_path = self.datasets_dir / "weekly_commit_aggregates.csv"
        aggregates_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Created weekly_commit_aggregates.csv")
        print(f"   Rows: {len(aggregates_df):,}")
        print(f"   Saved to: {output_path}")
        
        # Validation
        self._validate_dataset_4(aggregates_df)
        
        return aggregates_df
    
    def _validate_dataset_1(self, df):
        """Validate Dataset 1: Project Core Timeline."""
        print("\n   🔍 Validating Dataset 1...")
        
        issues = []
        
        # Check for nulls
        if df.isnull().any().any():
            issues.append("Contains null values")
        
        # Check week progression
        for project in df['project_name'].unique():
            project_df = df[df['project_name'] == project].sort_values('week_number')
            if not (project_df['week_number'].diff()[1:] == 1).all():
                issues.append(f"Non-sequential weeks in {project}")
                break
        
        # Check core contributor counts
        if (df['core_contributors_count'] == 0).any():
            issues.append("Some weeks have 0 core contributors")
        
        if issues:
            print(f"   ⚠️  Validation issues: {issues}")
        else:
            print("   ✅ Validation passed!")
    
    def _validate_dataset_2(self, df):
        """Validate Dataset 2: Contributor Activity."""
        print("\n   🔍 Validating Dataset 2...")
        
        issues = []
        
        # Check cumulative commits are non-decreasing
        for contributor in df['contributor_email'].unique()[:100]:  # Sample check
            contrib_df = df[df['contributor_email'] == contributor].sort_values('week_date')
            if not contrib_df['cumulative_commits'].is_monotonic_increasing:
                issues.append(f"Non-monotonic cumulative commits for {contributor}")
                break
        
        # Check contribution percentages
        if (df['contribution_percentage'] > 100).any():
            issues.append("Contribution percentage > 100%")
        
        if (df['contribution_percentage'] < 0).any():
            issues.append("Negative contribution percentage")
        
        if issues:
            print(f"   ⚠️  Validation issues: {issues}")
        else:
            print("   ✅ Validation passed!")
    
    def _validate_dataset_3(self, df):
        """Validate Dataset 3: Contributor Transitions."""
        print("\n   🔍 Validating Dataset 3...")
        
        issues = []
        
        # Check core logic
        core_df = df[df['became_core'] == True]
        if (core_df['weeks_to_core'].isna()).any():
            issues.append("Core contributors missing weeks_to_core")
        
        if (core_df['commits_at_core'].isna()).any():
            issues.append("Core contributors missing commits_at_core")
        
        # Check non-core logic
        non_core_df = df[df['became_core'] == False]
        if non_core_df['weeks_to_core'].notna().any():
            issues.append("Non-core contributors have weeks_to_core values")
        
        if issues:
            print(f"   ⚠️  Validation issues: {issues}")
        else:
            print("   ✅ Validation passed!")
    
    def _validate_dataset_4(self, df):
        """Validate Dataset 4: Weekly Aggregates."""
        print("\n   🔍 Validating Dataset 4...")
        
        issues = []
        
        # Check time of day commits sum
        time_cols = ['morning_commits', 'afternoon_commits', 'evening_commits', 'night_commits']
        df['total_time_commits'] = df[time_cols].sum(axis=1)
        df['total_weekend_weekday'] = df['weekend_commits'] + df['weekday_commits']
        
        # These should be equal (allowing for small differences due to data issues)
        if not np.allclose(df['total_time_commits'], df['total_weekend_weekday'], rtol=0.1):
            issues.append("Time-of-day commits don't sum correctly")
        
        # Check commit frequency
        if (df['commit_frequency'] > 7).any():
            issues.append("Commit frequency > 7 per week")
        
        if issues:
            print(f"   ⚠️  Validation issues: {issues}")
        else:
            print("   ✅ Validation passed!")
    
    def generate_summary_report(self, timeline_df, activity_df, transitions_df, aggregates_df):
        """
        Generate a comprehensive summary report of all datasets.
        """
        print("\n" + "="*80)
        print("📋 GENERATING SUMMARY REPORT")
        print("="*80)
        
        summary = {
            'dataset_creation_timestamp': datetime.now().isoformat(),
            'input_file': str(self.commits_csv_path),
            'output_directory': str(self.output_dir),
            
            'dataset_1_project_core_timeline': {
                'rows': len(timeline_df),
                'projects': timeline_df['project_name'].nunique(),
                'weeks_covered': timeline_df['week_number'].max(),
                'avg_core_contributors': timeline_df['core_contributors_count'].mean(),
                'file_size_mb': (self.datasets_dir / "project_core_timeline_weekly.csv").stat().st_size / (1024**2)
            },
            
            'dataset_2_contributor_activity': {
                'rows': len(activity_df),
                'contributors': activity_df['contributor_email'].nunique(),
                'projects': activity_df['project_name'].nunique(),
                'core_weeks': activity_df['is_core_this_week'].sum(),
                'avg_contribution_pct': activity_df['contribution_percentage'].mean(),
                'file_size_mb': (self.datasets_dir / "contributor_activity_weekly.csv").stat().st_size / (1024**2)
            },
            
            'dataset_3_transitions': {
                'rows': len(transitions_df),
                'became_core': transitions_df['became_core'].sum(),
                'core_ratio': transitions_df['became_core'].mean(),
                'median_weeks_to_core': transitions_df[transitions_df['became_core']]['weeks_to_core'].median(),
                'median_commits_at_core': transitions_df[transitions_df['became_core']]['commits_at_core'].median(),
                'file_size_mb': (self.datasets_dir / "contributor_transitions.csv").stat().st_size / (1024**2)
            },
            
            'dataset_4_aggregates': {
                'rows': len(aggregates_df),
                'avg_commit_size': aggregates_df['avg_commit_size'].mean(),
                'weekend_ratio': aggregates_df['weekend_commits'].sum() / (aggregates_df['weekend_commits'].sum() + aggregates_df['weekday_commits'].sum()),
                'file_size_mb': (self.datasets_dir / "weekly_commit_aggregates.csv").stat().st_size / (1024**2)
            },
            
            'by_project_type': {
                'OSS': {
                    'projects': transitions_df[transitions_df['project_type'] == 'OSS']['project_name'].nunique(),
                    'contributors': len(transitions_df[transitions_df['project_type'] == 'OSS']),
                    'core_contributors': transitions_df[(transitions_df['project_type'] == 'OSS') & (transitions_df['became_core'])].shape[0],
                    'core_ratio': transitions_df[transitions_df['project_type'] == 'OSS']['became_core'].mean()
                },
                'OSS4SG': {
                    'projects': transitions_df[transitions_df['project_type'] == 'OSS4SG']['project_name'].nunique(),
                    'contributors': len(transitions_df[transitions_df['project_type'] == 'OSS4SG']),
                    'core_contributors': transitions_df[(transitions_df['project_type'] == 'OSS4SG') & (transitions_df['became_core'])].shape[0],
                    'core_ratio': transitions_df[transitions_df['project_type'] == 'OSS4SG']['became_core'].mean()
                }
            }
        }
        
        # Save summary
        summary_path = self.output_dir / "dataset_creation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        print("\n📊 SUMMARY STATISTICS:")
        print("-" * 40)
        print(f"Total contributors analyzed: {summary['dataset_3_transitions']['rows']:,}")
        print(f"Became core: {summary['dataset_3_transitions']['became_core']:,} ({summary['dataset_3_transitions']['core_ratio']:.1%})")
        print(f"\nOSS Projects:")
        print(f"  Contributors: {summary['by_project_type']['OSS']['contributors']:,}")
        print(f"  Core ratio: {summary['by_project_type']['OSS']['core_ratio']:.1%}")
        print(f"\nOSS4SG Projects:")
        print(f"  Contributors: {summary['by_project_type']['OSS4SG']['contributors']:,}")
        print(f"  Core ratio: {summary['by_project_type']['OSS4SG']['core_ratio']:.1%}")
        
        print(f"\n📁 Summary saved to: {summary_path}")
        
        return summary
    
    def run_complete_pipeline(self):
        """
        Run the complete dataset creation pipeline.
        """
        start_time = datetime.now()
        
        try:
            # Load commits data
            self.load_commits_data()
            
            # Create datasets
            print("\n" + "="*80)
            print("🚀 CREATING ALL DATASETS")
            print("="*80)
            
            # Dataset 1: Project Core Timeline
            timeline_df = self.create_project_core_timeline()
            
            # Dataset 2: Contributor Activity
            activity_df = self.create_contributor_activity_weekly(timeline_df)
            
            # Dataset 3: Contributor Transitions
            transitions_df = self.create_contributor_transitions(activity_df)
            
            # Dataset 4: Weekly Aggregates
            aggregates_df = self.create_weekly_commit_aggregates(activity_df)
            
            # Generate summary report
            summary = self.generate_summary_report(
                timeline_df, activity_df, transitions_df, aggregates_df
            )
            
            # Calculate total time
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            print("\n" + "="*80)
            print("✅ DATASET CREATION COMPLETE!")
            print("="*80)
            print(f"Total execution time: {total_time/60:.1f} minutes")
            print(f"\nCreated 4 datasets in: {self.datasets_dir}")
            print("  1. project_core_timeline_weekly.csv")
            print("  2. contributor_activity_weekly.csv")
            print("  3. contributor_transitions.csv")
            print("  4. weekly_commit_aggregates.csv")
            print("\nNext steps:")
            print("  - Use these datasets for survival analysis")
            print("  - Train ML models on contributor patterns")
            print("  - Analyze OSS vs OSS4SG differences")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


# Main execution
if __name__ == "__main__":
    # Configuration - Updated paths for step5 subdirectory
    COMMITS_CSV = "../data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv"
    OUTPUT_DIR = "weekly_datasets"
    
    # Create dataset creator
    creator = WeeklyDatasetCreator(COMMITS_CSV, OUTPUT_DIR)
    
    # Run pipeline
    success = creator.run_complete_pipeline()
    
    if not success:
        sys.exit(1) 