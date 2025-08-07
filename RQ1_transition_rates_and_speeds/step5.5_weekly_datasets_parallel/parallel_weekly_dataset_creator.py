"""
Parallel version of dataset creator using multiprocessing.
Processes multiple projects simultaneously to speed up creation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from tqdm import tqdm
import warnings
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import sys

warnings.filterwarnings('ignore')

def process_project_timeline(args):
    """Process a single project for timeline - runs in parallel."""
    project_name, project_type, project_commits = args
    
    if len(project_commits) == 0:
        return []
    
    project_commits = project_commits.sort_values('commit_date')
    
    # Get all weeks for this project
    min_week = project_commits['week_date'].min()
    max_week = project_commits['week_date'].max()
    week_range = pd.date_range(start=min_week, end=max_week, freq='W-MON')
    
    timeline_records = []
    
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
        
        # Find core contributors (corrected logic)
        core_mask = cumsum >= threshold_80
        if core_mask.any():
            first_over_80_idx = np.where(core_mask)[0][0]
            core_contributors = list(contributor_counts.iloc[:first_over_80_idx + 1].index)
        else:
            core_contributors = list(contributor_counts.index)
        
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
            'core_contributors_emails': json.dumps(core_contributors)
        }
        
        timeline_records.append(record)
    
    return timeline_records

def process_project_activity(args):
    """Process a single project for activity - runs in parallel."""
    project_name, project_type, project_commits, core_lookup = args
    
    if len(project_commits) == 0:
        return []
    
    project_commits = project_commits.sort_values('commit_date')
    
    # Get project week range
    min_week = project_commits['week_date'].min()
    max_week = project_commits['week_date'].max()
    week_range = pd.date_range(start=min_week, end=max_week, freq='W-MON')
    
    # Get unique contributors in this project
    contributors = project_commits['author_email'].unique()
    
    activity_records = []
    
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
            
            activity_records.append(record)
    
    return activity_records

class ParallelWeeklyDatasetCreator:
    """Parallel version of the dataset creator."""
    
    def __init__(self, commits_csv_path, output_dir, n_cores=None):
        """
        Initialize with optional core count.
        
        Parameters:
        -----------
        n_cores : int, optional
            Number of CPU cores to use. Default is all available minus 1.
        """
        self.commits_csv_path = Path(commits_csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set number of cores
        if n_cores is None:
            self.n_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
        else:
            self.n_cores = n_cores
        
        # Create subdirectories
        self.datasets_dir = self.output_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        
        print("="*80)
        print("PARALLEL DATASET CREATOR")
        print("="*80)
        print(f"Using {self.n_cores} CPU cores")
        print(f"Input: {self.commits_csv_path}")
        print(f"Output: {self.output_dir}")
        print("-"*80)
    
    def load_commits_data(self):
        """Load and preprocess the commits dataset."""
        print("\n📂 LOADING COMMITS DATA...")
        
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
        
        # Clean email addresses
        print("   Cleaning contributor emails...")
        self.commits_df['author_email'] = self.commits_df['author_email'].str.lower().str.strip()
        
        # Get unique projects
        self.projects = self.commits_df.groupby(['project_name', 'project_type']).size().reset_index()[['project_name', 'project_type']]
        print(f"   Found {len(self.projects)} unique projects")
        
        return self.commits_df
    
    def create_project_core_timeline_parallel(self):
        """Create timeline using parallel processing."""
        print("\n📊 CREATING DATASET 1: PROJECT CORE TIMELINE (PARALLEL)")
        print("="*80)
        
        # Prepare project data for parallel processing
        project_data = []
        for _, row in self.projects.iterrows():
            project_commits = self.commits_df[
                self.commits_df['project_name'] == row['project_name']
            ]
            project_data.append((row['project_name'], row['project_type'], project_commits))
        
        # Process in parallel
        print(f"Processing {len(project_data)} projects on {self.n_cores} cores...")
        
        with mp.Pool(processes=self.n_cores) as pool:
            results = list(tqdm(
                pool.imap(process_project_timeline, project_data),
                total=len(project_data),
                desc="Processing projects"
            ))
        
        # Combine results
        all_timeline_records = []
        for project_records in results:
            all_timeline_records.extend(project_records)
        
        # Create DataFrame
        timeline_df = pd.DataFrame(all_timeline_records)
        
        # Save to CSV
        output_path = self.datasets_dir / "project_core_timeline_weekly.csv"
        timeline_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Created project_core_timeline_weekly.csv")
        print(f"   Rows: {len(timeline_df):,}")
        
        return timeline_df
    
    def create_contributor_activity_weekly_parallel(self, timeline_df):
        """Create activity records using parallel processing."""
        print("\n📊 CREATING DATASET 2: CONTRIBUTOR ACTIVITY (PARALLEL)")
        print("="*80)
        
        # Create core lookup
        print("   Preparing core status lookup...")
        core_lookup = {}
        for _, row in timeline_df.iterrows():
            key = (row['project_name'], row['week_date'])
            core_lookup[key] = json.loads(row['core_contributors_emails'])
        
        # Prepare project data for parallel processing
        project_data = []
        for _, row in self.projects.iterrows():
            project_commits = self.commits_df[
                self.commits_df['project_name'] == row['project_name']
            ]
            # Filter core_lookup for this project
            project_core_lookup = {k: v for k, v in core_lookup.items() if k[0] == row['project_name']}
            project_data.append((row['project_name'], row['project_type'], project_commits, project_core_lookup))
        
        # Process in parallel
        print(f"Processing {len(project_data)} projects on {self.n_cores} cores...")
        
        with mp.Pool(processes=self.n_cores) as pool:
            results = list(tqdm(
                pool.imap(process_project_activity, project_data),
                total=len(project_data),
                desc="Processing projects"
            ))
        
        # Combine results
        all_activity_records = []
        for project_records in results:
            all_activity_records.extend(project_records)
        
        # Create DataFrame
        activity_df = pd.DataFrame(all_activity_records)
        
        # Save to CSV
        output_path = self.datasets_dir / "contributor_activity_weekly.csv"
        activity_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Created contributor_activity_weekly.csv")
        print(f"   Rows: {len(activity_df):,}")
        
        return activity_df
    
    def run_complete_pipeline(self):
        """Run the complete parallel pipeline."""
        start_time = datetime.now()
        
        try:
            # Load commits data
            self.load_commits_data()
            
            print("\n🚀 CREATING DATASETS IN PARALLEL")
            print("="*80)
            
            # Dataset 1: Project Core Timeline (Parallel)
            timeline_df = self.create_project_core_timeline_parallel()
            
            # Dataset 2: Contributor Activity (Parallel)
            activity_df = self.create_contributor_activity_weekly_parallel(timeline_df)
            
            # Note: Datasets 3 and 4 can use the original non-parallel methods
            # as they're much faster (just aggregating existing data)
            print("\nUse the original creator for datasets 3 and 4 (they're fast)")
            
            # Calculate total time
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            print("\n" + "="*80)
            print("✅ PARALLEL PROCESSING COMPLETE!")
            print(f"Total execution time: {total_time/60:.1f} minutes")
            print(f"Speedup: ~{self.n_cores}x faster")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Configuration
    COMMITS_CSV = "../data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv"
    OUTPUT_DIR = "weekly_datasets_parallel"
    N_CORES = None  # Use all available cores minus 1
    
    # Create parallel dataset creator
    creator = ParallelWeeklyDatasetCreator(COMMITS_CSV, OUTPUT_DIR, n_cores=N_CORES)
    
    # Run parallel pipeline
    success = creator.run_complete_pipeline() 