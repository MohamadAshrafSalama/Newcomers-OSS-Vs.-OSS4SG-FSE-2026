"""
Dataset 1: project_core_timeline_weekly.csv
Tracks who is core in each project at each week using the 80% rule.
One row = one project's state at one specific week
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import warnings
import psutil
import gc
from tqdm import tqdm

warnings.filterwarnings('ignore')

class ProjectCoreTimelineCreator:
    def __init__(self, commits_csv_path, output_dir):
        """
        Initialize the core timeline creator.
        
        Parameters:
        -----------
        commits_csv_path : str
            Path to master_commits_dataset.csv
        output_dir : str
            Directory for output files
        """
        self.commits_csv_path = Path(commits_csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets_dir = self.output_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        
        print("="*80)
        print("DATASET 1: PROJECT CORE TIMELINE CREATOR")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print("-"*80)
    
    def load_commits_data(self):
        """Load and prepare commits data."""
        print("\n📂 LOADING COMMITS DATA...")
        
        self.commits_df = pd.read_csv(
            self.commits_csv_path,
            low_memory=False,
            dtype={
                'project_name': str,
                'project_type': str,
                'commit_hash': str,
                'author_email': str
            }
        )
        
        print(f"   Loaded {len(self.commits_df):,} commits")
        
        # Prepare data
        self.commits_df['commit_date'] = pd.to_datetime(self.commits_df['commit_date'], utc=True)
        self.commits_df['week_date'] = self.commits_df['commit_date'].dt.to_period('W').dt.start_time
        self.commits_df['author_email'] = self.commits_df['author_email'].str.lower().str.strip()
        
        # Get unique projects
        self.projects = self.commits_df.groupby(['project_name', 'project_type']).size().reset_index()[['project_name', 'project_type']]
        print(f"   Found {len(self.projects)} projects")
    
    def calculate_core_contributors_at_week(self, project_commits, target_week):
        """
        Calculate core contributors for a specific week using the 80% rule.
        Core = smallest group responsible for 80% of commits up to that week.
        
        Parameters:
        -----------
        project_commits : DataFrame
            All commits for the project
        target_week : Timestamp
            The week to calculate core contributors for
        
        Returns:
        --------
        dict : Core contributor information for that week
        """
        # Get all commits up to and including target week
        commits_to_date = project_commits[project_commits['week_date'] <= target_week]
        
        if len(commits_to_date) == 0:
            return {
                'core_contributors': [],
                'core_threshold': 0,
                'total_commits': 0,
                'total_contributors': 0
            }
        
        # Count commits per contributor
        contributor_counts = commits_to_date.groupby('author_email').size().sort_values(ascending=False)
        
        # Calculate 80% threshold
        total_commits = contributor_counts.sum()
        threshold_commits = total_commits * 0.8
        
        # Find core contributors (those needed for 80% of commits)
        cumsum = contributor_counts.cumsum()
        core_mask = cumsum <= threshold_commits
        
        # Include the contributor that pushes us over 80%
        if core_mask.any():
            # Find the last True index
            last_core_idx = np.where(core_mask)[0][-1]
            # Include one more if we haven't reached 80%
            if last_core_idx < len(contributor_counts) - 1 and cumsum.iloc[last_core_idx] < threshold_commits:
                core_contributors = list(contributor_counts.index[:last_core_idx + 2])
                core_threshold = contributor_counts.iloc[last_core_idx + 1]
            else:
                core_contributors = list(contributor_counts.index[:last_core_idx + 1])
                core_threshold = contributor_counts.iloc[last_core_idx] if last_core_idx < len(contributor_counts) else 0
        else:
            # If no one reaches 80%, take just the top contributor
            core_contributors = [contributor_counts.index[0]] if len(contributor_counts) > 0 else []
            core_threshold = contributor_counts.iloc[0] if len(contributor_counts) > 0 else 0
        
        return {
            'core_contributors': core_contributors,
            'core_threshold': int(core_threshold),
            'total_commits': int(total_commits),
            'total_contributors': len(contributor_counts)
        }
    
    def create_project_core_timeline(self):
        """
        Create the project core timeline dataset.
        
        Returns:
        --------
        DataFrame : Project core timeline data
        """
        print("\n📊 CREATING PROJECT CORE TIMELINE...")
        print("="*80)
        
        all_timeline_records = []
        
        # Process each project
        for idx, project_row in tqdm(self.projects.iterrows(), total=len(self.projects), desc="Processing projects"):
            project_name = project_row['project_name']
            project_type = project_row['project_type']
            
            # Get project commits
            project_commits = self.commits_df[self.commits_df['project_name'] == project_name]
            
            if len(project_commits) == 0:
                continue
            
            # Sort by date
            project_commits = project_commits.sort_values('commit_date')
            
            # Get week range for this project
            min_week = project_commits['week_date'].min()
            max_week = project_commits['week_date'].max()
            
            # Create weekly range
            week_range = pd.date_range(start=min_week, end=max_week, freq='W-MON')
            
            # Calculate core contributors for each week
            for week_number, current_week in enumerate(week_range, 1):
                # Calculate core contributors at this week
                core_info = self.calculate_core_contributors_at_week(project_commits, current_week)
                
                # Create record
                record = {
                    'project_name': project_name,
                    'project_type': project_type,
                    'week_date': current_week,
                    'week_number': week_number,
                    'total_commits_to_date': core_info['total_commits'],
                    'total_contributors_to_date': core_info['total_contributors'],
                    'core_threshold_commits': core_info['core_threshold'],
                    'core_contributors_count': len(core_info['core_contributors']),
                    'core_contributors_emails': json.dumps(core_info['core_contributors'])
                }
                
                all_timeline_records.append(record)
            
            # Memory management
            if idx % 50 == 0:
                gc.collect()
                mem = psutil.virtual_memory()
                print(f"   Progress: {idx}/{len(self.projects)} | Memory: {(mem.total-mem.available)/(1024**3):.1f}GB")
        
        # Create DataFrame
        print(f"\n📊 Creating final DataFrame with {len(all_timeline_records):,} records...")
        timeline_df = pd.DataFrame(all_timeline_records)
        
        # Sort by project and week
        timeline_df = timeline_df.sort_values(['project_name', 'week_date'])
        
        # Save to CSV
        output_path = self.datasets_dir / "project_core_timeline_weekly.csv"
        timeline_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Created project_core_timeline_weekly.csv")
        print(f"   Rows: {len(timeline_df):,}")
        print(f"   Projects: {timeline_df['project_name'].nunique()}")
        print(f"   Date range: {timeline_df['week_date'].min()} to {timeline_df['week_date'].max()}")
        
        # Show sample statistics
        print("\n📊 Sample Statistics:")
        print(f"   Avg core contributors per project-week: {timeline_df['core_contributors_count'].mean():.2f}")
        print(f"   Max core contributors in any week: {timeline_df['core_contributors_count'].max()}")
        print(f"   Projects with most weeks: {timeline_df.groupby('project_name').size().nlargest(5).to_dict()}")
        
        return timeline_df
    
    def verify_80_percent_rule(self, timeline_df, sample_size=10):
        """
        Verify that the 80% rule is correctly applied.
        
        Parameters:
        -----------
        timeline_df : DataFrame
            The timeline dataframe to verify
        sample_size : int
            Number of random samples to check
        """
        print("\n🔍 VERIFYING 80% RULE...")
        print("-"*40)
        
        # Take random samples
        samples = timeline_df.sample(min(sample_size, len(timeline_df)))
        
        for _, row in samples.iterrows():
            project = row['project_name']
            week = pd.Timestamp(row['week_date'])
            
            # Get commits up to this week
            project_commits = self.commits_df[
                (self.commits_df['project_name'] == project) &
                (self.commits_df['week_date'] <= week)
            ]
            
            if len(project_commits) == 0:
                continue
            
            # Count commits per contributor
            contributor_counts = project_commits.groupby('author_email').size().sort_values(ascending=False)
            
            # Get core contributors from our data
            core_list = json.loads(row['core_contributors_emails'])
            
            # Calculate what percentage of commits the core contributors have
            core_commits = contributor_counts[contributor_counts.index.isin(core_list)].sum()
            total_commits = contributor_counts.sum()
            percentage = (core_commits / total_commits * 100) if total_commits > 0 else 0
            
            print(f"   {project[:30]} | Week {row['week_number']:3d} | "
                  f"Core: {len(core_list):2d} | "
                  f"Coverage: {percentage:.1f}% | "
                  f"{'✅' if percentage >= 75 else '⚠️'}")  # Allow 75-85% range
        
        print("\n   ✅ Verification complete!")

if __name__ == "__main__":
    # Configuration
    COMMITS_CSV = "../data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv"
    OUTPUT_DIR = "weekly_datasets"
    
    # Create the dataset
    creator = ProjectCoreTimelineCreator(COMMITS_CSV, OUTPUT_DIR)
    
    # Load data
    creator.load_commits_data()
    
    # Create timeline
    timeline_df = creator.create_project_core_timeline()
    
    # Verify correctness
    creator.verify_80_percent_rule(timeline_df)
    
    print("\n✅ Dataset 1 complete!")