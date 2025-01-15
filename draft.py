#!/usr/bin/env python3
"""
RQ1 Step 5: Project Core Timeline Weekly Analysis
=================================================

Creates project_core_timeline_weekly.csv tracking who is core in each project at each week.
Implements 80% cumulative commits rule for core contributor definition.

Author: Research Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import json
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'input_file': 'RQ1_transition_rates_and_speeds/data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv',
    'output_dir': 'RQ1_transition_rates_and_speeds/data_mining/step5_core_timeline_weekly',
    'checkpoint_file': 'step5_checkpoint.json',
    'core_threshold_percentile': 80,  # 80% rule for core contributors
    'batch_size': 1000000,  # Process commits in batches
}

class CoreTimelineProcessor:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.setup_logging()
        self.load_checkpoint()
        
    def setup_directories(self):
        """Create output directories if they don't exist."""
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'project_results').mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Configure logging for the process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'processing.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_checkpoint(self):
        """Load checkpoint to resume processing."""
        self.checkpoint_path = self.output_dir / self.config['checkpoint_file']
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                self.checkpoint = json.load(f)
        else:
            self.checkpoint = {
                'completed_projects': [],
                'failed_projects': [],
                'last_updated': None,
                'total_processed': 0
            }
    
    def save_checkpoint(self):
        """Save current progress."""
        self.checkpoint['last_updated'] = datetime.now().isoformat()
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def get_project_list(self) -> List[str]:
        """Get list of all projects to process."""
        self.logger.info("Loading project list from master dataset...")
        
        # Read just the project names to get unique projects
        df_sample = pd.read_csv(self.config['input_file'], nrows=10000)
        projects = df_sample['project_name'].unique().tolist()
        
        # Get all projects by reading in chunks
        chunk_iter = pd.read_csv(self.config['input_file'], chunksize=self.config['batch_size'])
        all_projects = set()
        
        for chunk in chunk_iter:
            all_projects.update(chunk['project_name'].unique())
            
        return sorted(list(all_projects))
    
    def calculate_weekly_core_status(self, project_commits: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate core contributor status for each week in a project.
        
        Args:
            project_commits: DataFrame with commits for one project
            
        Returns:
            DataFrame with weekly core status timeline
        """
        # Convert commit_date to datetime
        project_commits['commit_date'] = pd.to_datetime(project_commits['commit_date'])
        
        # Create week boundaries (Monday as start of week)
        min_date = project_commits['commit_date'].min()
        max_date = project_commits['commit_date'].max()
        
        # Get Monday of the first week
        start_monday = min_date - timedelta(days=min_date.weekday())
        end_monday = max_date - timedelta(days=max_date.weekday()) + timedelta(days=7)
        
        # Create weekly timeline
        weeks = pd.date_range(start=start_monday, end=end_monday, freq='W-MON')
        
        results = []
        
        for week_num, week_date in enumerate(weeks[:-1]):  # Exclude last boundary
            # Get commits up to end of this week
            week_end = week_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
            commits_to_date = project_commits[project_commits['commit_date'] <= week_end]
            
            if len(commits_to_date) == 0:
                continue
                
            # Calculate cumulative contributor statistics
            contributor_stats = commits_to_date.groupby('author_email').agg({
                'commit_hash': 'count',
                'total_lines_changed': 'sum'
            }).rename(columns={'commit_hash': 'commits'})
            
            # Calculate 80% threshold for core contributors
            total_commits = len(commits_to_date)
            contributor_stats = contributor_stats.sort_values('commits', ascending=False)
            contributor_stats['cumulative_commits'] = contributor_stats['commits'].cumsum()
            contributor_stats['cumulative_percentage'] = (contributor_stats['cumulative_commits'] / total_commits) * 100
            
            # Find core contributors (those in top 80% of commits)
            core_threshold_commits = contributor_stats[contributor_stats['cumulative_percentage'] <= self.config['core_threshold_percentile']]['commits'].min()
            if pd.isna(core_threshold_commits):
                core_threshold_commits = contributor_stats['commits'].max()
            
            core_contributors = contributor_stats[contributor_stats['commits'] >= core_threshold_commits]
            core_emails = core_contributors.index.tolist()
            
            # Create weekly record
            weekly_record = {
                'project_name': project_commits['project_name'].iloc[0],
                'project_type': project_commits['project_type'].iloc[0],
                'week_date': week_date.strftime('%Y-%m-%d'),
                'week_number': week_num + 1,
                'total_commits_to_date': total_commits,
                'total_contributors_to_date': len(contributor_stats),
                'core_threshold_commits': int(core_threshold_commits),
                'core_contributors_count': len(core_contributors),
                'core_contributors_emails': core_emails
            }
            
            results.append(weekly_record)
        
        return pd.DataFrame(results)
    
    def process_project(self, project_name: str) -> bool:
        """
        Process a single project to generate weekly core timeline.
        
        Args:
            project_name: Name of the project to process
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            self.logger.info(f"Processing project: {project_name}")
            
            # Load project commits in chunks to manage memory
            project_commits = []
            chunk_iter = pd.read_csv(self.config['input_file'], chunksize=self.config['batch_size'])
            
            for chunk in chunk_iter:
                project_chunk = chunk[chunk['project_name'] == project_name]
                if len(project_chunk) > 0:
                    project_commits.append(project_chunk)
            
            if not project_commits:
                self.logger.warning(f"No commits found for project: {project_name}")
                return False
            
            # Combine all chunks for this project
            project_df = pd.concat(project_commits, ignore_index=True)
            
            # Sort by commit date
            project_df = project_df.sort_values('commit_date')
            
            self.logger.info(f"Loaded {len(project_df)} commits for {project_name}")
            
            # Calculate weekly core status
            weekly_timeline = self.calculate_weekly_core_status(project_df)
            
            if len(weekly_timeline) == 0:
                self.logger.warning(f"No weekly data generated for project: {project_name}")
                return False
            
            # Save project results
            project_output_file = self.output_dir / 'project_results' / f"{project_name.replace('/', '_')}_timeline.csv"
            weekly_timeline.to_csv(project_output_file, index=False)
            
            self.logger.info(f"Saved {len(weekly_timeline)} weekly records for {project_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process project {project_name}: {str(e)}")
            return False
    
    def consolidate_results(self):
        """Consolidate all project results into final dataset."""
        self.logger.info("Consolidating all project results...")
        
        project_files = list((self.output_dir / 'project_results').glob('*_timeline.csv'))
        if not project_files:
            self.logger.error("No project result files found!")
            return
        
        all_timelines = []
        for file_path in tqdm(project_files, desc="Consolidating"):
            try:
                df = pd.read_csv(file_path)
                all_timelines.append(df)
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {str(e)}")
        
        # Combine all project timelines
        final_dataset = pd.concat(all_timelines, ignore_index=True)
        
        # Sort by project and week
        final_dataset = final_dataset.sort_values(['project_name', 'week_number'])
        
        # Save final dataset
        output_file = self.output_dir / 'project_core_timeline_weekly.csv'
        final_dataset.to_csv(output_file, index=False)
        
        # Generate summary statistics
        self.generate_summary_stats(final_dataset)
        
        self.logger.info(f"Final dataset saved: {output_file}")
        self.logger.info(f"Total records: {len(final_dataset)}")
        
    def generate_summary_stats(self, df: pd.DataFrame):
        """Generate and save summary statistics."""
        stats = {
            'total_records': len(df),
            'unique_projects': df['project_name'].nunique(),
            'total_weeks_analyzed': df['week_number'].sum(),
            'avg_weeks_per_project': df.groupby('project_name')['week_number'].max().mean(),
            'avg_core_contributors_per_week': df['core_contributors_count'].mean(),
            'max_core_contributors': df['core_contributors_count'].max(),
            'projects_by_type': df['project_type'].value_counts().to_dict(),
            'processing_date': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info("Summary statistics saved")
    
    def run(self):
        """Main execution function."""
        self.logger.info("Starting Project Core Timeline Weekly Analysis")
        
        # Get list of all projects
        projects = self.get_project_list()
        remaining_projects = [p for p in projects if p not in self.checkpoint['completed_projects']]
        
        self.logger.info(f"Total projects: {len(projects)}")
        self.logger.info(f"Completed projects: {len(self.checkpoint['completed_projects'])}")
        self.logger.info(f"Remaining projects: {len(remaining_projects)}")
        
        if not remaining_projects:
            self.logger.info("All projects already processed. Consolidating results...")
            self.consolidate_results()
            return
        
        # Process remaining projects
        for project_name in tqdm(remaining_projects, desc="Processing projects"):
            success = self.process_project(project_name)
            
            if success:
                self.checkpoint['completed_projects'].append(project_name)
                self.checkpoint['total_processed'] += 1
            else:
                self.checkpoint['failed_projects'].append(project_name)
            
            # Save checkpoint every 10 projects
            if len(self.checkpoint['completed_projects']) % 10 == 0:
                self.save_checkpoint()
        
        # Final checkpoint save
        self.save_checkpoint()
        
        # Consolidate all results
        self.consolidate_results()
        
        self.logger.info("Project Core Timeline Weekly Analysis completed!")
        self.logger.info(f"Successfully processed: {len(self.checkpoint['completed_projects'])} projects")
        self.logger.info(f"Failed projects: {len(self.checkpoint['failed_projects'])}")

if __name__ == "__main__":
    processor = CoreTimelineProcessor(CONFIG)
    processor.run()


    #!/usr/bin/env python3
"""
RQ1 Step 6: Contributor Activity Weekly Analysis
===============================================

Creates contributor_activity_weekly.csv tracking each contributor's activity week by week.
This is the most memory-intensive script - processes ~2.5M contributor-week records.

Author: Research Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import json
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple
import warnings
import gc
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'input_file': 'RQ1_transition_rates_and_speeds/data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv',
    'output_dir': 'RQ1_transition_rates_and_speeds/data_mining/step6_contributor_activity_weekly',
    'checkpoint_file': 'step6_checkpoint.json',
    'core_threshold_percentile': 80,
    'batch_size': 500000,  # Smaller batches for memory efficiency
    'save_frequency': 5,    # Save checkpoint every 5 projects
}

class ContributorActivityProcessor:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.setup_logging()
        self.load_checkpoint()
        
    def setup_directories(self):
        """Create output directories if they don't exist."""
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'project_results').mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Configure logging for the process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'processing.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_checkpoint(self):
        """Load checkpoint to resume processing."""
        self.checkpoint_path = self.output_dir / self.config['checkpoint_file']
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                self.checkpoint = json.load(f)
        else:
            self.checkpoint = {
                'completed_projects': [],
                'failed_projects': [],
                'last_updated': None,
                'total_processed': 0,
                'total_records_created': 0
            }
    
    def save_checkpoint(self):
        """Save current progress."""
        self.checkpoint['last_updated'] = datetime.now().isoformat()
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def get_project_list(self) -> List[str]:
        """Get list of all projects to process."""
        self.logger.info("Loading project list from master dataset...")
        
        chunk_iter = pd.read_csv(self.config['input_file'], chunksize=self.config['batch_size'])
        all_projects = set()
        
        for chunk in chunk_iter:
            all_projects.update(chunk['project_name'].unique())
            
        return sorted(list(all_projects))
    
    def calculate_core_status_timeline(self, project_commits: pd.DataFrame) -> Dict[str, List]:
        """
        Calculate when each contributor becomes core over time.
        
        Args:
            project_commits: DataFrame with commits for one project
            
        Returns:
            Dict mapping week_date to list of core contributor emails
        """
        # Convert commit_date to datetime
        project_commits['commit_date'] = pd.to_datetime(project_commits['commit_date'])
        
        # Create week boundaries
        min_date = project_commits['commit_date'].min()
        max_date = project_commits['commit_date'].max()
        start_monday = min_date - timedelta(days=min_date.weekday())
        end_monday = max_date - timedelta(days=max_date.weekday()) + timedelta(days=7)
        weeks = pd.date_range(start=start_monday, end=end_monday, freq='W-MON')
        
        core_timeline = {}
        
        for week_date in weeks[:-1]:
            week_end = week_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
            commits_to_date = project_commits[project_commits['commit_date'] <= week_end]
            
            if len(commits_to_date) == 0:
                core_timeline[week_date] = []
                continue
            
            # Calculate cumulative contributor statistics
            contributor_stats = commits_to_date.groupby('author_email')['commit_hash'].count()
            contributor_stats = contributor_stats.sort_values(ascending=False)
            
            # Calculate 80% threshold
            total_commits = len(commits_to_date)
            contributor_stats_df = pd.DataFrame({
                'commits': contributor_stats,
                'cumulative_commits': contributor_stats.cumsum(),
            })
            contributor_stats_df['cumulative_percentage'] = (contributor_stats_df['cumulative_commits'] / total_commits) * 100
            
            # Find core contributors
            core_threshold_commits = contributor_stats_df[
                contributor_stats_df['cumulative_percentage'] <= self.config['core_threshold_percentile']
            ]['commits'].min()
            
            if pd.isna(core_threshold_commits):
                core_threshold_commits = contributor_stats_df['commits'].max()
            
            core_contributors = contributor_stats_df[
                contributor_stats_df['commits'] >= core_threshold_commits
            ].index.tolist()
            
            core_timeline[week_date] = core_contributors
        
        return core_timeline
    
    def process_contributor_weekly_activity(self, project_commits: pd.DataFrame, core_timeline: Dict) -> pd.DataFrame:
        """
        Process weekly activity for each contributor in a project.
        
        Args:
            project_commits: DataFrame with commits for one project
            core_timeline: Dict mapping week dates to core contributor lists
            
        Returns:
            DataFrame with contributor weekly activity records
        """
        project_commits['commit_date'] = pd.to_datetime(project_commits['commit_date'])
        
        # Get all contributors
        contributors = project_commits['author_email'].unique()
        
        # Get week boundaries
        weeks = sorted(core_timeline.keys())
        
        results = []
        
        for contributor in contributors:
            contributor_commits = project_commits[project_commits['author_email'] == contributor]
            first_commit_date = contributor_commits['commit_date'].min()
            
            for week_num, week_date in enumerate(weeks):
                week_end = week_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
                
                # Get commits for this week
                week_commits = contributor_commits[
                    (contributor_commits['commit_date'] >= week_date) & 
                    (contributor_commits['commit_date'] <= week_end)
                ]
                
                # Get cumulative commits up to this week
                cumulative_commits = contributor_commits[
                    contributor_commits['commit_date'] <= week_end
                ]
                
                # Get total project commits up to this week
                project_commits_to_date = project_commits[
                    project_commits['commit_date'] <= week_end
                ]
                
                # Calculate metrics
                commits_this_week = len(week_commits)
                commit_hashes = week_commits['commit_hash'].tolist() if commits_this_week > 0 else []
                lines_added = week_commits['total_insertions'].sum() if commits_this_week > 0 else 0
                lines_deleted = week_commits['total_deletions'].sum() if commits_this_week > 0 else 0
                files_modified = week_commits['files_modified_count'].sum() if commits_this_week > 0 else 0
                
                cumulative_commits_count = len(cumulative_commits)
                cumulative_lines_changed = cumulative_commits['total_lines_changed'].sum()
                project_commits_count = len(project_commits_to_date)
                
                # Calculate contribution percentage
                contribution_percentage = (cumulative_commits_count / project_commits_count * 100) if project_commits_count > 0 else 0
                
                # Check if core this week
                is_core = contributor in core_timeline.get(week_date, [])
                
                # Calculate rank this week (based on commits this week)
                if commits_this_week > 0:
                    week_all_commits = project_commits[
                        (project_commits['commit_date'] >= week_date) & 
                        (project_commits['commit_date'] <= week_end)
                    ]
                    week_contributor_stats = week_all_commits.groupby('author_email')['commit_hash'].count().sort_values(ascending=False)
                    rank_this_week = list(week_contributor_stats.index).index(contributor) + 1
                else:
                    rank_this_week = None
                
                # Calculate weeks since first commit
                weeks_since_first = max(0, (week_date - first_commit_date).days // 7)
                
                # Create record
                record = {
                    'project_name': project_commits['project_name'].iloc[0],
                    'project_type': project_commits['project_type'].iloc[0],
                    'contributor_email': contributor,
                    'week_date': week_date.strftime('%Y-%m-%d'),
                    'week_number': week_num + 1,
                    'weeks_since_first_commit': weeks_since_first,
                    'commits_this_week': commits_this_week,
                    'commit_hashes': commit_hashes,
                    'lines_added_this_week': lines_added,
                    'lines_deleted_this_week': lines_deleted,
                    'files_modified_this_week': files_modified,
                    'cumulative_commits': cumulative_commits_count,
                    'cumulative_lines_changed': cumulative_lines_changed,
                    'project_commits_to_date': project_commits_count,
                    'contribution_percentage': round(contribution_percentage, 4),
                    'is_core_this_week': is_core,
                    'rank_this_week': rank_this_week
                }
                
                results.append(record)
        
        return pd.DataFrame(results)
    
    def process_project(self, project_name: str) -> bool:
        """
        Process a single project to generate contributor weekly activity.
        
        Args:
            project_name: Name of the project to process
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            self.logger.info(f"Processing project: {project_name}")
            
            # Load project commits efficiently
            project_commits = []
            chunk_iter = pd.read_csv(self.config['input_file'], chunksize=self.config['batch_size'])
            
            for chunk in chunk_iter:
                project_chunk = chunk[chunk['project_name'] == project_name]
                if len(project_chunk) > 0:
                    project_commits.append(project_chunk)
            
            if not project_commits:
                self.logger.warning(f"No commits found for project: {project_name}")
                return False
            
            # Combine chunks
            project_df = pd.concat(project_commits, ignore_index=True)
            project_df = project_df.sort_values('commit_date')
            
            self.logger.info(f"Loaded {len(project_df)} commits for {project_name}")
            
            # Calculate core status timeline
            self.logger.info(f"Calculating core timeline for {project_name}")
            core_timeline = self.calculate_core_status_timeline(project_df)
            
            # Process contributor weekly activity
            self.logger.info(f"Processing weekly activity for {project_name}")
            weekly_activity = self.process_contributor_weekly_activity(project_df, core_timeline)
            
            if len(weekly_activity) == 0:
                self.logger.warning(f"No weekly activity data generated for project: {project_name}")
                return False
            
            # Optimize data types to save memory
            weekly_activity['commits_this_week'] = weekly_activity['commits_this_week'].astype('int16')
            weekly_activity['lines_added_this_week'] = weekly_activity['lines_added_this_week'].astype('int32')
            weekly_activity['lines_deleted_this_week'] = weekly_activity['lines_deleted_this_week'].astype('int32')
            weekly_activity['files_modified_this_week'] = weekly_activity['files_modified_this_week'].astype('int16')
            weekly_activity['cumulative_commits'] = weekly_activity['cumulative_commits'].astype('int32')
            weekly_activity['cumulative_lines_changed'] = weekly_activity['cumulative_lines_changed'].astype('int32')
            weekly_activity['project_commits_to_date'] = weekly_activity['project_commits_to_date'].astype('int32')
            weekly_activity['contribution_percentage'] = weekly_activity['contribution_percentage'].astype('float32')
            weekly_activity['is_core_this_week'] = weekly_activity['is_core_this_week'].astype('bool')
            
            # Save project results
            project_output_file = self.output_dir / 'project_results' / f"{project_name.replace('/', '_')}_activity.csv"
            weekly_activity.to_csv(project_output_file, index=False)
            
            # Update checkpoint with record count
            self.checkpoint['total_records_created'] += len(weekly_activity)
            
            self.logger.info(f"Saved {len(weekly_activity)} activity records for {project_name}")
            
            # Clean up memory
            del project_df, weekly_activity, core_timeline
            gc.collect()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process project {project_name}: {str(e)}")
            return False
    
    def consolidate_results(self):
        """Consolidate all project results into final dataset."""
        self.logger.info("Consolidating all project results...")
        
        project_files = list((self.output_dir / 'project_results').glob('*_activity.csv'))
        if not project_files:
            self.logger.error("No project result files found!")
            return
        
        # Process files in batches to manage memory
        batch_size = 50
        all_batches = []
        
        for i in range(0, len(project_files), batch_size):
            batch_files = project_files[i:i+batch_size]
            batch_data = []
            
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(project_files)-1)//batch_size + 1}")
            
            for file_path in tqdm(batch_files, desc="Loading batch"):
                try:
                    df = pd.read_csv(file_path)
                    batch_data.append(df)
                except Exception as e:
                    self.logger.error(f"Error reading {file_path}: {str(e)}")
            
            if batch_data:
                batch_df = pd.concat(batch_data, ignore_index=True)
                
                # Save batch to temp file
                batch_file = self.output_dir / f'temp_batch_{i//batch_size}.csv'
                batch_df.to_csv(batch_file, index=False)
                all_batches.append(batch_file)
                
                del batch_data, batch_df
                gc.collect()
        
        # Final consolidation
        self.logger.info("Final consolidation of batches...")
        final_data = []
        
        for batch_file in all_batches:
            df = pd.read_csv(batch_file)
            final_data.append(df)
        
        final_dataset = pd.concat(final_data, ignore_index=True)
        final_dataset = final_dataset.sort_values(['project_name', 'contributor_email', 'week_number'])
        
        # Save final dataset
        output_file = self.output_dir / 'contributor_activity_weekly.csv'
        final_dataset.to_csv(output_file, index=False)
        
        # Clean up temp files
        for batch_file in all_batches:
            batch_file.unlink()
        
        # Generate summary statistics
        self.generate_summary_stats(final_dataset)
        
        self.logger.info(f"Final dataset saved: {output_file}")
        self.logger.info(f"Total records: {len(final_dataset)}")
        
    def generate_summary_stats(self, df: pd.DataFrame):
        """Generate and save summary statistics."""
        stats = {
            'total_records': len(df),
            'unique_projects': df['project_name'].nunique(),
            'unique_contributors': df['contributor_email'].nunique(),
            'total_contributor_weeks': len(df),
            'avg_weeks_per_contributor': df.groupby(['project_name', 'contributor_email'])['week_number'].max().mean(),
            'core_contributor_weeks': df['is_core_this_week'].sum(),
            'core_percentage': (df['is_core_this_week'].sum() / len(df) * 100),
            'projects_by_type': df['project_type'].value_counts().to_dict(),
            'processing_date': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info("Summary statistics saved")
    
    def run(self):
        """Main execution function."""
        self.logger.info("Starting Contributor Activity Weekly Analysis")
        
        # Get list of all projects
        projects = self.get_project_list()
        remaining_projects = [p for p in projects if p not in self.checkpoint['completed_projects']]
        
        self.logger.info(f"Total projects: {len(projects)}")
        self.logger.info(f"Completed projects: {len(self.checkpoint['completed_projects'])}")
        self.logger.info(f"Remaining projects: {len(remaining_projects)}")
        self.logger.info(f"Total records created so far: {self.checkpoint['total_records_created']}")
        
        if not remaining_projects:
            self.logger.info("All projects already processed. Consolidating results...")
            self.consolidate_results()
            return
        
        # Process remaining projects
        for i, project_name in enumerate(tqdm(remaining_projects, desc="Processing projects")):
            success = self.process_project(project_name)
            
            if success:
                self.checkpoint['completed_projects'].append(project_name)
                self.checkpoint['total_processed'] += 1
            else:
                self.checkpoint['failed_projects'].append(project_name)
            
            # Save checkpoint at specified frequency
            if (i + 1) % self.config['save_frequency'] == 0:
                self.save_checkpoint()
                self.logger.info(f"Checkpoint saved. Progress: {len(self.checkpoint['completed_projects'])}/{len(projects)} projects")
        
        # Final checkpoint save
        self.save_checkpoint()
        
        # Consolidate all results
        self.consolidate_results()
        
        self.logger.info("Contributor Activity Weekly Analysis completed!")
        self.logger.info(f"Successfully processed: {len(self.checkpoint['completed_projects'])} projects")
        self.logger.info(f"Failed projects: {len(self.checkpoint['failed_projects'])}")
        self.logger.info(f"Total records created: {self.checkpoint['total_records_created']}")

if __name__ == "__main__":
    processor = ContributorActivityProcessor(CONFIG)
    processor.run()






    #!/usr/bin/env python3
"""
RQ1 Step 7: Contributor Transitions Analysis
===========================================

Creates contributor_transitions.csv summarizing each contributor's journey to core status.
This script aggregates weekly activity data into transition summaries.

Author: Research Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import json
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple, Optional
import warnings
import gc
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'input_file': 'RQ1_transition_rates_and_speeds/data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv',
    'weekly_activity_file': 'RQ1_transition_rates_and_speeds/data_mining/step6_contributor_activity_weekly/contributor_activity_weekly.csv',
    'output_dir': 'RQ1_transition_rates_and_speeds/data_mining/step7_contributor_transitions',
    'checkpoint_file': 'step7_checkpoint.json',
    'core_threshold_percentile': 80,
    'batch_size': 500000,
    'use_weekly_data': False,  # Set to True if step 6 is completed
}

class ContributorTransitionsProcessor:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.setup_logging()
        self.load_checkpoint()
        
    def setup_directories(self):
        """Create output directories if they don't exist."""
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'project_results').mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Configure logging for the process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'processing.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_checkpoint(self):
        """Load checkpoint to resume processing."""
        self.checkpoint_path = self.output_dir / self.config['checkpoint_file']
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                self.checkpoint = json.load(f)
        else:
            self.checkpoint = {
                'completed_projects': [],
                'failed_projects': [],
                'last_updated': None,
                'total_processed': 0,
                'total_contributors_analyzed': 0
            }
    
    def save_checkpoint(self):
        """Save current progress."""
        self.checkpoint['last_updated'] = datetime.now().isoformat()
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def get_project_list(self) -> List[str]:
        """Get list of all projects to process."""
        self.logger.info("Loading project list...")
        
        if self.config['use_weekly_data'] and Path(self.config['weekly_activity_file']).exists():
            # Use weekly activity data if available
            df_sample = pd.read_csv(self.config['weekly_activity_file'], nrows=10000)
            projects = df_sample['project_name'].unique().tolist()
        else:
            # Use master commits dataset
            chunk_iter = pd.read_csv(self.config['input_file'], chunksize=self.config['batch_size'])
            all_projects = set()
            
            for chunk in chunk_iter:
                all_projects.update(chunk['project_name'].unique())
            
            projects = sorted(list(all_projects))
            
        return projects
    
    def calculate_core_timeline_from_commits(self, project_commits: pd.DataFrame) -> Dict[datetime, List[str]]:
        """
        Calculate core contributor timeline from commit data.
        
        Args:
            project_commits: DataFrame with commits for one project
            
        Returns:
            Dict mapping week dates to core contributor lists
        """
        project_commits['commit_date'] = pd.to_datetime(project_commits['commit_date'])
        
        # Create week boundaries
        min_date = project_commits['commit_date'].min()
        max_date = project_commits['commit_date'].max()
        start_monday = min_date - timedelta(days=min_date.weekday())
        end_monday = max_date - timedelta(days=max_date.weekday()) + timedelta(days=7)
        weeks = pd.date_range(start=start_monday, end=end_monday, freq='W-MON')
        
        core_timeline = {}
        
        for week_date in weeks[:-1]:
            week_end = week_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
            commits_to_date = project_commits[project_commits['commit_date'] <= week_end]
            
            if len(commits_to_date) == 0:
                core_timeline[week_date] = []
                continue
            
            # Calculate cumulative contributor statistics
            contributor_stats = commits_to_date.groupby('author_email')['commit_hash'].count()
            contributor_stats = contributor_stats.sort_values(ascending=False)
            
            # Calculate 80% threshold
            total_commits = len(commits_to_date)
            contributor_stats_df = pd.DataFrame({
                'commits': contributor_stats,
                'cumulative_commits': contributor_stats.cumsum(),
            })
            contributor_stats_df['cumulative_percentage'] = (contributor_stats_df['cumulative_commits'] / total_commits) * 100
            
            # Find core contributors
            core_threshold_commits = contributor_stats_df[
                contributor_stats_df['cumulative_percentage'] <= self.config['core_threshold_percentile']
            ]['commits'].min()
            
            if pd.isna(core_threshold_commits):
                core_threshold_commits = contributor_stats_df['commits'].max()
            
            core_contributors = contributor_stats_df[
                contributor_stats_df['commits'] >= core_threshold_commits
            ].index.tolist()
            
            core_timeline[week_date] = core_contributors
        
        return core_timeline
    
    def analyze_contributor_transitions_from_commits(self, project_commits: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze contributor transitions from raw commit data.
        
        Args:
            project_commits: DataFrame with commits for one project
            
        Returns:
            DataFrame with contributor transition summaries
        """
        project_commits['commit_date'] = pd.to_datetime(project_commits['commit_date'])
        
        # Calculate core timeline
        core_timeline = self.calculate_core_timeline_from_commits(project_commits)
        sorted_weeks = sorted(core_timeline.keys())
        
        # Get all contributors
        contributors = project_commits['author_email'].unique()
        
        results = []
        
        for contributor in contributors:
            contributor_commits = project_commits[project_commits['author_email'] == contributor]
            
            # Basic timeline info
            first_commit_date = contributor_commits['commit_date'].min()
            last_commit_date = contributor_commits['commit_date'].max()
            total_commits = len(contributor_commits)
            
            # Calculate active weeks (weeks with at least 1 commit)
            contributor_commits['week_start'] = contributor_commits['commit_date'].dt.to_period('W-MON').dt.start_time
            active_weeks = contributor_commits['week_start'].nunique()
            
            # Analyze core status progression
            became_core = False
            first_core_week_date = None
            weeks_to_core = None
            commits_at_core = None
            maintained_core_status = False
            total_weeks_as_core = 0
            
            # Check each week for core status
            core_weeks = []
            for week_date in sorted_weeks:
                if contributor in core_timeline[week_date]:
                    core_weeks.append(week_date)
                    if not became_core:
                        became_core = True
                        first_core_week_date = week_date
                        weeks_to_core = (week_date - first_commit_date).days // 7
                        
                        # Count commits up to this week
                        commits_at_core = len(contributor_commits[
                            contributor_commits['commit_date'] <= week_date
                        ])
            
            if core_weeks:
                total_weeks_as_core = len(core_weeks)
                # Check if still core at the end
                maintained_core_status = sorted_weeks[-1] in core_weeks if sorted_weeks else False
            
            # Calculate maximum contribution percentage
            max_contribution_percentage = 0
            for week_date in sorted_weeks:
                week_end = week_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
                commits_to_date = project_commits[project_commits['commit_date'] <= week_end]
                contributor_commits_to_date = contributor_commits[contributor_commits['commit_date'] <= week_end]
                
                if len(commits_to_date) > 0:
                    contribution_percentage = len(contributor_commits_to_date) / len(commits_to_date) * 100
                    max_contribution_percentage = max(max_contribution_percentage, contribution_percentage)
            
            # Determine final status
            if maintained_core_status:
                final_status = "core"
            elif total_commits >= 10:  # Arbitrary threshold for "active"
                final_status = "active"
            else:
                final_status = "inactive"
            
            # Create transition record
            record = {
                'project_name': project_commits['project_name'].iloc[0],
                'project_type': project_commits['project_type'].iloc[0],
                'contributor_email': contributor,
                'first_commit_date': first_commit_date.strftime('%Y-%m-%d'),
                'last_commit_date': last_commit_date.strftime('%Y-%m-%d'),
                'total_commits': total_commits,
                'total_active_weeks': active_weeks,
                'became_core': became_core,
                'first_core_week_date': first_core_week_date.strftime('%Y-%m-%d') if first_core_week_date else None,
                'weeks_to_core': weeks_to_core,
                'commits_at_core': commits_at_core,
                'maintained_core_status': maintained_core_status,
                'total_weeks_as_core': total_weeks_as_core,
                'max_contribution_percentage': round(max_contribution_percentage, 4),
                'final_status': final_status
            }
            
            results.append(record)
        
        return pd.DataFrame(results)
    
    def analyze_contributor_transitions_from_weekly_data(self, project_name: str) -> pd.DataFrame:
        """
        Analyze contributor transitions from pre-computed weekly activity data.
        
        Args:
            project_name: Name of the project to analyze
            
        Returns:
            DataFrame with contributor transition summaries
        """
        # Load weekly activity data for this project
        weekly_data = pd.read_csv(self.config['weekly_activity_file'])
        project_data = weekly_data[weekly_data['project_name'] == project_name]
        
        if len(project_data) == 0:
            return pd.DataFrame()
        
        # Group by contributor
        results = []
        
        for contributor, contributor_data in project_data.groupby('contributor_email'):
            contributor_data = contributor_data.sort_values('week_number')
            
            # Basic timeline info
            first_commit_date = pd.to_datetime(contributor_data['week_date']).min()
            last_commit_date = pd.to_datetime(contributor_data['week_date']).max()
            total_commits = contributor_data['cumulative_commits'].max()
            total_active_weeks = (contributor_data['commits_this_week'] > 0).sum()
            
            # Analyze core status
            core_weeks = contributor_data[contributor_data['is_core_this_week']]
            became_core = len(core_weeks) > 0
            
            first_core_week_date = None
            weeks_to_core = None
            commits_at_core = None
            
            if became_core:
                first_core_week = core_weeks.iloc[0]
                first_core_week_date = pd.to_datetime(first_core_week['week_date'])
                weeks_to_core = first_core_week['weeks_since_first_commit']
                commits_at_core = first_core_week['cumulative_commits']
            
            total_weeks_as_core = len(core_weeks)
            maintained_core_status = contributor_data.iloc[-1]['is_core_this_week'] if len(contributor_data) > 0 else False
            max_contribution_percentage = contributor_data['contribution_percentage'].max()
            
            # Determine final status
            if maintained_core_status:
                final_status = "core"
            elif total_commits >= 10:
                final_status = "active"
            else:
                final_status = "inactive"
            
            # Create transition record
            record = {
                'project_name': project_name,
                'project_type': contributor_data['project_type'].iloc[0],
                'contributor_email': contributor,
                'first_commit_date': first_commit_date.strftime('%Y-%m-%d'),
                'last_commit_date': last_commit_date.strftime('%Y-%m-%d'),
                'total_commits': total_commits,
                'total_active_weeks': total_active_weeks,
                'became_core': became_core,
                'first_core_week_date': first_core_week_date.strftime('%Y-%m-%d') if first_core_week_date else None,
                'weeks_to_core': weeks_to_core,
                'commits_at_core': commits_at_core,
                'maintained_core_status': maintained_core_status,
                'total_weeks_as_core': total_weeks_as_core,
                'max_contribution_percentage': round(max_contribution_percentage, 4),
                'final_status': final_status
            }
            
            results.append(record)
        
        return pd.DataFrame(results)
    
    def process_project(self, project_name: str) -> bool:
        """
        Process a single project to generate contributor transitions.
        
        Args:
            project_name: Name of the project to process
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            self.logger.info(f"Processing project: {project_name}")
            
            if self.config['use_weekly_data'] and Path(self.config['weekly_activity_file']).exists():
                # Use pre-computed weekly data
                transitions_df = self.analyze_contributor_transitions_from_weekly_data(project_name)
            else:
                # Calculate from raw commit data
                project_commits = []
                chunk_iter = pd.read_csv(self.config['input_file'], chunksize=self.config['batch_size'])
                
                for chunk in chunk_iter:
                    project_chunk = chunk[chunk['project_name'] == project_name]
                    if len(project_chunk) > 0:
                        project_commits.append(project_chunk)
                
                if not project_commits:
                    self.logger.warning(f"No commits found for project: {project_name}")
                    return False
                
                project_df = pd.concat(project_commits, ignore_index=True)
                project_df = project_df.sort_values('commit_date')
                
                self.logger.info(f"Loaded {len(project_df)} commits for {project_name}")
                
                transitions_df = self.analyze_contributor_transitions_from_commits(project_df)
            
            if len(transitions_df) == 0:
                self.logger.warning(f"No transition data generated for project: {project_name}")
                return False
            
            # Optimize data types
            transitions_df['total_commits'] = transitions_df['total_commits'].astype('int32')
            transitions_df['total_active_weeks'] = transitions_df['total_active_weeks'].astype('int16')
            transitions_df['weeks_to_core'] = transitions_df['weeks_to_core'].astype('Int16')  # Nullable integer
            transitions_df['commits_at_core'] = transitions_df['commits_at_core'].astype('Int32')  # Nullable integer
            transitions_df['total_weeks_as_core'] = transitions_df['total_weeks_as_core'].astype('int16')
            transitions_df['max_contribution_percentage'] = transitions_df['max_contribution_percentage'].astype('float32')
            transitions_df['became_core'] = transitions_df['became_core'].astype('bool')
            transitions_df['maintained_core_status'] = transitions_df['maintained_core_status'].astype('bool')
            
            # Save project results
            project_output_file = self.output_dir / 'project_results' / f"{project_name.replace('/', '_')}_transitions.csv"
            transitions_df.to_csv(project_output_file, index=False)
            
            # Update checkpoint
            self.checkpoint['total_contributors_analyzed'] += len(transitions_df)
            
            self.logger.info(f"Saved {len(transitions_df)} contributor transitions for {project_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process project {project_name}: {str(e)}")
            return False
    
    def consolidate_results(self):
        """Consolidate all project results into final dataset."""
        self.logger.info("Consolidating all project results...")
        
        project_files = list((self.output_dir / 'project_results').glob('*_transitions.csv'))
        if not project_files:
            self.logger.error("No project result files found!")
            return
        
        all_transitions = []
        for file_path in tqdm(project_files, desc="Consolidating"):
            try:
                df = pd.read_csv(file_path)
                all_transitions.append(df)
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {str(e)}")
        
        # Combine all transitions
        final_dataset = pd.concat(all_transitions, ignore_index=True)
        
        # Sort by project and contributor
        final_dataset = final_dataset.sort_values(['project_name', 'contributor_email'])
        
        # Save final dataset
        output_file = self.output_dir / 'contributor_transitions.csv'
        final_dataset.to_csv(output_file, index=False)
        
        # Generate summary statistics
        self.generate_summary_stats(final_dataset)
        
        self.logger.info(f"Final dataset saved: {output_file}")
        self.logger.info(f"Total records: {len(final_dataset)}")
        
    def generate_summary_stats(self, df: pd.DataFrame):
        """Generate and save summary statistics."""
        stats = {
            'total_contributors': len(df),
            'unique_projects': df['project_name'].nunique(),
            'contributors_who_became_core': df['became_core'].sum(),
            'core_transition_rate': (df['became_core'].sum() / len(df) * 100),
            'avg_weeks_to_core': df[df['became_core']]['weeks_to_core'].mean(),
            'median_weeks_to_core': df[df['became_core']]['weeks_to_core'].median(),
            'avg_commits_to_core': df[df['became_core']]['commits_at_core'].mean(),
            'median_commits_to_core': df[df['became_core']]['commits_at_core'].median(),
            'final_status_distribution': df['final_status'].value_counts().to_dict(),
            'projects_by_type': df['project_type'].value_counts().to_dict(),
            'processing_date': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info("Summary statistics saved")
    
    def run(self):
        """Main execution function."""
        self.logger.info("Starting Contributor Transitions Analysis")
        
        # Check if we can use weekly data
        if Path(self.config['weekly_activity_file']).exists():
            self.config['use_weekly_data'] = True
            self.logger.info("Using pre-computed weekly activity data")
        else:
            self.logger.info("Using raw commit data (weekly data not available)")
        
        # Get list of all projects
        projects = self.get_project_list()
        remaining_projects = [p for p in projects if p not in self.checkpoint['completed_projects']]
        
        self.logger.info(f"Total projects: {len(projects)}")
        self.logger.info(f"Completed projects: {len(self.checkpoint['completed_projects'])}")
        self.logger.info(f"Remaining projects: {len(remaining_projects)}")
        self.logger.info(f"Total contributors analyzed so far: {self.checkpoint['total_contributors_analyzed']}")
        
        if not remaining_projects:
            self.logger.info("All projects already processed. Consolidating results...")
            self.consolidate_results()
            return
        
        # Process remaining projects
        for project_name in tqdm(remaining_projects, desc="Processing projects"):
            success = self.process_project(project_name)
            
            if success:
                self.checkpoint['completed_projects'].append(project_name)
                self.checkpoint['total_processed'] += 1
            else:
                self.checkpoint['failed_projects'].append(project_name)
            
            # Save checkpoint every 10 projects
            if len(self.checkpoint['completed_projects']) % 10 == 0:
                self.save_checkpoint()
        
        # Final checkpoint save
        self.save_checkpoint()
        
        # Consolidate all results
        self.consolidate_results()
        
        self.logger.info("Contributor Transitions Analysis completed!")
        self.logger.info(f"Successfully processed: {len(self.checkpoint['completed_projects'])} projects")
        self.logger.info(f"Failed projects: {len(self.checkpoint['failed_projects'])}")
        self.logger.info(f"Total contributors analyzed: {self.checkpoint['total_contributors_analyzed']}")

if __name__ == "__main__":
    processor = ContributorTransitionsProcessor(CONFIG)
    processor.run()




#!/usr/bin/env python3
"""
RQ1 Step 8: Weekly Commit Aggregates Analysis
=============================================

Creates weekly_commit_aggregates.csv with detailed behavioral features for machine learning.
Focuses on commit patterns, timing, and statistical features per contributor per week.

Author: Research Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import json
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple
import warnings
import gc
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'input_file': 'RQ1_transition_rates_and_speeds/data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv',
    'output_dir': 'RQ1_transition_rates_and_speeds/data_mining/step8_weekly_commit_aggregates',
    'checkpoint_file': 'step8_checkpoint.json',
    'batch_size': 500000,
    'save_frequency': 5,
}

class WeeklyCommitAggregatesProcessor:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.setup_logging()
        self.load_checkpoint()
        
    def setup_directories(self):
        """Create output directories if they don't exist."""
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'project_results').mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Configure logging for the process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'processing.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_checkpoint(self):
        """Load checkpoint to resume processing."""
        self.checkpoint_path = self.output_dir / self.config['checkpoint_file']
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                self.checkpoint = json.load(f)
        else:
            self.checkpoint = {
                'completed_projects': [],
                'failed_projects': [],
                'last_updated': None,
                'total_processed': 0,
                'total_records_created': 0
            }
    
    def save_checkpoint(self):
        """Save current progress."""
        self.checkpoint['last_updated'] = datetime.now().isoformat()
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def get_project_list(self) -> List[str]:
        """Get list of all projects to process."""
        self.logger.info("Loading project list from master dataset...")
        
        chunk_iter = pd.read_csv(self.config['input_file'], chunksize=self.config['batch_size'])
        all_projects = set()
        
        for chunk in chunk_iter:
            all_projects.update(chunk['project_name'].unique())
            
        return sorted(list(all_projects))
    
    def calculate_commit_aggregates(self, week_commits: pd.DataFrame) -> Dict:
        """
        Calculate rich behavioral features for commits in a single week.
        
        Args:
            week_commits: DataFrame with commits for one contributor in one week
            
        Returns:
            Dict with aggregated features
        """
        if len(week_commits) == 0:
            return {
                'avg_commit_size': 0.0,
                'max_commit_size': 0,
                'commit_frequency': 0.0,
                'avg_files_per_commit': 0.0,
                'weekend_commits': 0,
                'weekday_commits': 0,
                'morning_commits': 0,
                'afternoon_commits': 0,
                'evening_commits': 0,
                'night_commits': 0,
                'avg_message_length': 0.0,
                'code_churn': 0.0,
                'commit_size_variance': 0.0,
                'files_touched_total': 0,
                'avg_insertions_per_commit': 0.0,
                'avg_deletions_per_commit': 0.0,
                'churn_ratio_avg': 0.0,
                'unique_commit_days': 0,
                'commit_spread_hours': 0.0,
                'message_word_count_avg': 0.0
            }
        
        # Basic size metrics
        week_commits['commit_size'] = week_commits['total_lines_changed']
        avg_commit_size = week_commits['commit_size'].mean()
        max_commit_size = week_commits['commit_size'].max()
        commit_size_variance = week_commits['commit_size'].var() if len(week_commits) > 1 else 0.0
        
        # Frequency metrics (commits per day in the week)
        commit_frequency = len(week_commits) / 7.0
        
        # File metrics
        avg_files_per_commit = week_commits['files_modified_count'].mean()
        files_touched_total = week_commits['files_modified_count'].sum()
        
        # Temporal patterns - weekend vs weekday
        weekend_commits = (week_commits['commit_is_weekend'] == True).sum()
        weekday_commits = (week_commits['commit_is_weekend'] == False).sum()
        
        # Time of day patterns (using commit_hour)
        morning_commits = ((week_commits['commit_hour'] >= 6) & (week_commits['commit_hour'] < 12)).sum()
        afternoon_commits = ((week_commits['commit_hour'] >= 12) & (week_commits['commit_hour'] < 18)).sum()
        evening_commits = ((week_commits['commit_hour'] >= 18) & (week_commits['commit_hour'] < 24)).sum()
        night_commits = ((week_commits['commit_hour'] >= 0) & (week_commits['commit_hour'] < 6)).sum()
        
        # Message characteristics
        avg_message_length = week_commits['message_length_chars'].mean()
        message_word_count_avg = week_commits['message_length_words'].mean()
        
        # Code change patterns
        code_churn = (week_commits['total_insertions'] + week_commits['total_deletions']).sum() / 2.0
        avg_insertions_per_commit = week_commits['total_insertions'].mean()
        avg_deletions_per_commit = week_commits['total_deletions'].mean()
        
        # Churn ratio (deletions/insertions)
        total_insertions = week_commits['total_insertions'].sum()
        total_deletions = week_commits['total_deletions'].sum()
        churn_ratio_avg = total_deletions / total_insertions if total_insertions > 0 else 0.0
        
        # Temporal spread
        unique_commit_days = week_commits['commit_day_of_week'].nunique()
        
        # Commit timing spread (how spread out commits are within the week)
        if len(week_commits) > 1:
            commit_hours = week_commits['commit_hour'].values
            commit_spread_hours = np.std(commit_hours)
        else:
            commit_spread_hours = 0.0
        
        return {
            'avg_commit_size': float(avg_commit_size),
            'max_commit_size': int(max_commit_size),
            'commit_frequency': float(commit_frequency),
            'avg_files_per_commit': float(avg_files_per_commit),
            'weekend_commits': int(weekend_commits),
            'weekday_commits': int(weekday_commits),
            'morning_commits': int(morning_commits),
            'afternoon_commits': int(afternoon_commits),
            'evening_commits': int(evening_commits),
            'night_commits': int(night_commits),
            'avg_message_length': float(avg_message_length),
            'code_churn': float(code_churn),
            'commit_size_variance': float(commit_size_variance),
            'files_touched_total': int(files_touched_total),
            'avg_insertions_per_commit': float(avg_insertions_per_commit),
            'avg_deletions_per_commit': float(avg_deletions_per_commit),
            'churn_ratio_avg': float(churn_ratio_avg),
            'unique_commit_days': int(unique_commit_days),
            'commit_spread_hours': float(commit_spread_hours),
            'message_word_count_avg': float(message_word_count_avg)
        }
    
    def process_project_weekly_aggregates(self, project_commits: pd.DataFrame) -> pd.DataFrame:
        """
        Process weekly aggregates for all contributors in a project.
        
        Args:
            project_commits: DataFrame with commits for one project
            
        Returns:
            DataFrame with weekly aggregate features
        """
        project_commits['commit_date'] = pd.to_datetime(project_commits['commit_date'])
        
        # Create week boundaries
        min_date = project_commits['commit_date'].min()
        max_date = project_commits['commit_date'].max()
        start_monday = min_date - timedelta(days=min_date.weekday())
        end_monday = max_date - timedelta(days=max_date.weekday()) + timedelta(days=7)
        weeks = pd.date_range(start=start_monday, end=end_monday, freq='W-MON')
        
        # Get all contributors
        contributors = project_commits['author_email'].unique()
        
        results = []
        
        for contributor in contributors:
            contributor_commits = project_commits[project_commits['author_email'] == contributor]
            
            for week_date in weeks[:-1]:  # Exclude last boundary
                week_end = week_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
                
                # Get commits for this week
                week_commits = contributor_commits[
                    (contributor_commits['commit_date'] >= week_date) & 
                    (contributor_commits['commit_date'] <= week_end)
                ]
                
                # Calculate aggregates (even if no commits - creates zero record)
                aggregates = self.calculate_commit_aggregates(week_commits)
                
                # Create record
                record = {
                    'project_name': project_commits['project_name'].iloc[0],
                    'contributor_email': contributor,
                    'week_date': week_date.strftime('%Y-%m-%d'),
                }
                
                # Add all aggregate features
                record.update(aggregates)
                
                results.append(record)
        
        return pd.DataFrame(results)
    
    def process_project(self, project_name: str) -> bool:
        """
        Process a single project to generate weekly commit aggregates.
        
        Args:
            project_name: Name of the project to process
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            self.logger.info(f"Processing project: {project_name}")
            
            # Load project commits efficiently
            project_commits = []
            chunk_iter = pd.read_csv(self.config['input_file'], chunksize=self.config['batch_size'])
            
            for chunk in chunk_iter:
                project_chunk = chunk[chunk['project_name'] == project_name]
                if len(project_chunk) > 0:
                    project_commits.append(project_chunk)
            
            if not project_commits:
                self.logger.warning(f"No commits found for project: {project_name}")
                return False
            
            # Combine chunks
            project_df = pd.concat(project_commits, ignore_index=True)
            project_df = project_df.sort_values('commit_date')
            
            self.logger.info(f"Loaded {len(project_df)} commits for {project_name}")
            
            # Process weekly aggregates
            self.logger.info(f"Calculating weekly aggregates for {project_name}")
            weekly_aggregates = self.process_project_weekly_aggregates(project_df)
            
            if len(weekly_aggregates) == 0:
                self.logger.warning(f"No weekly aggregate data generated for project: {project_name}")
                return False
            
            # Optimize data types to save memory
            weekly_aggregates['max_commit_size'] = weekly_aggregates['max_commit_size'].astype('int32')
            weekly_aggregates['weekend_commits'] = weekly_aggregates['weekend_commits'].astype('int8')
            weekly_aggregates['weekday_commits'] = weekly_aggregates['weekday_commits'].astype('int8')
            weekly_aggregates['morning_commits'] = weekly_aggregates['morning_commits'].astype('int8')
            weekly_aggregates['afternoon_commits'] = weekly_aggregates['afternoon_commits'].astype('int8')
            weekly_aggregates['evening_commits'] = weekly_aggregates['evening_commits'].astype('int8')
            weekly_aggregates['night_commits'] = weekly_aggregates['night_commits'].astype('int8')
            weekly_aggregates['files_touched_total'] = weekly_aggregates['files_touched_total'].astype('int16')
            weekly_aggregates['unique_commit_days'] = weekly_aggregates['unique_commit_days'].astype('int8')
            
            # Convert float columns to float32
            float_columns = [
                'avg_commit_size', 'commit_frequency', 'avg_files_per_commit',
                'avg_message_length', 'code_churn', 'commit_size_variance',
                'avg_insertions_per_commit', 'avg_deletions_per_commit',
                'churn_ratio_avg', 'commit_spread_hours', 'message_word_count_avg'
            ]
            for col in float_columns:
                weekly_aggregates[col] = weekly_aggregates[col].astype('float32')
            
            # Save project results
            project_output_file = self.output_dir / 'project_results' / f"{project_name.replace('/', '_')}_aggregates.csv"
            weekly_aggregates.to_csv(project_output_file, index=False)
            
            # Update checkpoint
            self.checkpoint['total_records_created'] += len(weekly_aggregates)
            
            self.logger.info(f"Saved {len(weekly_aggregates)} weekly aggregate records for {project_name}")
            
            # Clean up memory
            del project_df, weekly_aggregates
            gc.collect()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process project {project_name}: {str(e)}")
            return False
    
    def consolidate_results(self):
        """Consolidate all project results into final dataset."""
        self.logger.info("Consolidating all project results...")
        
        project_files = list((self.output_dir / 'project_results').glob('*_aggregates.csv'))
        if not project_files:
            self.logger.error("No project result files found!")
            return
        
        # Process files in batches to manage memory
        batch_size = 50
        all_batches = []
        
        for i in range(0, len(project_files), batch_size):
            batch_files = project_files[i:i+batch_size]
            batch_data = []
            
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(project_files)-1)//batch_size + 1}")
            
            for file_path in tqdm(batch_files, desc="Loading batch"):
                try:
                    df = pd.read_csv(file_path)
                    batch_data.append(df)
                except Exception as e:
                    self.logger.error(f"Error reading {file_path}: {str(e)}")
            
            if batch_data:
                batch_df = pd.concat(batch_data, ignore_index=True)
                
                # Save batch to temp file
                batch_file = self.output_dir / f'temp_batch_{i//batch_size}.csv'
                batch_df.to_csv(batch_file, index=False)
                all_batches.append(batch_file)
                
                del batch_data, batch_df
                gc.collect()
        
        # Final consolidation
        self.logger.info("Final consolidation of batches...")
        final_data = []
        
        for batch_file in all_batches:
            df = pd.read_csv(batch_file)
            final_data.append(df)
        
        final_dataset = pd.concat(final_data, ignore_index=True)
        final_dataset = final_dataset.sort_values(['project_name', 'contributor_email', 'week_date'])
        
        # Save final dataset
        output_file = self.output_dir / 'weekly_commit_aggregates.csv'
        final_dataset.to_csv(output_file, index=False)
        
        # Clean up temp files
        for batch_file in all_batches:
            batch_file.unlink()
        
        # Generate summary statistics
        self.generate_summary_stats(final_dataset)
        
        self.logger.info(f"Final dataset saved: {output_file}")
        self.logger.info(f"Total records: {len(final_dataset)}")
        
    def generate_summary_stats(self, df: pd.DataFrame):
        """Generate and save summary statistics."""
        # Calculate statistics for non-zero weeks (weeks with commits)
        active_weeks = df[df['commit_frequency'] > 0]
        
        stats = {
            'total_records': len(df),
            'active_contributor_weeks': len(active_weeks),
            'unique_projects': df['project_name'].nunique(),
            'unique_contributors': df['contributor_email'].nunique(),
            'avg_commits_per_active_week': active_weeks['commit_frequency'].mean() * 7 if len(active_weeks) > 0 else 0,
            'avg_commit_size': active_weeks['avg_commit_size'].mean() if len(active_weeks) > 0 else 0,
            'avg_files_per_commit': active_weeks['avg_files_per_commit'].mean() if len(active_weeks) > 0 else 0,
            'weekend_vs_weekday_ratio': (active_weeks['weekend_commits'].sum() / active_weeks['weekday_commits'].sum()) if active_weeks['weekday_commits'].sum() > 0 else 0,
            'temporal_patterns': {
                'morning_commits': int(df['morning_commits'].sum()),
                'afternoon_commits': int(df['afternoon_commits'].sum()),
                'evening_commits': int(df['evening_commits'].sum()),
                'night_commits': int(df['night_commits'].sum())
            },
            'feature_distributions': {
                'avg_commit_size_stats': {
                    'mean': float(active_weeks['avg_commit_size'].mean()) if len(active_weeks) > 0 else 0,
                    'median': float(active_weeks['avg_commit_size'].median()) if len(active_weeks) > 0 else 0,
                    'std': float(active_weeks['avg_commit_size'].std()) if len(active_weeks) > 0 else 0
                },
                'commit_frequency_stats': {
                    'mean': float(active_weeks['commit_frequency'].mean()) if len(active_weeks) > 0 else 0,
                    'median': float(active_weeks['commit_frequency'].median()) if len(active_weeks) > 0 else 0,
                    'std': float(active_weeks['commit_frequency'].std()) if len(active_weeks) > 0 else 0
                }
            },
            'processing_date': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info("Summary statistics saved")
    
    def run(self):
        """Main execution function."""
        self.logger.info("Starting Weekly Commit Aggregates Analysis")
        
        # Get list of all projects
        projects = self.get_project_list()
        remaining_projects = [p for p in projects if p not in self.checkpoint['completed_projects']]
        
        self.logger.info(f"Total projects: {len(projects)}")
        self.logger.info(f"Completed projects: {len(self.checkpoint['completed_projects'])}")
        self.logger.info(f"Remaining projects: {len(remaining_projects)}")
        self.logger.info(f"Total records created so far: {self.checkpoint['total_records_created']}")
        
        if not remaining_projects:
            self.logger.info("All projects already processed. Consolidating results...")
            self.consolidate_results()
            return
        
        # Process remaining projects
        for i, project_name in enumerate(tqdm(remaining_projects, desc="Processing projects")):
            success = self.process_project(project_name)
            
            if success:
                self.checkpoint['completed_projects'].append(project_name)
                self.checkpoint['total_processed'] += 1
            else:
                self.checkpoint['failed_projects'].append(project_name)
            
            # Save checkpoint at specified frequency
            if (i + 1) % self.config['save_frequency'] == 0:
                self.save_checkpoint()
                self.logger.info(f"Checkpoint saved. Progress: {len(self.checkpoint['completed_projects'])}/{len(projects)} projects")
        
        # Final checkpoint save
        self.save_checkpoint()
        
        # Consolidate all results
        self.consolidate_results()
        
        self.logger.info("Weekly Commit Aggregates Analysis completed!")
        self.logger.info(f"Successfully processed: {len(self.checkpoint['completed_projects'])} projects")
        self.logger.info(f"Failed projects: {len(self.checkpoint['failed_projects'])}")
        self.logger.info(f"Total records created: {self.checkpoint['total_records_created']}")

if __name__ == "__main__":
    processor = WeeklyCommitAggregatesProcessor(CONFIG)
    processor.run()





    #!/usr/bin/env python3
"""
RQ1 Master Pipeline Execution Script
===================================

Orchestrates the execution of all RQ1 data processing steps for contributor evolution analysis.
Can run steps individually or as a complete pipeline with dependency management.

Author: Research Team
Date: August 2025
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import json
from datetime import datetime
import logging

# Pipeline configuration
PIPELINE_STEPS = {
    'step5': {
        'name': 'Project Core Timeline Weekly',
        'script': 'project_core_timeline_weekly.py',
        'description': 'Track who is core in each project at each week',
        'estimated_time': '10-15 minutes',
        'output': 'project_core_timeline_weekly.csv',
        'dependencies': []
    },
    'step6': {
        'name': 'Contributor Activity Weekly', 
        'script': 'contributor_activity_weekly.py',
        'description': 'Track each contributor\'s activity week by week',
        'estimated_time': '6-12 hours',
        'output': 'contributor_activity_weekly.csv',
        'dependencies': []
    },
    'step7': {
        'name': 'Contributor Transitions',
        'script': 'contributor_transitions.py', 
        'description': 'Summary of each contributor\'s journey to core',
        'estimated_time': '30-60 minutes',
        'output': 'contributor_transitions.csv',
        'dependencies': ['step6']  # Optional dependency on step6
    },
    'step8': {
        'name': 'Weekly Commit Aggregates',
        'script': 'weekly_commit_aggregates.py',
        'description': 'ML features per contributor per week',
        'estimated_time': '4-8 hours', 
        'output': 'weekly_commit_aggregates.csv',
        'dependencies': []
    }
}

class RQ1PipelineManager:
    def __init__(self):
        self.setup_logging()
        self.scripts_dir = Path(__file__).parent
        self.start_time = datetime.now()
        
    def setup_logging(self):
        """Configure logging for pipeline execution."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rq1_pipeline_execution.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def check_dependencies(self, step_id: str) -> bool:
        """
        Check if all dependencies for a step are satisfied.
        
        Args:
            step_id: ID of the step to check
            
        Returns:
            bool: True if dependencies are satisfied
        """
        step_info = PIPELINE_STEPS[step_id]
        dependencies = step_info.get('dependencies', [])
        
        if not dependencies:
            return True
            
        for dep_step in dependencies:
            dep_info = PIPELINE_STEPS[dep_step]
            output_dir = f"RQ1_transition_rates_and_speeds/data_mining/{dep_step}_{dep_info['name'].lower().replace(' ', '_')}"
            output_file = Path(output_dir) / dep_info['output']
            
            if not output_file.exists():
                self.logger.warning(f"Dependency {dep_step} output not found: {output_file}")
                self.logger.info(f"Step {step_id} can still run but may take longer without pre-computed data")
                
        return True
    
    def get_step_status(self, step_id: str) -> dict:
        """
        Get the current status of a pipeline step.
        
        Args:
            step_id: ID of the step to check
            
        Returns:
            dict: Status information
        """
        step_info = PIPELINE_STEPS[step_id]
        script_name = step_info['script']
        
        # Check if output exists
        output_dir = f"RQ1_transition_rates_and_speeds/data_mining/{step_id}_{step_info['name'].lower().replace(' ', '_')}"
        output_file = Path(output_dir) / step_info['output']
        checkpoint_file = Path(output_dir) / f"{step_id}_checkpoint.json"
        
        status = {
            'step_id': step_id,
            'name': step_info['name'],
            'completed': output_file.exists(),
            'checkpoint_exists': checkpoint_file.exists(),
            'output_file': str(output_file),
            'checkpoint_file': str(checkpoint_file)
        }
        
        # Load checkpoint if it exists
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                status['progress'] = {
                    'completed_projects': len(checkpoint.get('completed_projects', [])),
                    'failed_projects': len(checkpoint.get('failed_projects', [])),
                    'last_updated': checkpoint.get('last_updated'),
                    'total_processed': checkpoint.get('total_processed', 0)
                }
            except Exception as e:
                self.logger.error(f"Error reading checkpoint for {step_id}: {str(e)}")
                
        return status
    
    def print_pipeline_status(self):
        """Print current status of all pipeline steps."""
        self.logger.info("=" * 80)
        self.logger.info("RQ1 PIPELINE STATUS")
        self.logger.info("=" * 80)
        
        for step_id, step_info in PIPELINE_STEPS.items():
            status = self.get_step_status(step_id)
            
            status_icon = "✅" if status['completed'] else "⏳" if status['checkpoint_exists'] else "❌"
            
            self.logger.info(f"{status_icon} {step_id.upper()}: {step_info['name']}")
            self.logger.info(f"   Description: {step_info['description']}")
            self.logger.info(f"   Estimated Time: {step_info['estimated_time']}")
            
            if status['checkpoint_exists'] and not status['completed']:
                progress = status.get('progress', {})
                completed = progress.get('completed_projects', 0)
                failed = progress.get('failed_projects', 0)
                self.logger.info(f"   Progress: {completed} completed, {failed} failed projects")
                
            if status['completed']:
                self.logger.info(f"   Output: {status['output_file']}")
                
            self.logger.info("")
            
        self.logger.info("=" * 80)
    
    def run_step(self, step_id: str, force: bool = False) -> bool:
        """
        Execute a single pipeline step.
        
        Args:
            step_id: ID of the step to run
            force: Force execution even if already completed
            
        Returns:
            bool: True if successful
        """
        if step_id not in PIPELINE_STEPS:
            self.logger.error(f"Unknown step: {step_id}")
            return False
            
        step_info = PIPELINE_STEPS[step_id]
        status = self.get_step_status(step_id)
        
        # Check if already completed
        if status['completed'] and not force:
            self.logger.info(f"Step {step_id} already completed. Use --force to re-run.")
            return True
            
        # Check dependencies
        if not self.check_dependencies(step_id):
            self.logger.error(f"Dependencies not satisfied for step {step_id}")
            return False
            
        script_path = self.scripts_dir / step_info['script']
        if not script_path.exists():
            self.logger.error(f"Script not found: {script_path}")
            return False
            
        self.logger.info(f"Starting step {step_id}: {step_info['name']}")
        self.logger.info(f"Estimated time: {step_info['estimated_time']}")
        self.logger.info(f"Script: {script_path}")
        
        step_start_time = time.time()
        
        try:
            # Execute the script
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, timeout=None)
            
            step_duration = time.time() - step_start_time
            
            if result.returncode == 0:
                self.logger.info(f"✅ Step {step_id} completed successfully in {step_duration/3600:.2f} hours")
                self.logger.info("Script output (last 10 lines):")
                for line in result.stdout.split('\n')[-10:]:
                    if line.strip():
                        self.logger.info(f"  {line}")
                return True
            else:
                self.logger.error(f"❌ Step {step_id} failed with return code {result.returncode}")
                self.logger.error("Error output:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        self.logger.error(f"  {line}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Step {step_id} timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error executing step {step_id}: {str(e)}")
            return False
    
    def run_pipeline(self, steps: list = None, force: bool = False):
        """
        Run the complete pipeline or specific steps.
        
        Args:
            steps: List of step IDs to run (None for all)
            force: Force execution even if already completed
        """
        if steps is None:
            steps = list(PIPELINE_STEPS.keys())
            
        self.logger.info("🚀 Starting RQ1 Pipeline Execution")
        self.logger.info(f"Steps to execute: {', '.join(steps)}")
        
        # Print initial status
        self.print_pipeline_status()
        
        results = {}
        
        for step_id in steps:
            if step_id not in PIPELINE_STEPS:
                self.logger.error(f"Unknown step: {step_id}")
                continue
                
            success = self.run_step(step_id, force)
            results[step_id] = success
            
            if not success:
                self.logger.error(f"Pipeline stopped due to failure in step {step_id}")
                break
                
        # Final status
        total_duration = time.time() - self.start_time.timestamp()
        
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        
        for step_id, success in results.items():
            status_icon = "✅" if success else "❌"
            self.logger.info(f"{status_icon} {step_id}: {PIPELINE_STEPS[step_id]['name']}")
            
        self.logger.info(f"\nTotal execution time: {total_duration/3600:.2f} hours")
        
        # Print final status
        self.print_pipeline_status()
        
        return all(results.values())

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='RQ1 Contributor Evolution Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_rq1_pipeline.py --status                    # Check pipeline status
  python run_rq1_pipeline.py --run-all                   # Run complete pipeline  
  python run_rq1_pipeline.py --step step5                # Run specific step
  python run_rq1_pipeline.py --step step6 --force        # Force re-run step
  python run_rq1_pipeline.py --steps step5,step7         # Run multiple steps
        """
    )
    
    parser.add_argument('--status', action='store_true',
                       help='Show current pipeline status')
    parser.add_argument('--run-all', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--step', type=str,
                       help='Run specific step (step5, step6, step7, step8)')
    parser.add_argument('--steps', type=str,
                       help='Run multiple steps (comma-separated)')
    parser.add_argument('--force', action='store_true',
                       help='Force execution even if already completed')
    
    args = parser.parse_args()
    
    # Initialize pipeline manager
    manager = RQ1PipelineManager()
    
    # Show status
    if args.status:
        manager.print_pipeline_status()
        return
    
    # Determine steps to run
    steps_to_run = None
    
    if args.run_all:
        steps_to_run = list(PIPELINE_STEPS.keys())
    elif args.step:
        steps_to_run = [args.step]
    elif args.steps:
        steps_to_run = [s.strip() for s in args.steps.split(',')]
    else:
        # No action specified, show help
        parser.print_help()
        return
    
    # Validate steps
    invalid_steps = [s for s in steps_to_run if s not in PIPELINE_STEPS]
    if invalid_steps:
        print(f"Error: Invalid steps: {', '.join(invalid_steps)}")
        print(f"Valid steps: {', '.join(PIPELINE_STEPS.keys())}")
        return
    
    # Run pipeline
    success = manager.run_pipeline(steps_to_run, args.force)
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()