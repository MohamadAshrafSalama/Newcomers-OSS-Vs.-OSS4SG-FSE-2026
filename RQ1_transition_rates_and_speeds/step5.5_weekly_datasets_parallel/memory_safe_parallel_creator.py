"""
Memory-safe parallel dataset creator with monitoring and resume capability.
Designed for 16GB RAM systems.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import warnings
import multiprocessing as mp
from tqdm import tqdm
import psutil
import gc
import pickle
import sys

warnings.filterwarnings('ignore')

def get_memory_usage():
    """Get current memory usage in GB."""
    return psutil.Process().memory_info().rss / (1024**3)

def check_memory_safety(threshold_gb=14):
    """Check if we have enough memory to continue."""
    mem = psutil.virtual_memory()
    used_gb = (mem.total - mem.available) / (1024**3)
    if used_gb > threshold_gb:
        print(f"⚠️  Memory usage high: {used_gb:.1f}GB / {mem.total/(1024**3):.1f}GB")
        gc.collect()
        return False
    return True

def process_project_activity_safe(args):
    """Memory-safe version of activity processing."""
    project_name, project_type, project_commits, core_lookup = args
    
    if len(project_commits) == 0:
        return []
    
    try:
        project_commits = project_commits.sort_values('commit_date')
        
        # Get project week range
        min_week = project_commits['week_date'].min()
        max_week = project_commits['week_date'].max()
        week_range = pd.date_range(start=min_week, end=max_week, freq='W-MON')
        
        contributors = project_commits['author_email'].unique()
        
        # Process in chunks if too many contributors
        if len(contributors) > 100:
            # Process contributors in batches
            activity_records = []
            for i in range(0, len(contributors), 50):
                batch = contributors[i:i+50]
                batch_records = process_contributor_batch(
                    batch, project_commits, week_range, 
                    project_name, project_type, core_lookup
                )
                activity_records.extend(batch_records)
                gc.collect()  # Force garbage collection
        else:
            activity_records = process_contributor_batch(
                contributors, project_commits, week_range,
                project_name, project_type, core_lookup
            )
        
        return activity_records
        
    except Exception as e:
        print(f"Error processing {project_name}: {e}")
        return []

def process_contributor_batch(contributors, project_commits, week_range, 
                             project_name, project_type, core_lookup):
    """Process a batch of contributors."""
    activity_records = []
    
    for contributor in contributors:
        contributor_commits = project_commits[
            project_commits['author_email'] == contributor
        ]
        
        first_commit_week = contributor_commits['week_date'].min()
        cumulative_commits = 0
        cumulative_lines = 0
        
        for week_idx, current_week in enumerate(week_range):
            if current_week < first_commit_week:
                continue
            
            week_commits = contributor_commits[
                contributor_commits['week_date'] == current_week
            ]
            
            commits_this_week = len(week_commits)
            
            if commits_this_week > 0:
                commit_hashes = week_commits['commit_hash'].tolist()
                lines_added = int(week_commits['total_insertions'].sum())
                lines_deleted = int(week_commits['total_deletions'].sum())
                files_modified = int(week_commits['files_modified_count'].sum())
            else:
                commit_hashes = []
                lines_added = 0
                lines_deleted = 0
                files_modified = 0
            
            cumulative_commits += commits_this_week
            cumulative_lines += lines_added + lines_deleted
            
            project_commits_to_date = len(project_commits[
                project_commits['week_date'] <= current_week
            ])
            
            contribution_percentage = (cumulative_commits / project_commits_to_date * 100) if project_commits_to_date > 0 else 0
            
            core_list = core_lookup.get((project_name, current_week), [])
            is_core_this_week = contributor in core_list
            
            week_contributors_cumulative = project_commits[
                project_commits['week_date'] <= current_week
            ].groupby('author_email').size().sort_values(ascending=False)
            
            if contributor in week_contributors_cumulative.index:
                rank = week_contributors_cumulative.index.get_loc(contributor) + 1
            else:
                rank = len(week_contributors_cumulative) + 1
            
            weeks_since_first = (current_week - first_commit_week).days // 7
            
            record = {
                'project_name': project_name,
                'project_type': project_type,
                'contributor_email': contributor,
                'week_date': current_week,
                'week_number': week_idx + 1,
                'weeks_since_first_commit': weeks_since_first,
                'commits_this_week': commits_this_week,
                'commit_hashes': json.dumps(commit_hashes),
                'lines_added_this_week': lines_added,
                'lines_deleted_this_week': lines_deleted,
                'files_modified_this_week': files_modified,
                'cumulative_commits': cumulative_commits,
                'cumulative_lines_changed': cumulative_lines,
                'project_commits_to_date': project_commits_to_date,
                'contribution_percentage': round(contribution_percentage, 2),
                'is_core_this_week': is_core_this_week,
                'rank_this_week': rank
            }
            
            activity_records.append(record)
    
    return activity_records

class MemorySafeParallelCreator:
    def __init__(self, commits_csv_path, output_dir, n_cores=4, checkpoint_dir="checkpoints"):
        """
        Use fewer cores (4) for memory safety on 16GB systems.
        """
        self.commits_csv_path = Path(commits_csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_cores = n_cores  # Use 4 cores for 16GB RAM
        
        self.datasets_dir = self.output_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print("="*80)
        print("MEMORY-SAFE PARALLEL DATASET CREATOR")
        print("="*80)
        print(f"System RAM: {psutil.virtual_memory().total/(1024**3):.1f}GB")
        print(f"Using {self.n_cores} CPU cores (reduced for memory safety)")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print("-"*80)
    
    def save_checkpoint(self, data, name):
        """Save checkpoint for recovery."""
        checkpoint_file = self.checkpoint_dir / f"{name}_checkpoint.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Checkpoint saved: {name}")
    
    def load_checkpoint(self, name):
        """Load checkpoint if exists."""
        checkpoint_file = self.checkpoint_dir / f"{name}_checkpoint.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            print(f"📂 Checkpoint loaded: {name}")
            return data
        return None
    
    def create_activity_with_resume(self, timeline_df):
        """Create activity dataset with resume capability."""
        print("\n📊 CREATING DATASET 2: CONTRIBUTOR ACTIVITY (MEMORY-SAFE)")
        print("="*80)
        
        # Check for existing checkpoint
        checkpoint = self.load_checkpoint("activity_progress")
        if checkpoint:
            completed_projects = checkpoint['completed']
            all_activity_records = checkpoint['records']
            print(f"📂 Resuming from checkpoint: {len(completed_projects)}/{len(self.projects)} projects done")
            print(f"📊 Records saved so far: {len(all_activity_records):,}")
        else:
            completed_projects = set()
            all_activity_records = []
            print(f"🚀 Starting fresh: 0/{len(self.projects)} projects done")
        
        # Create core lookup
        print("   Preparing core status lookup...")
        core_lookup = {}
        for _, row in timeline_df.iterrows():
            key = (row['project_name'], pd.Timestamp(row['week_date']))
            core_lookup[key] = json.loads(row['core_contributors_emails'])
        print(f"   ✅ Core lookup prepared for {len(core_lookup)} project-week combinations")
        
        # Process remaining projects
        remaining_projects = [
            row for _, row in self.projects.iterrows() 
            if row['project_name'] not in completed_projects
        ]
        
        if not remaining_projects:
            print("🎉 All projects already processed!")
        else:
            print(f"🔄 Processing {len(remaining_projects)} remaining projects...")
            
            # Process in batches to control memory
            batch_size = 10  # Process 10 projects at a time
            total_batches = (len(remaining_projects) - 1) // batch_size + 1
            
            for i in range(0, len(remaining_projects), batch_size):
                batch = remaining_projects[i:i+batch_size]
                current_batch_num = i // batch_size + 1
                
                print(f"\n📦 Processing batch {current_batch_num}/{total_batches} ({len(batch)} projects)")
                print(f"   Progress: {len(completed_projects)}/{len(self.projects)} projects completed ({len(completed_projects)/len(self.projects)*100:.1f}%)")
                
                # Check memory before processing
                mem = psutil.virtual_memory()
                current_mem = (mem.total - mem.available) / (1024**3)
                print(f"   Memory: {current_mem:.1f}GB / {mem.total/(1024**3):.1f}GB ({current_mem/(mem.total/(1024**3))*100:.1f}%)")
                
                if not check_memory_safety(threshold_gb=14):
                    print("⚠️  Memory limit reached. Saving checkpoint...")
                    self.save_checkpoint({
                        'completed': completed_projects,
                        'records': all_activity_records
                    }, "activity_progress")
                    gc.collect()
                
                # Prepare batch data
                batch_data = []
                for row in batch:
                    project_commits = self.commits_df[
                        self.commits_df['project_name'] == row['project_name']
                    ]
                    project_core_lookup = {
                        k: v for k, v in core_lookup.items() 
                        if k[0] == row['project_name']
                    }
                    batch_data.append((
                        row['project_name'], 
                        row['project_type'], 
                        project_commits, 
                        project_core_lookup
                    ))
                
                print(f"   Processing {len(batch_data)} projects in parallel...")
                
                # Process batch in parallel
                with mp.Pool(processes=self.n_cores) as pool:
                    batch_results = list(tqdm(
                        pool.imap(process_project_activity_safe, batch_data),
                        total=len(batch_data),
                        desc=f"Batch {current_batch_num}/{total_batches}",
                        unit="project"
                    ))
                
                # Collect results
                batch_records = 0
                for idx, project_records in enumerate(batch_results):
                    all_activity_records.extend(project_records)
                    completed_projects.add(batch[idx]['project_name'])
                    batch_records += len(project_records)
                
                print(f"   ✅ Batch {current_batch_num} completed: {batch_records:,} records added")
                print(f"   📊 Total records so far: {len(all_activity_records):,}")
                
                # Save checkpoint every batch
                if current_batch_num % 5 == 0:  # Every 5 batches
                    self.save_checkpoint({
                        'completed': completed_projects,
                        'records': all_activity_records
                    }, "activity_progress")
                    print(f"   💾 Checkpoint saved after batch {current_batch_num}")
                
                # Force garbage collection
                gc.collect()
                
                # Show memory status
                mem = psutil.virtual_memory()
                print(f"   Memory after batch: {(mem.total-mem.available)/(1024**3):.1f}GB / {mem.total/(1024**3):.1f}GB")
        
        # Create final DataFrame
        print(f"\n📊 Creating final DataFrame with {len(all_activity_records):,} records...")
        activity_df = pd.DataFrame(all_activity_records)
        
        # Save to CSV
        output_path = self.datasets_dir / "contributor_activity_weekly.csv"
        activity_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Created contributor_activity_weekly.csv")
        print(f"   Rows: {len(activity_df):,}")
        print(f"   Projects processed: {len(completed_projects)}/{len(self.projects)}")
        
        # Clean up checkpoint
        checkpoint_file = self.checkpoint_dir / "activity_progress_checkpoint.pkl"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("   🧹 Checkpoint cleaned up")
        
        return activity_df
    
    def load_commits_data(self):
        """Load commits data."""
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
        
        self.commits_df['commit_date'] = pd.to_datetime(self.commits_df['commit_date'], utc=True)
        self.commits_df['week_date'] = self.commits_df['commit_date'].dt.to_period('W').dt.start_time
        self.commits_df['author_email'] = self.commits_df['author_email'].str.lower().str.strip()
        
        self.projects = self.commits_df.groupby(['project_name', 'project_type']).size().reset_index()[['project_name', 'project_type']]
        print(f"   Found {len(self.projects)} projects")

if __name__ == "__main__":
    # Configuration for 16GB RAM system
    COMMITS_CSV = "../data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv"
    OUTPUT_DIR = "weekly_datasets_safe"
    N_CORES = 4  # Reduced for memory safety
    
    creator = MemorySafeParallelCreator(COMMITS_CSV, OUTPUT_DIR, n_cores=N_CORES)
    
    # Load data
    creator.load_commits_data()
    
    # Load timeline (assuming it's already created)
    timeline_df = pd.read_csv("weekly_datasets_parallel/datasets/project_core_timeline_weekly.csv")
    
    # Create activity dataset with resume capability
    activity_df = creator.create_activity_with_resume(timeline_df)
    
    print("\n✅ Memory-safe processing complete!") 