#!/usr/bin/env python3
"""
Consolidate all JSON commit files into one master CSV with metrics
This script reads all JSON files and creates a comprehensive dataset
"""

import csv
import json
import os
import pandas as pd
import re
from datetime import datetime
from tqdm import tqdm

class CommitConsolidator:
    def __init__(self):
        self.clean_dataset = "clean_projects_mining_status.csv"
        self.output_csv = "master_commits_dataset.csv"
        self.failed_projects = []
        
    def calculate_commit_metrics(self, commit, project_name, project_type):
        """Calculate ONLY objective metrics for a single commit"""
        
        # Basic commit info
        metrics = {
            'project_name': project_name,
            'project_type': project_type,
            'commit_hash': commit.get('hash', ''),
            'author_name': commit.get('author_name', ''),
            'author_email': commit.get('author_email', ''),
            'commit_date': commit.get('date', ''),
            'commit_message': commit.get('message', '')
        }
        
        # Message metrics (objective only)
        message = commit.get('message', '')
        metrics['message_length_chars'] = len(message)
        metrics['message_length_words'] = len(message.split()) if message else 0
        
        # File change metrics (objective only)
        files_changed = commit.get('files_changed', [])
        metrics['files_modified_count'] = len(files_changed)
        
        total_insertions = 0
        total_deletions = 0
        
        for file_change in files_changed:
            try:
                insertions = int(file_change.get('insertions', 0))
                deletions = int(file_change.get('deletions', 0))
                total_insertions += insertions
                total_deletions += deletions
            except (ValueError, TypeError):
                # Handle cases where insertions/deletions are not valid numbers
                pass
        
        metrics['total_insertions'] = total_insertions
        metrics['total_deletions'] = total_deletions
        metrics['total_lines_changed'] = total_insertions + total_deletions
        metrics['churn_ratio'] = total_deletions / max(total_insertions, 1)  # Avoid division by zero
        
        # Temporal metrics (objective only)
        try:
            dt = pd.to_datetime(commit.get('date', ''))
            metrics['commit_hour'] = dt.hour
            metrics['commit_day_of_week'] = dt.weekday()  # Monday=0, Sunday=6
            metrics['commit_day_of_month'] = dt.day
            metrics['commit_day_of_year'] = dt.dayofyear
            metrics['commit_month'] = dt.month
            metrics['commit_year'] = dt.year
            metrics['commit_is_weekend'] = dt.weekday() >= 5  # Saturday=5, Sunday=6
        except:
            metrics['commit_hour'] = None
            metrics['commit_day_of_week'] = None
            metrics['commit_day_of_month'] = None
            metrics['commit_day_of_year'] = None
            metrics['commit_month'] = None
            metrics['commit_year'] = None
            metrics['commit_is_weekend'] = None
        
        return metrics
    
    def process_project_json(self, project_row):
        """Process a single project's JSON file and extract commits with metrics"""
        
        project_name = project_row['project_name']
        project_type = project_row['type']
        json_file_path = project_row['json_file_path']
        
        if not os.path.exists(json_file_path):
            self.failed_projects.append(f"{project_name}: JSON file not found")
            return []
        
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            commits_with_metrics = []
            commits = data.get('commits', [])
            
            for commit in commits:
                metrics = self.calculate_commit_metrics(commit, project_name, project_type)
                commits_with_metrics.append(metrics)
            
            return commits_with_metrics
            
        except Exception as e:
            self.failed_projects.append(f"{project_name}: Error reading JSON - {str(e)}")
            return []
    
    def consolidate_all_commits(self):
        """Main method to consolidate all commits into master CSV"""
        
        print("=" * 70)
        print("CONSOLIDATING ALL COMMITS INTO MASTER DATASET")
        print("=" * 70)
        
        if not os.path.exists(self.clean_dataset):
            print(f"ERROR: Clean dataset not found: {self.clean_dataset}")
            return False
        
        # Read clean dataset
        projects = []
        with open(self.clean_dataset, 'r') as f:
            reader = csv.DictReader(f)
            projects = [row for row in reader]
        
        print(f"Processing {len(projects)} successfully mined projects...")
        
        all_commits = []
        total_commits = 0
        
        # Process each project with progress bar
        progress_bar = tqdm(
            projects,
            desc="Consolidating commits",
            unit=" project",
            ncols=120,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} projects | ETA: {remaining} | Speed: {rate_fmt}"
        )
        
        for project_row in progress_bar:
            project_name = project_row['project_name']
            progress_bar.set_description(f"Processing {project_name}")
            
            project_commits = self.process_project_json(project_row)
            all_commits.extend(project_commits)
            total_commits += len(project_commits)
        
        progress_bar.close()
        
        # Save to CSV
        if all_commits:
            print(f"\nSaving {total_commits} commits to CSV...")
            
            df = pd.DataFrame(all_commits)
            df.to_csv(self.output_csv, index=False)
            
            print(f"\n" + "="*70)
            print("CONSOLIDATION SUMMARY")
            print("="*70)
            print(f"Total projects processed: {len(projects)}")
            print(f"Total commits consolidated: {total_commits}")
            print(f"Average commits per project: {total_commits/len(projects):.1f}")
            print(f"Failed projects: {len(self.failed_projects)}")
            
            if self.failed_projects:
                print(f"\nFailed projects:")
                for failure in self.failed_projects:
                    print(f"  - {failure}")
            
            # Show breakdown by project type
            oss_commits = len([c for c in all_commits if c['project_type'] == 'OSS'])
            oss4sg_commits = len([c for c in all_commits if c['project_type'] == 'OSS4SG'])
            
            print(f"\nCommit breakdown:")
            print(f"  OSS commits: {oss_commits:,}")
            print(f"  OSS4SG commits: {oss4sg_commits:,}")
            print(f"  Ratio: 1:{oss4sg_commits/oss_commits:.2f}")
            
            print(f"\nMaster dataset saved: {self.output_csv}")
            print(f"Dataset size: {os.path.getsize(self.output_csv) / (1024*1024):.1f} MB")
            
            return True
        else:
            print("ERROR: No commits were successfully processed!")
            return False

if __name__ == "__main__":
    consolidator = CommitConsolidator()
    consolidator.consolidate_all_commits()