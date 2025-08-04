#!/usr/bin/env python3
"""
Repository Cloning Script for RQ1 Data Preparation
=================================================

This script clones all 375 projects from the final_balanced_dataset.csv for 
comprehensive commit analysis in the newcomer transition study.

Features:
- Progress bar with real-time statistics
- Resume capability (skips already cloned repos)
- Error handling and detailed logging
- Organized directory structure
- Progress tracking in JSON format

Usage:
    python clone_all_projects.py [--force-reclone] [--github-token TOKEN]

Author: FSE 2026 Research Project
"""

import os
import sys
import csv
import json
import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

class RepositoryCloner:
    def __init__(self, base_dir=None, github_token=None):
        """Initialize the repository cloner."""
        if base_dir is None:
            base_dir = Path(__file__).parent
        
        self.base_dir = Path(base_dir)
        self.dataset_path = self.base_dir.parent.parent.parent.parent / "preparing_dataset" / "data" / "final_balanced_dataset.csv"
        self.clone_dir = self.base_dir / "cloned_repositories"
        self.progress_file = self.base_dir / "clone_progress.json"
        self.log_file = self.base_dir / "clone_log.txt"
        
        self.github_token = github_token
        self.progress_data = self.load_progress()
        
        # Ensure directories exist
        self.clone_dir.mkdir(exist_ok=True)
        
    def load_progress(self):
        """Load existing progress or create new progress tracking."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "total_projects": 0,
                "completed": [],
                "failed": [],
                "skipped": [],
                "start_time": None,
                "last_update": None
            }
    
    def save_progress(self):
        """Save current progress to JSON file."""
        self.progress_data["last_update"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress_data, f, indent=2)
    
    def log_message(self, message, level="INFO"):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")
    
    def read_dataset(self):
        """Read and parse the final_balanced_dataset.csv file."""
        projects = []
        try:
            with open(self.dataset_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    project_name = row['project_name']
                    project_type = row['type']
                    projects.append({
                        'name': project_name,
                        'type': project_type,
                        'owner': project_name.split('/')[0],
                        'repo': project_name.split('/')[1]
                    })
            
            self.log_message(f"Successfully loaded {len(projects)} projects from dataset")
            return projects
            
        except FileNotFoundError:
            self.log_message(f"Dataset file not found: {self.dataset_path}", "ERROR")
            sys.exit(1)
        except Exception as e:
            self.log_message(f"Error reading dataset: {str(e)}", "ERROR")
            sys.exit(1)
    
    def get_clone_url(self, project_name):
        """Generate GitHub clone URL with optional token authentication."""
        if self.github_token:
            return f"https://{self.github_token}@github.com/{project_name}.git"
        else:
            return f"https://github.com/{project_name}.git"
    
    def clone_repository(self, project):
        """Clone a single repository with error handling."""
        project_name = project['name']
        safe_name = project_name.replace('/', '_')
        clone_path = self.clone_dir / safe_name
        
        # Skip if already cloned and not forcing reclone
        if clone_path.exists() and project_name in self.progress_data["completed"]:
            self.progress_data["skipped"].append(project_name)
            return "skipped"
        
        try:
            clone_url = self.get_clone_url(project_name)
            
            # Remove existing directory if it exists but was incomplete
            if clone_path.exists():
                import shutil
                shutil.rmtree(clone_path)
            
            # Clone the repository
            result = subprocess.run([
                'git', 'clone', '--quiet', clone_url, str(clone_path)
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                self.progress_data["completed"].append(project_name)
                return "success"
            else:
                error_msg = result.stderr.strip()
                self.log_message(f"Failed to clone {project_name}: {error_msg}", "WARNING")
                self.progress_data["failed"].append({
                    "project": project_name,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                return "failed"
                
        except subprocess.TimeoutExpired:
            self.log_message(f"Timeout cloning {project_name}", "WARNING")
            self.progress_data["failed"].append({
                "project": project_name,
                "error": "Timeout (>5 minutes)",
                "timestamp": datetime.now().isoformat()
            })
            return "failed"
            
        except Exception as e:
            self.log_message(f"Exception cloning {project_name}: {str(e)}", "WARNING")
            self.progress_data["failed"].append({
                "project": project_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return "failed"
    
    def clone_all_repositories(self, force_reclone=False):
        """Clone all repositories with progress tracking."""
        projects = self.read_dataset()
        self.progress_data["total_projects"] = len(projects)
        
        if self.progress_data["start_time"] is None:
            self.progress_data["start_time"] = datetime.now().isoformat()
        
        # Filter projects if resuming
        if not force_reclone:
            completed_set = set(self.progress_data["completed"])
            failed_set = {item["project"] if isinstance(item, dict) else item 
                         for item in self.progress_data["failed"]}
            projects_to_clone = [p for p in projects 
                               if p['name'] not in completed_set and p['name'] not in failed_set]
        else:
            projects_to_clone = projects
            # Reset progress if force reclone
            self.progress_data["completed"] = []
            self.progress_data["failed"] = []
            self.progress_data["skipped"] = []
        
        self.log_message(f"Starting to clone {len(projects_to_clone)} repositories")
        self.log_message(f"Total projects: {len(projects)}")
        self.log_message(f"Already completed: {len(self.progress_data['completed'])}")
        self.log_message(f"Previously failed: {len(self.progress_data['failed'])}")
        
        # Progress bar setup
        progress_bar = tqdm(
            projects_to_clone,
            desc="Cloning repositories",
            unit="repo",
            ncols=100,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} repos [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        success_count = len(self.progress_data["completed"])
        failed_count = len(self.progress_data["failed"])
        skipped_count = len(self.progress_data["skipped"])
        
        for project in progress_bar:
            result = self.clone_repository(project)
            
            if result == "success":
                success_count += 1
            elif result == "failed":
                failed_count += 1
            elif result == "skipped":
                skipped_count += 1
            
            # Update progress bar description with statistics
            progress_bar.set_description(
                f"Cloning repos [OK:{success_count} FAIL:{failed_count} SKIP:{skipped_count}]"
            )
            
            # Save progress every 10 repositories
            if (success_count + failed_count + skipped_count) % 10 == 0:
                self.save_progress()
        
        # Final save
        self.save_progress()
        
        # Summary
        total_processed = success_count + failed_count + skipped_count
        self.log_message("\n" + "="*60)
        self.log_message("CLONING SUMMARY")
        self.log_message("="*60)
        self.log_message(f"Total projects in dataset: {len(projects)}")
        self.log_message(f"Successfully cloned: {success_count}")
        self.log_message(f"Failed to clone: {failed_count}")
        self.log_message(f"Skipped (already exists): {skipped_count}")
        self.log_message(f"Total processed: {total_processed}")
        
        if failed_count > 0:
            self.log_message(f"\nFailed projects logged in: {self.progress_file}")
        
        return success_count, failed_count, skipped_count

def main():
    parser = argparse.ArgumentParser(description="Clone all repositories for RQ1 analysis")
    parser.add_argument("--force-reclone", action="store_true", 
                       help="Force reclone all repositories (ignore existing)")
    parser.add_argument("--github-token", type=str,
                       help="GitHub Personal Access Token for higher rate limits")
    
    args = parser.parse_args()
    
    cloner = RepositoryCloner(github_token=args.github_token)
    
    print("FSE 2026 Research Project - Repository Cloning")
    print("=" * 50)
    print(f"Dataset: {cloner.dataset_path}")
    print(f"Clone directory: {cloner.clone_dir}")
    print(f"Progress file: {cloner.progress_file}")
    print("=" * 50)
    
    try:
        success, failed, skipped = cloner.clone_all_repositories(args.force_reclone)
        
        if failed == 0:
            print("\nAll repositories cloned successfully!")
        else:
            print(f"\nCompleted with {failed} failures. Check {cloner.log_file} for details.")
            
    except KeyboardInterrupt:
        print("\n\nCloning interrupted by user. Progress saved.")
        print("   Run the script again to resume from where you left off.")
        cloner.save_progress()
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        cloner.log_message(f"Fatal error: {str(e)}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()