#!/usr/bin/env python3
"""
Main script to mine commit data from all projects
This script loops through the dataset, extracts commits, and tracks completion status
"""

import csv
import os
import sys
import json
from datetime import datetime
from tqdm import tqdm
from extract_single_project_commits import extract_commits_from_single_project

def create_updated_dataset_with_status():
    """Create updated dataset with mining status columns"""
    
    # Input dataset (from initiating_the_dataset folder)
    input_dataset = "../initiating_the_dataset/projects_with_absolute_paths.csv"
    
    # Output dataset with mining status (in this folder)
    output_dataset = "projects_mining_status.csv"
    
    print(f"Reading dataset: {input_dataset}")
    print(f"Creating status dataset: {output_dataset}")
    
    if not os.path.exists(input_dataset):
        print(f"ERROR: Input dataset not found: {input_dataset}")
        return False
    
    try:
        with open(input_dataset, 'r') as infile, open(output_dataset, 'w', newline='') as outfile:
            reader = csv.DictReader(infile)
            
            # Add mining status columns
            fieldnames = list(reader.fieldnames) + ['mining_completed', 'mining_date', 'commit_count', 'json_file_path', 'mining_error']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                # Initialize mining status columns
                row['mining_completed'] = False
                row['mining_date'] = ''
                row['commit_count'] = 0
                row['json_file_path'] = ''
                row['mining_error'] = ''
                
                writer.writerow(row)
        
        print("Status dataset created successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR creating status dataset: {str(e)}")
        return False

def mine_all_projects():
    """Mine commits from all projects and update status"""
    
    status_dataset = "projects_mining_status.csv"
    
    if not os.path.exists(status_dataset):
        print("Creating status dataset first...")
        if not create_updated_dataset_with_status():
            return False
    
    print("Starting commit mining for all projects...")
    
    # Read all projects
    projects = []
    with open(status_dataset, 'r') as f:
        reader = csv.DictReader(f)
        projects = list(reader)
    
    total_projects = len(projects)
    completed = 0
    failed = 0
    
    print(f"Total projects to mine: {total_projects}")
    
    # Process each project with progress bar
    progress_bar = tqdm(
        enumerate(projects, 1), 
        total=total_projects,
        desc="Mining commits",
        unit=" project",
        ncols=120,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} projects | ETA: {remaining} | Speed: {rate_fmt}"
    )
    
    for i, project in progress_bar:
        project_name = project['project_name']
        project_path = project['absolute_repo_path']
        repo_exists = project['repo_exists'].lower() == 'true'
        already_completed = project['mining_completed'].lower() == 'true'
        
        progress_bar.set_description(f"Mining {project_name}")
        
        # Skip if already completed
        if already_completed:
            completed += 1
            continue
        
        # Skip if repository doesn't exist
        if not repo_exists:
            project['mining_completed'] = False
            project['mining_error'] = 'Repository not found'
            failed += 1
            continue
        
        # Extract commits
        success, json_file, error = extract_commits_from_single_project(project_path, project_name)
        
        # Update project status
        project['mining_completed'] = success
        project['mining_date'] = datetime.now().isoformat()
        
        if success:
            # Get commit count from JSON
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    project['commit_count'] = data['total_commits']
            except:
                project['commit_count'] = 0
            
            project['json_file_path'] = json_file
            project['mining_error'] = ''
            completed += 1
        else:
            project['commit_count'] = 0
            project['json_file_path'] = ''
            project['mining_error'] = error
            failed += 1
        
        # Save progress every 10 projects
        if i % 10 == 0:
            save_progress(projects, status_dataset)
    
    # Close progress bar
    progress_bar.close()
    
    # Final save
    save_progress(projects, status_dataset)
    
    print(f"\n" + "="*70)
    print("MINING SUMMARY")
    print("="*70)
    print(f"Total projects: {total_projects}")
    print(f"Successfully completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {completed/total_projects*100:.1f}%")
    print(f"Updated dataset: {status_dataset}")
    
    return True

def save_progress(projects, output_file):
    """Save current progress to CSV"""
    try:
        with open(output_file, 'w', newline='') as f:
            if projects:
                fieldnames = projects[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(projects)
    except Exception as e:
        print(f"ERROR saving progress: {str(e)}")

def test_single_project():
    """Test the extraction on a single project first"""
    
    print("=" * 70)
    print("TESTING ON SINGLE PROJECT FIRST")
    print("=" * 70)
    
    # Read the first project from dataset for testing
    input_dataset = "../initiating_the_dataset/projects_with_absolute_paths.csv"
    
    if not os.path.exists(input_dataset):
        print(f"ERROR: Dataset not found: {input_dataset}")
        return False
    
    with open(input_dataset, 'r') as f:
        reader = csv.DictReader(f)
        first_project = next(reader)
    
    project_name = first_project['project_name']
    project_path = first_project['absolute_repo_path']
    
    print(f"Testing on project: {project_name}")
    
    success, json_file, error = extract_commits_from_single_project(project_path, project_name)
    
    if success:
        print(f"\nTEST SUCCESSFUL!")
        print(f"JSON file created: {json_file}")
        
        # Show some stats
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                print(f"Total commits extracted: {data['total_commits']}")
                if data['commits']:
                    print(f"First commit hash: {data['commits'][0]['hash'][:8]}...")
        except:
            pass
        
        return True
    else:
        print(f"\nTEST FAILED: {error}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        if test_single_project():
            print("\nTest successful! You can now run the full mining with:")
            print("python mine_all_projects.py")
        else:
            print("\nTest failed! Please fix issues before running full mining.")
    else:
        # Full mining mode
        print("=" * 70)
        print("MINING COMMITS FROM ALL PROJECTS")
        print("=" * 70)
        mine_all_projects()