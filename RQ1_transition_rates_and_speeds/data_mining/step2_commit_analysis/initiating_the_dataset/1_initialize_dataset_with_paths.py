#!/usr/bin/env python3
"""
Step 1: Initialize dataset with project info and absolute paths to cloned repositories
This script reads the original dataset and creates a new CSV with absolute paths to cloned repos
"""

import csv
import os
import sys

def create_dataset_with_paths():
    """Create new dataset CSV with project info and absolute paths"""
    
    # Input: Original cleaned dataset 
    input_dataset = "/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026/preparing_dataset/data/final_clean_cloned_dataset.csv"
    
    # Output: New dataset with absolute paths (in this folder)
    output_dataset = "projects_with_absolute_paths.csv"
    
    # Base path to cloned repositories
    cloned_repos_base = "/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026/RQ1_transition_rates_and_speeds/data_mining/step1_repository_cloning/cloned_repositories"
    
    print("Creating dataset with absolute paths...")
    print(f"Reading from: {input_dataset}")
    print(f"Writing to: {output_dataset}")
    print(f"Cloned repos base: {cloned_repos_base}")
    
    if not os.path.exists(input_dataset):
        print(f"ERROR: Input dataset not found: {input_dataset}")
        return False
    
    if not os.path.exists(cloned_repos_base):
        print(f"ERROR: Cloned repositories directory not found: {cloned_repos_base}")
        return False
    
    projects_found = 0
    projects_missing = 0
    
    try:
        with open(input_dataset, 'r') as infile, open(output_dataset, 'w', newline='') as outfile:
            reader = csv.DictReader(infile)
            
            # Create output CSV with additional columns
            fieldnames = list(reader.fieldnames) + ['absolute_repo_path', 'repo_exists']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                project_name = row['project_name']  # e.g., "5calls/ios"
                project_type = row['type']
                
                # Convert project name to safe directory name
                safe_name = project_name.replace('/', '_')
                
                # Create absolute path to cloned repository
                absolute_path = os.path.join(cloned_repos_base, safe_name)
                
                # Check if repository exists
                repo_exists = os.path.exists(absolute_path) and os.path.exists(os.path.join(absolute_path, '.git'))
                
                if repo_exists:
                    projects_found += 1
                else:
                    projects_missing += 1
                    print(f"WARNING: Repository not found: {safe_name} ({project_name})")
                
                # Add new columns to row
                row['absolute_repo_path'] = absolute_path
                row['repo_exists'] = repo_exists
                
                # Write updated row
                writer.writerow(row)
        
        print(f"\nDataset created successfully!")
        print(f"Projects found: {projects_found}")
        print(f"Projects missing: {projects_missing}")
        print(f"Total projects: {projects_found + projects_missing}")
        
        return True
        
    except Exception as e:
        print(f"ERROR creating dataset: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: INITIALIZING DATASET WITH ABSOLUTE PATHS")
    print("=" * 70)
    
    success = create_dataset_with_paths()
    
    if success:
        print("\nInitialization completed successfully!")
        print("Next step: Use the 'projects_with_absolute_paths.csv' for commit mining")
    else:
        print("\nInitialization failed!")
        sys.exit(1)