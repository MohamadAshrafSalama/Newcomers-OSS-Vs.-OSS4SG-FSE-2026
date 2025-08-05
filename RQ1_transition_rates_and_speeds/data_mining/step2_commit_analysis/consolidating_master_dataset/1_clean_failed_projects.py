#!/usr/bin/env python3
"""
Clean the mining status dataset by removing failed projects
This creates a clean dataset with only successfully mined projects
"""

import csv
import os

def clean_failed_projects():
    """Remove failed projects from the dataset and create clean version"""
    
    input_file = "projects_mining_status.csv"
    output_file = "clean_projects_mining_status.csv"
    
    print("=" * 70)
    print("CLEANING FAILED PROJECTS FROM DATASET")
    print("=" * 70)
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return False
    
    successful_projects = []
    failed_projects = []
    oss_successful = 0
    oss4sg_successful = 0
    oss_failed = 0
    oss4sg_failed = 0
    
    # Read and filter the dataset
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            project_name = row['project_name']
            project_type = row['type']
            mining_completed = row['mining_completed'].lower() == 'true'
            
            if mining_completed:
                successful_projects.append(row)
                if project_type == 'OSS':
                    oss_successful += 1
                else:
                    oss4sg_successful += 1
            else:
                failed_projects.append(row)
                if project_type == 'OSS':
                    oss_failed += 1
                else:
                    oss4sg_failed += 1
                print(f"REMOVING FAILED: {project_name} ({project_type}) - {row['mining_error']}")
    
    # Write clean dataset
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(successful_projects)
    
    # Print summary
    print(f"\n" + "="*70)
    print("CLEANING SUMMARY")
    print("="*70)
    print(f"Original projects: {len(successful_projects) + len(failed_projects)}")
    print(f"Successfully mined: {len(successful_projects)}")
    print(f"Failed projects removed: {len(failed_projects)}")
    print(f"")
    print(f"OSS projects - Successful: {oss_successful}, Failed: {oss_failed}")
    print(f"OSS4SG projects - Successful: {oss4sg_successful}, Failed: {oss4sg_failed}")
    print(f"")
    print(f"Final dataset balance:")
    print(f"  OSS: {oss_successful} projects")
    print(f"  OSS4SG: {oss4sg_successful} projects")
    print(f"  Ratio: 1:{oss4sg_successful/oss_successful:.2f}")
    print(f"")
    print(f"Clean dataset saved: {output_file}")
    
    return True

if __name__ == "__main__":
    clean_failed_projects()