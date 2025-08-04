#!/usr/bin/env python3
"""
Remove Failed Projects from Dataset
===================================

This script removes the 3 projects that failed to clone from the final_balanced_dataset.csv
and creates a clean dataset with only successfully cloned projects.

Failed projects (all OSS4SG):
- odoo/odoo (timeout)
- inasafe/inasafe (timeout)  
- hotosm/hotosm-website (timeout)

Result: 372 projects (185 OSS + 187 OSS4SG)
"""

import csv
import os
from pathlib import Path

def remove_failed_projects():
    """Remove failed projects from the dataset."""
    
    # Define failed projects
    failed_projects = {
        "odoo/odoo",
        "inasafe/inasafe", 
        "hotosm/hotosm-website"
    }
    
    # Paths
    original_file = Path("preparing_dataset/data/final_balanced_dataset.csv")
    clean_file = Path("preparing_dataset/data/final_clean_cloned_dataset.csv")
    
    print("Removing Failed Projects from Dataset")
    print("=" * 40)
    print(f"Original dataset: {original_file}")
    print(f"Clean dataset: {clean_file}")
    print(f"Failed projects to remove: {len(failed_projects)}")
    
    # Read original dataset and filter out failed projects
    clean_projects = []
    removed_count = 0
    oss_count = 0
    oss4sg_count = 0
    
    with open(original_file, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        
        for row in reader:
            project_name = row['project_name']
            project_type = row['type']
            
            if project_name in failed_projects:
                removed_count += 1
                print(f"  Removing: {project_name} ({project_type})")
            else:
                clean_projects.append(row)
                if project_type == 'OSS':
                    oss_count += 1
                elif project_type == 'OSS4SG':
                    oss4sg_count += 1
    
    # Write clean dataset
    with open(clean_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(clean_projects)
    
    # Summary
    total_clean = len(clean_projects)
    print("\n" + "=" * 40)
    print("CLEANING SUMMARY")
    print("=" * 40)
    print(f"Original projects: 375")
    print(f"Removed projects: {removed_count}")
    print(f"Clean dataset size: {total_clean}")
    print(f"  - OSS projects: {oss_count}")
    print(f"  - OSS4SG projects: {oss4sg_count}")
    print(f"  - Balance ratio: 1:{oss4sg_count/oss_count:.2f}")
    print(f"\nClean dataset saved to: {clean_file}")
    
    return total_clean, oss_count, oss4sg_count

if __name__ == "__main__":
    remove_failed_projects()