#!/usr/bin/env python3
"""
Create final clean dataset removing projects that failed verification
"""

import csv

def create_clean_dataset():
    """Remove failed projects and create final clean dataset"""
    
    # Projects that failed verification (insufficient closed PRs)
    failed_projects = {
        'openeemeter/eemeter',
        'somleng/somleng-scfm', 
        'sahana/eden'
    }
    
    clean_projects = []
    removed_count = 0
    
    print("Creating final clean dataset...")
    print(f"Removing {len(failed_projects)} projects that failed verification:")
    for project in failed_projects:
        print(f"  - {project}")
    
    # Read all projects and filter out failed ones
    with open('final_verified_dataset.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['project_name'] not in failed_projects:
                clean_projects.append(row)
            else:
                removed_count += 1
                print(f"    Removed: {row['project_name']} ({row['type']})")
    
    # Save clean dataset
    with open('final_clean_dataset.csv', 'w', newline='', encoding='utf-8') as f:
        if clean_projects:
            fieldnames = clean_projects[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_projects)
    
    # Generate summary
    oss_count = len([p for p in clean_projects if p['type'] == 'OSS'])
    oss4sg_count = len([p for p in clean_projects if p['type'] == 'OSS4SG'])
    total_count = len(clean_projects)
    
    print(f"\nFINAL CLEAN DATASET SUMMARY")
    print("=" * 50)
    print(f"Removed Failed Projects: {removed_count}")
    print(f"OSS Projects: {oss_count}")
    print(f"OSS4SG Projects: {oss4sg_count}")
    print(f"Total Clean Projects: {total_count}")
    print(f"Balance Ratio: 1 OSS : {oss4sg_count/oss_count:.2f} OSS4SG")
    print(f"Verification Rate: {total_count/(total_count + removed_count)*100:.1f}%")
    print(f"\nSaved to: final_clean_dataset.csv")
    print("All projects verified against 5 criteria")
    
    return {
        'total': total_count,
        'oss': oss_count, 
        'oss4sg': oss4sg_count,
        'removed': removed_count
    }

if __name__ == "__main__":
    stats = create_clean_dataset()