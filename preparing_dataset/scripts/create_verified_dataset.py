#!/usr/bin/env python3
"""
Create academically rigorous dataset using only verified projects
"""

import csv

def create_verified_dataset():
    """Create final dataset with only verified projects"""
    
    verified_projects = []
    
    print("Creating verified dataset with only original filtered projects...")
    
    # Add original OSS projects (already verified by your research)
    with open('Dataset/Filtered-OSS-Project-Info.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            verified_projects.append({
                'project_name': row['name'],
                'type': 'OSS',
                'source': 'verified_original',
                'verification_method': 'systematic_filtering_by_researchers'
            })
    
    # Add original OSS4SG projects (already verified by your research)  
    with open('Dataset/Filtered-OSS4SG-Project-Info.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            verified_projects.append({
                'project_name': row['name'],
                'type': 'OSS4SG',
                'source': 'verified_original',
                'verification_method': 'systematic_filtering_by_researchers'
            })
    
    # Save verified dataset
    with open('final_verified_dataset.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['project_name', 'type', 'source', 'verification_method']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(verified_projects)
    
    # Print summary
    oss_count = len([p for p in verified_projects if p['type'] == 'OSS'])
    oss4sg_count = len([p for p in verified_projects if p['type'] == 'OSS4SG'])
    total_count = len(verified_projects)
    
    print(f"\\nVERIFIED DATASET SUMMARY")
    print("=" * 50)
    print(f"OSS Projects: {oss_count}")
    print(f"OSS4SG Projects: {oss4sg_count}")
    print(f"Total Projects: {total_count}")
    print(f"Balance Ratio: 1 OSS : {oss4sg_count/oss_count:.2f} OSS4SG")
    print(f"\\nSaved to: final_verified_dataset.csv")
    print("\\nAll projects meet the 5 criteria (verified by systematic filtering)")

if __name__ == "__main__":
    create_verified_dataset()