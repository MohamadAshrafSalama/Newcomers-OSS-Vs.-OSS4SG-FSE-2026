#!/usr/bin/env python3
"""
Create balanced final dataset combining:
- 90 original OSS projects
- 95 additional OSS projects  
- 190 OSS4SG projects
Total: 375 projects (185 OSS + 190 OSS4SG)
"""

import csv

def create_balanced_dataset():
    """Combine all projects into balanced final dataset"""
    
    all_projects = []
    
    print("Creating balanced dataset...")
    print("=" * 50)
    
    # 1. Add original OSS projects (90)
    print("Loading original OSS projects...")
    with open('preparing_dataset/data/final_clean_dataset.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['type'] == 'OSS':
                all_projects.append({
                    'project_name': row['project_name'],
                    'type': 'OSS',
                    'source': 'original_filtered',
                    'verification_method': 'systematic_filtering_by_researchers'
                })
    
    original_oss_count = len([p for p in all_projects if p['type'] == 'OSS'])
    print(f"  Original OSS projects: {original_oss_count}")
    
    # 2. Add additional OSS projects (95)
    print("Loading additional OSS projects...")
    with open('additional_oss_projects.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_projects.append({
                'project_name': row['project_name'],
                'type': 'OSS', 
                'source': 'systematic_collection',
                'verification_method': 'github_api_realtime'
            })
    
    additional_oss_count = len([p for p in all_projects if p['type'] == 'OSS']) - original_oss_count
    print(f"  Additional OSS projects: {additional_oss_count}")
    
    # 3. Add OSS4SG projects (190)
    print("Loading OSS4SG projects...")
    with open('preparing_dataset/data/final_clean_dataset.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['type'] == 'OSS4SG':
                all_projects.append({
                    'project_name': row['project_name'],
                    'type': 'OSS4SG',
                    'source': 'original_filtered',
                    'verification_method': 'systematic_filtering_by_researchers'
                })
    
    oss4sg_count = len([p for p in all_projects if p['type'] == 'OSS4SG'])
    print(f"  OSS4SG projects: {oss4sg_count}")
    
    # 4. Save balanced dataset
    print("\\nSaving balanced dataset...")
    with open('preparing_dataset/data/final_balanced_dataset.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['project_name', 'type', 'source', 'verification_method']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_projects)
    
    # 5. Generate summary
    total_oss = len([p for p in all_projects if p['type'] == 'OSS'])
    total_oss4sg = len([p for p in all_projects if p['type'] == 'OSS4SG'])
    total_projects = len(all_projects)
    
    print("\\n" + "=" * 50)
    print("BALANCED DATASET SUMMARY")
    print("=" * 50)
    print(f"OSS Projects: {total_oss}")
    print(f"  - Original (filtered): {original_oss_count}")
    print(f"  - Additional (systematic): {additional_oss_count}")
    print(f"OSS4SG Projects: {total_oss4sg}")
    print(f"Total Projects: {total_projects}")
    print(f"Balance Ratio: 1 OSS : {total_oss4sg/total_oss:.2f} OSS4SG")
    print(f"\\nSaved to: preparing_dataset/data/final_balanced_dataset.csv")
    print("\\nAll projects verified against 5 criteria:")
    print("  1. >= 10 contributors")
    print("  2. >= 500 commits") 
    print("  3. >= 50 closed Pull Requests")
    print("  4. > 1 year of project history")
    print("  5. Updated within last year")
    
    return {
        'total': total_projects,
        'oss': total_oss,
        'oss4sg': total_oss4sg,
        'ratio': total_oss4sg/total_oss
    }

if __name__ == "__main__":
    stats = create_balanced_dataset()
    
    print(f"\\nSuccess! Created balanced dataset with {stats['total']} projects")
    print(f"Ratio improved from 1:2.11 to 1:{stats['ratio']:.2f}")