#!/usr/bin/env python3
"""
Analyze author matching between commits (email/name) and monthly core data (names)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_author_matching():
    print("=== AUTHOR MATCHING ANALYSIS ===")
    
    # Load sample of commits data
    print("Loading commits data...")
    commits_df = pd.read_csv(
        'RQ1_transition_rates_and_speeds/data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv',
        usecols=['project_name', 'author_email', 'author_name', 'commit_hash', 'commit_date'],
        nrows=100000
    )
    
    print(f"Sample commits loaded: {len(commits_df):,}")
    print(f"Unique emails: {commits_df['author_email'].nunique()}")
    print(f"Unique names: {commits_df['author_name'].nunique()}")
    
    # Check name-email mapping complexity
    print("\n=== EMAIL-NAME MAPPING COMPLEXITY ===")
    name_per_email = commits_df.groupby('author_email')['author_name'].nunique()
    email_per_name = commits_df.groupby('author_name')['author_email'].nunique()
    
    print(f"Max different names per email: {name_per_email.max()}")
    print(f"Max different emails per name: {email_per_name.max()}")
    
    # Show examples of complex mappings
    print("\n=== COMPLEX MAPPING EXAMPLES ===")
    complex_emails = name_per_email[name_per_email > 1].head(5)
    for email, name_count in complex_emails.items():
        names = commits_df[commits_df['author_email'] == email]['author_name'].unique()
        print(f"Email '{email}' has {name_count} names: {list(names)}")
    
    complex_names = email_per_name[email_per_name > 1].head(5)
    for name, email_count in complex_names.items():
        emails = commits_df[commits_df['author_name'] == name]['author_email'].unique()
        print(f"Name '{name}' has {email_count} emails: {list(emails)}")
    
    # Load monthly core data
    print("\n=== MONTHLY CORE DATA ANALYSIS ===")
    monthly_df = pd.read_csv(
        'RQ1_transition_rates_and_speeds/step4_newcomer_transition_rates/corrected_transition_results/monthly_transitions.csv'
    )
    
    # Extract all core names from monthly data
    core_names = set()
    for _, row in monthly_df.iterrows():
        if pd.notna(row['truly_new_core_names']) and row['truly_new_core_names'] != '[]':
            try:
                names = eval(row['truly_new_core_names'])  # Convert string list to actual list
                core_names.update(names)
            except:
                pass
    
    print(f"Total unique core contributor names: {len(core_names)}")
    print("Sample core names:")
    for name in list(core_names)[:10]:
        print(f"  '{name}'")
    
    # Check how many core names can be matched in commits data
    print("\n=== CORE NAME MATCHING IN COMMITS ===")
    commits_names = set(commits_df['author_name'].unique())
    matched_names = core_names.intersection(commits_names)
    
    print(f"Core names found in commits sample: {len(matched_names)}/{len(core_names)} ({len(matched_names)/len(core_names)*100:.1f}%)")
    
    # Create mapping strategy
    print("\n=== PROPOSED MATCHING STRATEGY ===")
    print("1. Primary: Match by author_name (most reliable)")
    print("2. Secondary: Create name normalization (trim spaces, handle case)")
    print("3. Verification: Use commit_hash to verify unique contributors")
    print("4. Fallback: Map emails to names when name matching fails")
    
    # Save analysis results
    results = {
        'sample_commits': len(commits_df),
        'unique_emails': commits_df['author_email'].nunique(),
        'unique_names': commits_df['author_name'].nunique(),
        'max_names_per_email': int(name_per_email.max()),
        'max_emails_per_name': int(email_per_name.max()),
        'total_core_names': len(core_names),
        'matched_core_names': len(matched_names),
        'match_rate': len(matched_names)/len(core_names) if core_names else 0
    }
    
    with open('RQ1_transition_rates_and_speeds/step10_ml_modeling/author_matching_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis saved to: author_matching_analysis.json")
    return results

if __name__ == "__main__":
    analyze_author_matching()

