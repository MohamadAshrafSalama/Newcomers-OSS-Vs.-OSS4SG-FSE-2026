#!/usr/bin/env python3
"""
Script to analyze comment counts in pull requests and issues 
for all newcomers before they become core contributors.

This script processes all timeline CSV files in the from_cache_timelines directory
and counts comments in PRs and issues where is_pre_core = True.
"""

import os
import json
import pandas as pd
from pathlib import Path
import glob

def parse_json_safely(json_str):
    """Safely parse JSON string, return empty dict if parsing fails."""
    try:
        if pd.isna(json_str) or json_str == '':
            return {}
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return {}

def count_comments_in_event(event_data, event_type):
    """Count comments in a single event based on event type."""
    if not event_data:
        return 0
    
    if event_type == 'pull_request':
        # Count both PR comments and review comments
        pr_comments = len(event_data.get('comments', {}).get('nodes', []))
        review_comments = len(event_data.get('reviews', {}).get('nodes', []))
        return pr_comments + review_comments
    
    elif event_type == 'issue':
        # Count issue comments
        return len(event_data.get('comments', {}).get('nodes', []))
    
    return 0

def analyze_timeline_file(file_path):
    """Analyze a single timeline file and return comment statistics."""
    try:
        df = pd.read_csv(file_path)
        
        # Filter for events before becoming core
        pre_core_events = df[df['is_pre_core'] == True]
        
        # Initialize counters
        total_pr_comments = 0
        total_issue_comments = 0
        pr_count = 0
        issue_count = 0
        
        # Process each event
        for _, row in pre_core_events.iterrows():
            event_type = row['event_type']
            event_data = parse_json_safely(row['event_data'])
            
            if event_type == 'pull_request':
                pr_count += 1
                total_pr_comments += count_comments_in_event(event_data, 'pull_request')
            
            elif event_type == 'issue':
                issue_count += 1
                total_issue_comments += count_comments_in_event(event_data, 'issue')
        
        return {
            'file': os.path.basename(file_path),
            'pr_count': pr_count,
            'issue_count': issue_count,
            'total_pr_comments': total_pr_comments,
            'total_issue_comments': total_issue_comments,
            'total_comments': total_pr_comments + total_issue_comments
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    """Main function to analyze all timeline files."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    timelines_dir = script_dir / 'from_cache_timelines'
    
    if not timelines_dir.exists():
        print(f"Timelines directory not found: {timelines_dir}")
        return
    
    # Find all timeline CSV files
    timeline_files = glob.glob(str(timelines_dir / 'timeline_*.csv'))
    
    if not timeline_files:
        print(f"No timeline files found in {timelines_dir}")
        return
    
    print(f"Found {len(timeline_files)} timeline files to analyze...")
    print("=" * 80)
    
    # Analyze each file
    results = []
    total_stats = {
        'total_pr_comments': 0,
        'total_issue_comments': 0,
        'total_comments': 0,
        'total_pr_count': 0,
        'total_issue_count': 0
    }
    
    for file_path in timeline_files:
        result = analyze_timeline_file(file_path)
        if result:
            results.append(result)
            
            # Update totals
            total_stats['total_pr_comments'] += result['total_pr_comments']
            total_stats['total_issue_comments'] += result['total_issue_comments']
            total_stats['total_comments'] += result['total_comments']
            total_stats['total_pr_count'] += result['pr_count']
            total_stats['total_issue_count'] += result['issue_count']
    
    # Display individual file results
    print("INDIVIDUAL FILE RESULTS:")
    print("-" * 80)
    print(f"{'Filename':<50} {'PRs':<5} {'Issues':<7} {'PR Comments':<12} {'Issue Comments':<15} {'Total Comments':<15}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x['total_comments'], reverse=True):
        print(f"{result['file']:<50} {result['pr_count']:<5} {result['issue_count']:<7} "
              f"{result['total_pr_comments']:<12} {result['total_issue_comments']:<15} {result['total_comments']:<15}")
    
    # Display summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (BEFORE BECOMING CORE):")
    print("=" * 80)
    print(f"Total Pull Requests: {total_stats['total_pr_count']:,}")
    print(f"Total Issues: {total_stats['total_issue_count']:,}")
    print(f"Total PR Comments: {total_stats['total_pr_comments']:,}")
    print(f"Total Issue Comments: {total_stats['total_issue_comments']:,}")
    print(f"Total Comments: {total_stats['total_comments']:,}")
    
    if total_stats['total_pr_count'] > 0:
        avg_pr_comments = total_stats['total_pr_comments'] / total_stats['total_pr_count']
        print(f"Average Comments per PR: {avg_pr_comments:.2f}")
    
    if total_stats['total_issue_count'] > 0:
        avg_issue_comments = total_stats['total_issue_comments'] / total_stats['total_issue_count']
        print(f"Average Comments per Issue: {avg_issue_comments:.2f}")
    
    # Save results to CSV
    output_file = script_dir / 'comment_analysis_results.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save summary to text file
    summary_file = script_dir / 'comment_analysis_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("COMMENT ANALYSIS SUMMARY (BEFORE BECOMING CORE)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Pull Requests: {total_stats['total_pr_count']:,}\n")
        f.write(f"Total Issues: {total_stats['total_issue_count']:,}\n")
        f.write(f"Total PR Comments: {total_stats['total_pr_comments']:,}\n")
        f.write(f"Total Issue Comments: {total_stats['total_issue_comments']:,}\n")
        f.write(f"Total Comments: {total_stats['total_comments']:,}\n")
        
        if total_stats['total_pr_count'] > 0:
            avg_pr_comments = total_stats['total_pr_comments'] / total_stats['total_pr_count']
            f.write(f"Average Comments per PR: {avg_pr_comments:.2f}\n")
        
        if total_stats['total_issue_count'] > 0:
            avg_issue_comments = total_stats['total_issue_comments'] / total_stats['total_issue_count']
            f.write(f"Average Comments per Issue: {avg_issue_comments:.2f}\n")
    
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()


