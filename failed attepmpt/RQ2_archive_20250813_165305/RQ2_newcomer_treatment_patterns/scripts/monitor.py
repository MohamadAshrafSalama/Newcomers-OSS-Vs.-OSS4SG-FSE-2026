#!/usr/bin/env python3
"""
Monitor and analyze RQ2 extraction progress and results.
Can run continuously or provide one-time analysis.
"""

import json
import pickle
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class ExtractionMonitor:
    """Monitor extraction progress and analyze results."""
    
    def __init__(self, output_dir: str):
        """Initialize monitor with output directory."""
        self.output_dir = Path(output_dir)
        self.state_file = self.output_dir / "extraction_state.pkl"
        self.cache_dir = self.output_dir / "cache"
        self.failed_dir = self.output_dir / "failed"
        self.log_file = self.output_dir / "extraction.log"
    
    def load_state(self) -> Optional[Dict]:
        """Load extraction state."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading state: {e}")
            return None
    
    def get_cache_stats(self) -> Dict:
        """Analyze cached results."""
        if not self.cache_dir.exists():
            return {}
        
        cache_files = list(self.cache_dir.glob("*.json"))
        
        stats = {
            'total_cached': len(cache_files),
            'by_project_type': {},
            'total_prs': 0,
            'total_issues': 0,
            'with_prs': 0,
            'with_issues': 0,
            'with_responses': 0,
            'response_times': [],
            'approval_rates': [],
            'review_counts': [],
            'comment_counts': []
        }
        
        # Analyze each cached result
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Count by type
                ptype = data.get('project_type', 'unknown')
                stats['by_project_type'][ptype] = stats['by_project_type'].get(ptype, 0) + 1
                
                # Count PRs/issues
                pr_count = len(data.get('pull_requests', []))
                issue_count = len(data.get('issues', []))
                
                stats['total_prs'] += pr_count
                stats['total_issues'] += issue_count
                
                if pr_count > 0:
                    stats['with_prs'] += 1
                if issue_count > 0:
                    stats['with_issues'] += 1
                
                # Analyze metrics
                metrics = data.get('treatment_metrics', {})
                
                if metrics.get('avg_response_hours'):
                    stats['response_times'].append(metrics['avg_response_hours'])
                    stats['with_responses'] += 1
                
                if metrics.get('approval_rate') is not None:
                    stats['approval_rates'].append(metrics['approval_rate'])
                
                if metrics.get('total_reviews'):
                    stats['review_counts'].append(metrics['total_reviews'])
                
                if metrics.get('total_comments'):
                    stats['comment_counts'].append(metrics['total_comments'])
                    
            except Exception as e:
                continue
        
        return stats
    
    def show_progress(self):
        """Display current extraction progress."""
        state = self.load_state()
        
        if not state:
            print("No extraction state found")
            print(f"   Looking in: {self.state_file}")
            return
        
        print("\n" + "="*70)
        print("Extraction Progress")
        print("="*70)
        
        # Basic counts
        print("\nProgress:")
        print(f"  Processed: {state.get('processed_count', 0):,}")
        print(f"  Skipped: {state.get('skipped_count', 0):,}")
        print(f"  Failed: {state.get('failed_count', 0):,}")
        
        total = (state.get('processed_count', 0) + 
                state.get('skipped_count', 0) + 
                state.get('failed_count', 0))
        
        if total > 0:
            success_rate = state.get('processed_count', 0) / total * 100
            print(f"  Success rate: {success_rate:.1f}%")
        
        # Time analysis
        if state.get('start_time'):
            start = datetime.fromisoformat(state['start_time'])
            elapsed = datetime.now() - start
            hours = elapsed.total_seconds() / 3600
            
            if hours > 0 and state.get('processed_count', 0) > 0:
                rate = state['processed_count'] / hours
                print(f"\nTime:")
                print(f"  Elapsed: {hours:.1f} hours")
                print(f"  Rate: {rate:.1f} contributors/hour")
                
                # Estimate remaining time
                if 'total_to_process' in state:
                    remaining = state['total_to_process'] - state['processed_count']
                    eta_hours = remaining / rate if rate > 0 else 0
                    print(f"  ETA: {eta_hours:.1f} hours")
        
        # Cache analysis
        cache_stats = self.get_cache_stats()
        
        if cache_stats:
            print(f"\nCollected Data:")
            print(f"  Cached results: {cache_stats['total_cached']:,}")
            print(f"  Total PRs: {cache_stats['total_prs']:,}")
            print(f"  Total Issues: {cache_stats['total_issues']:,}")
            print(f"  With PRs: {cache_stats['with_prs']:,}")
            print(f"  With Issues: {cache_stats['with_issues']:,}")
            print(f"  With responses: {cache_stats['with_responses']:,}")
            
            if cache_stats['by_project_type']:
                print(f"\n  By Project Type:")
                for ptype, count in cache_stats['by_project_type'].items():
                    print(f"    {ptype}: {count:,}")
            
            if cache_stats['response_times']:
                times = cache_stats['response_times']
                print(f"\n  Response Times (hours):")
                print(f"    Min: {min(times):.1f}")
                print(f"    Median: {sorted(times)[len(times)//2]:.1f}")
                print(f"    Mean: {sum(times)/len(times):.1f}")
                print(f"    Max: {max(times):.1f}")
        
        # Recent failures
        if state.get('failed_contributors'):
            print(f"\nRecent Failures (last 5):")
            recent = list(state['failed_contributors'].items())[-5:]
            for contrib_id, error in recent:
                print(f"  • {contrib_id[:50]}")
                print(f"    {str(error)[:60]}...")
        
        # Last save
        if state.get('last_save'):
            last = datetime.fromisoformat(state['last_save'])
            ago = (datetime.now() - last).total_seconds() / 60
            print(f"\nLast save: {ago:.1f} minutes ago")
    
    def analyze_results(self):
        """Analyze completed extraction results."""
        if not self.cache_dir.exists():
            print("No results found")
            return
        
        cache_files = list(self.cache_dir.glob("*.json"))
        
        if not cache_files:
            print("No cached results found")
            return
        
        print("\n" + "="*70)
        print("Extraction Results Analysis")
        print("="*70)
        
        print(f"\nFound {len(cache_files):,} contributor results")
        
        # Load all results
        results = []
        errors = 0
        
        print("Loading results...")
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                errors += 1
        
        if errors > 0:
            print(f"  Failed to load {errors} files")
        
        print(f"  Loaded {len(results):,} results")
        
        # Create DataFrame for analysis
        rows = []
        for data in results:
            metrics = data.get('treatment_metrics', {})
            
            row = {
                'username': data.get('username'),
                'project': data.get('project'),
                'project_type': data.get('project_type'),
                'weeks_to_core': data.get('weeks_to_core'),
                'commits_to_core': data.get('commits_to_core'),
                'total_prs': metrics.get('total_prs', 0),
                'total_issues': metrics.get('total_issues', 0),
                'total_items': metrics.get('total_items', 0),
                'total_reviews': metrics.get('total_reviews', 0),
                'total_comments': metrics.get('total_comments', 0),
                'unique_reviewers': metrics.get('unique_reviewers', 0),
                'unique_commenters': metrics.get('unique_commenters', 0),
                'avg_response_hours': metrics.get('avg_response_hours'),
                'median_response_hours': metrics.get('median_response_hours'),
                'approval_rate': metrics.get('approval_rate', 0),
                'changes_requested_rate': metrics.get('changes_requested_rate', 0)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Overall statistics
        print("\nOverall Statistics:")
        print(f"  Contributors analyzed: {len(df):,}")
        print(f"  Unique projects: {df['project'].nunique()}")
        print(f"  Total PRs: {df['total_prs'].sum():,}")
        print(f"  Total Issues: {df['total_issues'].sum():,}")
        print(f"  Total Reviews: {df['total_reviews'].sum():,}")
        print(f"  Total Comments: {df['total_comments'].sum():,}")
        
        # Activity rates
        with_prs = len(df[df['total_prs'] > 0])
        with_issues = len(df[df['total_issues'] > 0])
        with_activity = len(df[df['total_items'] > 0])
        
        print("\nActivity Rates:")
        print(f"  With PRs: {with_prs:,} ({with_prs/len(df)*100:.1f}%)")
        print(f"  With Issues: {with_issues:,} ({with_issues/len(df)*100:.1f}%)")
        print(f"  With any activity: {with_activity:,} ({with_activity/len(df)*100:.1f}%)")
        
        # By project type
        print("\nBy Project Type:")
        for ptype in df['project_type'].unique():
            if pd.isna(ptype) or ptype == '':
                continue
                
            type_df = df[df['project_type'] == ptype]
            print(f"\n  {ptype}:")
            print(f"    Contributors: {len(type_df):,}")
            print(f"    Avg PRs: {type_df['total_prs'].mean():.2f}")
            print(f"    Avg Issues: {type_df['total_issues'].mean():.2f}")
            print(f"    Avg Reviews: {type_df['total_reviews'].mean():.2f}")
            print(f"    Avg Reviewers: {type_df['unique_reviewers'].mean():.2f}")
            
            response_times = type_df['avg_response_hours'].dropna()
            if len(response_times) > 0:
                print(f"    Avg Response (hours): {response_times.mean():.1f}")
                print(f"    Median Response (hours): {response_times.median():.1f}")
            
            approval_rates = type_df['approval_rate'].dropna()
            if len(approval_rates) > 0:
                print(f"    Avg Approval Rate: {approval_rates.mean():.1%}")
        
        # Top contributors
        print("\nMost Active Contributors (by PRs):")
        top = df.nlargest(10, 'total_prs')[['username', 'project', 'total_prs', 'total_reviews']]
        for _, row in top.iterrows():
            print(f"  {str(row['username'])[:20]:20} ({str(row['project'])[:30]:30}): "
                  f"{row['total_prs']} PRs, {row['total_reviews']} reviews")
        
        # Save summary
        summary_file = self.output_dir / "analysis_summary.csv"
        df.to_csv(summary_file, index=False)
        print(f"\nSaved summary to {summary_file}")
    
    def watch_progress(self, interval: int = 30):
        """Continuously monitor progress."""
        print("Monitoring extraction progress... (Ctrl+C to stop)")
        
        try:
            while True:
                # Clear screen
                print("\033[2J\033[H", end="")
                
                # Show progress
                self.show_progress()
                
                # Wait
                print(f"\nRefreshing in {interval} seconds...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("Stopped")
    
    def generate_report(self):
        """Generate a comprehensive report."""
        report_file = self.output_dir / "extraction_report.txt"
        
        # Redirect stdout to capture output
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = report_output = StringIO()
        
        # Generate all analyses
        print("RQ2 EXTRACTION REPORT")
        print("=" * 70)
        print(f"Generated: {datetime.now().isoformat()}")
        print(f"Output Directory: {self.output_dir}")
        
        self.show_progress()
        self.analyze_results()
        
        # Get output
        report_content = report_output.getvalue()
        sys.stdout = old_stdout
        
        # Save report
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"Report saved to {report_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor RQ2 extraction progress')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Extraction output directory')
    parser.add_argument('--watch', action='store_true',
                       help='Continuously monitor progress')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze extraction results')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive report')
    parser.add_argument('--interval', type=int, default=30,
                       help='Refresh interval for watch mode (seconds)')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = ExtractionMonitor(args.output_dir)
    
    # Run requested action
    if args.report:
        monitor.generate_report()
    elif args.analyze:
        monitor.analyze_results()
    elif args.watch:
        monitor.watch_progress(args.interval)
    else:
        monitor.show_progress()


if __name__ == "__main__":
    main()


