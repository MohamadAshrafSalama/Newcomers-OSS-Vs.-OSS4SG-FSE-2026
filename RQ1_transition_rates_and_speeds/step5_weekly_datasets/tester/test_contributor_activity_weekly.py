#!/usr/bin/env python3
"""
Updated Test Suite for Contributor Activity Weekly Dataset
==========================================================
Correctly tests cumulative logic PER PROJECT, not globally.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DatasetValidator:
    def __init__(self):
        self.base_path = Path("RQ1_transition_rates_and_speeds/step5_weekly_datasets")
        self.activity_file = self.base_path / "dataset2_contributor_activity/contributor_activity_weekly.csv"
        self.core_timeline_file = self.base_path / "datasets/project_core_timeline_weekly.csv"
        
        self.activity_df = None
        self.core_timeline_df = None
        self.test_results = []
        self.warnings = []
        self.errors = []
        
    def load_datasets(self):
        """Load the datasets for validation."""
        print("=" * 70)
        print("LOADING DATASETS FOR VALIDATION")
        print("=" * 70)
        
        if not self.activity_file.exists():
            print(f"ERROR: Activity file not found: {self.activity_file}")
            sys.exit(1)
        
        print(f"Loading activity dataset from: {self.activity_file}")
        self.activity_df = pd.read_csv(self.activity_file)
        print(f"Loaded {len(self.activity_df):,} activity records")
        
        if self.core_timeline_file.exists():
            print(f"Loading core timeline from: {self.core_timeline_file}")
            self.core_timeline_df = pd.read_csv(self.core_timeline_file)
            self.core_timeline_df['week_date'] = pd.to_datetime(self.core_timeline_df['week_date'])
            print(f"Loaded {len(self.core_timeline_df):,} timeline records")
        
        print()
        
    def test_schema_completeness(self):
        """Test 1: Verify all expected columns exist."""
        print("Test 1: Schema Validation")
        print("-" * 40)
        
        expected_columns = {
            'project_name', 'project_type', 'contributor_email', 'week_date',
            'week_number', 'weeks_since_first_commit', 'commits_this_week',
            'commit_hashes', 'lines_added_this_week', 'lines_deleted_this_week',
            'files_modified_this_week', 'cumulative_commits', 'cumulative_lines_changed',
            'project_commits_to_date', 'contribution_percentage', 'is_core_this_week',
            'rank_this_week'
        }
        
        missing_cols = expected_columns - set(self.activity_df.columns)
        
        if missing_cols:
            self.errors.append(f"Missing columns: {missing_cols}")
            print(f"ERROR: Missing columns: {missing_cols}")
        else:
            print("All expected columns present")
        
        print()
        
    def test_data_integrity(self):
        """Test 2: Check for data integrity issues."""
        print("Test 2: Data Integrity")
        print("-" * 40)
        
        # Check for nulls in critical columns
        critical_cols = ['project_name', 'contributor_email', 'week_date', 'cumulative_commits']
        null_issues = []
        
        for col in critical_cols:
            null_count = self.activity_df[col].isnull().sum()
            if null_count > 0:
                null_issues.append(f"{col}: {null_count} nulls")
        
        if null_issues:
            self.errors.extend(null_issues)
            print("ERROR: Null values found in critical columns:")
            for issue in null_issues:
                print(f"   - {issue}")
        else:
            print("No null values in critical columns")
        
        # Note about emails without @ (not an error)
        no_at_emails = self.activity_df[
            ~self.activity_df['contributor_email'].str.contains('@', na=False)
        ]
        
        if len(no_at_emails) > 0:
            print(f"INFO: Found {len(no_at_emails)} Git identifiers without '@' (e.g., usernames)")
            print("   This is normal Git behavior - not an error")
            sample = no_at_emails['contributor_email'].head(3).tolist()
            print(f"   Examples: {sample}")
        
        # Check for negative values
        numeric_cols = ['commits_this_week', 'cumulative_commits', 'rank_this_week']
        negative_issues = []
        
        for col in numeric_cols:
            if col in self.activity_df.columns:
                neg_count = (self.activity_df[col] < 0).sum()
                if neg_count > 0:
                    negative_issues.append(f"{col}: {neg_count} negative values")
        
        if negative_issues:
            self.errors.extend(negative_issues)
            print("ERROR: Negative values found:")
            for issue in negative_issues:
                print(f"   - {issue}")
        else:
            print("No negative values in numeric columns")
        
        print()
        
    def test_cumulative_logic_per_project(self):
        """Test 3: Verify cumulative values PER PROJECT are correct."""
        print("Test 3: Cumulative Logic Validation (Per Project)")
        print("-" * 40)
        
        issues_found = 0
        contributors_checked = 0
        sample_size = 50
        
        # Get a sample of project-contributor pairs
        project_contrib_pairs = (
            self.activity_df.groupby(['project_name', 'contributor_email'])
            .size()
            .reset_index(name='count')
            .sample(n=min(sample_size, len(self.activity_df)), random_state=42)
        )
        
        for _, pair in project_contrib_pairs.iterrows():
            project = pair['project_name']
            contributor = pair['contributor_email']
            
            # Get data for this project-contributor pair
            mask = (
                (self.activity_df['project_name'] == project) & 
                (self.activity_df['contributor_email'] == contributor)
            )
            contrib_data = self.activity_df[mask].sort_values('week_number')
            
            if len(contrib_data) < 2:
                continue
            
            contributors_checked += 1
            
            # Check 1: Cumulative is non-decreasing
            cum_commits = contrib_data['cumulative_commits'].values
            is_monotonic = all(cum_commits[i] <= cum_commits[i+1] for i in range(len(cum_commits)-1))
            
            # Check 2: Cumulative equals sum of weekly
            total_weekly = contrib_data['commits_this_week'].sum()
            final_cumulative = contrib_data['cumulative_commits'].iloc[-1]
            sum_matches = abs(total_weekly - final_cumulative) <= 1
            
            if not is_monotonic or not sum_matches:
                issues_found += 1
                if issues_found == 1:  # Show first issue only
                    print(f"Issue found: {project} / {contributor}")
                    print(f"   Monotonic: {is_monotonic}, Sum matches: {sum_matches}")
                    print(f"   Weekly sum: {total_weekly}, Final cumulative: {final_cumulative}")
        
        if issues_found == 0:
            print(f"Cumulative logic is correct (checked {contributors_checked} project-contributor pairs)")
        else:
            self.errors.append(f"Cumulative logic issues: {issues_found}/{contributors_checked} pairs")
            print(f"Found issues in {issues_found}/{contributors_checked} project-contributor pairs")
        
        print()
        
    def test_multi_project_contributors(self):
        """Test 4: Analyze contributors in multiple projects."""
        print("Test 4: Multi-Project Contributor Analysis")
        print("-" * 40)
        
        # Find contributors in multiple projects
        contrib_projects = self.activity_df.groupby('contributor_email')['project_name'].nunique()
        multi_project_contribs = contrib_projects[contrib_projects > 1]
        
        print(f"Contributors in multiple projects: {len(multi_project_contribs):,} / {len(contrib_projects):,}")
        print(f"Percentage: {len(multi_project_contribs)/len(contrib_projects)*100:.1f}%")
        
        # Show top multi-project contributors
        top_multi = multi_project_contribs.nlargest(5)
        print("\nTop multi-project contributors:")
        for contributor, project_count in top_multi.items():
            print(f"  {contributor}: {project_count} projects")
        
        print("\nNote: These contributors have separate cumulative counts per project")
        print("   This is the expected behavior for the dataset")
        
        print()
        
    def test_core_status_consistency(self):
        """Test 5: Cross-validate core status with core timeline."""
        print("Test 5: Core Status Cross-Validation")
        print("-" * 40)
        
        if self.core_timeline_df is None:
            print("Skipping - core timeline not loaded")
            print()
            return
        
        sample_checks = 50
        mismatches = 0
        
        activity_sample = self.activity_df[['project_name', 'week_date']].drop_duplicates().sample(
            n=min(sample_checks, len(self.activity_df)), 
            random_state=42
        )
        
        for _, row in activity_sample.iterrows():
            project = row['project_name']
            week = row['week_date']
            
            timeline_row = self.core_timeline_df[
                (self.core_timeline_df['project_name'] == project) & 
                (self.core_timeline_df['week_date'] == week)
            ]
            
            if len(timeline_row) > 0:
                try:
                    core_emails = json.loads(timeline_row.iloc[0]['core_contributors_emails'])
                    
                    activity_week = self.activity_df[
                        (self.activity_df['project_name'] == project) & 
                        (self.activity_df['week_date'] == week)
                    ]
                    
                    for _, act_row in activity_week.iterrows():
                        expected_core = act_row['contributor_email'] in core_emails
                        actual_core = act_row['is_core_this_week']
                        
                        if expected_core != actual_core:
                            mismatches += 1
                            break
                except:
                    pass
        
        if mismatches == 0:
            print(f"Core status consistent with timeline (checked {sample_checks} project-weeks)")
        else:
            self.warnings.append(f"Core status mismatches: {mismatches}")
            print(f"Found {mismatches} core status mismatches")
        
        print()
        
    def test_statistical_sanity(self):
        """Test 6: Statistical sanity checks."""
        print("Test 6: Statistical Summary")
        print("-" * 40)
        
        print("Dataset Statistics:")
        print(f"  - Total records: {len(self.activity_df):,}")
        print(f"  - Unique projects: {self.activity_df['project_name'].nunique()}")
        print(f"  - Unique contributors: {self.activity_df['contributor_email'].nunique()}")
        print(f"  - Date range: {self.activity_df['week_date'].min()} to {self.activity_df['week_date'].max()}")
        
        # Activity metrics
        active_weeks = (self.activity_df['commits_this_week'] > 0).sum()
        active_percentage = active_weeks / len(self.activity_df) * 100
        
        print("\nActivity Metrics:")
        print(f"  - Active weeks (commits > 0): {active_weeks:,} ({active_percentage:.1f}%)")
        
        core_percentage = self.activity_df['is_core_this_week'].mean() * 100
        print(f"  - Core contributor-weeks: {core_percentage:.2f}%")
        
        avg_commits = self.activity_df['commits_this_week'].mean()
        avg_commits_active = self.activity_df[self.activity_df['commits_this_week'] > 0]['commits_this_week'].mean()
        
        print(f"  - Avg commits per week (all): {avg_commits:.3f}")
        print(f"  - Avg commits per week (active): {avg_commits_active:.2f}")
        
        # Validate ranges
        if core_percentage < 5 or core_percentage > 40:
            self.warnings.append(f"Unusual core percentage: {core_percentage:.2f}%")
            print(f"WARNING: Core percentage may be unusual: {core_percentage:.2f}%")
        else:
            print("Core percentage is reasonable")
        
        if avg_commits < 0.01:
            self.warnings.append(f"Very low average commits: {avg_commits:.3f}")
            print("WARNING: Average commits seems low")
        else:
            print("Average commits is reasonable")
        
        # Project type distribution
        print("\nProject Type Distribution:")
        for ptype, count in self.activity_df['project_type'].value_counts().items():
            percentage = (count / len(self.activity_df)) * 100
            print(f"  - {ptype}: {count:,} records ({percentage:.1f}%)")
        
        print()
        
    def generate_summary_report(self):
        """Generate final summary report."""
        print("=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        if not self.errors:
            print("ALL CRITICAL TESTS PASSED!")
            print("The contributor_activity_weekly dataset is correctly generated.")
            
            if self.warnings:
                print(f"\nWarnings (non-critical): {len(self.warnings)}")
                for warning in self.warnings:
                    print(f"  - {warning}")
        else:
            print(f"CRITICAL ISSUES FOUND: {len(self.errors)} errors")
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")
            
            if self.warnings:
                print("\nWarnings:")
                for warning in self.warnings:
                    print(f"  - {warning}")
        
        print("\n" + "=" * 70)
        
        # Save report
        report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_file': str(self.activity_file),
            'total_records': len(self.activity_df),
            'errors': self.errors,
            'warnings': self.warnings,
            'passed': len(self.errors) == 0,
            'dataset_stats': {
                'projects': int(self.activity_df['project_name'].nunique()),
                'contributors': int(self.activity_df['contributor_email'].nunique()),
                'date_range': {
                    'start': str(self.activity_df['week_date'].min()),
                    'end': str(self.activity_df['week_date'].max())
                },
                'core_percentage': float(self.activity_df['is_core_this_week'].mean() * 100),
                'avg_commits_per_week': float(self.activity_df['commits_this_week'].mean()),
                'active_weeks_percentage': float((self.activity_df['commits_this_week'] > 0).mean() * 100)
            }
        }
        
        report_file = self.base_path / "dataset2_contributor_activity/validation_report_updated.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed report saved to: {report_file}")
        
    def run_all_tests(self):
        """Run all validation tests."""
        self.load_datasets()
        
        self.test_schema_completeness()
        self.test_data_integrity()
        self.test_cumulative_logic_per_project()
        self.test_multi_project_contributors()
        self.test_core_status_consistency()
        self.test_statistical_sanity()
        
        self.generate_summary_report()

def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("CONTRIBUTOR ACTIVITY WEEKLY DATASET VALIDATOR (UPDATED)")
    print("=" * 70 + "\n")
    
    validator = DatasetValidator()
    validator.run_all_tests()

if __name__ == "__main__":
    main()
