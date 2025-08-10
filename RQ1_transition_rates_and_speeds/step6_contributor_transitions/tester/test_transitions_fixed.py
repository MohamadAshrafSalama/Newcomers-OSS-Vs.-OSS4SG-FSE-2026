#!/usr/bin/env python3
"""
Test and Verification Script for Contributor Transitions Dataset (Step 6)
=======================================================================

Comprehensive tests to validate the contributor_transitions.csv dataset.
Checks for data integrity, logical consistency, and statistical validity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class TransitionsValidator:
    def __init__(self):
        # Resolve paths relative to this file location to work from any CWD
        here = Path(__file__).resolve()
        self.base_path = here.parents[1]  # .../step6_contributor_transitions
        self.transitions_file = self.base_path / "results/contributor_transitions.csv"
        self.transitions_all_file = self.base_path / "results/contributor_transitions_including_instant.csv"
        self.activity_file = self.base_path / "../step5_weekly_datasets/dataset2_contributor_activity/contributor_activity_weekly.csv"

        self.transitions_df = None
        self.transitions_all_df = None
        self.errors = []
        self.warnings = []
        self.info = []

    def load_data(self):
        """Load the transitions datasets."""
        print("=" * 70)
        print("LOADING TRANSITIONS DATASETS FOR VALIDATION")
        print("=" * 70)

        # Load main (filtered) dataset
        if not self.transitions_file.exists():
            print(f"❌ ERROR: Transitions file not found: {self.transitions_file}")
            sys.exit(1)

        print(f"Loading main dataset: {self.transitions_file}")
        self.transitions_df = pd.read_csv(self.transitions_file, low_memory=False)
        print(f"✅ Loaded {len(self.transitions_df):,} transition records (excluding instant core)\n")

        # Load complete dataset if exists
        if self.transitions_all_file.exists():
            print(f"Loading complete dataset: {self.transitions_all_file}")
            self.transitions_all_df = pd.read_csv(self.transitions_all_file, low_memory=False)
            print(f"✅ Loaded {len(self.transitions_all_df):,} total records (including instant core)\n")

            instant_core_count = len(self.transitions_all_df) - len(self.transitions_df)
            print(f"ℹ️ Instant core contributors excluded: {instant_core_count:,}")

    def test_schema(self):
        """Test 1: Verify schema completeness and data types."""
        print("\n" + "=" * 70)
        print("TEST 1: SCHEMA VALIDATION")
        print("=" * 70)

        required_cols = [
            'project_name', 'project_type', 'contributor_email',
            'first_commit_date', 'first_commit_week', 'last_observed_date', 'last_observed_week',
            'total_weeks_observed', 'total_commits', 'total_lines_changed', 'total_active_weeks',
            'activity_rate', 'became_core', 'censored', 'time_to_event_weeks'
        ]

        core_specific_cols = [
            'first_core_date', 'first_core_week', 'weeks_to_core',
            'commits_to_core', 'lines_changed_to_core', 'active_weeks_to_core',
            'rank_at_first_core', 'contribution_percentage_at_first_core'
        ]

        # Check required columns
        missing = set(required_cols) - set(self.transitions_df.columns)
        if missing:
            self.errors.append(f"Missing required columns: {missing}")
            print(f"❌ Missing required columns: {missing}")
        else:
            print("✅ All required columns present")

        # Check data types
        type_issues = []

        # Boolean columns
        bool_cols = ['became_core', 'censored']
        for col in bool_cols:
            if col in self.transitions_df.columns:
                if self.transitions_df[col].dtype != bool:
                    type_issues.append(f"{col} should be boolean")

        # Numeric columns
        numeric_cols = ['total_commits', 'total_weeks_observed', 'weeks_to_core', 'commits_to_core']
        for col in numeric_cols:
            if col in self.transitions_df.columns:
                if not pd.api.types.is_numeric_dtype(self.transitions_df[col]):
                    type_issues.append(f"{col} should be numeric")

        if type_issues:
            self.warnings.extend(type_issues)
            print(f"⚠️ Data type issues: {len(type_issues)}")
            for issue in type_issues[:3]:
                print(f"   - {issue}")
        else:
            print("✅ All columns have appropriate data types")

    def test_instant_core_exclusion(self):
        """Test 2: Verify instant core contributors are excluded."""
        print("\n" + "=" * 70)
        print("TEST 2: INSTANT CORE EXCLUSION")
        print("=" * 70)

        # Check if any contributors have weeks_to_core = 0
        became_core = self.transitions_df[self.transitions_df['became_core'] == True]

        if len(became_core) > 0:
            instant_core = became_core[became_core['weeks_to_core'] == 0]

            if len(instant_core) > 0:
                self.errors.append(f"Found {len(instant_core)} instant core contributors (should be 0)")
                print(f"❌ Found {len(instant_core)} instant core contributors in filtered dataset")
                print(f"   These should have been excluded!")

                # Show samples
                print("\n   Sample instant core contributors:")
                for _, row in instant_core.head(3).iterrows():
                    print(f"   - {row['contributor_email'][:40]} in {row['project_name']}")
            else:
                print("✅ No instant core contributors found (correctly excluded)")

                # Show minimum weeks to core
                min_weeks = became_core['weeks_to_core'].min()
                print(f"   Minimum weeks to core: {min_weeks}")
        else:
            print("⚠️ No contributors became core in this dataset")

        # Compare with complete dataset if available
        if self.transitions_all_df is not None:
            all_became_core = self.transitions_all_df[self.transitions_all_df['became_core'] == True]
            all_instant = all_became_core[all_became_core['weeks_to_core'] == 0] if len(all_became_core) > 0 else pd.DataFrame()

            if len(all_instant) > 0:
                print(f"\nℹ️ Complete dataset has {len(all_instant)} instant core contributors")
                print(f"   These were correctly excluded from main dataset")

    def test_transition_logic(self):
        """Test 3: Verify transition logic and consistency."""
        print("\n" + "=" * 70)
        print("TEST 3: TRANSITION LOGIC VALIDATION")
        print("=" * 70)

        # Test censoring logic
        became_core = self.transitions_df[self.transitions_df['became_core'] == True]
        never_core = self.transitions_df[self.transitions_df['became_core'] == False]

        # Those who became core should NOT be censored
        if len(became_core) > 0:
            core_censored = became_core['censored'].sum()
            if core_censored > 0:
                self.errors.append(f"{core_censored} core contributors marked as censored")
                print(f"❌ {core_censored} core contributors incorrectly marked as censored")
            else:
                print("✅ Censoring correct for core contributors (all False)")

        # Those who never became core SHOULD be censored
        if len(never_core) > 0:
            never_core_not_censored = (~never_core['censored']).sum()
            if never_core_not_censored > 0:
                self.errors.append(f"{never_core_not_censored} non-core not censored")
                print(f"❌ {never_core_not_censored} non-core contributors not marked as censored")
            else:
                print("✅ Censoring correct for non-core contributors (all True)")

        # Test time_to_event logic
        issues = 0

        # For those who became core: time_to_event should equal weeks_to_core
        for _, row in became_core.iterrows():
            if pd.notna(row['weeks_to_core']):
                if row['time_to_event_weeks'] != row['weeks_to_core']:
                    issues += 1

        if issues > 0:
            self.errors.append(f"{issues} time_to_event mismatches for core contributors")
            print(f"❌ {issues} core contributors have time_to_event != weeks_to_core")
        else:
            print("✅ time_to_event correct for all core contributors")

        # For non-core: time_to_event should equal total_weeks_observed
        issues = 0
        for _, row in never_core.iterrows():
            if row['time_to_event_weeks'] != row['total_weeks_observed']:
                issues += 1

        if issues > 0:
            self.errors.append(f"{issues} time_to_event mismatches for non-core")
            print(f"❌ {issues} non-core contributors have incorrect time_to_event")
        else:
            print("✅ time_to_event correct for all non-core contributors")

    def test_data_integrity(self):
        """Test 4: Check data integrity and consistency."""
        print("\n" + "=" * 70)
        print("TEST 4: DATA INTEGRITY CHECKS")
        print("=" * 70)

        # Check for duplicates
        duplicates = self.transitions_df.duplicated(subset=['project_name', 'contributor_email']).sum()
        if duplicates > 0:
            self.errors.append(f"{duplicates} duplicate contributor-project pairs")
            print(f"❌ Found {duplicates} duplicate contributor-project pairs")
        else:
            print("✅ No duplicate contributor-project pairs")

        # Check for negative values where they shouldn't exist
        non_negative_cols = ['total_commits', 'total_weeks_observed', 'weeks_to_core', 
                            'commits_to_core', 'active_weeks_to_core']

        negative_issues = []
        for col in non_negative_cols:
            if col in self.transitions_df.columns:
                negative_count = (self.transitions_df[col] < 0).sum()
                if negative_count > 0:
                    negative_issues.append(f"{col}: {negative_count} negative values")

        if negative_issues:
            self.errors.extend(negative_issues)
            print(f"❌ Found negative values:")
            for issue in negative_issues:
                print(f"   - {issue}")
        else:
            print("✅ No negative values in numeric columns")

        # Check logical constraints
        logic_issues = []
        became_core = self.transitions_df[self.transitions_df['became_core'] == True]

        # commits_to_core <= total_commits for those who became core
        if len(became_core) > 0 and 'commits_to_core' in became_core.columns:
            invalid = (became_core['commits_to_core'] > became_core['total_commits']).sum()
            if invalid > 0:
                logic_issues.append(f"{invalid} have commits_to_core > total_commits")

        # active_weeks_to_core <= weeks_to_core + 1
        if len(became_core) > 0 and 'active_weeks_to_core' in became_core.columns:
            invalid = (became_core['active_weeks_to_core'] > became_core['weeks_to_core'] + 1).sum()
            if invalid > 0:
                logic_issues.append(f"{invalid} have active_weeks > weeks_to_core + 1")

        if logic_issues:
            self.warnings.extend(logic_issues)
            print(f"⚠️ Logical constraint issues:")
            for issue in logic_issues:
                print(f"   - {issue}")
        else:
            print("✅ All logical constraints satisfied")

        # Check for nulls in critical fields
        critical_non_null = ['project_name', 'contributor_email', 'became_core', 'censored']
        null_issues = []

        for col in critical_non_null:
            if col in self.transitions_df.columns:
                null_count = self.transitions_df[col].isnull().sum()
                if null_count > 0:
                    null_issues.append(f"{col}: {null_count} nulls")

        if null_issues:
            self.errors.extend(null_issues)
            print(f"❌ Null values in critical columns:")
            for issue in null_issues:
                print(f"   - {issue}")
        else:
            print("✅ No null values in critical columns")

    def test_statistical_validity(self):
        """Test 5: Statistical validity and distribution checks."""
        print("\n" + "=" * 70)
        print("TEST 5: STATISTICAL VALIDITY")
        print("=" * 70)

        total = len(self.transitions_df)
        became_core = self.transitions_df['became_core'].sum()
        core_rate = became_core / total * 100

        print(f"Dataset Overview:")
        print(f"  Total transitions: {total:,}")
        print(f"  Unique projects: {self.transitions_df['project_name'].nunique()}")
        print(f"  Unique contributors: {self.transitions_df['contributor_email'].nunique()}")
        print(f"  Became core: {became_core:,} ({core_rate:.1f}%)")

        # Check if core rate is reasonable (typically 5-30% for true newcomers)
        if core_rate < 1:
            self.warnings.append(f"Very low core rate: {core_rate:.1f}%")
            print(f"\n⚠️ Unusually low core achievement rate")
        elif core_rate > 50:
            self.warnings.append(f"Very high core rate: {core_rate:.1f}%")
            print(f"\n⚠️ Unusually high core achievement rate")
        else:
            print(f"\n✅ Core achievement rate seems reasonable for true newcomers")

        # Time to core statistics
        core_df = self.transitions_df[self.transitions_df['became_core'] == True]

        if len(core_df) > 0:
            print(f"\nTime to Core Statistics (n={len(core_df):,}):")
            print(f"  Median: {core_df['weeks_to_core'].median():.1f} weeks")
            print(f"  Mean: {core_df['weeks_to_core'].mean():.1f} weeks")
            print(f"  Std Dev: {core_df['weeks_to_core'].std():.1f} weeks")
            print(f"  Min: {core_df['weeks_to_core'].min()} weeks")
            print(f"  Max: {core_df['weeks_to_core'].max()} weeks")
            print(f"  Q25: {core_df['weeks_to_core'].quantile(0.25):.1f} weeks")
            print(f"  Q75: {core_df['weeks_to_core'].quantile(0.75):.1f} weeks")

            # Check if minimum is > 0 (no instant core)
            if core_df['weeks_to_core'].min() == 0:
                self.errors.append("Found instant core (weeks_to_core=0) in filtered dataset")
                print("\n❌ ERROR: Found instant core contributors (should be excluded)")
            else:
                print(f"\n✅ Minimum weeks to core is {core_df['weeks_to_core'].min()} (no instant core)")

            print(f"\nEffort to Core Statistics:")
            print(f"  Median: {core_df['commits_to_core'].median():.0f} commits")
            print(f"  Mean: {core_df['commits_to_core'].mean():.1f} commits")
            print(f"  Min: {core_df['commits_to_core'].min()} commits")
            print(f"  Max: {core_df['commits_to_core'].max()} commits")

    def test_project_type_comparison(self):
        """Test 6: Compare OSS vs OSS4SG statistics."""
        print("\n" + "=" * 70)
        print("TEST 6: PROJECT TYPE COMPARISON")
        print("=" * 70)

        for ptype in self.transitions_df['project_type'].unique():
            ptype_data = self.transitions_df[self.transitions_df['project_type'] == ptype]
            ptype_core = ptype_data[ptype_data['became_core'] == True]

            core_rate = len(ptype_core) / len(ptype_data) * 100 if len(ptype_data) > 0 else 0

            print(f"\n{ptype}:")
            print(f"  Total contributors: {len(ptype_data):,}")
            print(f"  Became core: {len(ptype_core):,} ({core_rate:.1f}%)")

            if len(ptype_core) > 0:
                print(f"  Time to core:")
                print(f"    Median: {ptype_core['weeks_to_core'].median():.1f} weeks")
                print(f"    Mean: {ptype_core['weeks_to_core'].mean():.1f} weeks")
                print(f"  Effort to core:")
                print(f"    Median: {ptype_core['commits_to_core'].median():.0f} commits")
                print(f"    Mean: {ptype_core['commits_to_core'].mean():.1f} commits")

                # Check minimum weeks (should be > 0)
                min_weeks = ptype_core['weeks_to_core'].min()
                if min_weeks == 0:
                    self.errors.append(f"{ptype} has instant core contributors")
                    print(f"  ❌ ERROR: Found instant core (min weeks = {min_weeks})")
                else:
                    print(f"  ✅ No instant core (min weeks = {min_weeks})")

    def test_sample_records(self):
        """Test 7: Inspect sample records for validity."""
        print("\n" + "=" * 70)
        print("TEST 7: SAMPLE RECORD INSPECTION")
        print("=" * 70)

        # Sample some core achievers
        core_df = self.transitions_df[self.transitions_df['became_core'] == True]

        if len(core_df) > 0:
            print("\nSample Core Achievers (excluding instant core):")
            sample = core_df.nsmallest(3, 'weeks_to_core')

            for idx, row in sample.iterrows():
                print(f"\n  {row['contributor_email'][:40]}...")
                print(f"    Project: {row['project_name']}")
                print(f"    Weeks to core: {row['weeks_to_core']}")
                print(f"    Commits to core: {row['commits_to_core']}")
                print(f"    Active weeks: {row['active_weeks_to_core']}")
                print(f"    Consistency: {row['commit_consistency_before_core']:.2f}")

        # Sample some who never became core
        never_core = self.transitions_df[self.transitions_df['became_core'] == False]

        if len(never_core) > 0:
            print("\nSample Never-Core Contributors:")
            sample = never_core.nlargest(3, 'total_commits')

            for idx, row in sample.iterrows():
                print(f"\n  {row['contributor_email'][:40]}...")
                print(f"    Project: {row['project_name']}")
                print(f"    Total commits: {row['total_commits']}")
                print(f"    Weeks observed: {row['total_weeks_observed']}")
                print(f"    Activity rate: {row['activity_rate']:.2f}")

    def generate_report(self):
        """Generate final validation report."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        if not self.errors:
            print("✅ ALL CRITICAL TESTS PASSED!")
            print("The contributor_transitions dataset is valid and ready for analysis.")

            if self.warnings:
                print(f"\n⚠️ {len(self.warnings)} warnings (non-critical):")
                for warning in self.warnings[:5]:
                    print(f"  - {warning}")
        else:
            print(f"❌ CRITICAL ISSUES FOUND: {len(self.errors)} errors")
            print("\nErrors must be fixed:")
            for error in self.errors:
                print(f"  - {error}")

            if self.warnings:
                print(f"\n⚠️ Also {len(self.warnings)} warnings to review")

        # Key statistics summary
        print("\n" + "-" * 40)
        print("KEY STATISTICS SUMMARY")
        print("-" * 40)

        core_df = self.transitions_df[self.transitions_df['became_core'] == True]

        print(f"Total records: {len(self.transitions_df):,}")
        print(f"Core achievement rate: {self.transitions_df['became_core'].mean()*100:.1f}%")

        if len(core_df) > 0:
            print(f"Median time to core: {core_df['weeks_to_core'].median():.1f} weeks")
            print(f"Median effort to core: {core_df['commits_to_core'].median():.0f} commits")
            print(f"Minimum weeks to core: {core_df['weeks_to_core'].min()} (should be > 0)")

        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'file_validated': str(self.transitions_file),
            'total_records': len(self.transitions_df),
            'errors': self.errors,
            'warnings': self.warnings,
            'passed': len(self.errors) == 0,
            'key_statistics': {
                'total_transitions': len(self.transitions_df),
                'became_core': int(self.transitions_df['became_core'].sum()),
                'core_rate_percent': float(self.transitions_df['became_core'].mean() * 100)
            }
        }

        if len(core_df) > 0:
            report['key_statistics'].update({
                'median_weeks_to_core': float(core_df['weeks_to_core'].median()),
                'min_weeks_to_core': int(core_df['weeks_to_core'].min()),
                'median_commits_to_core': float(core_df['commits_to_core'].median())
            })

        if self.transitions_all_df is not None:
            excluded_count = len(self.transitions_all_df) - len(self.transitions_df)
            report['instant_core_excluded'] = excluded_count

        report_path = self.transitions_file.parent / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Validation report saved: {report_path}")

        return len(self.errors) == 0

    def run_all_tests(self):
        """Run all validation tests."""
        self.load_data()
        self.test_schema()
        self.test_instant_core_exclusion()
        self.test_transition_logic()
        self.test_data_integrity()
        self.test_statistical_validity()
        self.test_project_type_comparison()
        self.test_sample_records()
        return self.generate_report()


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("CONTRIBUTOR TRANSITIONS DATASET VALIDATOR (Step 6)")
    print("=" * 70)

    validator = TransitionsValidator()
    success = validator.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


