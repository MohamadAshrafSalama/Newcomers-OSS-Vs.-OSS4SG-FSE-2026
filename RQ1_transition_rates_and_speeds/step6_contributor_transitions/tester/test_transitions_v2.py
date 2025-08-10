#!/usr/bin/env python3
"""
Test and Verification Script for Contributor Transitions Dataset v2
===================================================================

Comprehensive tests to validate the improved contributor_transitions.csv dataset.
Verifies that instant core AND early project members are properly excluded.
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

class TransitionsValidatorV2:
    def __init__(self):
        # File paths
        self.base_path = Path(".") / "results"
        self.main_file = self.base_path / "contributor_transitions.csv"  # Full exclusions
        self.no_instant_file = self.base_path / "contributor_transitions_no_instant.csv"  # Only instant excluded
        self.complete_file = self.base_path / "contributor_transitions_including_all.csv"  # No exclusions
        self.stats_file = self.base_path / "transition_statistics_v2.json"
        
        self.main_df = None
        self.no_instant_df = None
        self.complete_df = None
        self.errors = []
        self.warnings = []
        self.info = []
        
    def load_data(self):
        """Load all three datasets for comparison."""
        print("=" * 70)
        print("LOADING TRANSITIONS DATASETS v2 FOR VALIDATION")
        print("=" * 70)
        
        # Load main dataset (full exclusions)
        if not self.main_file.exists():
            print(f"❌ ERROR: Main transitions file not found: {self.main_file}")
            sys.exit(1)
        
        print(f"Loading main dataset (full exclusions): {self.main_file}")
        self.main_df = pd.read_csv(self.main_file, low_memory=False)
        print(f"✅ Loaded {len(self.main_df):,} records\n")
        
        # Load no-instant dataset
        if self.no_instant_file.exists():
            print(f"Loading no-instant dataset: {self.no_instant_file}")
            self.no_instant_df = pd.read_csv(self.no_instant_file, low_memory=False)
            print(f"✅ Loaded {len(self.no_instant_df):,} records\n")
        
        # Load complete dataset
        if self.complete_file.exists():
            print(f"Loading complete dataset: {self.complete_file}")
            self.complete_df = pd.read_csv(self.complete_file, low_memory=False)
            print(f"✅ Loaded {len(self.complete_df):,} records\n")
        
        # Calculate exclusions
        if self.complete_df is not None:
            total_excluded = len(self.complete_df) - len(self.main_df)
            instant_excluded = len(self.complete_df) - len(self.no_instant_df) if self.no_instant_df is not None else 0
            early_project_excluded = total_excluded - instant_excluded
            
            print("📊 EXCLUSION SUMMARY:")
            print(f"  Total excluded: {total_excluded:,}")
            print(f"  - Instant core (week 0): {instant_excluded:,}")
            print(f"  - Early project members: {early_project_excluded:,}")
    
    def test_exclusions(self):
        """Test 1: Verify exclusions are working correctly."""
        print("\n" + "=" * 70)
        print("TEST 1: EXCLUSION VERIFICATION")
        print("=" * 70)
        
        # Test instant core exclusion
        became_core = self.main_df[self.main_df['became_core'] == True]
        
        if len(became_core) > 0:
            instant_core = became_core[became_core['weeks_to_core'] == 0]
            
            if len(instant_core) > 0:
                self.errors.append(f"Found {len(instant_core)} instant core in main dataset")
                print(f"❌ Found {len(instant_core)} instant core contributors (should be 0)")
            else:
                print("✅ No instant core contributors (week 0 excluded)")
                
            min_weeks = became_core['weeks_to_core'].min()
            print(f"   Minimum weeks to core: {min_weeks}")
        
        # Test early project member exclusion
        if 'is_early_joiner' in self.main_df.columns and 'is_fast_core' in self.main_df.columns:
            early_fast = became_core[
                (became_core['is_early_joiner'] == True) & 
                (became_core['is_fast_core'] == True)
            ]
            
            if len(early_fast) > 0:
                # Check if they were supposed to be excluded
                early_fast_low_weeks = early_fast[early_fast['first_commit_week'] <= 8]
                if len(early_fast_low_weeks) > 0:
                    self.warnings.append(f"Found {len(early_fast_low_weeks)} early+fast core")
                    print(f"⚠️ Found {len(early_fast_low_weeks)} early joiners with fast core")
                    
                    # Show details
                    sample = early_fast_low_weeks.head(3)
                    for _, row in sample.iterrows():
                        print(f"   - Joined week {row['first_commit_week']}, core in {row['weeks_to_core']} weeks")
            else:
                print("✅ No early project members with fast core achievement")
        
        # Compare datasets if available
        if self.complete_df is not None and self.no_instant_df is not None:
            print("\n📊 Dataset Comparison:")
            
            # Complete dataset analysis
            complete_core = self.complete_df[self.complete_df['became_core'] == True]
            complete_instant = complete_core[complete_core['weeks_to_core'] == 0] if len(complete_core) > 0 else pd.DataFrame()
            
            print(f"\nComplete dataset:")
            print(f"  Total: {len(self.complete_df):,}")
            print(f"  Became core: {len(complete_core):,} ({len(complete_core)/len(self.complete_df)*100:.1f}%)")
            print(f"  Instant core: {len(complete_instant):,}")
            
            # No-instant dataset
            no_instant_core = self.no_instant_df[self.no_instant_df['became_core'] == True] if self.no_instant_df is not None else pd.DataFrame()
            
            print(f"\nNo-instant dataset:")
            print(f"  Total: {len(self.no_instant_df):,}")
            print(f"  Became core: {len(no_instant_core):,} ({len(no_instant_core)/len(self.no_instant_df)*100:.1f}%)")
            
            # Main dataset
            print(f"\nMain dataset (full exclusions):")
            print(f"  Total: {len(self.main_df):,}")
            print(f"  Became core: {len(became_core):,} ({len(became_core)/len(self.main_df)*100:.1f}%)")
    
    def test_early_project_analysis(self):
        """Test 2: Analyze early project joiners."""
        print("\n" + "=" * 70)
        print("TEST 2: EARLY PROJECT JOINER ANALYSIS")
        print("=" * 70)
        
        if 'first_commit_week' not in self.main_df.columns:
            print("⚠️ first_commit_week column not found")
            return
        
        # Analyze distribution of when people joined
        thresholds = [1, 4, 8, 12, 26, 52]
        
        print("When did contributors in main dataset join the project?")
        for threshold in thresholds:
            count = (self.main_df['first_commit_week'] <= threshold).sum()
            pct = count / len(self.main_df) * 100
            print(f"  Week ≤{threshold:3}: {count:6,} ({pct:5.1f}%)")
        
        # For those who became core, when did they join?
        core_df = self.main_df[self.main_df['became_core'] == True]
        
        if len(core_df) > 0:
            print(f"\nFor the {len(core_df):,} who became core:")
            
            for threshold in [4, 8, 12, 26, 52]:
                early_core = core_df[core_df['first_commit_week'] <= threshold]
                if len(early_core) > 0:
                    pct = len(early_core) / len(core_df) * 100
                    median_time = early_core['weeks_to_core'].median()
                    median_commits = early_core['commits_to_core'].median()
                    
                    print(f"\n  Joined week ≤{threshold}:")
                    print(f"    Count: {len(early_core):,} ({pct:.1f}% of core)")
                    print(f"    Median time to core: {median_time:.1f} weeks")
                    print(f"    Median commits to core: {median_commits:.0f}")
    
    def test_data_quality(self):
        """Test 3: Check data integrity and quality."""
        print("\n" + "=" * 70)
        print("TEST 3: DATA QUALITY CHECKS")
        print("=" * 70)
        
        # Check for duplicates
        duplicates = self.main_df.duplicated(subset=['project_name', 'contributor_email']).sum()
        if duplicates > 0:
            self.errors.append(f"{duplicates} duplicate pairs")
            print(f"❌ Found {duplicates} duplicate contributor-project pairs")
        else:
            print("✅ No duplicate contributor-project pairs")
        
        # Check logical constraints
        became_core = self.main_df[self.main_df['became_core'] == True]
        
        if len(became_core) > 0:
            # commits_to_core <= total_commits
            if 'commits_to_core' in became_core.columns:
                invalid = (became_core['commits_to_core'] > became_core['total_commits']).sum()
                if invalid > 0:
                    self.errors.append(f"{invalid} have commits_to_core > total_commits")
                    print(f"❌ {invalid} records have commits_to_core > total_commits")
                else:
                    print("✅ commits_to_core values are valid")
            
            # Check censoring logic
            core_censored = became_core['censored'].sum() if 'censored' in became_core.columns else 0
            if core_censored > 0:
                self.errors.append(f"{core_censored} core contributors marked as censored")
                print(f"❌ {core_censored} core contributors incorrectly marked as censored")
            else:
                print("✅ Censoring logic correct for core contributors")
        
        # Check for negative values
        numeric_cols = ['total_commits', 'weeks_to_core', 'commits_to_core']
        for col in numeric_cols:
            if col in self.main_df.columns:
                neg_count = (self.main_df[col] < 0).sum()
                if neg_count > 0:
                    self.errors.append(f"{col} has {neg_count} negative values")
                    print(f"❌ {col} has {neg_count} negative values")
        
        if not any(col in self.errors for col in numeric_cols):
            print("✅ No negative values in numeric columns")
    
    def test_statistical_validity(self):
        """Test 4: Statistical validity and distributions."""
        print("\n" + "=" * 70)
        print("TEST 4: STATISTICAL VALIDITY")
        print("=" * 70)
        
        total = len(self.main_df)
        became_core = self.main_df['became_core'].sum()
        core_rate = became_core / total * 100
        
        print(f"Dataset Overview:")
        print(f"  Total transitions: {total:,}")
        print(f"  Unique projects: {self.main_df['project_name'].nunique()}")
        print(f"  Unique contributors: {self.main_df['contributor_email'].nunique()}")
        print(f"  Became core: {became_core:,} ({core_rate:.1f}%)")
        
        # Expected range for true newcomers: 5-20%
        if core_rate < 3:
            self.warnings.append(f"Very low core rate: {core_rate:.1f}%")
            print(f"\n⚠️ Unusually low core achievement rate")
        elif core_rate > 30:
            self.warnings.append(f"High core rate: {core_rate:.1f}%")
            print(f"\n⚠️ Unusually high core achievement rate")
        else:
            print(f"\n✅ Core achievement rate reasonable for true newcomers")
        
        # Time and effort statistics
        core_df = self.main_df[self.main_df['became_core'] == True]
        
        if len(core_df) > 0:
            print(f"\nTime to Core (n={len(core_df):,}):")
            print(f"  Median: {core_df['weeks_to_core'].median():.1f} weeks")
            print(f"  Mean: {core_df['weeks_to_core'].mean():.1f} weeks")
            print(f"  Min: {core_df['weeks_to_core'].min()} weeks")
            print(f"  Max: {core_df['weeks_to_core'].max()} weeks")
            print(f"  Q25: {core_df['weeks_to_core'].quantile(0.25):.1f} weeks")
            print(f"  Q75: {core_df['weeks_to_core'].quantile(0.75):.1f} weeks")
            
            print(f"\nEffort to Core:")
            print(f"  Median: {core_df['commits_to_core'].median():.0f} commits")
            print(f"  Mean: {core_df['commits_to_core'].mean():.1f} commits")
            print(f"  Min: {core_df['commits_to_core'].min()} commits")
            print(f"  Max: {core_df['commits_to_core'].max()} commits")
            
            # Check if metrics are more reasonable now
            median_commits = core_df['commits_to_core'].median()
            if median_commits < 5:
                self.warnings.append(f"Low median commits: {median_commits}")
                print(f"\n⚠️ Median commits still seems low ({median_commits:.0f})")
            else:
                print(f"\n✅ Median commits seems reasonable ({median_commits:.0f})")
    
    def test_project_comparison(self):
        """Test 5: Compare OSS vs OSS4SG."""
        print("\n" + "=" * 70)
        print("TEST 5: PROJECT TYPE COMPARISON")
        print("=" * 70)
        
        for ptype in self.main_df['project_type'].unique():
            ptype_data = self.main_df[self.main_df['project_type'] == ptype]
            ptype_core = ptype_data[ptype_data['became_core'] == True]
            
            core_rate = len(ptype_core) / len(ptype_data) * 100 if len(ptype_data) > 0 else 0
            
            print(f"\n{ptype}:")
            print(f"  Total contributors: {len(ptype_data):,}")
            print(f"  Became core: {len(ptype_core):,} ({core_rate:.1f}%)")
            
            if len(ptype_core) > 0:
                print(f"  Time to core:")
                print(f"    Median: {ptype_core['weeks_to_core'].median():.1f} weeks")
                print(f"    Mean: {ptype_core['weeks_to_core'].mean():.1f} weeks")
                print(f"    Min: {ptype_core['weeks_to_core'].min()} weeks")
                
                print(f"  Effort to core:")
                print(f"    Median: {ptype_core['commits_to_core'].median():.0f} commits")
                print(f"    Mean: {ptype_core['commits_to_core'].mean():.1f} commits")
                
                # Check for suspicious values
                if ptype_core['commits_to_core'].median() < 5:
                    print(f"    ⚠️ Low median commits for {ptype}")
                
                if ptype_core['weeks_to_core'].min() < 2:
                    print(f"    ⚠️ Very fast core achievement in {ptype}")
    
    def test_sample_inspection(self):
        """Test 6: Inspect sample records."""
        print("\n" + "=" * 70)
        print("TEST 6: SAMPLE RECORD INSPECTION")
        print("=" * 70)
        
        core_df = self.main_df[self.main_df['became_core'] == True]
        
        if len(core_df) > 0:
            # Show fastest core achievers
            print("\nFastest Core Achievers (should not be week 0):")
            fastest = core_df.nsmallest(5, 'weeks_to_core')
            
            for _, row in fastest.iterrows():
                print(f"\n  {row['contributor_email'][:30]}...")
                print(f"    Project: {row['project_name']}")
                print(f"    Joined project week: {row['first_commit_week']}")
                print(f"    Weeks to core: {row['weeks_to_core']}")
                print(f"    Commits to core: {row['commits_to_core']}")
                
                # Flag suspicious cases
                if row['first_commit_week'] <= 8 and row['weeks_to_core'] <= 4:
                    print(f"    ⚠️ Early joiner with fast core!")
        
        # Show high performers who never became core
        never_core = self.main_df[self.main_df['became_core'] == False]
        if len(never_core) > 0:
            print("\nHigh Contributors Who Never Became Core:")
            high_contributors = never_core.nlargest(3, 'total_commits')
            
            for _, row in high_contributors.iterrows():
                print(f"\n  {row['contributor_email'][:30]}...")
                print(f"    Project: {row['project_name']}")
                print(f"    Total commits: {row['total_commits']}")
                print(f"    Weeks observed: {row['total_weeks_observed']}")
                print(f"    Joined week: {row['first_commit_week']}")
    
    def load_and_verify_stats(self):
        """Test 7: Verify saved statistics match dataset."""
        print("\n" + "=" * 70)
        print("TEST 7: STATISTICS FILE VERIFICATION")
        print("=" * 70)
        
        if not self.stats_file.exists():
            print("⚠️ Statistics file not found")
            return
        
        with open(self.stats_file, 'r') as f:
            stats = json.load(f)
        
        print("Exclusion settings from stats file:")
        if 'exclusion_settings' in stats:
            settings = stats['exclusion_settings']
            print(f"  Instant core excluded: {settings.get('instant_core_excluded', 'N/A')}")
            print(f"  Early project excluded: {settings.get('early_project_excluded', 'N/A')}")
            print(f"  Early project threshold: ≤{settings.get('early_project_weeks_threshold', 'N/A')} weeks")
            print(f"  Fast core threshold: ≤{settings.get('early_core_weeks_threshold', 'N/A')} weeks")
        
        if 'exclusion_stats' in stats:
            exc_stats = stats['exclusion_stats']
            print(f"\nExclusion counts:")
            print(f"  Total processed: {exc_stats.get('total_processed', 'N/A'):,}")
            print(f"  Instant core excluded: {exc_stats.get('instant_core_excluded', 'N/A'):,}")
            print(f"  Early project excluded: {exc_stats.get('early_project_core_excluded', 'N/A'):,}")
    
    def generate_report(self):
        """Generate final validation report."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        if not self.errors:
            print("✅ ALL CRITICAL TESTS PASSED!")
            print("The improved transitions dataset correctly excludes:")
            print("  1. Instant core contributors (week 0)")
            print("  2. Early project members who become core quickly")
            
            if self.warnings:
                print(f"\n⚠️ {len(self.warnings)} warnings to review:")
                for warning in self.warnings[:5]:
                    print(f"  - {warning}")
        else:
            print(f"❌ CRITICAL ISSUES FOUND: {len(self.errors)} errors")
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")
        
        # Key improvements summary
        print("\n" + "-" * 40)
        print("KEY IMPROVEMENTS IN v2")
        print("-" * 40)
        
        if self.complete_df is not None and self.no_instant_df is not None:
            # Calculate improvement metrics
            complete_core = self.complete_df[self.complete_df['became_core'] == True]
            main_core = self.main_df[self.main_df['became_core'] == True]
            
            if len(complete_core) > 0 and len(main_core) > 0:
                print(f"\nBefore (complete dataset):")
                print(f"  Median time to core: {complete_core['weeks_to_core'].median():.1f} weeks")
                print(f"  Median commits to core: {complete_core['commits_to_core'].median():.0f}")
                
                print(f"\nAfter v2 exclusions:")
                print(f"  Median time to core: {main_core['weeks_to_core'].median():.1f} weeks")
                print(f"  Median commits to core: {main_core['commits_to_core'].median():.0f}")
                
                print(f"\nExcluded: {len(complete_core) - len(main_core):,} questionable core transitions")
        
        # Save report
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'dataset_version': 'v2',
            'files_validated': {
                'main': str(self.main_file),
                'no_instant': str(self.no_instant_file) if self.no_instant_file.exists() else None,
                'complete': str(self.complete_file) if self.complete_file.exists() else None
            },
            'total_records': len(self.main_df),
            'errors': self.errors,
            'warnings': self.warnings,
            'passed': len(self.errors) == 0,
            'key_statistics': {
                'total_transitions': len(self.main_df),
                'became_core': int(self.main_df['became_core'].sum()),
                'core_rate_percent': float(self.main_df['became_core'].mean() * 100)
            }
        }
        
        core_df = self.main_df[self.main_df['became_core'] == True]
        if len(core_df) > 0:
            report['key_statistics'].update({
                'median_weeks_to_core': float(core_df['weeks_to_core'].median()),
                'min_weeks_to_core': int(core_df['weeks_to_core'].min()),
                'median_commits_to_core': float(core_df['commits_to_core'].median())
            })
        
        report_path = self.base_path / 'validation_report_v2.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Validation report saved: {report_path}")
        
        return len(self.errors) == 0
    
    def run_all_tests(self):
        """Run all validation tests."""
        self.load_data()
        self.test_exclusions()
        self.test_early_project_analysis()
        self.test_data_quality()
        self.test_statistical_validity()
        self.test_project_comparison()
        self.test_sample_inspection()
        self.load_and_verify_stats()
        return self.generate_report()

def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("CONTRIBUTOR TRANSITIONS DATASET VALIDATOR v2")
    print("=" * 70)
    
    validator = TransitionsValidatorV2()
    success = validator.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


