#!/usr/bin/env python3
"""
Test Case for project_core_timeline_weekly.py
=============================================

Comprehensive test suite to validate core timeline generation logic.
Tests the 80% cumulative commits rule with controlled synthetic data.

Author: Research Team
Date: August 2025
"""

import pandas as pd
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# Add the scripts directory (parent of tester) to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the functions we want to test
from project_core_timeline_weekly import compute_weekly_core_timeline


class TestCoreTimelineWeekly:
    def __init__(self):
        self.test_dir = Path(tempfile.mkdtemp())
        print(f"Test directory: {self.test_dir}")
        
    def cleanup(self):
        """Clean up test files."""
        shutil.rmtree(self.test_dir)
        
    def create_test_data(self, scenario: str) -> pd.DataFrame:
        """Create controlled test data for different scenarios."""
        
        if scenario == "basic":
            # Simple scenario: 3 contributors over 2 weeks
            # Week 1: Alice=5, Bob=3, Charlie=2 commits (total=10)
            # Week 2: Alice=8, Bob=5, Charlie=3 commits (total=16)
            
            base_date = datetime(2023, 1, 2)  # Monday
            
            commits = []
            
            # Week 1 commits (Jan 2-8, 2023)
            for i in range(5):  # Alice: 5 commits
                commits.append({
                    'project_name': 'test/project',
                    'project_type': 'OSS',
                    'author_email': 'alice@test.com',
                    'commit_hash': f'alice_w1_{i}',
                    'commit_date': base_date + timedelta(hours=i*2)
                })
            
            for i in range(3):  # Bob: 3 commits
                commits.append({
                    'project_name': 'test/project',
                    'project_type': 'OSS',
                    'author_email': 'bob@test.com',
                    'commit_hash': f'bob_w1_{i}',
                    'commit_date': base_date + timedelta(days=1, hours=i*2)
                })
            
            for i in range(2):  # Charlie: 2 commits
                commits.append({
                    'project_name': 'test/project',
                    'project_type': 'OSS',
                    'author_email': 'charlie@test.com',
                    'commit_hash': f'charlie_w1_{i}',
                    'commit_date': base_date + timedelta(days=2, hours=i*2)
                })
            
            # Week 2 commits (Jan 9-15, 2023)
            week2_base = base_date + timedelta(days=7)
            
            for i in range(3):  # Alice: +3 more commits (total=8)
                commits.append({
                    'project_name': 'test/project',
                    'project_type': 'OSS',
                    'author_email': 'alice@test.com',
                    'commit_hash': f'alice_w2_{i}',
                    'commit_date': week2_base + timedelta(hours=i*2)
                })
            
            for i in range(2):  # Bob: +2 more commits (total=5)
                commits.append({
                    'project_name': 'test/project',
                    'project_type': 'OSS',
                    'author_email': 'bob@test.com',
                    'commit_hash': f'bob_w2_{i}',
                    'commit_date': week2_base + timedelta(days=1, hours=i*2)
                })
            
            for i in range(1):  # Charlie: +1 more commit (total=3)
                commits.append({
                    'project_name': 'test/project',
                    'project_type': 'OSS',
                    'author_email': 'charlie@test.com',
                    'commit_hash': f'charlie_w2_{i}',
                    'commit_date': week2_base + timedelta(days=2, hours=i*2)
                })
            
            return pd.DataFrame(commits)
        
        elif scenario == "edge_case_80_percent":
            # Designed to test exact 80% threshold behavior
            # 10 contributors, carefully constructed commits
            
            base_date = datetime(2023, 1, 2)  # Monday
            commits = []
            
            # Create commits such that exactly 80% of commits come from specific contributors
            contributors = [f'user{i}@test.com' for i in range(10)]
            commit_counts = [20, 15, 10, 8, 7, 5, 4, 3, 2, 1]  # Total = 75 commits
            
            commit_id = 0
            for i, (contributor, count) in enumerate(zip(contributors, commit_counts)):
                for j in range(count):
                    commits.append({
                        'project_name': 'test/edge',
                        'project_type': 'OSS4SG',
                        'author_email': contributor,
                        'commit_hash': f'commit_{commit_id}',
                        'commit_date': base_date + timedelta(hours=commit_id)
                    })
                    commit_id += 1
            
            return pd.DataFrame(commits)
        
        elif scenario == "single_contributor":
            # Edge case: only one contributor
            base_date = datetime(2023, 1, 2)
            
            commits = []
            for i in range(5):
                commits.append({
                    'project_name': 'test/single',
                    'project_type': 'OSS',
                    'author_email': 'solo@test.com',
                    'commit_hash': f'solo_{i}',
                    'commit_date': base_date + timedelta(hours=i*6)
                })
            
            return pd.DataFrame(commits)
        
        elif scenario == "empty_project":
            # Edge case: no commits
            return pd.DataFrame(columns=['project_name', 'project_type', 'author_email', 'commit_hash', 'commit_date'])
        
        else:
            raise ValueError(f"Unknown test scenario: {scenario}")
    
    def test_basic_scenario(self):
        """Test basic weekly core timeline generation."""
        print("\n=== Testing Basic Scenario ===")
        
        test_data = self.create_test_data("basic")
        print(f"Created test data with {len(test_data)} commits")
        print("Expected behavior:")
        print("Week 1: Alice=5, Bob=3, Charlie=2 (total=10)")
        print("  80% of 10 = 8 commits")
        print("  Alice(5) + Bob(3) = 8 commits exactly = 80%")
        print("  Core contributors: Alice, Bob")
        print()
        print("Week 2: Alice=8, Bob=5, Charlie=3 (total=16)")
        print("  80% of 16 = 12.8 commits")
        print("  Alice(8) + Bob(5) = 13 commits > 80%")
        print("  Core contributors: Alice, Bob")
        
        # Run the function
        result = compute_weekly_core_timeline(test_data, core_threshold_percentile=80)
        
        print(f"\nActual results:")
        print(result[['week_number', 'total_commits_to_date', 'core_threshold_commits', 'core_contributors_count']].to_string())
        
        # Validate Week 1
        week1 = result[result['week_number'] == 1].iloc[0]
        week1_core_emails = json.loads(week1['core_contributors_emails'])
        
        assert week1['total_commits_to_date'] == 10, f"Week 1 should have 10 commits, got {week1['total_commits_to_date']}"
        assert week1['core_contributors_count'] == 2, f"Week 1 should have 2 core contributors, got {week1['core_contributors_count']}"
        assert set(week1_core_emails) == {'alice@test.com', 'bob@test.com'}, f"Week 1 core should be Alice and Bob, got {week1_core_emails}"
        
        # Validate Week 2
        week2 = result[result['week_number'] == 2].iloc[0]
        week2_core_emails = json.loads(week2['core_contributors_emails'])
        
        assert week2['total_commits_to_date'] == 16, f"Week 2 should have 16 commits, got {week2['total_commits_to_date']}"
        assert week2['core_contributors_count'] == 2, f"Week 2 should have 2 core contributors, got {week2['core_contributors_count']}"
        assert set(week2_core_emails) == {'alice@test.com', 'bob@test.com'}, f"Week 2 core should be Alice and Bob, got {week2_core_emails}"
        
        print("Basic scenario test PASSED")
        return True
    
    def test_80_percent_threshold(self):
        """Test exact 80% threshold calculation."""
        print("\n=== Testing 80% Threshold Logic ===")
        
        test_data = self.create_test_data("edge_case_80_percent")
        print(f"Created test data with {len(test_data)} commits")
        print("Contributors and commits: [20, 15, 10, 8, 7, 5, 4, 3, 2, 1] = 75 total")
        print("80% of 75 = 60 commits")
        print("Cumulative: user0(20)=20, user1(15)=35, user2(10)=45, user3(8)=53, user4(7)=60")
        print("Expected core: user0, user1, user2, user3, user4 (5 contributors)")
        
        result = compute_weekly_core_timeline(test_data, core_threshold_percentile=80)
        
        print(f"\nActual results:")
        print(result[['week_number', 'total_commits_to_date', 'core_threshold_commits', 'core_contributors_count']].to_string())
        
        week1 = result[result['week_number'] == 1].iloc[0]
        core_emails = json.loads(week1['core_contributors_emails'])
        
        assert week1['total_commits_to_date'] == 75, f"Should have 75 commits, got {week1['total_commits_to_date']}"
        assert week1['core_contributors_count'] == 5, f"Should have 5 core contributors, got {week1['core_contributors_count']}"
        
        expected_core = {'user0@test.com', 'user1@test.com', 'user2@test.com', 'user3@test.com', 'user4@test.com'}
        assert set(core_emails) == expected_core, f"Core should be top 5 contributors, got {set(core_emails)}"
        
        print("80% threshold test PASSED")
        return True
    
    def test_single_contributor(self):
        """Test edge case with only one contributor."""
        print("\n=== Testing Single Contributor Edge Case ===")
        
        test_data = self.create_test_data("single_contributor")
        result = compute_weekly_core_timeline(test_data, core_threshold_percentile=80)
        
        print(f"Results:")
        print(result[['week_number', 'total_commits_to_date', 'core_threshold_commits', 'core_contributors_count']].to_string())
        
        week1 = result[result['week_number'] == 1].iloc[0]
        core_emails = json.loads(week1['core_contributors_emails'])
        
        assert week1['core_contributors_count'] == 1, f"Should have 1 core contributor, got {week1['core_contributors_count']}"
        assert core_emails == ['solo@test.com'], f"Core should be solo contributor, got {core_emails}"
        
        print("Single contributor test PASSED")
        return True
    
    def test_empty_project(self):
        """Test edge case with no commits."""
        print("\n=== Testing Empty Project Edge Case ===")
        
        test_data = self.create_test_data("empty_project")
        result = compute_weekly_core_timeline(test_data, core_threshold_percentile=80)
        
        assert len(result) == 0, f"Empty project should produce no results, got {len(result)} rows"
        
        print("Empty project test PASSED")
        return True
    
    def test_data_integrity(self):
        """Test data integrity and format validation."""
        print("\n=== Testing Data Integrity ===")
        
        test_data = self.create_test_data("basic")
        result = compute_weekly_core_timeline(test_data, core_threshold_percentile=80)
        
        # Check required columns
        required_columns = [
            'project_name', 'project_type', 'week_date', 'week_number',
            'total_commits_to_date', 'total_contributors_to_date',
            'core_threshold_commits', 'core_contributors_count',
            'core_contributors_emails', 'core_method', 'core_threshold_percentile'
        ]
        
        for col in required_columns:
            assert col in result.columns, f"Missing required column: {col}"
        
        # Check data types and values
        assert result['week_number'].dtype == 'int64', "week_number should be integer"
        assert result['core_threshold_percentile'].iloc[0] == 80, "core_threshold_percentile should be 80"
        assert result['core_method'].iloc[0] == 'cumulative_commits_to_date_percentile', "core_method should be cumulative_commits_to_date_percentile"
        
        # Validate JSON format of core_contributors_emails
        for idx, row in result.iterrows():
            try:
                emails = json.loads(row['core_contributors_emails'])
                assert isinstance(emails, list), f"core_contributors_emails should be JSON list, got {type(emails)}"
                assert len(emails) == row['core_contributors_count'], f"JSON list length should match core_contributors_count"
            except json.JSONDecodeError as e:
                assert False, f"core_contributors_emails should be valid JSON: {e}"
        
        # Check week numbering
        assert result['week_number'].min() == 1, "Week numbering should start at 1"
        assert (result['week_number'].diff().dropna() >= 0).all(), "Week numbers should be non-decreasing"
        
        print("Data integrity test PASSED")
        return True
    
    def test_different_percentiles(self):
        """Test different core threshold percentiles."""
        print("\n=== Testing Different Percentiles ===")
        
        test_data = self.create_test_data("basic")
        
        # Test 50%, 80%, 90% thresholds
        for percentile in [50, 80, 90]:
            result = compute_weekly_core_timeline(test_data, core_threshold_percentile=percentile)
            
            assert len(result) > 0, f"Should generate results for {percentile}% threshold"
            assert result['core_threshold_percentile'].iloc[0] == percentile, f"Should record correct percentile: {percentile}"
            
            # Higher percentiles should generally have fewer or same core contributors
            if percentile == 50:
                count_50 = result.iloc[0]['core_contributors_count']
            elif percentile == 80:
                count_80 = result.iloc[0]['core_contributors_count']
            elif percentile == 90:
                count_90 = result.iloc[0]['core_contributors_count']
        
        # Logical relationship under minimal-top-set rule: core size is non-decreasing with higher percentiles
        assert count_50 <= count_80 <= count_90, f"Core count should increase (or stay the same) with higher percentiles: 50%({count_50}) <= 80%({count_80}) <= 90%({count_90})"
        
        print("Different percentiles test PASSED")
        return True
    
    def run_all_tests(self):
        """Run all test cases."""
        print("RUNNING CORE TIMELINE WEEKLY TESTS")
        print("=" * 50)
        
        tests = [
            self.test_basic_scenario,
            self.test_80_percent_threshold,
            self.test_single_contributor,
            self.test_empty_project,
            self.test_data_integrity,
            self.test_different_percentiles
        ]
        
        passed = 0
        failed = 0
        
        for test_func in tests:
            try:
                result = test_func()
                if result:
                    passed += 1
            except Exception as e:
                print(f"ERROR: {test_func.__name__} FAILED: {str(e)}")
                failed += 1
        
        print("\n" + "=" * 50)
        print(f"TEST SUMMARY: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("ALL TESTS PASSED!")
            return True
        else:
            print("SOME TESTS FAILED!")
            return False


def main():
    """Main test execution."""
    tester = TestCoreTimelineWeekly()
    
    try:
        success = tester.run_all_tests()
        return 0 if success else 1
    finally:
        tester.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

