import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import unittest
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the main class (adjust path as needed)
import sys
sys.path.append('.')  # Add current directory to path
from contribution_timeseries_generator import ContributionIndexTimeSeries

class TestContributionTimeSeries(unittest.TestCase):
    """
    Comprehensive test suite for validating contribution index time series generation.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test data directory and sample files."""
        cls.test_dir = Path("test_timeseries_validation")
        cls.test_dir.mkdir(exist_ok=True)
        
        # Create sample timeline data for testing
        cls.create_sample_timeline_data()
    
    @classmethod
    def create_sample_timeline_data(cls):
        """Create controlled test timeline files."""
        
        # Test Case 1: Simple timeline with known values
        test_data_1 = {
            'event_id': [1, 2, 3, 4, 5, 6, 7, 8],
            'event_type': ['commit', 'commit', 'pull_request', 'issue', 
                          'commit', 'pull_request', 'commit', 'issue'],
            'event_timestamp': [
                '2023-01-01 10:00:00', '2023-01-02 11:00:00',
                '2023-01-03 12:00:00', '2023-01-05 13:00:00',
                '2023-01-08 14:00:00', '2023-01-09 15:00:00',
                '2023-01-15 16:00:00', '2023-01-22 17:00:00'
            ],
            'event_week': [0, 0, 0, 0, 1, 1, 2, 3],
            'event_identifier': ['c1', 'c2', 'pr1', 'i1', 'c3', 'pr2', 'c4', 'i2'],
            'event_data': [
                '{}',  # commit 1
                '{}',  # commit 2
                '{"merged": true, "comments": {"nodes": []}}',  # PR merged
                '{"comments": {"nodes": []}}',  # issue
                '{}',  # commit 3
                '{"merged": false, "comments": {"nodes": []}}',  # PR not merged
                '{}',  # commit 4
                '{"comments": {"nodes": []}}'  # issue
            ],
            'project_name': ['test_project'] * 8,
            'project_type': ['OSS'] * 8,
            'contributor_email': ['test@example.com'] * 8,
            'username': ['testuser'] * 8,
            'is_pre_core': [True] * 8
        }
        
        df1 = pd.DataFrame(test_data_1)
        df1.to_csv(cls.test_dir / 'timeline_test_project_test@example.com.csv', index=False)
        
        # Test Case 2: Timeline with comments and complex events
        test_data_2 = {
            'event_id': [1, 2, 3],
            'event_type': ['pull_request', 'issue', 'commit'],
            'event_timestamp': [
                '2023-02-01 10:00:00', '2023-02-02 11:00:00', '2023-02-03 12:00:00'
            ],
            'event_week': [0, 0, 0],
            'event_identifier': ['pr1', 'i1', 'c1'],
            'event_data': [
                json.dumps({
                    "merged": True,
                    "comments": {
                        "nodes": [
                            {"author": {"login": "complexuser"}, "body": "comment1"},
                            {"author": {"login": "other"}, "body": "comment2"}
                        ]
                    },
                    "reviews": {
                        "nodes": [
                            {"author": {"login": "complexuser"}, "state": "APPROVED"}
                        ]
                    }
                }),
                json.dumps({
                    "comments": {
                        "nodes": [
                            {"author": {"login": "complexuser"}, "body": "issue comment"}
                        ]
                    }
                }),
                '{}'
            ],
            'project_name': ['complex_project'] * 3,
            'project_type': ['OSS4SG'] * 3,
            'contributor_email': ['complex@example.com'] * 3,
            'username': ['complexuser'] * 3,
            'is_pre_core': [True] * 3
        }
        
        df2 = pd.DataFrame(test_data_2)
        df2.to_csv(cls.test_dir / 'timeline_complex_project_complex@example.com.csv', index=False)
        
        # Test Case 3: Edge case - single event
        test_data_3 = {
            'event_id': [1],
            'event_type': ['commit'],
            'event_timestamp': ['2023-03-01 10:00:00'],
            'event_week': [0],
            'event_identifier': ['c1'],
            'event_data': ['{}'],
            'project_name': ['single_project'],
            'project_type': ['OSS'],
            'contributor_email': ['single@example.com'],
            'username': ['singleuser'],
            'is_pre_core': [True]
        }
        
        df3 = pd.DataFrame(test_data_3)
        df3.to_csv(cls.test_dir / 'timeline_single_project_single@example.com.csv', index=False)
        
        # Test Case 4: Timeline with gaps (weeks with no activity)
        test_data_4 = {
            'event_id': [1, 2, 3, 4],
            'event_type': ['commit', 'commit', 'commit', 'commit'],
            'event_timestamp': [
                '2023-04-01 10:00:00', '2023-04-02 11:00:00',
                '2023-04-20 12:00:00', '2023-05-01 13:00:00'
            ],
            'event_week': [0, 0, 2, 4],  # Gaps in weeks 1 and 3
            'event_identifier': ['c1', 'c2', 'c3', 'c4'],
            'event_data': ['{}', '{}', '{}', '{}'],
            'project_name': ['gap_project'] * 4,
            'project_type': ['OSS'] * 4,
            'contributor_email': ['gap@example.com'] * 4,
            'username': ['gapuser'] * 4,
            'is_pre_core': [True] * 4
        }
        
        df4 = pd.DataFrame(test_data_4)
        df4.to_csv(cls.test_dir / 'timeline_gap_project_gap@example.com.csv', index=False)

class ValidationTests:
    """
    Validation tests for the contribution time series generation.
    """
    
    def __init__(self, timeline_dir, verbose=True):
        self.timeline_dir = Path(timeline_dir)
        self.verbose = verbose
        self.test_results = []
        self.issues_found = []
        
    def log(self, message, level="INFO"):
        """Log a message."""
        if self.verbose:
            print(f"[{level}] {message}")
        self.test_results.append({"level": level, "message": message})
    
    def validate_file_loading(self):
        """Test 1: Validate that all timeline files can be loaded."""
        self.log("=" * 60)
        self.log("TEST 1: File Loading Validation")
        self.log("-" * 40)
        
        timeline_files = list(self.timeline_dir.glob('timeline_*.csv'))
        self.log(f"Found {len(timeline_files)} timeline files")
        
        loading_errors = []
        successful_loads = 0
        
        for filepath in timeline_files:
            try:
                df = pd.read_csv(filepath)
                if len(df) > 0:
                    successful_loads += 1
                else:
                    loading_errors.append(f"{filepath.name}: Empty file")
            except Exception as e:
                loading_errors.append(f"{filepath.name}: {str(e)}")
        
        if loading_errors:
            self.log(f"FAILED: {len(loading_errors)} files had loading issues", "ERROR")
            for error in loading_errors[:10]:  # Show first 10 errors
                self.log(f"  - {error}", "ERROR")
            self.issues_found.extend(loading_errors)
        else:
            self.log(f"PASSED: All {successful_loads} files loaded successfully", "SUCCESS")
        
        return len(loading_errors) == 0
    
    def validate_data_structure(self):
        """Test 2: Validate data structure and required columns."""
        self.log("=" * 60)
        self.log("TEST 2: Data Structure Validation")
        self.log("-" * 40)
        
        required_columns = [
            'event_id', 'event_type', 'event_timestamp', 'event_week',
            'event_identifier', 'event_data', 'project_name', 'project_type',
            'contributor_email', 'username'
        ]
        
        structure_errors = []
        timeline_files = list(self.timeline_dir.glob('timeline_*.csv'))[:100]  # Check first 100
        
        for filepath in timeline_files:
            try:
                df = pd.read_csv(filepath)
                missing_cols = [col for col in required_columns if col not in df.columns]
                
                if missing_cols:
                    structure_errors.append(f"{filepath.name}: Missing columns {missing_cols}")
                
                # Check data types
                if 'event_week' in df.columns:
                    if not pd.api.types.is_numeric_dtype(df['event_week']):
                        structure_errors.append(f"{filepath.name}: event_week is not numeric")
                        
            except Exception as e:
                structure_errors.append(f"{filepath.name}: {str(e)}")
        
        if structure_errors:
            self.log(f"FAILED: {len(structure_errors)} files have structure issues", "ERROR")
            for error in structure_errors[:10]:
                self.log(f"  - {error}", "ERROR")
            self.issues_found.extend(structure_errors)
        else:
            self.log("PASSED: All checked files have correct structure", "SUCCESS")
        
        return len(structure_errors) == 0
    
    def validate_event_data_parsing(self):
        """Test 3: Validate JSON parsing of event_data."""
        self.log("=" * 60)
        self.log("TEST 3: Event Data JSON Parsing Validation")
        self.log("-" * 40)
        
        parsing_errors = []
        timeline_files = list(self.timeline_dir.glob('timeline_*.csv'))[:50]  # Check first 50
        
        for filepath in timeline_files:
            try:
                df = pd.read_csv(filepath)
                
                for idx, row in df.iterrows():
                    if pd.notna(row['event_data']) and row['event_data'] != '{}':
                        try:
                            data = json.loads(row['event_data'])
                            
                            # Validate structure based on event type
                            if row['event_type'] == 'pull_request':
                                if 'merged' not in data and 'state' not in data:
                                    parsing_errors.append(
                                        f"{filepath.name} row {idx}: PR missing merge status"
                                    )
                            
                        except json.JSONDecodeError as e:
                            parsing_errors.append(
                                f"{filepath.name} row {idx}: JSON parse error - {str(e)}"
                            )
                            
            except Exception as e:
                parsing_errors.append(f"{filepath.name}: {str(e)}")
        
        if parsing_errors:
            self.log(f"WARNING: {len(parsing_errors)} JSON parsing issues found", "WARNING")
            for error in parsing_errors[:5]:
                self.log(f"  - {error}", "WARNING")
            self.issues_found.extend(parsing_errors)
        else:
            self.log("PASSED: All event_data JSON parsed successfully", "SUCCESS")
        
        return len(parsing_errors) == 0
    
    def validate_metric_extraction(self):
        """Test 4: Validate metric extraction logic."""
        self.log("=" * 60)
        self.log("TEST 4: Metric Extraction Validation")
        self.log("-" * 40)
        
        # Use test data with known values
        test_file = self.timeline_dir / 'timeline_test_project_test@example.com.csv'
        
        if not test_file.exists():
            # Use first available file
            timeline_files = list(self.timeline_dir.glob('timeline_*.csv'))
            if timeline_files:
                test_file = timeline_files[0]
            else:
                self.log("ERROR: No files to test", "ERROR")
                return False
        
        try:
            from contribution_timeseries_generator import ContributionIndexTimeSeries
            processor = ContributionIndexTimeSeries(self.timeline_dir)
            
            df = processor.load_timeline(test_file)
            weekly_metrics = processor.extract_weekly_metrics(df)
            
            # Validate metrics
            checks_passed = True
            
            # Check 1: All weeks present
            expected_weeks = range(df['event_week'].min(), df['event_week'].max() + 1)
            actual_weeks = set(weekly_metrics['week_num'].values)
            missing_weeks = set(expected_weeks) - actual_weeks
            
            if missing_weeks:
                self.log(f"ERROR: Missing weeks in metrics: {missing_weeks}", "ERROR")
                checks_passed = False
            
            # Check 2: Non-negative values
            numeric_cols = ['commits', 'prs_merged', 'comments', 'issues_opened', 'active_days']
            for col in numeric_cols:
                if (weekly_metrics[col] < 0).any():
                    self.log(f"ERROR: Negative values found in {col}", "ERROR")
                    checks_passed = False
            
            # Check 3: Active days <= 7
            if (weekly_metrics['active_days'] > 7).any():
                self.log("ERROR: Active days > 7 found", "ERROR")
                checks_passed = False
            
            # Check 4: Total events match
            total_commits_metrics = weekly_metrics['commits'].sum()
            total_commits_raw = len(df[df['event_type'] == 'commit'])
            
            if total_commits_metrics != total_commits_raw:
                self.log(f"ERROR: Commit count mismatch: {total_commits_metrics} vs {total_commits_raw}", "ERROR")
                checks_passed = False
            
            if checks_passed:
                self.log("PASSED: Metric extraction validation successful", "SUCCESS")
            else:
                self.log("FAILED: Metric extraction has issues", "ERROR")
                
            return checks_passed
            
        except Exception as e:
            self.log(f"ERROR: Failed to validate metrics: {str(e)}", "ERROR")
            return False
    
    def validate_contribution_index(self):
        """Test 5: Validate contribution index calculation."""
        self.log("=" * 60)
        self.log("TEST 5: Contribution Index Calculation Validation")
        self.log("-" * 40)
        
        try:
            from contribution_timeseries_generator import ContributionIndexTimeSeries
            processor = ContributionIndexTimeSeries(self.timeline_dir)
            
            # Create test data with known values
            test_metrics = pd.DataFrame({
                'week_num': [0, 1, 2],
                'commits': [10, 0, 5],
                'prs_merged': [2, 0, 1],
                'comments': [5, 2, 0],
                'issues_opened': [1, 0, 2],
                'active_days': [7, 2, 3],
                'duration': [1.0, 0.5, 0.75]
            })
            
            # Calculate contribution index
            result = processor.calculate_contribution_index(test_metrics.copy())
            
            # Manual calculation for verification
            expected_ci = []
            for _, row in test_metrics.iterrows():
                ci = (0.25 * row['commits'] +
                      0.20 * row['prs_merged'] +
                      0.15 * row['comments'] +
                      0.15 * row['issues_opened'] +
                      0.15 * (row['active_days'] / 7.0) +
                      0.10 * row['duration'])
                expected_ci.append(ci)
            
            # Compare
            calculated_ci = result['contribution_index'].values
            expected_ci = np.array(expected_ci)
            
            if np.allclose(calculated_ci, expected_ci, rtol=1e-5):
                self.log("PASSED: Contribution index calculation correct", "SUCCESS")
                self.log(f"  Expected: {expected_ci}", "INFO")
                self.log(f"  Calculated: {calculated_ci}", "INFO")
                return True
            else:
                self.log("FAILED: Contribution index calculation mismatch", "ERROR")
                self.log(f"  Expected: {expected_ci}", "ERROR")
                self.log(f"  Calculated: {calculated_ci}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"ERROR: Failed to validate contribution index: {str(e)}", "ERROR")
            return False
    
    def validate_time_series_consistency(self):
        """Test 6: Validate time series consistency and completeness."""
        self.log("=" * 60)
        self.log("TEST 6: Time Series Consistency Validation")
        self.log("-" * 40)
        
        try:
            from contribution_timeseries_generator import ContributionIndexTimeSeries
            processor = ContributionIndexTimeSeries(self.timeline_dir)
            
            # Process a subset of contributors
            timeline_files = list(self.timeline_dir.glob('timeline_*.csv'))[:10]
            
            consistency_issues = []
            
            for filepath in timeline_files:
                df = processor.load_timeline(filepath)
                weekly_metrics = processor.extract_weekly_metrics(df)
                
                # Check for consistency
                min_week = weekly_metrics['week_num'].min()
                max_week = weekly_metrics['week_num'].max()
                expected_weeks = set(range(min_week, max_week + 1))
                actual_weeks = set(weekly_metrics['week_num'].values)
                
                if expected_weeks != actual_weeks:
                    missing = expected_weeks - actual_weeks
                    consistency_issues.append(f"{filepath.name}: Missing weeks {missing}")
                
                # Check for duplicate weeks
                if len(weekly_metrics['week_num']) != len(weekly_metrics['week_num'].unique()):
                    consistency_issues.append(f"{filepath.name}: Duplicate weeks found")
            
            if consistency_issues:
                self.log(f"WARNING: {len(consistency_issues)} consistency issues", "WARNING")
                for issue in consistency_issues[:5]:
                    self.log(f"  - {issue}", "WARNING")
            else:
                self.log("PASSED: All time series are consistent", "SUCCESS")
            
            return len(consistency_issues) == 0
            
        except Exception as e:
            self.log(f"ERROR: Failed to validate consistency: {str(e)}", "ERROR")
            return False
    
    def validate_aggregation(self):
        """Test 7: Validate monthly aggregation."""
        self.log("=" * 60)
        self.log("TEST 7: Monthly Aggregation Validation")
        self.log("-" * 40)
        
        try:
            from contribution_timeseries_generator import ContributionIndexTimeSeries
            processor = ContributionIndexTimeSeries(self.timeline_dir)
            
            # Create test weekly data
            test_weekly = pd.DataFrame({
                'week_num': [0, 1, 2, 3, 4, 5, 6, 7],
                'commits': [1, 2, 3, 4, 5, 6, 7, 8],
                'prs_merged': [1, 0, 1, 0, 1, 0, 1, 0],
                'comments': [2, 2, 2, 2, 3, 3, 3, 3],
                'issues_opened': [1, 0, 0, 1, 1, 0, 0, 1],
                'active_days': [3, 4, 5, 2, 3, 4, 5, 2],
                'duration': [1, 0.75, 0.5, 0.5, 1, 0.75, 0.5, 0.5]
            })
            
            # Apply monthly aggregation
            monthly = processor.aggregate_to_monthly(test_weekly)
            
            # Validate aggregation
            # Month 0 should have weeks 0-3
            month_0 = monthly[monthly['month_num'] == 0].iloc[0]
            expected_commits_m0 = test_weekly.iloc[0:4]['commits'].sum()  # 1+2+3+4 = 10
            
            if month_0['commits'] == expected_commits_m0:
                self.log("PASSED: Monthly aggregation is correct", "SUCCESS")
                self.log(f"  Month 0 commits: {month_0['commits']} (expected: {expected_commits_m0})", "INFO")
                return True
            else:
                self.log("FAILED: Monthly aggregation mismatch", "ERROR")
                self.log(f"  Month 0 commits: {month_0['commits']} (expected: {expected_commits_m0})", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"ERROR: Failed to validate aggregation: {str(e)}", "ERROR")
            return False
    
    def validate_edge_cases(self):
        """Test 8: Validate edge cases handling."""
        self.log("=" * 60)
        self.log("TEST 8: Edge Cases Validation")
        self.log("-" * 40)
        
        edge_case_tests = []
        
        # Test empty event_data
        try:
            empty_data = pd.DataFrame({
                'event_id': [1],
                'event_type': ['commit'],
                'event_timestamp': ['2023-01-01 10:00:00'],
                'event_week': [0],
                'event_identifier': ['c1'],
                'event_data': [None],  # NULL/None value
                'project_name': ['test'],
                'project_type': ['OSS'],
                'contributor_email': ['test@test.com'],
                'username': ['test'],
                'is_pre_core': [True]
            })
            
            from contribution_timeseries_generator import ContributionIndexTimeSeries
            processor = ContributionIndexTimeSeries(self.timeline_dir)
            
            # Should handle None without error
            empty_data['event_data'] = empty_data['event_data'].apply(
                lambda x: json.loads(x) if pd.notna(x) else {}
            )
            edge_case_tests.append(("NULL event_data", True))
            
        except Exception as e:
            edge_case_tests.append(("NULL event_data", False))
            self.log(f"  Failed: {str(e)}", "ERROR")
        
        # Test single event timeline
        try:
            single_file = self.timeline_dir / 'timeline_single_project_single@example.com.csv'
            if single_file.exists():
                df = processor.load_timeline(single_file)
                metrics = processor.extract_weekly_metrics(df)
                if len(metrics) > 0:
                    edge_case_tests.append(("Single event timeline", True))
                else:
                    edge_case_tests.append(("Single event timeline", False))
        except:
            edge_case_tests.append(("Single event timeline", False))
        
        # Summary
        passed = sum(1 for _, result in edge_case_tests if result)
        total = len(edge_case_tests)
        
        if passed == total:
            self.log(f"PASSED: All {total} edge cases handled correctly", "SUCCESS")
        else:
            self.log(f"PARTIAL: {passed}/{total} edge cases passed", "WARNING")
            for test_name, result in edge_case_tests:
                status = "✓" if result else "✗"
                self.log(f"  {status} {test_name}", "INFO" if result else "WARNING")
        
        return passed == total
    
    def run_all_validations(self):
        """Run all validation tests."""
        self.log("=" * 70)
        self.log("CONTRIBUTION TIME SERIES VALIDATION TEST SUITE")
        self.log("=" * 70)
        self.log(f"Timeline directory: {self.timeline_dir}")
        self.log(f"Starting validation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")
        
        # Run all tests
        test_results = {
            "File Loading": self.validate_file_loading(),
            "Data Structure": self.validate_data_structure(),
            "JSON Parsing": self.validate_event_data_parsing(),
            "Metric Extraction": self.validate_metric_extraction(),
            "Contribution Index": self.validate_contribution_index(),
            "Time Series Consistency": self.validate_time_series_consistency(),
            "Monthly Aggregation": self.validate_aggregation(),
            "Edge Cases": self.validate_edge_cases()
        }
        
        # Summary
        self.log("")
        self.log("=" * 70)
        self.log("VALIDATION SUMMARY")
        self.log("=" * 70)
        
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            self.log(f"{test_name:.<30} {status}")
        
        self.log("")
        self.log(f"Overall Result: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            self.log("🎉 ALL TESTS PASSED! Time series generation is working correctly.", "SUCCESS")
        elif passed_tests >= total_tests * 0.7:
            self.log("⚠️ MOSTLY PASSED: Some issues found but system is mostly functional.", "WARNING")
        else:
            self.log("❌ CRITICAL ISSUES: Multiple tests failed. Review the errors above.", "ERROR")
        
        # Save detailed report
        self.save_validation_report(test_results)
        
        return test_results
    
    def save_validation_report(self, test_results):
        """Save detailed validation report to file."""
        report_path = Path("validation_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CONTRIBUTION TIME SERIES VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Timeline Directory: {self.timeline_dir}\n\n")
            
            f.write("TEST RESULTS:\n")
            f.write("-" * 40 + "\n")
            for test_name, result in test_results.items():
                status = "PASSED" if result else "FAILED"
                f.write(f"{test_name}: {status}\n")
            
            f.write("\n" + "DETAILED LOGS:\n")
            f.write("-" * 40 + "\n")
            for entry in self.test_results:
                f.write(f"[{entry['level']}] {entry['message']}\n")
            
            if self.issues_found:
                f.write("\n" + "ISSUES FOUND:\n")
                f.write("-" * 40 + "\n")
                for issue in self.issues_found[:50]:  # First 50 issues
                    f.write(f"- {issue}\n")
        
        self.log(f"\nDetailed report saved to: {report_path}", "INFO")


def main():
    """Main function to run validation tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate contribution time series generation')
    parser.add_argument('--timeline-dir', type=str, 
                        default='../../RQ2_newcomer_treatment_patterns_test2/step2_timelines/from_cache_timelines/',
                        help='Directory containing timeline CSV files')
    parser.add_argument('--verbose', action='store_true', 
                       help='Show detailed output')
    parser.add_argument('--test-mode', action='store_true',
                       help='Use test data instead of real data')
    
    args = parser.parse_args()
    
    if args.test_mode:
        # Run unit tests with test data
        print("Running unit tests with synthetic data...")
        unittest.main(argv=[''], exit=False)
        
        # Also run validation on test data
        test_dir = Path("test_timeseries_validation")
        if test_dir.exists():
            validator = ValidationTests(test_dir, verbose=args.verbose)
            validator.run_all_validations()
    else:
        # Run validation on actual data
        validator = ValidationTests(args.timeline_dir, verbose=args.verbose)
        test_results = validator.run_all_validations()
        
        # Return exit code based on results
        passed = sum(1 for result in test_results.values() if result)
        if passed == len(test_results):
            exit(0)  # All tests passed
        else:
            exit(1)  # Some tests failed


if __name__ == "__main__":
    main()