#!/usr/bin/env python3
"""
Step 4: Validate and Test Survival Analysis Results
This script validates that all survival analysis steps ran correctly and results make sense.
It performs sanity checks, consistency tests, and generates a validation report.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # noqa: F401  (kept for potential extensions)
import warnings

warnings.filterwarnings('ignore')

# Define paths
BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
STEP8_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step8_survival_analysis"
DATA_DIR = STEP8_DIR / "data"
RESULTS_DIR = STEP8_DIR / "results"
PLOTS_DIR = STEP8_DIR / "visualizations"


class SurvivalAnalysisValidator:
    """Comprehensive validator for survival analysis results."""

    def __init__(self):
        self.validation_results = {
            'data_checks': {},
            'statistical_checks': {},
            'consistency_checks': {},
            'interpretation_checks': {},
            'warnings': [],
            'errors': [],
        }

    def validate_data_preparation(self) -> bool:
        """Validate Step 1: Data Preparation."""
        print("\n=== Validating Data Preparation ===")

        main_file = DATA_DIR / "survival_data.csv"
        if not main_file.exists():
            self.validation_results['errors'].append("Main survival data file not found")
            return False

        df = pd.read_csv(main_file)

        checks = {
            'file_exists': main_file.exists(),
            'has_data': len(df) > 0,
            'has_required_columns': all(col in df.columns for col in ['duration', 'event', 'project_type']),
            'valid_event_values': set(df['event'].unique()).issubset({0, 1}),
            'positive_durations': (df['duration'] > 0).all(),
            'project_types_present': len(df['project_type'].unique()) >= 2,
        }
        self.validation_results['data_checks'] = checks

        print(f"OK: Data loaded: {len(df):,} contributors")
        print(f"OK: Event rate: {df['event'].mean():.1%}")
        print(f"OK: Project types: {df['project_type'].unique()}")

        # Balance warning
        type_counts = df['project_type'].value_counts()
        if len(type_counts) >= 2:
            balance_ratio = type_counts.min() / type_counts.max()
            if balance_ratio < 0.3:
                self.validation_results['warnings'].append(f"Imbalanced data: {balance_ratio:.2f}")

        # Outlier warning
        duration_outliers = df['duration'] > df['duration'].quantile(0.99)
        if duration_outliers.sum() > len(df) * 0.05:
            self.validation_results['warnings'].append(
                f"Many duration outliers: {int(duration_outliers.sum())}"
            )

        # Consistency with Step 6
        step6_file = BASE_DIR / "RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv"
        if step6_file.exists():
            original = pd.read_csv(step6_file)
            if len(original) != len(df):
                self.validation_results['warnings'].append(
                    f"Row count mismatch: {len(df)} vs {len(original)} original"
                )

        return all(checks.values())

    def validate_kaplan_meier(self) -> bool:
        """Validate Step 2: Kaplan-Meier Analysis."""
        print("\n=== Validating Kaplan-Meier Results ===")

        results_file = RESULTS_DIR / "kaplan_meier_results.json"
        if not results_file.exists():
            self.validation_results['errors'].append("Kaplan-Meier results not found")
            return False

        with open(results_file) as f:
            km_results = json.load(f)

        checks = {
            'has_log_rank': 'log_rank_test' in km_results,
            'has_medians': 'median_survival' in km_results,
            'significant_difference': km_results.get('log_rank_test', {}).get('p_value', 1) < 0.05,
            'has_timepoint_data': 'survival_at_timepoints' in km_results,
        }
        self.validation_results['statistical_checks']['kaplan_meier'] = checks

        if km_results.get('median_survival'):
            oss_median = km_results['median_survival'].get('OSS')
            oss4sg_median = km_results['median_survival'].get('OSS4SG')
            if oss_median and oss4sg_median:
                print(f"OK: Median survival: OSS={oss_median:.1f}, OSS4SG={oss4sg_median:.1f} weeks")

        if 'survival_at_timepoints' in km_results:
            for week_data in km_results['survival_at_timepoints'].values():
                if not (0 <= week_data['oss_survival'] <= 1):
                    self.validation_results['errors'].append("Invalid survival probability")

        for plot_file in ['survival_curves_main.png', 'cumulative_incidence.png']:
            if not (PLOTS_DIR / plot_file).exists():
                self.validation_results['warnings'].append(f"Missing plot: {plot_file}")

        print(f"OK: Log-rank p-value: {km_results.get('log_rank_test', {}).get('p_value', 'N/A')}")
        return all(checks.values())

    def validate_cox_regression(self) -> bool:
        """Validate Step 3: Cox Regression."""
        print("\n=== Validating Cox Regression Results ===")

        results_file = RESULTS_DIR / "cox_regression_results.json"
        if not results_file.exists():
            self.validation_results['errors'].append("Cox regression results not found")
            return False

        with open(results_file) as f:
            cox_results = json.load(f)

        simple = cox_results.get('simple_model', {})
        checks = {
            'has_hazard_ratio': 'hazard_ratio_oss4sg' in simple,
            'has_confidence_interval': 'ci_lower' in simple and 'ci_upper' in simple,
            'has_concordance': 'concordance' in simple,
            'valid_concordance': 0.5 <= simple.get('concordance', 0) <= 1.0,
        }
        self.validation_results['statistical_checks']['cox_regression'] = checks

        hr = simple.get('hazard_ratio_oss4sg', 0)
        if hr > 0:
            print(f"OK: Hazard ratio (OSS4SG): {hr:.3f}")
            if hr < 0.5 or hr > 5:
                self.validation_results['warnings'].append(f"Extreme hazard ratio: {hr:.3f}")

        full = cox_results.get('full_model', {})
        if 'concordance' in full:
            c_index = full['concordance']
            print(f"OK: Full model concordance: {c_index:.3f}")
            if c_index < 0.6:
                self.validation_results['warnings'].append(f"Low concordance: {c_index:.3f}")
            elif c_index > 0.9:
                self.validation_results['warnings'].append(
                    f"Very high concordance (possible overfitting): {c_index:.3f}"
                )

        if 'covariates' in full:
            print(f"OK: Number of covariates: {len(full['covariates'])}")
            for covar, stats in full['covariates'].items():
                if abs(stats.get('coefficient', 0)) > 10:
                    self.validation_results['warnings'].append(
                        f"Extreme coefficient for {covar}"
                    )

        return all(checks.values())

    def validate_consistency(self) -> bool:
        """Check consistency across different analyses."""
        print("\n=== Validating Consistency Across Analyses ===")
        consistency_checks: dict[str, bool] = {}
        try:
            with open(RESULTS_DIR / "kaplan_meier_results.json") as f:
                km_results = json.load(f)
            with open(RESULTS_DIR / "cox_regression_results.json") as f:
                cox_results = json.load(f)

            km_n = km_results.get('sample_sizes', {})
            total_km = sum(km_n.values())
            df = pd.read_csv(DATA_DIR / "survival_data.csv")
            consistency_checks['sample_size_match'] = abs(total_km - len(df)) < 10

            km_events = km_results.get('events', {})
            km_event_rate = sum(km_events.values()) / total_km if total_km > 0 else 0
            actual_event_rate = df['event'].mean()
            consistency_checks['event_rate_match'] = abs(km_event_rate - actual_event_rate) < 0.01

            km_medians = km_results.get('median_survival', {})
            cox_hr = cox_results.get('simple_model', {}).get('hazard_ratio_oss4sg', 1)
            oss_med = km_medians.get('OSS')
            oss4sg_med = km_medians.get('OSS4SG')
            # Only check direction if both medians are finite (not None/inf)
            if (
                oss_med is not None
                and oss4sg_med is not None
                and np.isfinite(oss_med)
                and np.isfinite(oss4sg_med)
            ):
                km_faster_oss4sg = oss4sg_med < oss_med
                cox_faster_oss4sg = cox_hr > 1
                consistency_checks['direction_consistency'] = km_faster_oss4sg == cox_faster_oss4sg
            else:
                consistency_checks['direction_consistency'] = True

            self.validation_results['consistency_checks'] = consistency_checks
            for check, passed in consistency_checks.items():
                status = "OK" if passed else "FAIL"
                print(f"{status}: {check}")

            if not all(consistency_checks.values()):
                self.validation_results['warnings'].append(
                    "Inconsistencies detected between analyses"
                )
        except Exception as e:
            self.validation_results['errors'].append(f"Error checking consistency: {e}")
            return False

        return all(consistency_checks.values())

    def validate_interpretation(self) -> bool:
        """Check if results make sense and are interpretable."""
        print("\n=== Validating Result Interpretation ===")
        interpretation_checks: dict[str, bool] = {}
        try:
            df = pd.read_csv(DATA_DIR / "survival_data.csv")
            with open(RESULTS_DIR / "kaplan_meier_results.json") as f:
                km_results = json.load(f)

            week_52 = km_results.get('survival_at_timepoints', {}).get('week_52', {})
            if week_52:
                oss_surv = week_52.get('oss_survival', 1)
                oss4sg_surv = week_52.get('oss4sg_survival', 1)
                interpretation_checks['oss4sg_faster'] = oss4sg_surv < oss_surv
                effect_size = abs(oss_surv - oss4sg_surv)
                interpretation_checks['meaningful_effect'] = effect_size > 0.05
                print(f"OK: At 52 weeks: OSS survival={oss_surv:.3f}, OSS4SG survival={oss4sg_surv:.3f}")
                print(f"OK: Effect size: {effect_size:.3f}")

            event_rate = df['event'].mean()
            interpretation_checks['reasonable_event_rate'] = 0.01 < event_rate < 0.5
            max_duration = df['duration'].max()
            interpretation_checks['reasonable_duration'] = 52 < max_duration < 520

            self.validation_results['interpretation_checks'] = interpretation_checks
            if all(interpretation_checks.values()):
                print("\nOK: Results are interpretable and align with research questions")
            else:
                print("\nReview: Some interpretation checks did not pass")
        except Exception as e:
            self.validation_results['errors'].append(f"Error validating interpretation: {e}")
            return False

        return len(self.validation_results['errors']) == 0

    def generate_validation_plots(self) -> None:
        """Generate diagnostic plots for validation."""
        print("\n=== Generating Validation Plots ===")
        try:
            df = pd.read_csv(DATA_DIR / "survival_data.csv")
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            ax = axes[0, 0]
            for event in [0, 1]:
                subset = df[df['event'] == event]['duration']
                label = 'Became Core' if event == 1 else 'Censored'
                ax.hist(subset, alpha=0.6, label=label, bins=30)
            ax.set_xlabel('Duration (weeks)')
            ax.set_ylabel('Count')
            ax.set_title('Duration Distribution by Outcome')
            ax.legend()

            ax = axes[0, 1]
            event_rates = df.groupby('project_type')['event'].mean()
            event_rates.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
            ax.set_ylabel('Event Rate')
            ax.set_title('Core Transition Rate by Type')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

            ax = axes[0, 2]
            type_counts = df['project_type'].value_counts()
            type_counts.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
            ax.set_ylabel('Count')
            ax.set_title('Sample Size by Type')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

            ax = axes[1, 0]
            oss_dur = df[df['project_type'] == 'OSS']['duration']
            oss4sg_dur = df[df['project_type'] == 'OSS4SG']['duration']
            ax.violinplot([oss_dur, oss4sg_dur], positions=[0, 1])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['OSS', 'OSS4SG'])
            ax.set_ylabel('Duration (weeks)')
            ax.set_title('Duration Distribution by Type')

            if 'total_commits' in df.columns:
                ax = axes[1, 1]
                for ptype, color in [('OSS', '#1f77b4'), ('OSS4SG', '#ff7f0e')]:
                    subset = df[df['project_type'] == ptype]
                    ax.scatter(
                        subset['duration'],
                        subset['total_commits'],
                        alpha=0.3,
                        s=10,
                        label=ptype,
                        color=color,
                    )
                ax.set_xlabel('Duration (weeks)')
                ax.set_ylabel('Total Commits')
                ax.set_title('Activity vs Duration')
                ax.set_yscale('log')
                ax.legend()

            ax = axes[1, 2]
            censoring_rate = 1 - df.groupby('project_type')['event'].mean()
            censoring_rate.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
            ax.set_ylabel('Censoring Rate')
            ax.set_title('Censoring Rate by Type')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

            plt.suptitle('Survival Analysis Validation Plots', fontsize=14, fontweight='bold')
            plt.tight_layout()
            save_path = PLOTS_DIR / "validation_diagnostics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved validation plots to: {save_path}")
            plt.close(fig)
        except Exception as e:
            self.validation_results['warnings'].append(
                f"Could not generate validation plots: {e}"
            )

    def generate_report(self) -> bool:
        """Generate comprehensive validation report."""
        print("\n=== Generating Validation Report ===")
        report_path = RESULTS_DIR / "validation_report.txt"
        with open(report_path, 'w') as f:
            f.write("SURVIVAL ANALYSIS VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            n_errors = len(self.validation_results['errors'])
            n_warnings = len(self.validation_results['warnings'])
            f.write("STATUS: ")
            f.write("ANALYSIS VALID\n\n" if n_errors == 0 else "ERRORS FOUND\n\n")
            f.write(f"Errors: {n_errors}\n")
            f.write(f"Warnings: {n_warnings}\n\n")
            f.write("DATA PREPARATION CHECKS:\n")
            for check, passed in self.validation_results['data_checks'].items():
                status = "OK" if passed else "FAIL"
                f.write(f"  {status} {check}\n")
            f.write("\n")
            f.write("STATISTICAL CHECKS:\n")
            for analysis, checks in self.validation_results['statistical_checks'].items():
                f.write(f"  {analysis}:\n")
                for check, passed in checks.items():
                    status = "OK" if passed else "FAIL"
                    f.write(f"    {status} {check}\n")
            f.write("\n")
            f.write("CONSISTENCY CHECKS:\n")
            for check, passed in self.validation_results['consistency_checks'].items():
                status = "OK" if passed else "FAIL"
                f.write(f"  {status} {check}\n")
            f.write("\n")
            f.write("INTERPRETATION CHECKS:\n")
            for check, passed in self.validation_results['interpretation_checks'].items():
                status = "OK" if passed else "FAIL"
                f.write(f"  {status} {check}\n")
            f.write("\n")
            if self.validation_results['errors']:
                f.write("ERRORS:\n")
                for error in self.validation_results['errors']:
                    f.write(f"  - {error}\n")
                f.write("\n")
            if self.validation_results['warnings']:
                f.write("WARNINGS:\n")
                for warning in self.validation_results['warnings']:
                    f.write(f"  - {warning}\n")
                f.write("\n")
            f.write("RECOMMENDATIONS:\n")
            if n_errors > 0:
                f.write("  - Fix errors before proceeding with analysis\n")
            if n_warnings > 0:
                f.write("  - Review warnings and determine if action needed\n")
            if n_errors == 0 and n_warnings == 0:
                f.write("  - Analysis appears valid and ready for publication\n")
        print(f"Saved validation report to: {report_path}")

        # Save sanitized JSON
        def _safe(obj):
            if isinstance(obj, dict):
                return {str(k): _safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_safe(v) for v in obj]
            if isinstance(obj, (np.generic,)):
                return obj.item()
            return obj

        json_path = RESULTS_DIR / "validation_results.json"
        with open(json_path, 'w') as f:
            json.dump(_safe(self.validation_results), f, indent=2)
        print(f"Saved validation results to: {json_path}")
        return len(self.validation_results['errors']) == 0


def main() -> bool:
    print("=" * 60)
    print("STEP 4: SURVIVAL ANALYSIS VALIDATION")
    print("=" * 60)

    validator = SurvivalAnalysisValidator()
    checks_passed = []
    checks_passed.append(validator.validate_data_preparation())
    checks_passed.append(validator.validate_kaplan_meier())
    checks_passed.append(validator.validate_cox_regression())
    checks_passed.append(validator.validate_consistency())
    checks_passed.append(validator.validate_interpretation())
    validator.generate_validation_plots()
    all_valid = validator.generate_report()

    print("\n" + "=" * 60)
    if all_valid and all(checks_passed):
        print("SURVIVAL ANALYSIS VALIDATION COMPLETE - ALL CHECKS PASSED")
    else:
        print("VALIDATION COMPLETE - REVIEW WARNINGS AND ERRORS")
    print("=" * 60)
    return all_valid


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)


