#!/usr/bin/env python3
"""
MASTER SCRIPT: Run Complete Survival Analysis Pipeline
This runs all survival analysis steps in sequence and generates a final summary report.
"""

import sys
import subprocess
from pathlib import Path
import time
import json

# Define paths
BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
STEP8_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step8_survival_analysis"
SCRIPTS_DIR = STEP8_DIR / "scripts"
RESULTS_DIR = STEP8_DIR / "results"

# Create necessary directories
for dir_path in [STEP8_DIR, SCRIPTS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and capture results."""
    print("\n" + "=" * 60)
    print(f"RUNNING: {description}")
    print("=" * 60)

    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        print("Please ensure all scripts are in the scripts/ directory")
        return False

    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)], capture_output=True, text=True, cwd=str(BASE_DIR)
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f"\n{description} completed in {elapsed:.1f} seconds")
            return True
        else:
            print(f"\n{description} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"\nError running {script_name}: {e}")
        return False


def generate_final_summary() -> bool:
    """Generate a comprehensive summary of all results."""
    print("\n" + "=" * 60)
    print("GENERATING FINAL SUMMARY")
    print("=" * 60)

    summary: dict = {'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'), 'pipeline_status': 'Complete', 'key_findings': {}}

    try:
        stats_file = RESULTS_DIR / "data_preparation_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                prep_stats = json.load(f)
            summary['data_stats'] = {
                'total_contributors': prep_stats.get('total_contributors'),
                'became_core': prep_stats.get('became_core'),
                'core_rate': prep_stats.get('core_rate'),
            }

        km_file = RESULTS_DIR / "kaplan_meier_results.json"
        if km_file.exists():
            with open(km_file) as f:
                km_results = json.load(f)
            summary['kaplan_meier'] = {
                'log_rank_p_value': km_results.get('log_rank_test', {}).get('p_value'),
                'median_survival_oss': km_results.get('median_survival', {}).get('OSS'),
                'median_survival_oss4sg': km_results.get('median_survival', {}).get('OSS4SG'),
            }

        cox_file = RESULTS_DIR / "cox_regression_results.json"
        if cox_file.exists():
            with open(cox_file) as f:
                cox_results = json.load(f)
            summary['cox_regression'] = {
                'hazard_ratio_oss4sg': cox_results.get('simple_model', {}).get('hazard_ratio_oss4sg'),
                'concordance': cox_results.get('simple_model', {}).get('concordance'),
            }

        val_file = RESULTS_DIR / "validation_results.json"
        if val_file.exists():
            with open(val_file) as f:
                val_results = json.load(f)
            summary['validation'] = {
                'errors': len(val_results.get('errors', [])),
                'warnings': len(val_results.get('warnings', [])),
            }

        if 'cox_regression' in summary and summary['cox_regression'].get('hazard_ratio_oss4sg'):
            hr = summary['cox_regression']['hazard_ratio_oss4sg']
            summary['key_findings']['main'] = f"OSS4SG contributors are {hr:.2f}x more likely to become core"

        if 'kaplan_meier' in summary:
            oss_med = summary['kaplan_meier'].get('median_survival_oss')
            oss4sg_med = summary['kaplan_meier'].get('median_survival_oss4sg')
            if oss_med and oss4sg_med:
                diff = oss_med - oss4sg_med
                summary['key_findings']['median_difference'] = (
                    f"OSS takes {diff:.1f} weeks longer to reach core"
                )

        summary_file = RESULTS_DIR / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        report_file = RESULTS_DIR / "FINAL_REPORT.txt"
        with open(report_file, 'w') as f:
            f.write("SURVIVAL ANALYSIS - FINAL REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {summary['analysis_date']}\n\n")
            f.write("KEY FINDINGS:\n")
            f.write("-" * 40 + "\n")
            for _, finding in summary.get('key_findings', {}).items():
                f.write(f"- {finding}\n")
            f.write("\n")
            if 'data_stats' in summary:
                f.write("DATA SUMMARY:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Contributors: {summary['data_stats']['total_contributors']:,}\n")
                f.write(f"Became Core: {summary['data_stats']['became_core']:,}\n")
                f.write(f"Core Rate: {summary['data_stats']['core_rate']:.1%}\n\n")
            if 'kaplan_meier' in summary:
                f.write("KAPLAN-MEIER RESULTS:\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"Log-rank test p-value: {summary['kaplan_meier'].get('log_rank_p_value', 'N/A')}\n"
                )
                f.write("Median time to core:\n")
                f.write(
                    f"  OSS: {summary['kaplan_meier'].get('median_survival_oss', 'Not reached')} weeks\n"
                )
                f.write(
                    f"  OSS4SG: {summary['kaplan_meier'].get('median_survival_oss4sg', 'Not reached')} weeks\n\n"
                )
            if 'cox_regression' in summary:
                f.write("COX REGRESSION RESULTS:\n")
                f.write("-" * 40 + "\n")
                hr_val = summary['cox_regression'].get('hazard_ratio_oss4sg')
                c_val = summary['cox_regression'].get('concordance')
                f.write(f"Hazard Ratio (OSS4SG): {hr_val:.3f}\n" if hr_val else "")
                f.write(f"Model Concordance: {c_val:.3f}\n\n" if c_val else "\n")
            if 'validation' in summary:
                f.write("VALIDATION SUMMARY:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Errors: {summary['validation']['errors']}\n")
                f.write(f"Warnings: {summary['validation']['warnings']}\n\n")
            f.write("=" * 60 + "\n")
            f.write("Analysis pipeline completed successfully\n")

        print(f"\nFinal report saved to: {report_file}")
        print(f"Summary JSON saved to: {summary_file}")

        print("\n" + "=" * 40)
        print("KEY FINDINGS:")
        print("=" * 40)
        for _, finding in summary.get('key_findings', {}).items():
            print(f"- {finding}")
    except Exception as e:
        print(f"Error generating summary: {e}")
        return False
    return True


def main() -> bool:
    print("=" * 60)
    print("SURVIVAL ANALYSIS PIPELINE")
    print("OSS vs OSS4SG Newcomer-to-Core Transitions")
    print("=" * 60)

    all_successful = True
    steps = [
        ("1_prepare_survival_data.py", "Step 1: Data Preparation"),
        ("2_kaplan_meier_analysis.py", "Step 2: Kaplan-Meier Analysis"),
        ("3_cox_regression.py", "Step 3: Cox Proportional Hazards Regression"),
        ("4_validate_results.py", "Step 4: Validation and Testing"),
    ]

    print("\nChecking for required scripts...")
    scripts_missing = False
    for script, _ in steps:
        if not (SCRIPTS_DIR / script).exists():
            print(f"  Missing: {script}")
            scripts_missing = True
        else:
            print(f"  Found: {script}")
    if scripts_missing:
        print("\nPlease copy all survival analysis scripts to:")
        print(f"  {SCRIPTS_DIR}")
        print("Scripts needed:")
        for script, _ in steps:
            print(f"  - {script}")
        return False

    print("\nStarting analysis pipeline...")
    start_time = time.time()
    for script, description in steps:
        success = run_script(script, description)
        if not success:
            all_successful = False
            print(f"\nPipeline stopped due to error in {description}")
            break

    if all_successful:
        generate_final_summary()

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    if all_successful:
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"  Total time: {total_time:.1f} seconds")
        print(f"  Results saved to: {RESULTS_DIR}")
        print(f"  Visualizations saved to: {STEP8_DIR / 'visualizations'}")
    else:
        print("PIPELINE FAILED")
        print("  Check error messages above")
    print("=" * 60)
    return all_successful


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


