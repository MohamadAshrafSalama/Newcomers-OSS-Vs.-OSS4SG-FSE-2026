#!/usr/bin/env python3
"""
QUICKSTART: Run Survival Analysis with One Command
This runs the complete survival analysis pipeline.
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
SCRIPTS_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step8_survival_analysis/scripts"


def main() -> bool:
    print("=" * 60)
    print("SURVIVAL ANALYSIS QUICKSTART")
    print("=" * 60)

    data_file = BASE_DIR / "RQ1_transition_rates_and_speeds/step8_survival_analysis/data/survival_data.csv"
    scripts = [
        "1_prepare_survival_data.py",
        "2_kaplan_meier_analysis.py",
        "3_cox_regression.py",
        "4_validate_results.py",
    ]

    scripts_to_run = scripts if not data_file.exists() else scripts[1:]
    if not data_file.exists():
        print("\nRunning complete analysis pipeline...")
    else:
        print("\nData already prepared. Running analysis only...")

    for script in scripts_to_run:
        script_path = SCRIPTS_DIR / script
        print(f"\nRunning: {script}")
        print("-" * 40)
        result = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))
        if result.returncode != 0:
            print(f"\nError in {script}")
            return False

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nResults saved in:")
    print(f"  {BASE_DIR / 'RQ1_transition_rates_and_speeds/step8_survival_analysis/results/'}")
    print("\nPlots saved in:")
    print(f"  {BASE_DIR / 'RQ1_transition_rates_and_speeds/step8_survival_analysis/visualizations/'}")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


