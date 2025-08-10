#!/usr/bin/env python3
"""
SETUP SCRIPT: Prepare Environment for Survival Analysis
This script sets up the directory structure and saves helper scripts.
Run this first to prepare everything.
"""

import os
import sys
from pathlib import Path

# Define base paths
BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
STEP8_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step8_survival_analysis"
SCRIPTS_DIR = STEP8_DIR / "scripts"
DATA_DIR = STEP8_DIR / "data"
RESULTS_DIR = STEP8_DIR / "results"
PLOTS_DIR = STEP8_DIR / "visualizations"


def create_directory_structure() -> bool:
    print("Creating directory structure...")
    directories = [STEP8_DIR, SCRIPTS_DIR, DATA_DIR, RESULTS_DIR, PLOTS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  Created/verified: {directory.relative_to(BASE_DIR)}")
    return True


def check_dependencies() -> bool:
    print("\nChecking Python dependencies...")
    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical operations',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical plots',
        'scipy': 'Statistical tests',
        'lifelines': 'Survival analysis',
        'sklearn': 'Machine learning utilities',
    }
    missing_packages: list[str] = []
    for package, description in required_packages.items():
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  {package:<12} - {description}")
        except ImportError:
            print(f"  MISSING  {package:<12} - {description}")
            missing_packages.append(package)
    if missing_packages:
        print("\nMissing packages detected. Install them using:")
        venv_path = BASE_DIR / ".venv/bin/pip"
        if venv_path.exists():
            print(f'  {venv_path} install {" ".join(missing_packages)}')
        else:
            print(f'  pip install {" ".join(missing_packages)}')
        return False
    return True


def check_input_data() -> bool:
    print("\nChecking input data...")
    step6_file = BASE_DIR / (
        "RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv"
    )
    if step6_file.exists():
        import pandas as pd
        df = pd.read_csv(step6_file)
        print(f"  Found transitions data: {len(df):,} contributor-project rows")
        required_cols = [
            'time_to_event_weeks',
            'became_core',
            'project_type',
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  Missing columns: {missing_cols}")
            print("  Re-run Step 6 generation if needed")
            return False
        print("  Required columns present")
        print(f"  Event rate: {df['became_core'].mean():.1%}")
        print(f"  Median observation time: {df['time_to_event_weeks'].median():.1f} weeks")
        return True
    else:
        print(f"  Transitions data not found at: {step6_file}")
        print("  Please run Step 6 first.")
        return False


def main() -> bool:
    print("=" * 60)
    print("SURVIVAL ANALYSIS SETUP")
    print("=" * 60)
    os.chdir(BASE_DIR)
    structure_ok = create_directory_structure()
    deps_ok = check_dependencies()
    data_ok = check_input_data()
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    print(f"  Directory structure: {'OK' if structure_ok else 'FAIL'}")
    print(f"  Dependencies: {'OK' if deps_ok else 'MISSING'}")
    print(f"  Input data: {'OK' if data_ok else 'MISSING'}")
    ready = structure_ok and deps_ok and data_ok
    if ready:
        print("\nREADY FOR SURVIVAL ANALYSIS")
    else:
        print("\nPlease fix the issues above before running analysis")
    return ready


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


