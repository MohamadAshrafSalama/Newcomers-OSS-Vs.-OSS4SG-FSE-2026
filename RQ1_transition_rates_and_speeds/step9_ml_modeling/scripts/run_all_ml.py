#!/usr/bin/env python3
"""
MASTER: Run Step 9 ML Modeling Pipeline
"""

import sys
import subprocess
from pathlib import Path
import time

BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
STEP9_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step9_ml_modeling"
SCRIPTS_DIR = STEP9_DIR / "scripts"


def run(script):
    print("\n" + "=" * 60)
    print(f"RUNNING: {script}")
    print("=" * 60)
    t0 = time.time()
    res = subprocess.run([sys.executable, str(SCRIPTS_DIR / script)], cwd=str(BASE_DIR), text=True)
    ok = res.returncode == 0
    print(f"Completed in {time.time() - t0:.1f}s. {'OK' if ok else 'FAIL'}")
    return ok


def main():
    steps = [
        "1_prepare_ml_dataset.py",
        "2_train_and_evaluate_models.py",
        "3_feature_importance_and_correlations.py",
    ]
    for s in steps:
        if not run(s):
            print(f"FAILED at {s}")
            return 1
    print("\nAll ML steps completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


